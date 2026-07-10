"""ParallelMuse stage 1 (functionality-specified partial rollout) for BrowseComp-Plus.

Adapted from functionality_specified_partial_rollout.py with three changes:

1. Fixed retriever. The only tool is `search`, backed by the deepsearch_edi
   baseline Milvus retriever (inference/baseline_retriever.py). No visit /
   Serper / Jina / live web. Retrieval params are locked in the retriever
   config; every trajectory records the docids/chunk_ids it retrieved.

2. Model-agnostic LLM endpoint. The agent LLM is any OpenAI-compatible
   endpoint (OpenRouter, local vLLM, ...) selected via --llm-base-url /
   --llm-model / --llm-api-key-env, e.g. deepseek-v4-flash on OpenRouter.

3. Robustness fixes needed off the Qwen/vLLM happy path:
   - logprobs are optional: if the provider does not return them, the step
     gets step_ppl=None and branch selection falls back to evenly spaced
     tool-call steps (deterministic) instead of crashing.
   - <think>/<tool_call> spans are located by character offsets mapped onto
     token boundaries, not by assuming each tag is a single token.
   - reasoning-model output in the `reasoning` field is folded back into the
     recorded content as <think>...</think> (PPL spans still use raw content).
   - max_tokens degradation uses int(); messages are sanitized to
     {role, content} before each API call.

Method logic (branch-point selection math, sampling budget accounting,
prompts minus the removed tools, sampling params) is kept identical to the
original so results remain comparable to the paper's method description.

Usage (stage A: initial trajectory-level rollouts):
  python bcp_partial_rollout.py \
    --qa_file_path /path/to/BrowseComp-Plus/topics-qrels/queries.tsv \
    --output_dir ./bcp_results \
    --llm-model deepseek/deepseek-v4-flash \
    --partial_sampling_mode none --sampling_budget 2

Stage B (uncertainty-guided partial rollouts, after stage A):
  python bcp_partial_rollout.py ... \
    --partial_sampling_mode tool_call_ppl \
    --initial_rollout_num 2 --partial_sampling_topk 2 \
    --partial_sampling_times_per_pos 1 --sampling_budget 8
"""

import argparse
import asyncio
import copy
import csv
import datetime
import json
import math
import os
import re
import sys
import time
import traceback
from collections import Counter

import json5
import numpy as np
from openai import AsyncOpenAI
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_INFERENCE = os.path.abspath(os.path.join(_HERE, "..", "..", "inference"))
sys.path.insert(0, _REPO_INFERENCE)

from baseline_retriever import BaselineMilvusRetriever, load_retriever_config  # noqa: E402


def today_date():
    return datetime.date.today().strftime("%Y-%m-%d")


SYSTEM_PROMPT = """You are a deep research assistant. Your core function is to conduct thorough, multi-step investigations into any topic by searching a fixed knowledge corpus. For every request, synthesize information from the retrieved evidence snippets to deliver a comprehensive, accurate, and objective response. When you have gathered sufficient information and are ready to provide the definitive response, you must enclose the entire final answer within <answer></answer> tags.

# Tools

You may call the search tool one or multiple times to assist with the user query. All evidence comes from a fixed BrowseComp-Plus knowledge corpus; there is no web browsing.

You are provided with the search tool, its signature within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "search", "description": "Search the fixed BrowseComp-Plus corpus with the baseline retriever. It returns top relevant evidence snippets.", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query."}}, "required": ["query"]}}}
</tools>

For each search tool call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": "search", "arguments": {"query": "your query"}}
</tool_call>

Current date: """


# ---------------------------------------------------------------------------
# Search tool: async wrapper around the shared baseline retriever
# ---------------------------------------------------------------------------

class BCPSearchTool:
    """Search-only tool over the fixed baseline retriever.

    call() returns (response_text, docids, chunk_ids); docids is None when the
    call failed (invalid arguments / retrieval error), mirroring the contract
    used by inference/tool_search_baseline.py.
    """

    def __init__(self, retriever: BaselineMilvusRetriever):
        self.retriever = retriever

    def _format_query_result(self, entry: dict) -> str:
        query, hits = entry["query"], entry["hits"]
        if not hits:
            return f"No results found for '{query}'. Try with a more general query."
        formatted = []
        for hit in hits:
            title = hit["title"] or hit["docid"]
            formatted.append(
                f"{hit['rank']}. [{title}]\n"
                f"docid: {hit['docid']}\n"
                f"chunk_id: {hit['chunk_id']}\n"
                f"score: {hit['score']:.4f}\n"
                f"{hit['text']}"
            )
        return (
            f"A search for '{query}' found {len(hits)} results:\n\n"
            "## Search Results\n" + "\n\n".join(formatted)
        )

    async def call(self, tool_args: dict):
        query = tool_args.get("query") if isinstance(tool_args, dict) else None
        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list) and query and all(isinstance(q, str) for q in query):
            queries = query
        else:
            return (
                "[Search] Invalid request format: Input must be a JSON object "
                "containing a string 'query' field",
                None,
                None,
            )
        try:
            entries = await asyncio.to_thread(self.retriever.search, queries)
        except Exception as e:  # noqa: BLE001
            return f"[Search] Retrieval error: {e}", None, None

        docids, chunk_ids = [], []
        for entry in entries:
            for hit in entry["hits"]:
                docids.append(hit["docid"])
                chunk_ids.append(hit["chunk_id"])
        text = "\n=======\n".join(self._format_query_result(e) for e in entries)
        return text, docids, chunk_ids


# ---------------------------------------------------------------------------
# LLM call with optional logprobs + char-offset PPL spans
# ---------------------------------------------------------------------------

_rr_index = 0


def get_next_base_url(pool):
    global _rr_index
    url = pool[_rr_index % len(pool)]
    _rr_index += 1
    return url


def sanitize_messages(messages):
    """Strip bookkeeping keys (step_ppl, timings, ...) before the API call."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def char_span_to_token_span(tokens, text, start_char, end_char):
    """Map a [start_char, end_char) span of text onto (start_tok, end_tok) indices.

    tokens must concatenate to text. Returns None if the span cannot be mapped.
    """
    offsets = []
    pos = 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok)
    if pos != len(text):
        return None
    start_tok = end_tok = None
    for i, (s, e) in enumerate(offsets):
        if start_tok is None and e > start_char:
            start_tok = i
        if e >= end_char:
            end_tok = i
            break
    if start_tok is None or end_tok is None:
        return None
    return start_tok, end_tok


def compute_step_ppl(result_text, result_tokens, result_toplogprobs):
    """Same entropy->PPL statistic as the original, with char-offset spans."""
    all_entropies = []
    for toplogprobs in result_toplogprobs:
        logprob_values = np.array([tlp.logprob for tlp in toplogprobs], dtype=np.float64)
        probs = np.exp(logprob_values)
        probs = probs / probs.sum()
        entropy = -np.sum(probs * logprob_values)
        all_entropies.append(entropy)
    entropies = np.array(all_entropies, dtype=np.float64)

    def span_ppl(open_tag, close_tag):
        start_char = result_text.find(open_tag)
        end_char = result_text.find(close_tag)
        if start_char == -1 or end_char == -1 or end_char <= start_char:
            return -1
        span = char_span_to_token_span(
            result_tokens, result_text, start_char + len(open_tag), end_char
        )
        if span is None:
            return -1
        start_tok, end_tok = span  # end_tok is inclusive (token containing end_char)
        if end_tok < start_tok:
            return -1
        return float(np.exp(np.mean(entropies[start_tok:end_tok + 1])))

    if "<tool_call>" in result_text:
        return {
            "think_ppl": span_ppl("<think>", "</think>"),
            "tool_call_ppl": span_ppl("<tool_call>", "</tool_call>"),
            "all_ppl": float(np.exp(np.mean(entropies))) if len(entropies) else -1,
        }
    return {"think_ppl": -1, "tool_call_ppl": -1, "all_ppl": -1}


async def call_llm(sem, messages, args):
    max_tokens = args.max_completion_tokens

    async with sem:
        for _retry in range(10):
            base_url = get_next_base_url(args.llm_base_url_pool)
            client = AsyncOpenAI(api_key=args.llm_api_key, base_url=base_url)
            try:
                response = await client.chat.completions.create(
                    model=args.llm_model,
                    messages=sanitize_messages(messages),
                    stop=["\n<tool_response>", "<tool_response>"],
                    temperature=args.temperature,
                    top_p=args.top_p,
                    presence_penalty=args.presence_penalty,
                    logprobs=True,
                    top_logprobs=16,
                    max_tokens=int(max_tokens),
                    extra_body=args.extra_body_dict or None,
                )
                choice = response.choices[0]
                result_text = choice.message.content or ""
                if not result_text.strip():
                    raise ValueError("empty completion")
                break
            except Exception as e:  # noqa: BLE001
                print(f"[{args.llm_model} async error] {e}")
                if "time out" not in str(e).lower():
                    max_tokens = max(int(max_tokens / 2), 1024)
        else:
            return None, None

        # PPL from logprobs when the provider returns them; None otherwise.
        step_ppl = None
        logprobs = getattr(choice, "logprobs", None)
        logprobs_content = getattr(logprobs, "content", None) if logprobs else None
        if logprobs_content:
            tokens = [item.token for item in logprobs_content]
            toplogprobs = [item.top_logprobs for item in logprobs_content]
            if any(toplogprobs):
                try:
                    step_ppl = compute_step_ppl(result_text, tokens, toplogprobs)
                except Exception as e:  # noqa: BLE001
                    print(f"[step_ppl error] {e}")
                    step_ppl = None

        # Reasoning-model support (e.g. deepseek/OpenRouter): fold the
        # `reasoning` field into the record as <think> so trajectory records
        # and stage-2 reports see the thinking. PPL above used raw content.
        reasoning = getattr(choice.message, "reasoning", None)
        if reasoning and "<think>" not in result_text:
            result_text = f"<think>\n{reasoning.strip()}\n</think>\n{result_text}"

    return result_text, step_ppl


def count_tokens_approx(messages):
    """Char/4 approximation; no local checkpoint tokenizer required."""
    return sum(len(m.get("content") or "") for m in messages) // 4


# ---------------------------------------------------------------------------
# Rollout (method logic identical to the original)
# ---------------------------------------------------------------------------

async def rollout_single_traj(llm_sem, tool_sem, search_tool, data, messages, args,
                              max_turn_given=None, rollout_type="traj_level_rollout"):
    max_context_length = args.max_context_length
    max_turn = int(args.max_turn if not max_turn_given else max_turn_given)

    question = data["question"]
    answer = data.get("answer", "")
    query_id = data.get("query_id", "")

    record = copy.deepcopy(messages)

    termination = "max_turn_exceeded"
    prediction = "[No Prediction]"
    retrieved_docids = []
    retrieved_chunk_ids = []
    search_call_count = 0

    def result(term):
        return {
            "question": question,
            "answer": answer,
            "query_id": query_id,
            "prediction": prediction,
            "rollout": record,
            "termination": term,
            "prepand_msg": messages,
            "rollout_type": rollout_type,
            "retrieved_docids": retrieved_docids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "search_call_count": search_call_count,
        }

    for _turn in range(max_turn):
        if count_tokens_approx(record) > max_context_length:
            return result("max_length_exceeded")

        llm_response, step_ppl = await call_llm(llm_sem, record, args)
        if llm_response is None:
            return result("llm_error_occurred")

        record.append({"role": "assistant", "content": llm_response, "step_ppl": step_ppl})

        if "<tool_call>" in llm_response and "</tool_call>" in llm_response:
            tool_call_str = llm_response.split("<tool_call>")[-1].split("</tool_call>")[0]
            try:
                tool_call = json5.loads(tool_call_str)
                tool_name = tool_call["name"]
                tool_args = tool_call["arguments"]

                if tool_name == "search":
                    async with tool_sem["search"]:
                        tool_response, docids, chunk_ids = await search_tool.call(tool_args)
                    if docids is not None:
                        search_call_count += 1
                        retrieved_docids.extend(docids)
                        retrieved_chunk_ids.extend(chunk_ids)
                else:
                    tool_response = (
                        f"Tool {tool_name} does not exist. Only `search` is available."
                    )
            except Exception as e:  # noqa: BLE001
                tool_response = (
                    'Error: Tool call is not a valid JSON. Tool call must contain a '
                    'valid "name" and "arguments" field.'
                )
                print(f"Tool call error {e}")

            record.append({"role": "user", "content": f"<tool_response>\n{tool_response}\n</tool_response>"})
        else:
            prediction = llm_response.strip()
            prediction = prediction.split("<answer>")[-1].split("</answer>")[0].strip()
            return result("answer")

    return result(termination)


# ---------------------------------------------------------------------------
# Branch-point selection (original math + deterministic no-logprobs fallback)
# ---------------------------------------------------------------------------

def _valid_ppl_steps(rollout, mode):
    steps = []
    for i, msg in enumerate(rollout):
        ppl = msg.get("step_ppl") or None
        if ppl and ppl.get(mode, -1) > 0:
            steps.append({"step_id": i, "step_ppl": ppl[mode]})
    return steps


def _fallback_steps(rollout, topk):
    """Deterministic fallback when the provider returned no logprobs:
    evenly spaced assistant tool-call steps."""
    tool_call_steps = [
        i for i, msg in enumerate(rollout)
        if msg["role"] == "assistant" and "<tool_call>" in (msg.get("content") or "")
    ]
    if not tool_call_steps:
        return []
    if len(tool_call_steps) <= topk:
        chosen = tool_call_steps
    else:
        idxs = np.linspace(0, len(tool_call_steps) - 1, topk).round().astype(int)
        chosen = [tool_call_steps[i] for i in sorted(set(idxs.tolist()))]
    return [{"step_id": i, "step_ppl": -1} for i in chosen]


def branch_high_uncertainty_steps(rollout, partial_sampling_topk, partial_sampling_mode):
    if partial_sampling_mode != "mixed_ppl":
        branch_step = _valid_ppl_steps(rollout, partial_sampling_mode)
        branch_step = sorted(branch_step, key=lambda x: x["step_ppl"], reverse=True)[:partial_sampling_topk]
        if not branch_step:
            branch_step = _fallback_steps(rollout, partial_sampling_topk)
        return branch_step

    tool_call_topk = math.ceil(partial_sampling_topk / 2)
    think_topk = math.floor(partial_sampling_topk / 2)

    tool_call_steps = sorted(
        _valid_ppl_steps(rollout, "tool_call_ppl"), key=lambda x: x["step_ppl"], reverse=True
    )[:tool_call_topk]
    think_steps = sorted(
        _valid_ppl_steps(rollout, "think_ppl"), key=lambda x: x["step_ppl"], reverse=True
    )[:think_topk]

    branch_step = tool_call_steps + think_steps
    if not branch_step:
        branch_step = _fallback_steps(rollout, partial_sampling_topk)
    return branch_step


# ---------------------------------------------------------------------------
# Dataset loading (BCP queries.tsv or jsonl) + IO helpers
# ---------------------------------------------------------------------------

def read_jsonl(file_path):
    result = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def load_dataset(path):
    """Returns list of {question, answer, query_id}."""
    if path.endswith(".tsv"):
        items = []
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) >= 2:
                    items.append({
                        "query_id": row[0].strip(),
                        "question": row[1].strip(),
                        "answer": "",
                    })
        return items
    items = []
    for data in read_jsonl(path):
        items.append({
            "query_id": str(data.get("query_id", data.get("id", ""))),
            "question": data.get("question", data.get("query", "")),
            "answer": data.get("answer", ""),
        })
    return [it for it in items if it["question"]]


def get_initial_rollouts(question, existing_rollouts, initial_rollout_num):
    all_rollouts = [item for item in existing_rollouts if item["question"] == question]
    initial_rollouts = []
    for rollout in all_rollouts:
        if rollout["termination"] == "answer":
            initial_rollouts.append(rollout)
        if len(initial_rollouts) == initial_rollout_num:
            break
    if len(initial_rollouts) != initial_rollout_num:
        for rollout in all_rollouts:
            if rollout not in initial_rollouts and rollout["termination"] not in (
                "max_length_exceeded", "llm_error_occurred"
            ):
                initial_rollouts.append(rollout)
            if len(initial_rollouts) == initial_rollout_num:
                break
    if len(initial_rollouts) != initial_rollout_num:
        for rollout in all_rollouts:
            if rollout not in initial_rollouts and rollout["termination"] != "llm_error_occurred":
                initial_rollouts.append(rollout)
            if len(initial_rollouts) == initial_rollout_num:
                break
    return initial_rollouts


# ---------------------------------------------------------------------------
# Main (same two-phase structure as the original)
# ---------------------------------------------------------------------------

async def main(args, search_tool, retriever_meta):
    llm_sem = asyncio.Semaphore(args.max_llm_workers)
    tool_sem = {"search": asyncio.Semaphore(args.max_search_workers)}

    dataset = load_dataset(args.qa_file_path)
    if args.limit > 0:
        dataset = dataset[: args.limit]
    dataset_tag = os.path.basename(args.qa_file_path).rsplit(".", 1)[0]
    model_tag = args.llm_model.replace("/", "_")

    full_traj_rollout_output_file_path = os.path.join(
        args.output_dir, f"{dataset_tag}_{model_tag}_1_none_initial_rollout.jsonl"
    )
    if args.partial_sampling_mode == "none":
        output_file_path = full_traj_rollout_output_file_path
    else:
        output_file_path = os.path.join(
            args.output_dir,
            f"{dataset_tag}_{model_tag}_{args.initial_rollout_num}_{args.partial_sampling_mode}"
            f"_{args.partial_sampling_topk}_{args.partial_sampling_rounds}"
            f"_{args.partial_sampling_times_per_pos}.jsonl",
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Retriever: {json.dumps(retriever_meta, ensure_ascii=False)}")
    print(f"Output: {output_file_path}")

    existing_rollouts = []
    visited_counter = Counter()
    if os.path.exists(full_traj_rollout_output_file_path):
        existing_rollouts = read_jsonl(full_traj_rollout_output_file_path)
        for visited_data in existing_rollouts:
            # Errored trajectories don't count toward the budget: a transient
            # API outage must not leave a question permanently short of usable
            # rollouts (stage B asserts on this count). Re-running stage A
            # regenerates them.
            if visited_data.get("termination") != "llm_error_occurred":
                visited_counter[visited_data["question"]] += 1

    # resume partial rollout (generic: dataset size from the input file itself)
    fully_visited_question = []
    visited_initial_rollouts = []
    if os.path.exists(output_file_path) and output_file_path != full_traj_rollout_output_file_path:
        initial_num = len(dataset)
        visited_rollouts = read_jsonl(output_file_path)
        visited_initial_rollouts = visited_rollouts[: initial_num * args.initial_rollout_num]
        visited_rollouts = visited_rollouts[initial_num * args.initial_rollout_num:]

        visited_rollouts_counter = Counter()
        for visited_data in visited_rollouts:
            visited_rollouts_counter[visited_data["question"]] += 1

        fully_visited_question = [
            q for q, count in visited_rollouts_counter.items()
            if count == args.sampling_budget - args.initial_rollout_num
        ]
        fully_visited_rollouts = [r for r in visited_rollouts if r["question"] in fully_visited_question]

        os.remove(output_file_path)
        with open(output_file_path, "a") as f:
            for r in visited_initial_rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            for r in fully_visited_rollouts:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tasks = []
    if args.partial_sampling_mode == "none":
        pending_counter = Counter()
        for data in dataset:
            question = data["question"]
            need_to_submit = args.sampling_budget - visited_counter[question] - pending_counter[question]
            for _ in range(need_to_submit):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT + str(today_date())},
                    {"role": "user", "content": question},
                ]
                tasks.append(rollout_single_traj(llm_sem, tool_sem, search_tool, data, messages, args))
                pending_counter[question] += 1
    else:
        sampling_budget = args.sampling_budget
        initial_rollout_num = args.initial_rollout_num
        partial_sampling_topk = args.partial_sampling_topk
        partial_sampling_rounds = args.partial_sampling_rounds
        partial_sampling_times_per_pos = args.partial_sampling_times_per_pos

        assert sampling_budget >= initial_rollout_num + initial_rollout_num * partial_sampling_topk * \
            partial_sampling_rounds * partial_sampling_times_per_pos, "Sampling budget is not set correctly."
        assert all(visited_counter[d["question"]] >= initial_rollout_num for d in dataset), \
            "Initial rollouts are not sufficient; run --partial_sampling_mode none first."

        for data in tqdm(dataset, desc="Detecting branching point ..."):
            question = data["question"]
            if question in fully_visited_question:
                continue

            if not visited_initial_rollouts:
                initial_rollouts = get_initial_rollouts(question, existing_rollouts, initial_rollout_num)
                with open(output_file_path, "a") as f:
                    for r in initial_rollouts:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                initial_rollouts = [r for r in visited_initial_rollouts if r["question"] == question]

            assert len(initial_rollouts) == initial_rollout_num, "Initial rollouts are not sufficient."

            for _sampling_round in range(partial_sampling_rounds):
                for r in initial_rollouts:
                    tmp_tasks_count = 1

                    partial_completion_count = None
                    branch_step = branch_high_uncertainty_steps(
                        r["rollout"], partial_sampling_topk, args.partial_sampling_mode
                    )
                    if len(branch_step) != partial_sampling_topk:
                        partial_completion_count = len(branch_step)

                    tasks.extend(
                        rollout_single_traj(
                            llm_sem, tool_sem, search_tool, data,
                            r["rollout"][: int(b["step_id"])],
                            args, args.max_turn - (int(b["step_id"]) - 2) / 2,
                            "partial_rollout",
                        )
                        for b in branch_step
                        for _times in range(partial_sampling_times_per_pos)
                    )
                    tmp_tasks_count += len(branch_step) * partial_sampling_times_per_pos

                    sampling_budget_per_initial_rollout = int(sampling_budget / initial_rollout_num)
                    if sampling_budget_per_initial_rollout > 1 + partial_sampling_topk * \
                            partial_sampling_rounds * partial_sampling_times_per_pos:
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT + str(today_date())},
                            {"role": "user", "content": question},
                        ]
                        completed = (partial_completion_count if partial_completion_count is not None
                                     else partial_sampling_topk)
                        extra = (sampling_budget_per_initial_rollout - 1 -
                                 completed * partial_sampling_rounds * partial_sampling_times_per_pos)
                        tasks.extend(
                            rollout_single_traj(llm_sem, tool_sem, search_tool, data, messages, args,
                                                None, "traj_level_rollout")
                            for _count in range(extra)
                        )
                        tmp_tasks_count += extra

                    if partial_completion_count is None:
                        assert tmp_tasks_count == sampling_budget_per_initial_rollout, "Sampling budget mismatch."

    print(f"Total number of tasks: {len(tasks)}")

    error_log_path = os.path.join(args.output_dir, "inference_error.txt")
    with open(output_file_path, "a") as f, open(error_log_path, "a") as log:
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="No Blocking Rollout ..."):
            try:
                result = await future
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            except Exception as e:  # noqa: BLE001
                error_message = (
                    f"{type(e).__name__}: {e}\n"
                    f"Traceback:\n{''.join(traceback.format_tb(e.__traceback__))}"
                )
                print(f"[ERROR]: {error_message}")
                log.write(f"[ERROR]: {error_message}\n\n")
                log.flush()
                os.fsync(log.fileno())


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--qa_file_path", type=str, required=True,
                        help="BCP queries.tsv (query_id<TAB>query) or JSONL with question/answer/query_id")
    parser.add_argument("--output_dir", type=str, default="./bcp_results")
    parser.add_argument("--retriever-config", type=str,
                        default=os.path.join(_REPO_INFERENCE, "configs", "bcp_baseline_retriever.json"))

    # LLM endpoint (OpenRouter / any OpenAI-compatible server)
    parser.add_argument("--llm-model", type=str, required=True,
                        help="e.g. deepseek/deepseek-v4-flash (OpenRouter) or a local served model name")
    parser.add_argument("--llm-base-url", type=str, action="append", dest="llm_base_url_pool",
                        help="Repeatable. Default: https://openrouter.ai/api/v1")
    parser.add_argument("--llm-api-key-env", type=str, default="OPENROUTER_API_KEY",
                        help="Env var holding the API key ('EMPTY' is used if the var is unset)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--max_completion_tokens", type=int, default=32 * 1024)
    parser.add_argument("--extra-body", type=str, default="",
                        help='JSON passed as extra_body, e.g. \'{"reasoning": {"enabled": true}}\'')
    parser.add_argument("--limit", type=int, default=0,
                        help="Only run the first N questions (0 = all); for smoke tests")

    # rollout
    parser.add_argument("--max_llm_workers", type=int, default=16)
    parser.add_argument("--max_search_workers", type=int, default=8)
    parser.add_argument("--sampling_budget", type=int, default=8)
    parser.add_argument("--max_turn", type=int, default=100)
    parser.add_argument("--max_context_length", type=int, default=100 * 1024,
                        help="Approximate token budget (chars/4) for a trajectory")

    # partial sampling
    parser.add_argument("--initial_rollout_num", type=int, default=1)
    parser.add_argument("--partial_sampling_mode", type=str, default="tool_call_ppl",
                        choices=["none", "all_ppl", "think_ppl", "tool_call_ppl", "mixed_ppl"])
    parser.add_argument("--partial_sampling_topk", type=int, default=2)
    parser.add_argument("--partial_sampling_rounds", type=int, default=1)
    parser.add_argument("--partial_sampling_times_per_pos", type=int, default=3)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if not args.llm_base_url_pool:
        args.llm_base_url_pool = ["https://openrouter.ai/api/v1"]
    args.llm_api_key = os.environ.get(args.llm_api_key_env, "") or "EMPTY"
    args.extra_body_dict = json.loads(args.extra_body) if args.extra_body else None

    retriever = BaselineMilvusRetriever(load_retriever_config(args.retriever_config))
    search_tool = BCPSearchTool(retriever)

    asyncio.run(main(args, search_tool, retriever.run_config_summary()))
