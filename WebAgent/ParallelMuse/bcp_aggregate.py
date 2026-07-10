"""ParallelMuse stage 2 (compressed reasoning aggregation) for BrowseComp-Plus.

Adapted from compressed_reasoning_aggregation.py:

- Fixes two latent bugs in the released script (get_llm_response and
  call_converge were invoked with an extra positional argument).
- The report/integration LLM is any OpenAI-compatible endpoint
  (same flags as bcp_partial_rollout.py); by default use the same model that
  produced the rollouts.
- Emits one BCP-evaluator-compatible run_*.json per question:
  query_id / status / retrieved_docids / result[output_text], with
  retriever + method provenance in metadata. retrieved_docids is the union
  over all parallel trajectories of the question — both the docids recorded
  by stage 1 and any docids parsed from tool_response texts inside reused
  prefixes (partial rollouts inherit searches from their initial rollout).

Prompts (REPORT_PROMPT / INTEGRATE_PROMPT / CONVERGE_SYSTEM_PROMPT) are
verbatim from the original.

Usage:
  python bcp_aggregate.py \
    --rollout-file ./bcp_results/queries_deepseek_deepseek-v4-flash_..._.jsonl \
    --output-dir ./bcp_results/aggregated \
    --llm-model deepseek/deepseek-v4-flash
"""

import argparse
import asyncio
import datetime
import json
import os
import random
import re
import sys
import traceback
from collections import defaultdict
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_INFERENCE = os.path.abspath(os.path.join(_HERE, "..", "..", "inference"))
sys.path.insert(0, _REPO_INFERENCE)

from baseline_retriever import load_retriever_config  # noqa: E402


CONVERGE_SYSTEM_PROMPT = """Follow the user's instructions and present your answer in the format they request.
""".strip()

REPORT_PROMPT = """Given the following problem-solving trajectory:
{traj}

Your task is to distill this trajectory into a concise yet sufficiently detailed problem-solving report.
You must only use the information contained within the provided trajectory — no additional or external information is allowed.

The report must include:

1. **Solution Planning**: Identify how the main problem is decomposed into subproblems, and describe the sequence and dependency relationships among those subproblems.
2. **Solution Methods**: For each subproblem, indicate which tools were invoked to solve it, the parameters used in those tool calls, and any resulting partial answers that contributed directly or indirectly to progress toward the final answer.
   _Do not repeat the full output of the tools; only include the specific fragments of tool results that were essential in deriving the subanswers._
3. **Final Reasoning**: Clearly outline the reasoning process by which the subproblems and their associated subanswers led to the derivation of the final answer.

Additional requirements:
- The report must remain **concise** and **focused**.
- Remove any content unrelated to problem-solving or any ineffective tool calls.
- Ensure the final report has clear logical structure, with each step traceable and analyzable.

Finally, present the complete report in **Markdown format**, and wrap the entire report content within <report> </report> tags.
""".strip()

INTEGRATE_PROMPT = """You are tasked with solving the question: {question}.

Multiple independent teams have provided detailed process reports describing their approaches to solving this problem. As the final analyst, your role is to consolidate these reports, carefully examine the problem-solving methods they contain, and identify the key information obtained in each.

Your goal is to produce a final answer that perfectly resolves the question. Note that some of the reports may contain inconsistencies — you must critically evaluate which report(s) are reasonable and trustworthy.

If multiple reports reach the same conclusion, this increases the likelihood that the conclusion is correct; however, this is not guaranteed. You must still carefully verify and reflect to ensure that the final selected answer is truly the most accurate possible.

Wrap your final answer in <answer> </answer> tags.

Important:
- Every question has a definitive, certain answer.
- You are not allowed to decline answering on the grounds of uncertainty.
- For any report that does not provide a clear and definite final answer, its confidence level should be significantly reduced.
- You must ultimately select one report as having the most correct answer.
- You are not allowed to call or use any external tools for verification. You must rely solely on the information already provided, conduct in-depth analysis, and then produce the final answer.
- You are not allowed to merge multiple different answers, nor are you allowed to produce an overly broad answer that attempts to encompass all candidate answers — such an answer should be eliminated first.
- You do not need to restate or summarize the reports; instead, provide a short-form answer that directly answers the question.

Below is the content of these reports:
""".strip()


_DOCID_LINE = re.compile(r"^docid: (.+)$", re.MULTILINE)
_CHUNKID_LINE = re.compile(r"^chunk_id: (.+)$", re.MULTILINE)


def read_jsonl(file_path):
    result = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                result.append(json.loads(line))
    return result


def cluster_by_question(dataset):
    cluster = defaultdict(list)
    for item in dataset:
        cluster[item["question"]].append(item)
    return list(cluster.values())


def collect_retrieved_ids(traj_group):
    """Union of retrieved docids/chunk_ids across all trajectories of a question.

    Sources: (a) stage-1 recorded lists (new searches made by each rollout),
    (b) docid:/chunk_id: lines inside tool_response messages — this recovers
    searches inherited via reused prefixes in partial rollouts.
    """
    docids, chunk_ids = set(), set()
    for traj in traj_group:
        docids.update(traj.get("retrieved_docids", []))
        chunk_ids.update(traj.get("retrieved_chunk_ids", []))
        for msg in traj.get("rollout", []):
            content = msg.get("content") or ""
            if msg.get("role") == "user" and "<tool_response>" in content:
                docids.update(_DOCID_LINE.findall(content))
                chunk_ids.update(_CHUNKID_LINE.findall(content))
    return sorted(docids), sorted(chunk_ids)


def construct_interaction_from_record(record):
    interaction = ""
    for r in record:
        content = r.get("content") or ""
        if r["role"] == "assistant" and "<tool_call>" in content and "</tool_call>" in content:
            thinking = content.split("<think>")[-1].split("</think>")[0].strip()
            tool_call = content.split("<tool_call>")[-1].split("</tool_call>")[0].strip()
            interaction += f"**{r['role']}:**\n*Thinking:* {thinking}\n*Tool Call:* {tool_call}\n\n"
        elif r["role"] == "user" and "<tool_response>" in content and "</tool_response>" in content:
            tool_response = content.split("<tool_response>")[-1].split("</tool_response>")[0].strip()
            interaction += f"**Tool Response:**\n{tool_response}\n\n"
        elif r["role"] == "assistant":
            thinking = content.split("<think>")[-1].split("</think>")[0].strip()
            prediction = content.split("</think>")[-1].split("<answer>")[-1].split("</answer>")[0].strip()
            if thinking == prediction:
                prediction = "[No Prediction]"
            interaction += f"**{r['role']}:**\n*Thinking:* {thinking}\n*Answer:* {prediction}\n\n"
        else:
            interaction += f"**{r['role']}:**\n{content}\n\n"
    return interaction.strip()


async def get_llm_response(args, messages, max_tokens):
    client = AsyncOpenAI(
        base_url=random.choice(args.llm_base_url_pool),
        api_key=args.llm_api_key,
    )
    try:
        response = await client.chat.completions.create(
            model=args.llm_model,
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=args.temperature,
            top_p=args.top_p,
            presence_penalty=args.presence_penalty,
            extra_body=args.extra_body_dict or None,
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return content
        raise ValueError("empty completion")
    except Exception as e:  # noqa: BLE001
        print(f"[REPORT CONVERGE CLIENT ERROR]: {e}")
        await asyncio.sleep(2)
        if "time out" in str(e).lower():
            return "[Request time out]"
    return "[Error getting converge response]"


async def call_state_report(args, sem, traj, max_retries=10):
    max_tokens = args.report_max_tokens
    if traj.get("prediction", "[No Prediction]") == "[No Prediction]":
        return "[Error getting state report]"

    interaction = construct_interaction_from_record(traj["rollout"])
    user_input = REPORT_PROMPT.format(traj=interaction)

    response = None
    async with sem:
        for _retry in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": CONVERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ]
                response = await get_llm_response(args, messages, max_tokens)
                response = response.split("</think>")[-1].strip()
                if "[Error getting converge response]" in response or "[Request time out]" in response:
                    raise ValueError(response)
                break
            except Exception as e:  # noqa: BLE001
                await asyncio.sleep(2)
                if "time out" not in str(e).lower():
                    max_tokens = max(int(max_tokens / 2), 1024)
                response = None

    if response is None:
        return "[Error getting state report]"
    return response.split("<report>")[-1].split("</report>")[0].strip()


async def call_info_integrate(args, sem, question, report_group, max_retries=10):
    max_tokens = args.integrate_max_tokens
    user_input = INTEGRATE_PROMPT.format(question=question)

    report_group = [r for r in report_group if r != "[Error getting state report]"]
    if len(report_group) == 0:
        return 0, "[No Valid Answer]"

    for i, report in enumerate(report_group):
        user_input += f"\n\n[Report {i + 1}]: {report}"

    response = None
    async with sem:
        for _retry in range(max_retries):
            try:
                messages = [
                    {"role": "system", "content": CONVERGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_input},
                ]
                response = await get_llm_response(args, messages, max_tokens)
                response = response.split("</think>")[-1].strip()
                if "[Error getting converge response]" in response or "[Request time out]" in response:
                    raise ValueError(response)
                break
            except Exception as e:  # noqa: BLE001
                await asyncio.sleep(2)
                if "time out" not in str(e).lower():
                    max_tokens = max(int(max_tokens / 2), 1024)
                response = None

    if response is None:
        return len(report_group), "[Error getting integrated answer]"

    final_answer = response.split("<answer>")[-1].split("</answer>")[0].strip()
    return len(report_group), final_answer


async def call_converge(args, sem, traj_group):
    question = traj_group[0]["question"]
    answer = traj_group[0].get("answer", "")
    query_id = next((t.get("query_id") for t in traj_group if t.get("query_id")), "")

    report_group = []
    for traj in traj_group:
        report = await call_state_report(args, sem["report"], traj)
        report_group.append(report)

    merge_num, prediction = await call_info_integrate(args, sem["merge"], question, report_group)
    if merge_num == 0:
        prediction = "[No Prediction]"

    retrieved_docids, retrieved_chunk_ids = collect_retrieved_ids(traj_group)
    return {
        "question": question,
        "answer": answer,
        "query_id": query_id,
        "prediction": prediction,
        "merge_num": merge_num,
        "report_group": report_group,
        "retrieved_docids": retrieved_docids,
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "n_trajectories": len(traj_group),
        "total_search_calls": sum(t.get("search_call_count", 0) for t in traj_group),
        "traj_terminations": [t.get("termination", "") for t in traj_group],
    }


def persist_bcp_output(output_dir: Path, result: dict, args, retriever_meta: dict):
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = output_dir / f"run_{ts}.json"

    prediction = result["prediction"]
    ok = prediction not in ("[No Prediction]", "[No Valid Answer]", "[Error getting integrated answer]")
    output_data = {
        "metadata": {
            "method": "parallelmuse_compressed_reasoning_aggregation",
            "model": args.llm_model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
            "n_trajectories": result["n_trajectories"],
            "merge_num": result["merge_num"],
            **retriever_meta,
        },
        "query_id": result["query_id"] or None,
        "question": result["question"],
        "tool_call_counts": {"search": result["total_search_calls"]},
        "tool_call_counts_all": {"search": result["total_search_calls"]},
        "status": "completed" if ok else "no_valid_answer",
        "retrieved_docids": result["retrieved_docids"],
        "retrieved_chunk_ids": result["retrieved_chunk_ids"],
        "result": [
            {"type": "output_text", "tool_name": None, "arguments": None, "output": prediction}
        ],
        "traj_terminations": result["traj_terminations"],
    }
    if args.store_reports:
        output_data["report_group"] = result["report_group"]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {result['query_id'] or result['question'][:40]!r} -> {filename}")


def collect_processed_query_ids(output_dir: Path) -> set:
    done = set()
    if not output_dir.exists():
        return done
    for json_path in output_dir.glob("run_*.json"):
        try:
            with json_path.open(encoding="utf-8") as jf:
                data = json.load(jf)
            if data.get("query_id") and data.get("status") == "completed":
                done.add(str(data["query_id"]))
        except Exception:
            continue
    return done


async def main(args):
    sem = {
        "report": asyncio.Semaphore(args.max_report_workers),
        "merge": asyncio.Semaphore(args.max_merge_workers),
    }

    retriever_meta = {}
    try:
        cfg = load_retriever_config(args.retriever_config)
        retriever_meta = {
            "retriever": cfg.get("retriever_name", "deepsearch_edi_baseline_milvus"),
            "collection_name": cfg.get("milvus", {}).get("collection_name"),
            "mode": cfg.get("retrieval", {}).get("mode"),
            "top_k": cfg.get("retrieval", {}).get("top_k"),
            "baseline": cfg.get("retrieval", {}).get("baseline"),
            "add_instruction": cfg.get("retrieval", {}).get("add_instruction"),
        }
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load retriever config for provenance: {e}")

    dataset = read_jsonl(args.rollout_file)
    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)

    done_query_ids = collect_processed_query_ids(output_dir)

    tasks = []
    skipped = 0
    for cluster in cluster_by_question(dataset):
        query_id = next((t.get("query_id") for t in cluster if t.get("query_id")), "")
        if query_id and query_id in done_query_ids:
            skipped += 1
            continue
        filtered = [t for t in cluster if t.get("prediction") and t["prediction"] != "[No Prediction]"]
        if not filtered:
            filtered = cluster  # still emit a no_valid_answer record for the evaluator
        tasks.append(call_converge(args, sem, filtered))

    print(f"Questions: {len(tasks)} to aggregate, {skipped} already done")

    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Converging ..."):
        try:
            result = await future
            persist_bcp_output(output_dir, result, args, retriever_meta)
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR]: {type(e).__name__}: {e}\n{''.join(traceback.format_tb(e.__traceback__))}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-file", type=str, required=True,
                        help="Stage-1 output jsonl (all trajectories, all questions)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for per-question BCP evaluator run_*.json files")
    parser.add_argument("--retriever-config", type=str,
                        default=os.path.join(_REPO_INFERENCE, "configs", "bcp_baseline_retriever.json"))

    parser.add_argument("--llm-model", type=str, required=True)
    parser.add_argument("--llm-base-url", type=str, action="append", dest="llm_base_url_pool")
    parser.add_argument("--llm-api-key-env", type=str, default="OPENROUTER_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--report_max_tokens", type=int, default=32 * 1024)
    parser.add_argument("--integrate_max_tokens", type=int, default=64 * 1024)
    parser.add_argument("--extra-body", type=str, default="",
                        help='JSON passed as extra_body, e.g. \'{"reasoning": {"enabled": true}}\'')

    parser.add_argument("--max_report_workers", type=int, default=16)
    parser.add_argument("--max_merge_workers", type=int, default=8)
    parser.add_argument("--store-reports", action="store_true",
                        help="Include the compressed reports in the output JSON")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if not args.llm_base_url_pool:
        args.llm_base_url_pool = ["https://openrouter.ai/api/v1"]
    args.llm_api_key = os.environ.get(args.llm_api_key_env, "") or "EMPTY"
    args.extra_body_dict = json.loads(args.extra_body) if args.extra_body else None
    asyncio.run(main(args))
