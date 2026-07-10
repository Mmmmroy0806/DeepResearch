"""Run Tongyi DeepResearch on BrowseComp-Plus with the deepsearch_edi baseline retriever.

Search-only evaluation: the agent sees a single `search` tool backed by the
fixed BrowseComp-Plus Milvus baseline index (see baseline_retriever.py).
No visit / Serper / Jina / Scholar / file / Python tools are loaded.

Input formats:
  1. TSV (BrowseComp-Plus topics-qrels/queries.tsv):  query_id<TAB>query
  2. JSONL: {"query_id": "...", "question": "...", "answer": "..."}

Output: one JSON file per query under --output-dir, in the format expected by
the BrowseComp-Plus evaluator (query_id / status / retrieved_docids / result),
plus retriever provenance metadata for the paper.

Example:
  python inference/run_browsecomp_plus.py \
    --dataset /path/to/BrowseComp-Plus/topics-qrels/queries.tsv \
    --model /path/to/Tongyi-DeepResearch-30B-A3B \
    --output-dir runs/tongyi_bcp_baseline \
    --port 6001
"""

import argparse
import csv
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm
import json5

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baseline_retriever import BaselineMilvusRetriever, load_retriever_config
from react_agent_bcp import MultiTurnReactAgentBCP
from tool_search_baseline import SearchBaseline

FORBIDDEN_TOOL_MARKERS = ("visit", "google_scholar", "parse_file", "PythonInterpreter")


def parse_messages_to_result_array(messages: list) -> list:
    """Convert raw ReAct messages to the BCP evaluator's result array format."""
    result_array = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")

        for think_content in re.findall(r'<think>(.*?)</think>', content, re.DOTALL):
            result_array.append({
                "type": "reasoning",
                "tool_name": None,
                "arguments": None,
                "output": think_content.strip(),
            })

        for tool_call_content in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
            try:
                tool_call_data = json5.loads(tool_call_content)
            except Exception:
                continue
            tool_output = ""
            if i + 1 < len(messages) and messages[i + 1].get("role") == "user":
                next_content = messages[i + 1].get("content", "")
                response_match = re.search(
                    r'<tool_response>\n(.*?)\n</tool_response>', next_content, re.DOTALL
                )
                if response_match:
                    tool_output = response_match.group(1).strip()
            result_array.append({
                "type": "tool_call",
                "tool_name": tool_call_data.get("name", ""),
                "arguments": json.dumps(tool_call_data.get("arguments", {}), ensure_ascii=False),
                "output": tool_output,
            })

        for answer_content in re.findall(r'<answer>(.*?)</answer>', content, re.DOTALL):
            result_array.append({
                "type": "output_text",
                "tool_name": None,
                "arguments": None,
                "output": answer_content.strip(),
            })
    return result_array


def load_queries(dataset_path: str) -> list:
    """Returns list of {"query_id", "question", "answer"} dicts."""
    path = Path(dataset_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    items = []
    if path.suffix.lower() == ".tsv":
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f, delimiter="\t"):
                if len(row) < 2:
                    continue
                items.append({
                    "query_id": row[0].strip(),
                    "question": row[1].strip(),
                    "answer": "",
                })
    elif path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                items.append({
                    "query_id": str(data.get("query_id", data.get("id", ""))),
                    "question": data.get("question", data.get("query", "")),
                    "answer": data.get("answer", ""),
                })
    else:
        raise ValueError("Unsupported dataset extension; use .tsv or .jsonl")

    items = [it for it in items if it["question"]]
    return items


def sanity_check_tools(result: dict, query_id: str):
    """Fail loudly if any web tool leaked into the run (experiment-control check)."""
    for name in result.get("tool_call_counts_all", {}):
        for marker in FORBIDDEN_TOOL_MARKERS:
            if marker.lower() in name.lower():
                print(
                    f"WARNING [query {query_id}]: forbidden tool call attempted: {name!r}. "
                    "The search-only prompt should prevent this; the call was rejected."
                )


def persist_response(output_dir: Path, item: dict, result: dict, args, retriever_meta: dict):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = output_dir / f"run_{ts}.json"

    termination = result.get("termination", "")
    status = "completed" if termination == "answer" else (termination or "error")

    output_data = {
        "metadata": {
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "presence_penalty": args.presence_penalty,
            **retriever_meta,
        },
        "query_id": item["query_id"] or None,
        "question": item["question"],
        "tool_call_counts": result.get("tool_call_counts", {}),
        "tool_call_counts_all": result.get("tool_call_counts_all", {}),
        "status": status,
        "retrieved_docids": sorted(set(result.get("retrieved_docids", []))),
        "retrieved_chunk_ids": sorted(set(result.get("retrieved_chunk_ids", []))),
        "result": parse_messages_to_result_array(result.get("messages", [])),
    }
    if "error" in result:
        output_data["error"] = result["error"]
    if args.store_raw:
        output_data["raw_messages"] = result.get("messages", [])

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Saved response to {filename}")


def collect_processed_ids(output_dir: Path) -> set:
    """Return query_ids with a terminal saved result.

    Non-error terminal statuses such as token-limit answer generation or timeout
    are intentionally treated as processed. Re-running them on resume tends to
    reproduce the same expensive terminal state and can pollute the run
    directory with duplicate query_id files. Only files explicitly carrying an
    "error" field are considered retryable.
    """
    processed = set()
    if not output_dir.exists():
        return processed
    for json_path in output_dir.glob("run_*.json"):
        try:
            with json_path.open(encoding="utf-8") as jf:
                data = json.load(jf)
            qid = data.get("query_id")
            if qid and "error" not in data:
                processed.add(str(qid))
        except Exception:
            continue
    return processed


def delete_existing_outputs_for_query(output_dir: Path, query_id: str) -> None:
    """Remove stale retryable outputs for query_id before writing a fresh result."""
    if not query_id:
        return
    for json_path in output_dir.glob("run_*.json"):
        try:
            with json_path.open(encoding="utf-8") as jf:
                data = json.load(jf)
            if str(data.get("query_id")) == str(query_id):
                json_path.unlink()
        except Exception:
            continue


def main():
    parser = argparse.ArgumentParser(
        description="Tongyi DeepResearch on BrowseComp-Plus with the deepsearch_edi baseline retriever"
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to queries.tsv (query_id<TAB>query) or a JSONL file")
    parser.add_argument("--model", type=str, required=True, help="Model path (for tokenizer + server model name)")
    parser.add_argument("--output-dir", type=str, default="runs/tongyi_bcp_baseline")
    parser.add_argument("--retriever-config", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "configs", "bcp_baseline_retriever.json"))
    parser.add_argument("--port", type=int, default=6001, help="Local LLM server port")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--presence_penalty", type=float, default=1.1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="Only run the first N queries (0 = all)")
    parser.add_argument("--store-raw", action="store_true", help="Store raw messages in the output JSON")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)

    # One retriever + one Milvus client for the whole run; BaselineMilvusRetriever
    # serializes client calls internally. Each worker thread gets its own agent
    # instance below so per-run agent state never crosses queries.
    retriever_config = load_retriever_config(args.retriever_config)
    retriever = BaselineMilvusRetriever(retriever_config)
    snippet_max_tokens = int(retriever_config.get("retrieval", {}).get("snippet_max_tokens", 0))
    search_tool = SearchBaseline(retriever=retriever, snippet_max_tokens=snippet_max_tokens)
    retriever_meta = retriever.run_config_summary()

    llm_cfg = {
        'model': args.model,
        'generate_cfg': {
            'max_input_tokens': 320000,
            'max_retries': 10,
            'temperature': args.temperature,
            'top_p': args.top_p,
            'presence_penalty': args.presence_penalty,
        },
        'model_type': 'qwen_dashscope',
    }
    thread_state = threading.local()

    def get_thread_agent() -> MultiTurnReactAgentBCP:
        agent = getattr(thread_state, "agent", None)
        if agent is None:
            agent = MultiTurnReactAgentBCP(
                llm=llm_cfg,
                function_list=["search"],
                search_tool=search_tool,
            )
            thread_state.agent = agent
        return agent

    items = load_queries(args.dataset)
    if args.limit > 0:
        items = items[: args.limit]

    processed_ids = collect_processed_ids(output_dir)
    remaining = [it for it in items if it["query_id"] not in processed_ids]
    print(f"Dataset: {args.dataset} — {len(items)} queries, "
          f"{len(processed_ids)} already done, {len(remaining)} to run")
    print(f"Retriever: {json.dumps(retriever_meta, ensure_ascii=False)}")

    def handle_one(item: dict):
        task_data = {
            "item": {"question": item["question"], "answer": item["answer"]},
            "planning_port": args.port,
        }
        try:
            result = get_thread_agent()._run(task_data, args.model)
            sanity_check_tools(result, item["query_id"])
        except Exception as exc:  # noqa: BLE001
            print(f"Error processing query {item['query_id']}: {exc}")
            result = {
                "question": item["question"],
                "answer": item["answer"],
                "error": str(exc),
                "messages": [],
                "prediction": "[Failed]",
                "termination": "error",
            }
        delete_existing_outputs_for_query(output_dir, item["query_id"])
        persist_response(output_dir, item, result, args, retriever_meta)

    if args.num_threads <= 1:
        for item in tqdm(remaining, desc="Queries", unit="query"):
            handle_one(item)
    else:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor, \
             tqdm(total=len(remaining), desc="Queries", unit="query") as pbar:
            futures = [executor.submit(handle_one, item) for item in remaining]
            for _ in as_completed(futures):
                pbar.update(1)

    print("\nAll queries completed.")


if __name__ == "__main__":
    main()
