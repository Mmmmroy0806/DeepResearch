"""Convert stage-1 trajectory jsonl into BCP-evaluator run_*.json files.

Purpose: the single-trajectory ReAct control entry. Running
bcp_partial_rollout.py with --partial_sampling_mode none --sampling_budget 1
produces exactly one ReAct trajectory per question with the same model, same
prompt, and same fixed retriever as the ParallelMuse entry; this script turns
those trajectories into per-question evaluator files so the ReAct baseline and
the ParallelMuse entry share an identical evaluation path.

Trajectory choice per question: the first trajectory with termination ==
"answer" (file order), else the first non-error trajectory. With
sampling_budget=1 there is only one.

Usage:
  python bcp_traj_to_eval.py \
    --rollout-file ./bcp_results/queries_deepseek_deepseek-v4-flash_1_none_initial_rollout.jsonl \
    --output-dir ./bcp_results/react_baseline_eval \
    --model-label deepseek/deepseek-v4-flash
"""

import argparse
import datetime
import json
import os
from pathlib import Path

from bcp_aggregate import (  # noqa: E402
    cluster_by_question,
    collect_retrieved_ids,
    collect_processed_query_ids,
    read_jsonl,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_INFERENCE = os.path.abspath(os.path.join(_HERE, "..", "..", "inference"))


def load_retriever_meta(config_path):
    try:
        import sys
        sys.path.insert(0, _REPO_INFERENCE)
        from baseline_retriever import load_retriever_config
        cfg = load_retriever_config(config_path)
        return {
            "retriever": cfg.get("retriever_name", "deepsearch_edi_baseline_milvus"),
            "collection_name": cfg.get("milvus", {}).get("collection_name"),
            "mode": cfg.get("retrieval", {}).get("mode"),
            "top_k": cfg.get("retrieval", {}).get("top_k"),
            "baseline": cfg.get("retrieval", {}).get("baseline"),
            "add_instruction": cfg.get("retrieval", {}).get("add_instruction"),
        }
    except Exception as e:  # noqa: BLE001
        print(f"Warning: could not load retriever config for provenance: {e}")
        return {}


def pick_trajectory(cluster):
    for traj in cluster:
        if traj.get("termination") == "answer":
            return traj
    for traj in cluster:
        if traj.get("termination") != "llm_error_occurred":
            return traj
    return cluster[0] if cluster else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-label", type=str, default="",
                        help="Model id recorded in metadata (defaults to the rollout filename)")
    parser.add_argument("--retriever-config", type=str,
                        default=os.path.join(_REPO_INFERENCE, "configs", "bcp_baseline_retriever.json"))
    args = parser.parse_args()

    model_label = args.model_label or os.path.basename(args.rollout_file)
    retriever_meta = load_retriever_meta(args.retriever_config)

    output_dir = Path(args.output_dir).expanduser().resolve()
    os.makedirs(output_dir, exist_ok=True)
    done = collect_processed_query_ids(output_dir)

    dataset = read_jsonl(args.rollout_file)
    written = skipped = 0
    for cluster in cluster_by_question(dataset):
        traj = pick_trajectory(cluster)
        if traj is None:
            continue
        query_id = traj.get("query_id", "")
        if query_id and query_id in done:
            skipped += 1
            continue

        prediction = traj.get("prediction", "[No Prediction]")
        termination = traj.get("termination", "")
        status = "completed" if termination == "answer" else (termination or "error")
        retrieved_docids, retrieved_chunk_ids = collect_retrieved_ids([traj])

        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_data = {
            "metadata": {
                "method": "react_single_trajectory",
                "model": model_label,
                **retriever_meta,
            },
            "query_id": query_id or None,
            "question": traj["question"],
            "tool_call_counts": {"search": traj.get("search_call_count", 0)},
            "tool_call_counts_all": {"search": traj.get("search_call_count", 0)},
            "status": status,
            "retrieved_docids": retrieved_docids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "result": [
                {"type": "output_text", "tool_name": None, "arguments": None, "output": prediction}
            ],
        }
        with open(output_dir / f"run_{ts}.json", "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        written += 1

    print(f"Wrote {written} evaluator files to {output_dir} ({skipped} already done)")


if __name__ == "__main__":
    main()
