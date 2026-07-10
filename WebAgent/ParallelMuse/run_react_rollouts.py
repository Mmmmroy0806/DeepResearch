"""YAML-driven orchestrator for N independent ParallelMuse-ReAct rollout runs.

Each entry under `runs:` is one environment: a full pass over the dataset with
bcp_partial_rollout.py (--partial_sampling_mode none --sampling_budget 1),
writing one ReAct trajectory per question into its own output_dir. Environments
run as subprocesses, at most --max-parallel at a time, until all are complete.

Per-environment artifacts:
  <output_dir>/<dataset>_<model>_1_none_initial_rollout.jsonl   trajectories
  <output_dir>/orchestrator.log                                  subprocess output
  <output_dir>/env_meta.json                                     resolved config + timings
  <output_dir>/eval/run_*.json                                   evaluator files (convert_to_eval)

Resume: re-running the orchestrator skips complete environments; incomplete
ones continue from where they stopped (bcp_partial_rollout's own per-question
resume). Ctrl-C terminates running children; completed work is kept.

Usage:
  python run_react_rollouts.py --config configs/react8_gpt_oss_20b.yaml --max-parallel 3
  python run_react_rollouts.py --config configs/react8_gpt_oss_20b.yaml --status
  python run_react_rollouts.py --config configs/react8_gpt_oss_20b.yaml --only env1,env2 --max-parallel 2
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from collections import Counter

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "retriever_config": os.path.join(_HERE, "..", "..", "inference", "configs",
                                     "bcp_baseline_retriever.json"),
    "llm_base_urls": ["https://openrouter.ai/api/v1"],
    "llm_api_key_env": "OPENROUTER_API_KEY",
    "extra_body": "",
    "temperature": 0.6,
    "top_p": 0.95,
    "presence_penalty": 1.1,
    "max_turn": 100,
    "max_completion_tokens": 32 * 1024,
    "max_context_length": 100 * 1024,
    "partial_sampling_mode": "none",
    "sampling_budget": 1,
    "max_llm_workers": 8,
    "max_search_workers": 4,
    "limit": 0,
    "convert_to_eval": True,
}


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    common = {**DEFAULTS, **(cfg.get("common") or {})}
    runs = []
    seen_names, seen_dirs = set(), set()
    for entry in cfg.get("runs") or []:
        merged = {**common, **entry}
        name = merged.get("name")
        out_dir = merged.get("output_dir")
        if not name or not out_dir:
            raise ValueError(f"Every run needs `name` and `output_dir`: {entry}")
        if name in seen_names or out_dir in seen_dirs:
            raise ValueError(f"Duplicate run name or output_dir: {name} / {out_dir}")
        seen_names.add(name)
        seen_dirs.add(out_dir)
        merged["output_dir"] = os.path.abspath(os.path.join(os.path.dirname(path), "..", out_dir)
                                               if not os.path.isabs(out_dir) else out_dir)
        merged["retriever_config"] = os.path.abspath(
            os.path.join(_HERE, merged["retriever_config"])
            if not os.path.isabs(merged["retriever_config"]) else merged["retriever_config"])
        runs.append(merged)
    if not runs:
        raise ValueError("No runs defined in the YAML.")
    return runs


def dataset_question_count(run):
    sys.path.insert(0, _HERE)
    from bcp_partial_rollout import load_dataset
    items = load_dataset(run["qa_file_path"])
    if run["limit"] and int(run["limit"]) > 0:
        items = items[: int(run["limit"])]
    return len(items), items


def rollout_file_path(run):
    dataset_tag = os.path.basename(run["qa_file_path"]).rsplit(".", 1)[0]
    model_tag = run["llm_model"].replace("/", "_")
    return os.path.join(run["output_dir"], f"{dataset_tag}_{model_tag}_1_none_initial_rollout.jsonl")


def env_progress(run):
    """Returns (done_questions, total_questions)."""
    total, _items = dataset_question_count(run)
    path = rollout_file_path(run)
    counter = Counter()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    counter[json.loads(line)["question"]] += 1
                except Exception:
                    continue
    budget = int(run["sampling_budget"])
    done = sum(1 for _q, c in counter.items() if c >= budget)
    return done, total


def build_command(run):
    cmd = [
        sys.executable, os.path.join(_HERE, "bcp_partial_rollout.py"),
        "--qa_file_path", run["qa_file_path"],
        "--output_dir", run["output_dir"],
        "--retriever-config", run["retriever_config"],
        "--llm-model", run["llm_model"],
        "--llm-api-key-env", run["llm_api_key_env"],
        "--temperature", str(run["temperature"]),
        "--top_p", str(run["top_p"]),
        "--presence_penalty", str(run["presence_penalty"]),
        "--max_turn", str(run["max_turn"]),
        "--max_completion_tokens", str(run["max_completion_tokens"]),
        "--max_context_length", str(run["max_context_length"]),
        "--partial_sampling_mode", str(run["partial_sampling_mode"]),
        "--sampling_budget", str(run["sampling_budget"]),
        "--max_llm_workers", str(run["max_llm_workers"]),
        "--max_search_workers", str(run["max_search_workers"]),
        "--limit", str(run["limit"]),
    ]
    for url in run["llm_base_urls"]:
        cmd += ["--llm-base-url", url]
    if run["extra_body"]:
        cmd += ["--extra-body", run["extra_body"] if isinstance(run["extra_body"], str)
                else json.dumps(run["extra_body"])]
    return cmd


def write_meta(run, extra):
    meta = {k: v for k, v in run.items() if k != "llm_api_key"}
    meta.update(extra)
    os.makedirs(run["output_dir"], exist_ok=True)
    with open(os.path.join(run["output_dir"], "env_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def convert_env(run):
    cmd = [
        sys.executable, os.path.join(_HERE, "bcp_traj_to_eval.py"),
        "--rollout-file", rollout_file_path(run),
        "--output-dir", os.path.join(run["output_dir"], "eval"),
        "--model-label", run["llm_model"],
        "--retriever-config", run["retriever_config"],
    ]
    subprocess.run(cmd, cwd=_HERE, check=False)


def print_status(runs):
    print(f"{'env':<8} {'progress':<12} state")
    for run in runs:
        done, total = env_progress(run)
        state = "complete" if done >= total else ("in progress" if done else "not started")
        print(f"{run['name']:<8} {done}/{total:<10} {state}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated run names to restrict to")
    parser.add_argument("--status", action="store_true", help="Print progress and exit")
    args = parser.parse_args()

    runs = load_config(args.config)
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        runs = [r for r in runs if r["name"] in wanted]
        if not runs:
            raise ValueError(f"--only matched no runs: {args.only}")

    if args.status:
        print_status(runs)
        return

    api_key_envs = {r["llm_api_key_env"] for r in runs}
    for env_name in api_key_envs:
        if not os.environ.get(env_name):
            print(f"WARNING: env var {env_name} is not set; API calls will use 'EMPTY'.")

    pending = []
    for run in runs:
        done, total = env_progress(run)
        if done >= total:
            print(f"[{run['name']}] already complete ({done}/{total}), skipping")
            if run["convert_to_eval"]:
                convert_env(run)
            continue
        pending.append(run)

    print(f"{len(pending)} environment(s) to run, max {args.max_parallel} in parallel")
    active = {}  # name -> (proc, run, logfile, start_ts)
    try:
        while pending or active:
            while pending and len(active) < args.max_parallel:
                run = pending.pop(0)
                os.makedirs(run["output_dir"], exist_ok=True)
                log_path = os.path.join(run["output_dir"], "orchestrator.log")
                logf = open(log_path, "a", encoding="utf-8")
                cmd = build_command(run)
                logf.write(f"\n===== {datetime.datetime.now().isoformat()} launch =====\n"
                           f"{' '.join(cmd)}\n\n")
                logf.flush()
                proc = subprocess.Popen(cmd, cwd=_HERE, stdout=logf, stderr=subprocess.STDOUT)
                start_ts = time.time()
                write_meta(run, {"pid": proc.pid, "started_at": datetime.datetime.now().isoformat(),
                                 "status": "running"})
                active[run["name"]] = (proc, run, logf, start_ts)
                print(f"[{run['name']}] started (pid {proc.pid}) -> {log_path}")

            time.sleep(10)
            for name in list(active):
                proc, run, logf, start_ts = active[name]
                if proc.poll() is None:
                    continue
                logf.close()
                del active[name]
                done, total = env_progress(run)
                elapsed = round(time.time() - start_ts)
                complete = done >= total
                write_meta(run, {"finished_at": datetime.datetime.now().isoformat(),
                                 "elapsed_seconds": elapsed, "exit_code": proc.returncode,
                                 "progress": f"{done}/{total}",
                                 "status": "complete" if complete else "incomplete"})
                print(f"[{name}] exited code={proc.returncode}, progress {done}/{total}, {elapsed}s")
                if complete:
                    if run["convert_to_eval"]:
                        convert_env(run)
                elif proc.returncode != 0:
                    print(f"[{name}] incomplete — re-run the orchestrator to resume it")
    except KeyboardInterrupt:
        print("\nInterrupted; terminating running environments (progress is kept)...")
        for name, (proc, run, logf, _ts) in active.items():
            proc.terminate()
            logf.close()
            write_meta(run, {"status": "interrupted"})
        for name, (proc, _run, _logf, _ts) in active.items():
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(130)

    print("\nAll environments processed. Final status:")
    print_status(runs)


if __name__ == "__main__":
    main()
