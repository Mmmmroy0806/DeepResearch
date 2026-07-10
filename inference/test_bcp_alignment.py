"""Alignment test: Tongyi adapter vs deepsearch_edi baseline retriever.

Acceptance criterion (plan §12.2): for the same query, same config, and same
Milvus collection, the top-k chunk_ids returned by both retrievers must be
identical (same ids, same order).

The adapter (baseline_retriever.py) only needs pymilvus + requests, so this
script is designed to run inside the deepsearch_edi virtualenv, where the
original retriever is importable:

  cd /Users/mmmroym/Downloads/huawei/gitcode/deepsearch_edi
  .venv/bin/python /Users/mmmroym/Downloads/huawei/DeepResearch/inference/test_bcp_alignment.py \
      --config /Users/mmmroym/Downloads/huawei/DeepResearch/inference/configs/bcp_baseline_retriever.json

Requires: reachable Milvus (config milvus.uri) and the embedding API key env
var referenced by the config (e.g. OPENROUTER_API_KEY).
"""

import argparse
import json
import os
import sys
import urllib.parse

DEFAULT_QUERIES = [
    "Aloinopsis Washingtonia difference",
    "tower built in the 1340s wooden house city",
    "research group founded in 2009 coordinator December 2023",
    "three-day event Thursday to Saturday 2002 learning institution",
    "individual born in the 1910s father occupation",
]

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def run_adapter(config: dict, queries: list) -> dict:
    """Returns {query: [chunk_id, ...]} from the Tongyi adapter."""
    from baseline_retriever import BaselineMilvusRetriever

    retriever = BaselineMilvusRetriever(config)
    out = {}
    for entry in retriever.search(queries):
        out[entry["query"]] = [hit["chunk_id"] for hit in entry["hits"]]
    return out


def run_deepsearch_edi(config: dict, queries: list) -> dict:
    """Returns {query: [chunk_id, ...]} from the original deepsearch_edi retriever."""
    import openjiuwen_deepsearch.algorithm.search_tools.retrieval.base_retriever as base_retriever_module

    RetrieveConfig = base_retriever_module.RetrieveConfig
    from openjiuwen_deepsearch.algorithm.search_tools.retrieval.embedder import (
        RemoteQwenEmbedder,
    )
    from openjiuwen_deepsearch.algorithm.search_tools.retrieval.retriever import (
        BrowsecompPlusMilvusRetriever,
    )

    milvus_cfg = config["milvus"]
    embedding_cfg = config["embedding"]
    retrieval_cfg = config["retrieval"]

    uri = milvus_cfg.get("uri")
    if uri:
        parsed = urllib.parse.urlparse(uri)
        host, port = parsed.hostname, str(parsed.port or 19530)
    else:
        host, port = milvus_cfg.get("host", "localhost"), str(milvus_cfg.get("port", 19530))

    embedder = RemoteQwenEmbedder(
        pretrained_model=embedding_cfg["model_name"],
        api_token=embedding_cfg["api_key"],
        api_url=embedding_cfg["base_url"],
        timeout=int(embedding_cfg.get("timeout", 100)),
    )
    token = milvus_cfg.get("token") or ""
    original_milvus_client = None
    if token:
        # deepsearch_edi's BaseRetriever currently only accepts host/port and
        # constructs MilvusClient(uri=...). For alignment testing only, inject
        # the configured token without modifying the deepsearch_edi source tree.
        original_milvus_client = base_retriever_module.MilvusClient

        def _milvus_client_with_token(*args, **kwargs):
            kwargs.setdefault("token", token)
            return original_milvus_client(*args, **kwargs)

        base_retriever_module.MilvusClient = _milvus_client_with_token
    try:
        retriever = BrowsecompPlusMilvusRetriever(
            milvus_host=host,
            milvus_port=port,
            database_name=milvus_cfg.get("database_name", ""),
            collection_name=milvus_cfg.get("collection_name", ""),
            embedder=embedder,
        )
    finally:
        if original_milvus_client is not None:
            base_retriever_module.MilvusClient = original_milvus_client
    out = {}
    for query in queries:
        # One query per call to mirror how the agent's search tool is used.
        _, id_list = retriever.retrieve(
            RetrieveConfig(
                query=[query],
                top_k=int(retrieval_cfg.get("top_k", 3)),
                add_instruction=bool(retrieval_cfg.get("add_instruction", True)),
                mode=retrieval_cfg.get("mode", "hybrid"),
                top_k_multiply_factor=int(retrieval_cfg.get("top_k_multiply_factor", 5)),
                baseline=True,
                include_leading_chunk=bool(retrieval_cfg.get("include_leading_chunk", False)),
            )
        )
        out[query] = id_list
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default=os.path.join(HERE, "configs", "bcp_baseline_retriever.json")
    )
    parser.add_argument("--query", action="append", default=None,
                        help="Query to compare (repeatable); defaults to 5 built-in queries")
    args = parser.parse_args()

    from baseline_retriever import load_retriever_config

    config = load_retriever_config(args.config)
    queries = args.query or DEFAULT_QUERIES

    print(f"Comparing {len(queries)} queries on collection "
          f"{config['milvus']['collection_name']!r} (mode={config['retrieval']['mode']}, "
          f"top_k={config['retrieval']['top_k']})\n")

    adapter_ids = run_adapter(config, queries)
    edi_ids = run_deepsearch_edi(config, queries)

    all_match = True
    for query in queries:
        a, b = adapter_ids.get(query, []), edi_ids.get(query, [])
        match = a == b
        all_match &= match
        print(f"[{'MATCH' if match else 'MISMATCH'}] {query!r}")
        print(f"  adapter       : {a}")
        print(f"  deepsearch_edi: {b}")

    print()
    if all_match:
        print("ALIGNMENT OK: all top-k chunk_ids identical.")
        sys.exit(0)
    print("ALIGNMENT FAILED: check collection / embedding config / mode / top_k / "
          "WeightedRanker weights per plan §12.2.")
    sys.exit(1)


if __name__ == "__main__":
    main()
