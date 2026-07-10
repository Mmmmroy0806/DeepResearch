"""Baseline Milvus retriever for BrowseComp-Plus, ported from deepsearch_edi.

This is a minimal, self-contained copy of the *baseline* retrieval path of
`openjiuwen_deepsearch.algorithm.search_tools.retrieval` (RemoteQwenEmbedder +
BrowsecompPlusMilvusRetriever with RetrieveConfig.baseline=True), so that
Tongyi DeepResearch can be evaluated against the exact same fixed retriever
without importing the deepsearch_edi package.

Baseline semantics (must stay aligned with deepsearch_edi):
  - search_limit == top_k (no top_k_multiply_factor over-fetch)
  - one returned hit per Milvus hit: no per-doc merge, no dedup,
    no leading-chunk backfill, no rerank, no full-document expansion
  - hybrid mode = dense (ef=500) + sparse (BM25) AnnSearchRequests combined
    with WeightedRanker(0.6, 0.4)
  - query instruction (when add_instruction=True):
    "Instruct: Given a web search query, retrieve relevant passages that
     answer the query\nQuery:{query}"

Retrieval parameters are locked at construction time from the config file;
the agent/model can only supply query strings.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

logger = logging.getLogger(__name__)

DEFAULT_QUERY_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
BASELINE_HYBRID_RANKER_WEIGHTS = (0.6, 0.4)

# Same table as deepsearch_edi RemoteQwenEmbedder._model2embed_dim
_MODEL2EMBED_DIM = {
    "qwen3-embedding-0.6b": 1024,
    "qwen3-embedding-8b": 4096,
    "qwen3/qwen3-embedding-8b": 4096,
}

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class EmbeddingResponseShapeError(RuntimeError):
    """Embedding API returned a malformed successful response; retrying won't fix it."""


def expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} references in config values from the environment."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        def _sub(m: "re.Match[str]") -> str:
            return os.environ.get(m.group(1), "")
        return _ENV_VAR_PATTERN.sub(_sub, obj)
    return obj


def load_retriever_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return expand_env_vars(config)


def chunk_id_to_docid(chunk_id: str) -> str:
    """Milvus primary key is '{docid}__{chunk_idx}'; BCP evaluator needs docid."""
    if "__" in chunk_id:
        return chunk_id.rsplit("__", 1)[0]
    return chunk_id


class RemoteQwenEmbedder:
    """Minimal copy of deepsearch_edi RemoteQwenEmbedder (same payload, same instruction)."""

    def __init__(
        self,
        model_name: str,
        api_key: str,
        base_url: str,
        timeout: int = 100,
        model_dim: Optional[int] = None,
        instruction: str = DEFAULT_QUERY_INSTRUCTION,
    ):
        if not api_key:
            raise ValueError(
                "Embedding API key not provided. Set it in the config file "
                "(e.g. \"${OPENROUTER_API_KEY}\") and export the env var."
            )
        if not base_url:
            raise ValueError("Embedding API base_url not provided.")

        matched_dim = None
        for supported, dim in _MODEL2EMBED_DIM.items():
            if supported in model_name.lower():
                matched_dim = dim
                break
        self.embed_dim = model_dim or matched_dim
        if not self.embed_dim:
            raise ValueError(
                f"Unknown embedding dim for model {model_name!r}; "
                f"supported: {sorted(_MODEL2EMBED_DIM)} or set embedding.model_dim in config."
            )

        self.model_name = model_name
        self.instruction = instruction
        self.timeout = timeout
        self._api_url = base_url.strip().strip('"').strip("'")
        self._headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        }

    def get_query_instruction(self, query: str) -> str:
        return f"Instruct: {self.instruction}\nQuery:{query}"

    def encode(self, input_texts: List[str], is_query: bool = False) -> List[List[float]]:
        if is_query:
            input_texts = [self.get_query_instruction(t) for t in input_texts]

        payload = {
            "model": self.model_name,
            "input": input_texts,
            "dimensions": self.embed_dim,
            "encoding_format": "float",
        }
        max_try = 10
        last_exception: Optional[Exception] = None
        for attempt in range(max_try):
            try:
                response = requests.post(
                    self._api_url,
                    json=payload,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json().get("data", [])
                data = sorted(data, key=lambda d: d.get("index", 0))
                expected = list(range(len(input_texts)))
                actual = [d.get("index", i) for i, d in enumerate(data)]
                if len(data) != len(input_texts) or actual != expected:
                    raise EmbeddingResponseShapeError(
                        "Embedding API returned unexpected data shape: "
                        f"expected indices={expected}, got indices={actual}, "
                        f"n_embeddings={len(data)}, n_inputs={len(input_texts)}"
                    )
                return [d["embedding"] for d in data]
            except EmbeddingResponseShapeError:
                raise
            except Exception as e:  # noqa: BLE001 - retried, re-raised below
                last_exception = e
                logger.warning(
                    "Embedder encode failed (attempt=%d/%d model=%s url=%s): %s",
                    attempt + 1,
                    max_try,
                    self.model_name,
                    self._api_url,
                    e,
                )
                if attempt < max_try - 1:
                    time.sleep(10)
        raise RuntimeError(
            f"Embedding API call failed after {max_try} attempts: {last_exception}"
        ) from last_exception


class BaselineMilvusRetriever:
    """deepsearch_edi BrowsecompPlusMilvusRetriever, baseline mode only."""

    def __init__(self, config: dict):
        milvus_cfg = config.get("milvus", {})
        embedding_cfg = config.get("embedding", {})
        retrieval_cfg = config.get("retrieval", {})

        self.retriever_name = config.get("retriever_name", "deepsearch_edi_baseline_milvus")

        # --- retrieval params: locked here, never model-controlled ---
        self.top_k = int(retrieval_cfg.get("top_k", 3))
        self.top_k_multiply_factor = int(retrieval_cfg.get("top_k_multiply_factor", 5))
        self.mode = retrieval_cfg.get("mode", "hybrid")
        if self.mode not in ("dense", "sparse", "hybrid"):
            raise ValueError(f"Unsupported retrieval mode for BCP baseline: {self.mode!r}")
        self.add_instruction = bool(retrieval_cfg.get("add_instruction", True))
        self.include_leading_chunk = bool(retrieval_cfg.get("include_leading_chunk", False))
        self.snippet_max_tokens = int(retrieval_cfg.get("snippet_max_tokens", 0))
        configured_weights = tuple(retrieval_cfg.get("hybrid_ranker_weights", BASELINE_HYBRID_RANKER_WEIGHTS))
        if len(configured_weights) != 2:
            raise ValueError("retrieval.hybrid_ranker_weights must contain exactly two weights")
        if tuple(float(w) for w in configured_weights) != BASELINE_HYBRID_RANKER_WEIGHTS:
            raise ValueError(
                "This adapter implements the deepsearch_edi baseline only; "
                f"hybrid_ranker_weights must be {BASELINE_HYBRID_RANKER_WEIGHTS}."
            )
        self.hybrid_ranker_weights = BASELINE_HYBRID_RANKER_WEIGHTS
        if not retrieval_cfg.get("baseline", True):
            raise ValueError("This adapter only implements baseline=true retrieval.")

        # --- schema fields (match create_browsecompplus_index.py) ---
        self.vector_field = "embedding"
        self.text_field = "content"
        self.sparse_field = "content_sparse"
        self.title_field = "title"
        self.id_field = "id"
        self.metric_type = "COSINE"

        # --- embedder (not needed for sparse-only) ---
        self.embedder: Optional[RemoteQwenEmbedder] = None
        if self.mode != "sparse":
            self.embedder = RemoteQwenEmbedder(
                model_name=embedding_cfg.get("model_name", "qwen/qwen3-embedding-8b"),
                api_key=embedding_cfg.get("api_key", ""),
                base_url=embedding_cfg.get("base_url", ""),
                timeout=int(embedding_cfg.get("timeout", 100)),
                model_dim=embedding_cfg.get("model_dim"),
            )

        # --- Milvus client ---
        uri = milvus_cfg.get("uri")
        if not uri:
            host = milvus_cfg.get("host", "localhost")
            port = milvus_cfg.get("port", 19530)
            uri = f"http://{host}:{port}"
        token = milvus_cfg.get("token") or ""
        if token:
            self.client = MilvusClient(uri=uri, token=token)
        else:
            self.client = MilvusClient(uri=uri)

        database_name = milvus_cfg.get("database_name", "")
        if database_name and database_name != "default":
            if database_name not in self.client.list_databases():
                raise ValueError(
                    f"Milvus has no database named {database_name!r}. "
                    f"Available: {sorted(self.client.list_databases())}"
                )
            self.client.use_database(database_name)
        self.database_name = database_name

        collection_name = milvus_cfg.get("collection_name", "")
        existing = set(self.client.list_collections())
        if collection_name not in existing:
            raise ValueError(
                f"Milvus has no collection named {collection_name!r} "
                f"(database={database_name!r}). Available: {sorted(existing) or '(none)'}"
            )
        self.client.load_collection(collection_name)
        self.collection_name = collection_name
        # MilvusClient thread-safety is not guaranteed across all versions. The
        # BCP runner may use multiple worker threads, so serialize actual client
        # calls while keeping each agent loop independent.
        self._search_lock = threading.Lock()

        logger.info(
            "BaselineMilvusRetriever ready: retriever=%s collection=%s db=%s mode=%s "
            "top_k=%d baseline=true add_instruction=%s embedding_model=%s embedding_base_url=%s",
            self.retriever_name,
            self.collection_name,
            self.database_name,
            self.mode,
            self.top_k,
            self.add_instruction,
            embedding_cfg.get("model_name"),
            embedding_cfg.get("base_url"),
        )

    def _hit_to_dict(self, hit: Any, rank: int) -> Dict[str, Any]:
        # pymilvus MilvusClient hits are Hit objects (attribute access) in some
        # versions and plain dicts ({"id", "distance", "entity"}) in others.
        if isinstance(hit, dict):
            chunk_id = str(hit.get("id"))
            score = float(hit.get("distance", 0.0))
            entity = hit.get("entity") or {}
        else:
            chunk_id = str(hit.id)
            score = float(hit.distance)
            entity = hit.entity
        title = entity.get(self.title_field) or ""
        docid = entity.get("docid") or chunk_id_to_docid(chunk_id)
        text = entity.get(self.text_field) or ""
        metadata = entity.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "rank": rank,
            "chunk_id": chunk_id,
            "docid": str(docid),
            "title": title,
            "text": text,
            "score": score,
            "metadata": metadata,
        }

    def search(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Baseline retrieval: returns [{"query": ..., "hits": [...]}] per query."""
        if isinstance(queries, str):
            queries = [queries]
        queries = [q for q in queries if isinstance(q, str) and q.strip()]
        if not queries:
            return []

        if self.mode == "sparse":
            query_vecs: List[List[float]] = [[0.0]] * len(queries)
        else:
            query_vecs = self.embedder.encode(queries, is_query=self.add_instruction)

        with self._search_lock:
            return self._search_locked(queries, query_vecs)

    def _search_locked(self, queries: List[str], query_vecs: List[List[float]]) -> List[Dict[str, Any]]:
        # Baseline: request exactly top_k hits (no over-fetch).
        search_limit = self.top_k

        search_kwargs = {
            "collection_name": self.collection_name,
            "output_fields": [
                self.text_field,
                self.title_field,
                self.id_field,
                "docid",
                "metadata",
            ],
        }

        results: List[Any] = []
        if self.mode == "dense":
            search_params = {"metric_type": self.metric_type, "params": {"ef": 500}}
            results += self.client.search(
                data=query_vecs,
                anns_field=self.vector_field,
                search_params=search_params,
                limit=search_limit,
                filter="",
                **search_kwargs,
            )
        elif self.mode == "sparse":
            results += self.client.search(
                data=queries,
                anns_field=self.sparse_field,
                search_params={"metric_type": "BM25"},
                limit=search_limit,
                filter="",
                **search_kwargs,
            )
        elif self.mode == "hybrid":
            dense_request = AnnSearchRequest(
                data=query_vecs,
                anns_field=self.vector_field,
                limit=search_limit,
                param={"ef": 500},
                expr="",
            )
            sparse_request = AnnSearchRequest(
                data=queries,
                anns_field=self.sparse_field,
                limit=search_limit,
                param={},
                expr="",
            )
            results += self.client.hybrid_search(
                reqs=[dense_request, sparse_request],
                ranker=WeightedRanker(*self.hybrid_ranker_weights),
                limit=search_limit,
                **search_kwargs,
            )

        if len(results) != len(queries):
            raise ValueError(
                f"Milvus results length ({len(results)}) does not match query length ({len(queries)})"
            )

        out: List[Dict[str, Any]] = []
        for query, result in zip(queries, results):
            hits = [
                self._hit_to_dict(hit, rank)
                for rank, hit in enumerate(result[: self.top_k], start=1)
            ]
            out.append({"query": query, "hits": hits})
            logger.info(
                "retrieve query=%r retriever=%s collection=%s mode=%s top_k=%d "
                "chunk_ids=%s docids=%s",
                query,
                self.retriever_name,
                self.collection_name,
                self.mode,
                self.top_k,
                [h["chunk_id"] for h in hits],
                [h["docid"] for h in hits],
            )
        return out

    def run_config_summary(self) -> Dict[str, Any]:
        """Config provenance for run outputs (no secrets)."""
        summary = {
            "retriever": self.retriever_name,
            "database_name": self.database_name,
            "collection_name": self.collection_name,
            "mode": self.mode,
            "top_k": self.top_k,
            "top_k_multiply_factor": self.top_k_multiply_factor,
            "baseline": True,
            "add_instruction": self.add_instruction,
            "include_leading_chunk": self.include_leading_chunk,
            "snippet_max_tokens": self.snippet_max_tokens,
        }
        if self.mode == "hybrid":
            summary["hybrid_ranker_weights"] = list(self.hybrid_ranker_weights)
        if self.embedder is not None:
            summary["embedding_model"] = self.embedder.model_name
            summary["embedding_base_url"] = self.embedder._api_url
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Smoke-test the BCP baseline retriever")
    parser.add_argument(
        "--config",
        default=os.path.join(os.path.dirname(__file__), "configs", "bcp_baseline_retriever.json"),
    )
    parser.add_argument("--query", required=True, action="append", help="Query text (repeatable)")
    args = parser.parse_args()

    retriever = BaselineMilvusRetriever(load_retriever_config(args.config))
    for entry in retriever.search(args.query):
        print(f"\n=== query: {entry['query']} ===")
        for hit in entry["hits"]:
            preview = hit["text"][:200].replace("\n", " ")
            print(
                f"  #{hit['rank']} score={hit['score']:.4f} docid={hit['docid']} "
                f"chunk_id={hit['chunk_id']}\n     title={hit['title'][:80]!r}\n     text={preview!r}"
            )
