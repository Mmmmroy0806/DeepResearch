"""Search-only tool for BrowseComp-Plus, backed by the deepsearch_edi baseline retriever.

Replaces Tongyi's original Serper/Google `search` tool. The tool name stays
`search` so the model-facing protocol is unchanged, but every result comes
from the fixed BrowseComp-Plus Milvus baseline index — no web access.

The model may only supply query strings; top_k / mode / collection /
add_instruction are locked in the retriever config.
"""

import logging
from typing import List, Optional, Tuple, Union

from qwen_agent.tools.base import BaseTool, register_tool

from baseline_retriever import BaselineMilvusRetriever

logger = logging.getLogger(__name__)


@register_tool("search", allow_overwrite=True)
class SearchBaseline(BaseTool):
    name = "search"
    description = (
        "Search the fixed BrowseComp-Plus corpus with the baseline retriever. "
        "It returns top relevant evidence snippets."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, retriever: BaselineMilvusRetriever, snippet_max_tokens: int = 0):
        super().__init__()
        # One retriever (and Milvus client) per process, reused across calls.
        self.retriever = retriever
        self.snippet_max_tokens = snippet_max_tokens
        self.tokenizer = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            try:
                from transformers import AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "snippet_max_tokens=%d requested but tokenizer unavailable (%s); "
                    "returning full chunks (matches deepsearch_edi baseline behavior).",
                    snippet_max_tokens,
                    e,
                )

    def _truncate(self, text: str) -> str:
        if not self.tokenizer or self.snippet_max_tokens <= 0:
            return text
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) > self.snippet_max_tokens:
            return self.tokenizer.decode(
                tokens[: self.snippet_max_tokens], skip_special_tokens=True
            )
        return text

    def _format_query_result(self, entry: dict) -> str:
        query = entry["query"]
        hits = entry["hits"]
        if not hits:
            return f"No results found for '{query}'. Try with a more general query."
        formatted = []
        for hit in hits:
            title = hit["title"] or hit["docid"]
            snippet = self._truncate(hit["text"])
            formatted.append(
                f"{hit['rank']}. [{title}]\n"
                f"docid: {hit['docid']}\n"
                f"chunk_id: {hit['chunk_id']}\n"
                f"score: {hit['score']:.4f}\n"
                f"{snippet}"
            )
        return (
            f"A search for '{query}' found {len(hits)} results:\n\n"
            "## Search Results\n" + "\n\n".join(formatted)
        )

    def call(
        self, params: Union[str, dict], **kwargs
    ) -> Tuple[str, Optional[List[str]], Optional[List[str]]]:
        """Returns (response_text, docids, chunk_ids); docids is None on failure."""
        try:
            query = params["query"]
        except Exception:  # noqa: BLE001
            return (
                "[Search] Invalid request format: Input must be a JSON object containing 'query' field",
                None,
                None,
            )

        if isinstance(query, str):
            queries = [query]
        elif isinstance(query, list) and query and all(isinstance(q, str) for q in query):
            queries = query
        else:
            return (
                "[Search] Invalid request format: 'query' must be a string (or an array of strings)",
                None,
                None,
            )

        try:
            entries = self.retriever.search(queries)
        except Exception as e:  # noqa: BLE001
            logger.exception("Baseline retriever search failed for %r", queries)
            return f"[Search] Retrieval error: {e}", None, None

        docids: List[str] = []
        chunk_ids: List[str] = []
        for entry in entries:
            for hit in entry["hits"]:
                docids.append(hit["docid"])
                chunk_ids.append(hit["chunk_id"])

        response = "\n=======\n".join(self._format_query_result(e) for e in entries)
        return response, docids, chunk_ids
