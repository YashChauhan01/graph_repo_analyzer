"""
retrieval/vector_retriever.py
──────────────────────────────
Performs semantic similarity search against ChromaDB.

Returns ranked code chunks with metadata, formatted as LLM-ready context.
Supports optional metadata filtering (by language, file_path, node_kind).
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from core.config import cfg

logger = logging.getLogger(__name__)

# Default number of chunks to retrieve per query
_DEFAULT_TOP_K = 6


def _build_embedding_fn():
    if cfg.embedding_backend == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=cfg.openai_api_key,
            model_name="text-embedding-3-small",
        )
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


class VectorRetriever:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
        self._embed_fn = _build_embedding_fn()
        self._collection = self._client.get_or_create_collection(
            name=cfg.chroma_collection,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = _DEFAULT_TOP_K,
        where: dict[str, Any] | None = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of dicts with keys:
          id, text, score, file_path, node_name, node_kind, language, start_line, end_line
        """
        if self._collection.count() == 0:
            logger.warning("ChromaDB collection is empty — has ingestion run?")
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(top_k, self._collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "text": doc,
                    "score": round(1 - distance, 4),   # cosine similarity
                    "file_path": meta.get("file_path", ""),
                    "node_name": meta.get("node_name", ""),
                    "node_kind": meta.get("node_kind", ""),
                    "language": meta.get("language", ""),
                    "start_line": int(meta.get("start_line", 0)),
                    "end_line": int(meta.get("end_line", 0)),
                }
            )
        return hits

    def search_by_language(self, query: str, language: str, top_k: int = _DEFAULT_TOP_K) -> list[dict]:
        """Narrow search to a specific language (python | java)."""
        return self.search(query, top_k=top_k, where={"language": language})

    def search_by_file(self, query: str, file_path: str, top_k: int = _DEFAULT_TOP_K) -> list[dict]:
        """Narrow search to a specific source file."""
        return self.search(query, top_k=top_k, where={"file_path": file_path})

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_for_context(self, hits: list[dict], max_chars: int = 3000) -> str:
        """
        Convert search hits into a readable context block for the LLM.
        Respects a character budget to avoid token bloat.
        """
        if not hits:
            return "No semantically similar code found."

        lines = ["[Vector DB — semantic matches]"]
        total_chars = 0

        for hit in hits:
            header = (
                f"\n── {hit['node_kind']} `{hit['node_name']}`  "
                f"(score={hit['score']})  "
                f"{hit['file_path']}:{hit['start_line']}–{hit['end_line']}\n"
            )
            body = hit["text"]

            if total_chars + len(header) + len(body) > max_chars:
                # Truncate body to fit within budget
                remaining = max_chars - total_chars - len(header) - 20
                if remaining > 100:
                    body = body[:remaining] + "\n... [truncated]"
                else:
                    break

            lines.append(header + body)
            total_chars += len(header) + len(body)

        return "\n".join(lines)
