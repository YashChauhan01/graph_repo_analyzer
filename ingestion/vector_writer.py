"""
ingestion/vector_writer.py
──────────────────────────
Stores semantic code chunks in ChromaDB as dense vector embeddings.

Embedding backends
──────────────────
  "local"  — sentence-transformers (no API key needed, runs on CPU)
  "openai" — OpenAI text-embedding-3-small (requires OPENAI_API_KEY)

The embedding function is injected into ChromaDB's collection so
retrieval uses the same model as ingestion automatically.
"""

from __future__ import annotations

import logging
from typing import Callable

import chromadb
from chromadb.utils import embedding_functions

from core.config import cfg
from ingestion.parser import SemanticChunk

logger = logging.getLogger(__name__)

# Max documents per ChromaDB upsert batch
_BATCH_SIZE = 100


def _build_embedding_fn() -> Callable:
    if cfg.embedding_backend == "openai":
        if not cfg.openai_api_key:
            raise EnvironmentError("OPENAI_API_KEY is required when EMBEDDING_BACKEND=openai")
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=cfg.openai_api_key,
            model_name="text-embedding-3-small",
        )
    # Default: local sentence-transformers (all-MiniLM-L6-v2)
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


class VectorWriter:
    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=cfg.chroma_persist_dir)
        self._embed_fn = _build_embedding_fn()
        self._collection = self._client.get_or_create_collection(
            name=cfg.chroma_collection,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def write_chunks(self, chunks: list[SemanticChunk]) -> None:
        """Upsert chunks in batches. Existing IDs are overwritten (idempotent)."""
        if not chunks:
            return

        for batch_start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _BATCH_SIZE]
            self._upsert_batch(batch)
            logger.debug("Upserted batch %d–%d", batch_start, batch_start + len(batch))

    def delete_file(self, file_path: str) -> None:
        """Remove all chunks belonging to a specific file (re-ingestion support)."""
        self._collection.delete(where={"file_path": file_path})

    def count(self) -> int:
        return self._collection.count()

    # ── Internals ─────────────────────────────────────────────────────────────

    def _upsert_batch(self, chunks: list[SemanticChunk]) -> None:
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "file_path": c.file_path,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "node_name": c.node_name,
                    "node_kind": c.node_kind,
                    "language": c.language,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ],
        )
