"""
ingestion/pipeline.py
─────────────────────
Async ingestion orchestrator.

Flow per file:
  1. ASTExtractor.extract()         → structural nodes + semantic chunks
  2. GraphWriter.write_nodes()      → Neo4j
  3. GraphWriter.write_edges()      → Neo4j (after all nodes written)
  4. VectorWriter.write_chunks()    → ChromaDB

Concurrency model:
  - asyncio.Semaphore limits parallel file processing to MAX_WORKERS.
  - CPU-bound parsing runs in a thread pool via loop.run_in_executor.
  - All Neo4j and ChromaDB I/O runs inside the executor as well
    (their clients are synchronous; wrapping keeps the event loop free).

Resilience:
  - Per-file errors are caught and logged; one bad file never aborts the run.
  - GraphWriter already applies exponential backoff on Neo4j writes.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.config import cfg
from ingestion.parser import ASTExtractor
from ingestion.graph_writer import GraphWriter
from ingestion.vector_writer import VectorWriter

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".py", ".java"}


class IngestionPipeline:
    def __init__(self) -> None:
        self._extractor = ASTExtractor(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )
        self._graph = GraphWriter()
        self._vector = VectorWriter()
        self._semaphore = asyncio.Semaphore(cfg.max_workers)
        self._executor = ThreadPoolExecutor(max_workers=cfg.max_workers)

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest_repo(self, repo_path: str | Path) -> dict:
        """
        Walk a repository directory and ingest all supported source files.
        Returns a summary dict with counts and timing.
        """
        repo_path = Path(repo_path)
        if not repo_path.exists():
            raise FileNotFoundError(f"Repository path not found: {repo_path}")

        files = [
            p for p in repo_path.rglob("*")
            if p.is_file() and p.suffix in _SUPPORTED_EXTENSIONS
        ]

        logger.info("Found %d source files to ingest in %s", len(files), repo_path)
        start = time.perf_counter()

        tasks = [self._ingest_file(f) for f in files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ok = sum(1 for r in results if r is True)
        failed = sum(1 for r in results if r is not True)
        elapsed = time.perf_counter() - start

        summary = {
            "total_files": len(files),
            "succeeded": ok,
            "failed": failed,
            "elapsed_seconds": round(elapsed, 2),
            "vector_chunks": self._vector.count(),
        }
        logger.info("Ingestion complete: %s", summary)
        return summary

    async def ingest_file(self, file_path: str | Path) -> bool:
        """Ingest a single file. Useful for incremental updates."""
        return await self._ingest_file(Path(file_path))

    def close(self) -> None:
        self._graph.close()
        self._executor.shutdown(wait=False)

    # ── Internals ─────────────────────────────────────────────────────────────

    async def _ingest_file(self, file_path: Path) -> bool:
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            try:
                # Parse (CPU-bound) in thread pool
                nodes, chunks = await loop.run_in_executor(
                    self._executor,
                    self._extractor.extract,
                    file_path,
                )

                if not nodes and not chunks:
                    logger.debug("No nodes extracted from %s (skipping)", file_path)
                    return True

                # Clear old data for this file before re-writing
                await loop.run_in_executor(
                    self._executor, self._graph.clear_file, str(file_path)
                )
                await loop.run_in_executor(
                    self._executor, self._vector.delete_file, str(file_path)
                )

                # Write to both databases
                await loop.run_in_executor(
                    self._executor, self._graph.write_nodes, nodes
                )
                await loop.run_in_executor(
                    self._executor, self._graph.write_edges, nodes
                )
                await loop.run_in_executor(
                    self._executor, self._vector.write_chunks, chunks
                )

                logger.info(
                    "Ingested %s → %d nodes, %d chunks",
                    file_path.name,
                    len(nodes),
                    len(chunks),
                )
                return True

            except Exception as e:
                logger.error("Failed to ingest %s: %s", file_path, e, exc_info=True)
                return False
