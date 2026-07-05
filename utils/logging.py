"""
ingestion/graph_writer.py
─────────────────────────
Writes structural nodes and dependency edges to Neo4j.

Schema
──────
  Nodes  : (:Class {name, qualified_name, file_path, language})
           (:Method {name, qualified_name, file_path, language, start_line, end_line})
           (:Function {name, qualified_name, file_path, language, start_line, end_line})

  Edges  : (:Method|Function)-[:CALLS]->(:Method|Function)
           (:Method)-[:BELONGS_TO]->(:Class)
           (:Class)-[:DEFINED_IN]->(:File)

Constraints and indexes are created on first run.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from neo4j import GraphDatabase, Session
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.config import cfg
from ingestion.parser import StructuralNode

logger = logging.getLogger(__name__)


class GraphWriter:
    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
        )
        self._ensure_schema()

    # ── Schema setup ──────────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create uniqueness constraints and indexes (idempotent)."""
        constraints = [
            "CREATE CONSTRAINT class_qname IF NOT EXISTS FOR (c:Class) REQUIRE c.qualified_name IS UNIQUE",
            "CREATE CONSTRAINT method_qname IF NOT EXISTS FOR (m:Method) REQUIRE m.qualified_name IS UNIQUE",
            "CREATE CONSTRAINT function_qname IF NOT EXISTS FOR (f:Function) REQUIRE f.qualified_name IS UNIQUE",
        ]
        with self._session() as session:
            for cql in constraints:
                try:
                    session.run(cql)
                except Exception as e:
                    logger.warning("Schema constraint skipped: %s", e)

    # ── Public API ────────────────────────────────────────────────────────────

    def write_nodes(self, nodes: list[StructuralNode]) -> None:
        """Upsert all structural nodes (classes, methods, functions)."""
        with self._session() as session:
            for node in nodes:
                self._upsert_node(session, node)

    def write_edges(self, nodes: list[StructuralNode]) -> None:
        """
        Write CALLS, BELONGS_TO, and DEFINED_IN relationships.
        Called after all nodes are written so targets are guaranteed to exist.
        """
        with self._session() as session:
            for node in nodes:
                if node.parent:
                    self._link_belongs_to(session, node)
                for callee in node.calls:
                    self._link_calls(session, node.qualified_name, callee)

    def clear_file(self, file_path: str) -> None:
        """Remove all nodes originating from a specific file (re-ingestion support)."""
        with self._session() as session:
            session.run(
                "MATCH (n {file_path: $fp}) DETACH DELETE n",
                fp=file_path,
            )

    def close(self) -> None:
        self._driver.close()

    # ── Internals ─────────────────────────────────────────────────────────────

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self._driver.session() as session:
            yield session

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(cfg.backoff_max_tries),
        reraise=True,
    )
    def _upsert_node(self, session: Session, node: StructuralNode) -> None:
        label = node.kind.capitalize()  # Class | Method | Function
        session.run(
            f"""
            MERGE (n:{label} {{qualified_name: $qname}})
            SET n.name        = $name,
                n.file_path   = $fp,
                n.language    = $lang,
                n.start_line  = $sl,
                n.end_line    = $el
            """,
            qname=node.qualified_name,
            name=node.name,
            fp=node.file_path,
            lang=node.language,
            sl=node.start_line,
            el=node.end_line,
        )

    def _link_belongs_to(self, session: Session, node: StructuralNode) -> None:
        label = node.kind.capitalize()
        session.run(
            f"""
            MATCH (m:{label} {{qualified_name: $mname}})
            MATCH (c:Class  {{qualified_name: $cname}})
            MERGE (m)-[:BELONGS_TO]->(c)
            """,
            mname=node.qualified_name,
            cname=node.parent,
        )

    def _link_calls(self, session: Session, caller: str, callee: str) -> None:
        """
        Create a CALLS edge. The callee may be a short name (e.g. "doSomething")
        or a qualified name. We attempt an exact match first, then a name match.
        """
        session.run(
            """
            MATCH (caller {qualified_name: $caller})
            OPTIONAL MATCH (callee_exact {qualified_name: $callee})
            OPTIONAL MATCH (callee_name  {name: $callee})
            WITH caller,
                 COALESCE(callee_exact, callee_name) AS callee
            WHERE callee IS NOT NULL
            MERGE (caller)-[:CALLS]->(callee)
            """,
            caller=caller,
            callee=callee,
        )
