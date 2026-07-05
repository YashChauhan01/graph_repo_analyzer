"""
retrieval/graph_retriever.py
────────────────────────────
Runs Cypher queries against Neo4j to answer architectural questions.

Query types covered:
  - Dependencies of a class/method
  - Callers of a method (reverse lookup)
  - Inheritance chains
  - All classes in a module/file
  - Full subgraph for a class (class + all its methods + their calls)
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from neo4j import GraphDatabase, Session
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from core.config import cfg

logger = logging.getLogger(__name__)


class GraphRetriever:
    def __init__(self) -> None:
        self._driver = GraphDatabase.driver(
            cfg.neo4j_uri,
            auth=(cfg.neo4j_user, cfg.neo4j_password),
        )

    # ── Public query API ──────────────────────────────────────────────────────

    def get_dependencies(self, qualified_name: str, depth: int = 2) -> list[dict]:
        """
        Return all nodes reachable via CALLS from the given qualified name,
        up to `depth` hops. Good for 'what does X depend on?'
        """
        with self._session() as session:
            result = session.run(
                """
                MATCH path = (start {qualified_name: $qname})-[:CALLS*1..$depth]->(dep)
                RETURN DISTINCT
                    dep.qualified_name AS name,
                    dep.file_path      AS file_path,
                    dep.language       AS language,
                    labels(dep)[0]     AS kind,
                    length(path)       AS hops
                ORDER BY hops
                """,
                qname=qualified_name,
                depth=depth,
            )
            return [dict(r) for r in result]

    def get_callers(self, qualified_name: str) -> list[dict]:
        """
        Who calls this method/function? Reverse CALLS traversal.
        Good for 'what are the entry points for X?'
        """
        with self._session() as session:
            result = session.run(
                """
                MATCH (caller)-[:CALLS]->(target {qualified_name: $qname})
                RETURN
                    caller.qualified_name AS name,
                    caller.file_path      AS file_path,
                    labels(caller)[0]     AS kind
                ORDER BY name
                """,
                qname=qualified_name,
            )
            return [dict(r) for r in result]

    def get_class_subgraph(self, class_name: str) -> dict:
        """
        Return a class and all its methods + their outgoing calls.
        Good for 'explain the structure of class X'.
        """
        with self._session() as session:
            # Get the class
            class_result = session.run(
                """
                MATCH (c:Class)
                WHERE c.name = $name OR c.qualified_name = $name
                RETURN c.qualified_name AS qualified_name,
                       c.file_path      AS file_path,
                       c.language       AS language
                LIMIT 1
                """,
                name=class_name,
            )
            class_row = class_result.single()
            if not class_row:
                return {}

            # Get methods belonging to the class
            methods_result = session.run(
                """
                MATCH (m:Method)-[:BELONGS_TO]->(c:Class)
                WHERE c.name = $name OR c.qualified_name = $name
                RETURN m.qualified_name AS method,
                       m.start_line     AS start_line,
                       m.end_line       AS end_line
                ORDER BY m.start_line
                """,
                name=class_name,
            )
            methods = [dict(r) for r in methods_result]

            # Get call edges for those methods
            calls_result = session.run(
                """
                MATCH (m:Method)-[:BELONGS_TO]->(c:Class)
                WHERE c.name = $name OR c.qualified_name = $name
                MATCH (m)-[:CALLS]->(callee)
                RETURN m.qualified_name      AS caller,
                       callee.qualified_name AS callee,
                       labels(callee)[0]     AS callee_kind
                """,
                name=class_name,
            )
            calls = [dict(r) for r in calls_result]

            return {
                "class": dict(class_row),
                "methods": methods,
                "calls": calls,
            }

    def get_file_overview(self, file_path: str) -> list[dict]:
        """Return all classes and top-level functions defined in a file."""
        with self._session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE n.file_path = $fp
                  AND labels(n)[0] IN ['Class', 'Function']
                RETURN labels(n)[0]     AS kind,
                       n.qualified_name AS name,
                       n.start_line     AS start_line
                ORDER BY start_line
                """,
                fp=file_path,
            )
            return [dict(r) for r in result]

    def search_by_name(self, name: str) -> list[dict]:
        """
        Fuzzy name search across all node types.
        Used by the router when a specific entity is mentioned.
        """
        with self._session() as session:
            result = session.run(
                """
                MATCH (n)
                WHERE toLower(n.name) CONTAINS toLower($name)
                   OR toLower(n.qualified_name) CONTAINS toLower($name)
                RETURN labels(n)[0]     AS kind,
                       n.qualified_name AS qualified_name,
                       n.file_path      AS file_path,
                       n.language       AS language
                ORDER BY size(n.qualified_name)
                LIMIT 20
                """,
                name=name,
            )
            return [dict(r) for r in result]

    def close(self) -> None:
        self._driver.close()

    # ── Internals ─────────────────────────────────────────────────────────────

    @contextmanager
    def _session(self) -> Iterator[Session]:
        with self._driver.session() as session:
            yield session

    def format_for_context(self, data: dict | list, query_type: str) -> str:
        """
        Convert raw Neo4j results into a readable context string for the LLM prompt.
        """
        if not data:
            return "No graph data found for this query."

        lines = [f"[Graph DB — {query_type}]"]

        if isinstance(data, list):
            for item in data:
                lines.append(
                    f"  • {item.get('kind', 'Node')} `{item.get('name') or item.get('qualified_name', '?')}`"
                    + (f"  [{item.get('file_path', '')}]" if item.get("file_path") else "")
                )
        elif isinstance(data, dict):
            if "class" in data:
                cls = data["class"]
                lines.append(f"  Class: `{cls.get('qualified_name')}`  ({cls.get('language')})")
                lines.append(f"  File : {cls.get('file_path')}")
                lines.append(f"  Methods ({len(data.get('methods', []))}):")
                for m in data.get("methods", []):
                    lines.append(f"    - {m['method']}  (lines {m['start_line']}–{m['end_line']})")
                if data.get("calls"):
                    lines.append("  Internal calls:")
                    for c in data["calls"][:15]:  # cap to avoid token bloat
                        lines.append(f"    {c['caller']} → {c['callee']}")

        return "\n".join(lines)
