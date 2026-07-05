"""
ingestion/parser.py
───────────────────
Tree-sitter based parser for Java and Python source files.

Responsibilities:
  1. Parse source into an AST.
  2. Extract structural nodes: classes, methods/functions, call expressions.
  3. Filter boilerplate (getters/setters, auto-imports, empty constructors).
  4. Produce semantic chunks suitable for embedding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

try:
    import tree_sitter_python as tspython
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class StructuralNode:
    """A class or method/function extracted from the AST."""
    kind: str           # "class" | "method" | "function"
    name: str
    qualified_name: str  # e.g. "MyClass.doSomething"
    file_path: str
    start_line: int
    end_line: int
    parent: str | None = None
    calls: list[str] = field(default_factory=list)
    language: str = "python"


@dataclass
class SemanticChunk:
    """A text chunk ready for embedding and storage in ChromaDB."""
    chunk_id: str
    text: str
    file_path: str
    start_line: int
    end_line: int
    node_name: str
    node_kind: str
    language: str
    metadata: dict = field(default_factory=dict)


# ── Boilerplate patterns ──────────────────────────────────────────────────────

_BOILERPLATE_METHOD_NAMES = re.compile(
    r"^(get|set|is|has|equals|hashCode|toString|clone|compareTo"
    r"|setUp|tearDown|__init__|__repr__|__str__|__eq__|__hash__)$"
)

_TRIVIAL_BODY_LINES = 4


def _is_boilerplate(name: str, body_lines: int) -> bool:
    return bool(_BOILERPLATE_METHOD_NAMES.match(name)) and body_lines <= _TRIVIAL_BODY_LINES


# ── Language setup ────────────────────────────────────────────────────────────

def _get_language(lang: str):
    if not TREE_SITTER_AVAILABLE:
        return None
    if lang == "python":
        return Language(tspython.language(), "python")
    if lang == "java":
        return Language(tsjava.language(), "java")
    return None


def _detect_language(path: Path) -> str | None:
    return {"py": "python", "java": "java"}.get(path.suffix.lstrip("."))


# ── Core extractor ────────────────────────────────────────────────────────────

class ASTExtractor:
    """
    Extracts structural nodes and semantic chunks from a single source file.
    Falls back to line-based chunking when Tree-sitter is unavailable.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, file_path: Path) -> tuple[list[StructuralNode], list[SemanticChunk]]:
        """
        Parse a file and return (structural_nodes, semantic_chunks).
        Gracefully falls back to text chunking if Tree-sitter is not installed.
        """
        lang = _detect_language(file_path)
        if lang is None:
            return [], []

        source = file_path.read_text(encoding="utf-8", errors="replace")

        if TREE_SITTER_AVAILABLE:
            return self._extract_with_treesitter(source, file_path, lang)
        else:
            return [], list(self._fallback_chunks(source, file_path, lang))

    # ── Tree-sitter path ──────────────────────────────────────────────────────

    def _extract_with_treesitter(
        self, source: str, file_path: Path, lang: str
    ) -> tuple[list[StructuralNode], list[SemanticChunk]]:
        language = _get_language(lang)
        parser = Parser()
        parser.set_language(language)

        tree = parser.parse(source.encode())

        lines = source.splitlines()
        nodes: list[StructuralNode] = []
        chunks: list[SemanticChunk] = []

        extract_fn = self._extract_python if lang == "python" else self._extract_java
        raw_nodes = list(extract_fn(tree.root_node, file_path, lines))

        for snode in raw_nodes:
            body_len = snode.end_line - snode.start_line
            if snode.kind in ("method", "function") and _is_boilerplate(snode.name, body_len):
                continue  # skip trivial boilerplate

            nodes.append(snode)

            # Build semantic chunk from node body
            body_text = "\n".join(lines[snode.start_line : snode.end_line + 1])
            chunk = SemanticChunk(
                chunk_id=f"{file_path}::{snode.qualified_name}",
                text=f"[{lang}] {snode.kind} `{snode.qualified_name}`:\n{body_text}",
                file_path=str(file_path),
                start_line=snode.start_line,
                end_line=snode.end_line,
                node_name=snode.qualified_name,
                node_kind=snode.kind,
                language=lang,
                metadata={"calls": snode.calls, "parent": snode.parent or ""},
            )
            chunks.append(chunk)

        return nodes, chunks

    def _extract_python(
        self, root: "Node", file_path: Path, lines: list[str]
    ) -> Iterator[StructuralNode]:
        """Walk a Python AST and yield structural nodes."""
        current_class: str | None = None

        def walk(node: "Node") -> Iterator[StructuralNode]:
            nonlocal current_class

            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                class_name = name_node.text.decode() if name_node else "Unknown"
                prev_class = current_class
                current_class = class_name
                yield StructuralNode(
                    kind="class",
                    name=class_name,
                    qualified_name=class_name,
                    file_path=str(file_path),
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    language="python",
                )
                for child in node.children:
                    yield from walk(child)
                current_class = prev_class

            elif node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                fn_name = name_node.text.decode() if name_node else "unknown"
                qualified = f"{current_class}.{fn_name}" if current_class else fn_name
                calls = _collect_calls_python(node)
                yield StructuralNode(
                    kind="method" if current_class else "function",
                    name=fn_name,
                    qualified_name=qualified,
                    file_path=str(file_path),
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    parent=current_class,
                    calls=calls,
                    language="python",
                )
                for child in node.children:
                    yield from walk(child)
            else:
                for child in node.children:
                    yield from walk(child)

        yield from walk(root)

    def _extract_java(
        self, root: "Node", file_path: Path, lines: list[str]
    ) -> Iterator[StructuralNode]:
        """Walk a Java AST and yield structural nodes."""
        current_class: str | None = None

        def walk(node: "Node") -> Iterator[StructuralNode]:
            nonlocal current_class

            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                class_name = name_node.text.decode() if name_node else "Unknown"
                prev_class = current_class
                current_class = class_name
                yield StructuralNode(
                    kind="class",
                    name=class_name,
                    qualified_name=class_name,
                    file_path=str(file_path),
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    language="java",
                )
                for child in node.children:
                    yield from walk(child)
                current_class = prev_class

            elif node.type == "method_declaration":
                name_node = node.child_by_field_name("name")
                method_name = name_node.text.decode() if name_node else "unknown"
                qualified = f"{current_class}.{method_name}" if current_class else method_name
                calls = _collect_calls_java(node)
                yield StructuralNode(
                    kind="method",
                    name=method_name,
                    qualified_name=qualified,
                    file_path=str(file_path),
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                    parent=current_class,
                    calls=calls,
                    language="java",
                )
            else:
                for child in node.children:
                    yield from walk(child)

        yield from walk(root)

    # ── Fallback: line-based chunking ─────────────────────────────────────────

    def _fallback_chunks(
        self, source: str, file_path: Path, lang: str
    ) -> Iterator[SemanticChunk]:
        """Split source into fixed-size line windows when Tree-sitter is unavailable."""
        lines = source.splitlines()
        step = max(1, self.chunk_size - self.chunk_overlap)

        for i, start in enumerate(range(0, len(lines), step)):
            end = min(start + self.chunk_size, len(lines))
            text = "\n".join(lines[start:end])
            yield SemanticChunk(
                chunk_id=f"{file_path}::chunk_{i}",
                text=text,
                file_path=str(file_path),
                start_line=start,
                end_line=end,
                node_name=f"chunk_{i}",
                node_kind="chunk",
                language=lang,
            )
            if end == len(lines):
                break


# ── Call-site collectors ──────────────────────────────────────────────────────

def _collect_calls_python(node: "Node") -> list[str]:
    calls: list[str] = []

    def walk(n: "Node") -> None:
        if n.type == "call":
            func = n.child_by_field_name("function")
            if func:
                calls.append(func.text.decode())
        for child in n.children:
            walk(child)

    walk(node)
    return list(dict.fromkeys(calls))


def _collect_calls_java(node: "Node") -> list[str]:
    calls: list[str] = []

    def walk(n: "Node") -> None:
        if n.type == "method_invocation":
            name_node = n.child_by_field_name("name")
            if name_node:
                calls.append(name_node.text.decode())
        for child in n.children:
            walk(child)

    walk(node)
    return list(dict.fromkeys(calls))