"""
agent/graph_agent.py
─────────────────────
LangGraph-powered agent that routes queries to the right retrieval engine
and synthesizes answers using Groq (Llama 3).

State graph
───────────
  [START]
     │
     ▼
  classify_query          → decide: "graph" | "vector" | "hybrid"
     │
     ├─ graph  ──► retrieve_from_graph  ──► build_context ──► generate_answer ──► [END]
     ├─ vector ──► retrieve_from_vector ──► build_context ──► generate_answer ──► [END]
     └─ hybrid ──► retrieve_from_graph  ──┐
                   retrieve_from_vector  ──► build_context ──► generate_answer ──► [END]

Routing heuristics
──────────────────
  Graph  : "depends on", "calls", "inherits", "architecture", "structure",
           "what classes", "module", "relationship", "imports"
  Vector : "how does", "implement", "logic", "what does this do",
           "explain", "example", "show me the code"
  Hybrid : anything ambiguous — both engines run, context is merged
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Literal, TypedDict

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from core.config import cfg
from retrieval.graph_retriever import GraphRetriever
from retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)

# ── Routing keyword sets ──────────────────────────────────────────────────────

_GRAPH_KEYWORDS = re.compile(
    r"\b(depend|call|inherit|extend|import|architecture|structure|module|relationship"
    r"|class|interface|package|coupling|hierarchy|subclass|override)\b",
    re.IGNORECASE,
)
_VECTOR_KEYWORDS = re.compile(
    r"\b(how does|implement|logic|algorithm|example|explain|show me|what does.*do"
    r"|works|detail|step|process|calculate|validate|parse|handle)\b",
    re.IGNORECASE,
)


# ── Agent state ───────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    route: str                   # "graph" | "vector" | "hybrid"
    graph_context: str
    vector_context: str
    final_context: str
    answer: str
    messages: Annotated[list, add_messages]


# ── Node functions ────────────────────────────────────────────────────────────

def _classify_query(state: AgentState) -> AgentState:
    query = state["query"]
    has_graph = bool(_GRAPH_KEYWORDS.search(query))
    has_vector = bool(_VECTOR_KEYWORDS.search(query))

    if has_graph and not has_vector:
        route = "graph"
    elif has_vector and not has_graph:
        route = "vector"
    else:
        route = "hybrid"   # ambiguous → hit both

    logger.info("Query routed to: %s", route)
    return {**state, "route": route}


def _make_graph_node(retriever: GraphRetriever):
    def retrieve_from_graph(state: AgentState) -> AgentState:
        query = state["query"]

        # Try to extract an entity name from the query for targeted lookup
        entity = _extract_entity(query)
        if entity:
            # Try class subgraph first, then dependency search
            data = retriever.get_class_subgraph(entity)
            if not data:
                data = retriever.get_dependencies(entity)
            context = retriever.format_for_context(data, "class subgraph" if data else "dependencies")
        else:
            # Fallback: name search across all nodes
            hits = retriever.search_by_name(query)
            context = retriever.format_for_context(hits, "name search")

        return {**state, "graph_context": context}

    return retrieve_from_graph


def _make_vector_node(retriever: VectorRetriever):
    def retrieve_from_vector(state: AgentState) -> AgentState:
        hits = retriever.search(state["query"])
        context = retriever.format_for_context(hits)
        return {**state, "vector_context": context}

    return retrieve_from_vector


def _build_context(state: AgentState) -> AgentState:
    route = state["route"]
    parts = []

    if route in ("graph", "hybrid") and state.get("graph_context"):
        parts.append(state["graph_context"])
    if route in ("vector", "hybrid") and state.get("vector_context"):
        parts.append(state["vector_context"])

    final = "\n\n".join(parts) if parts else "No relevant context found."
    return {**state, "final_context": final}


def _make_answer_node(llm: ChatGroq):
    system_prompt = SystemMessage(content="""You are an expert code analyst.
You are given context retrieved from a codebase (graph structure and/or semantic code chunks).
Answer the user's question accurately based ONLY on the provided context.
If the context is insufficient, say so clearly rather than guessing.
Format your answer in clear, concise prose. Use code references (class/method names) where helpful.
Do NOT invent code that isn't in the context.""")

    def generate_answer(state: AgentState) -> AgentState:
        user_message = HumanMessage(
            content=(
                f"Context from codebase:\n\n{state['final_context']}"
                f"\n\n---\nQuestion: {state['query']}"
            )
        )
        response = llm.invoke([system_prompt, user_message])
        answer = response.content
        return {
            **state,
            "answer": answer,
            "messages": [HumanMessage(content=state["query"]), response],
        }

    return generate_answer


# ── Conditional edge ──────────────────────────────────────────────────────────

def _route_edge(state: AgentState) -> Literal["retrieve_graph", "retrieve_vector", "retrieve_both"]:
    r = state["route"]
    if r == "graph":
        return "retrieve_graph"
    if r == "vector":
        return "retrieve_vector"
    return "retrieve_both"


# ── Agent builder ─────────────────────────────────────────────────────────────

class CodeAnalyzerAgent:
    """
    Main entry point. Build once, call .run(query) repeatedly.
    """

    def __init__(self) -> None:
        self._graph_retriever = GraphRetriever()
        self._vector_retriever = VectorRetriever()
        self._llm = ChatGroq(
            api_key=cfg.groq_api_key,
            model=cfg.groq_model,
            temperature=0.1,
        )
        self._app = self._build_graph()

    def run(self, query: str) -> dict:
        """
        Run the agent on a natural language query.
        Returns dict with keys: answer, route, graph_context, vector_context.
        """
        initial_state: AgentState = {
            "query": query,
            "route": "",
            "graph_context": "",
            "vector_context": "",
            "final_context": "",
            "answer": "",
            "messages": [],
        }
        result = self._app.invoke(initial_state)
        return {
            "answer": result["answer"],
            "route": result["route"],
            "graph_context": result.get("graph_context", ""),
            "vector_context": result.get("vector_context", ""),
        }

    def close(self) -> None:
        self._graph_retriever.close()

    # ── Graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> any:
        graph_node = _make_graph_node(self._graph_retriever)
        vector_node = _make_vector_node(self._vector_retriever)
        answer_node = _make_answer_node(self._llm)

        builder = StateGraph(AgentState)

        # Register nodes
        builder.add_node("classify_query",    _classify_query)
        builder.add_node("retrieve_graph",    graph_node)
        builder.add_node("retrieve_vector",   vector_node)
        builder.add_node("build_context",     _build_context)
        builder.add_node("generate_answer",   answer_node)

        # Entry
        builder.add_edge(START, "classify_query")

        # Conditional branching after classification
        builder.add_conditional_edges(
            "classify_query",
            _route_edge,
            {
                "retrieve_graph":  "retrieve_graph",
                "retrieve_vector": "retrieve_vector",
                "retrieve_both":   "retrieve_graph",   # hybrid: graph first
            },
        )

        # For hybrid, graph → vector → context
        # For graph-only, graph → context (vector node skipped)
        # For vector-only, vector → context

        # After graph retrieval: if hybrid, continue to vector; else go to context
        builder.add_conditional_edges(
            "retrieve_graph",
            lambda s: "retrieve_vector" if s["route"] == "hybrid" else "build_context",
            {
                "retrieve_vector": "retrieve_vector",
                "build_context":   "build_context",
            },
        )

        builder.add_edge("retrieve_vector", "build_context")
        builder.add_edge("build_context",   "generate_answer")
        builder.add_edge("generate_answer", END)

        return builder.compile()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_entity(query: str) -> str | None:
    """
    Attempt to pull a class/method name from a query.
    Looks for CamelCase words or quoted identifiers.
    """
    # Quoted identifier: "UserService" or 'processPayment'
    quoted = re.findall(r'["\']([A-Za-z_][A-Za-z0-9_.]+)["\']', query)
    if quoted:
        return quoted[0]

    # CamelCase word (likely a class name)
    camel = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)+)\b', query)
    if camel:
        return camel[0]

    # snake_case word preceded by "method" / "function" / "class"
    snake = re.findall(
        r'\b(?:method|function|class|module)\s+([a-z_][a-z0-9_]+)', query, re.IGNORECASE
    )
    if snake:
        return snake[0]

    return None
