"""LangGraph recommendation workflow — compile the StateGraph."""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.recommendation.nodes.explain import explain_node
from app.recommendation.nodes.fetch_watched import fetch_watched_node
from app.recommendation.nodes.format_output import format_output_node
from app.recommendation.nodes.rerank import rerank_node
from app.recommendation.nodes.retrieve_candidates import retrieve_candidates_node
from app.recommendation.nodes.theme_extract import theme_extract_node
from app.recommendation.state import GraphState
from app.utils.logging import get_logger


log = get_logger(__name__)


# Singleton compiled graph (lazy).
_graph = None


def _build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("fetch_watched", fetch_watched_node)
    builder.add_node("theme_extract", theme_extract_node)
    builder.add_node("retrieve_candidates", retrieve_candidates_node)
    builder.add_node("rerank", rerank_node)
    builder.add_node("explain", explain_node)
    builder.add_node("format_output", format_output_node)

    builder.add_edge(START, "fetch_watched")
    builder.add_edge("fetch_watched", "theme_extract")
    builder.add_edge("theme_extract", "retrieve_candidates")
    builder.add_edge("retrieve_candidates", "rerank")

    # Conditional branching: if rerank returned nothing, skip the LLM explain node.
    def _route_after_rerank(state: GraphState) -> str:
        reranked = state.get("reranked") or []
        if not reranked:
            log.info("No reranked candidates; routing to format_output directly")
            return "format_output"
        return "explain"

    builder.add_conditional_edges(
        "rerank",
        _route_after_rerank,
        {"explain": "explain", "format_output": "format_output"},
    )
    builder.add_edge("explain", "format_output")
    builder.add_edge("format_output", END)

    return builder.compile()


def get_graph():
    """Return the compiled graph (lazy + cached)."""
    global _graph
    if _graph is None:
        _graph = _build_graph()
    return _graph


def reset_graph_cache() -> None:
    """For tests."""
    global _graph
    _graph = None
