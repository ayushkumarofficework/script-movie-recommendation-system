"""retrieve_candidates node — query ChromaDB for the most similar movies."""
from __future__ import annotations

from app.config import settings
from app.core import retrieval
from app.recommendation.state import GraphState
from app.utils.logging import get_logger


log = get_logger("node.retrieve_candidates")


def retrieve_candidates_node(state: GraphState) -> GraphState:
    query_text = state.get("query_text", "")
    movie = state.get("watched_movie")
    if not query_text:
        return {"error": "query_text missing"}
    if not movie:
        return {"error": "watched_movie missing"}

    try:
        candidates = retrieval.retrieve_similar(
            query_text,
            watched_movie_id=movie.id,
            top_k=settings.top_k_retrieval,
        )
        return {
            "candidates": candidates,
            "trace": [
                {
                    "node": "retrieve_candidates",
                    "n_candidates": len(candidates),
                    "top_titles": [c.movie_title for c in candidates[:5]],
                }
            ],
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("retrieve_candidates failed")
        return {"error": str(exc), "candidates": []}
