"""Public entrypoint for the recommendation pipeline."""
from __future__ import annotations

import asyncio
import uuid

from app.core.models import Movie, RetrievalCandidate
from app.core.movie_bank import get_movie_bank
from app.recommendation.graph import get_graph
from app.recommendation.state import GraphState
from app.utils.logging import get_logger


log = get_logger("recommendation.service")


async def recommend(watched_movie_id: str, *, top_k: int = 5, include_explanations: bool = True) -> dict:
    """Run the LangGraph workflow and return a dict for the API layer.

    Raises RuntimeError if the watched movie is not in the bank.
    """
    bank = get_movie_bank()
    movie = bank.get(watched_movie_id)
    if not movie:
        raise RuntimeError(f"movie not found in bank: {watched_movie_id}")

    initial: GraphState = {
        "watched_movie": Movie(
            id=movie.id,
            title=movie.title,
            year=movie.year,
            director=movie.director,
            genres=movie.genres,
            runtime_min=movie.runtime_min,
            source_uri=movie.source_uri,
        ),
        "top_k": top_k,
        "include_explanations": include_explanations,
        "trace_id": str(uuid.uuid4()),
        "candidates": [],
        "reranked": [],
        "explanations": {},
        "final_recommendations": [],
        "themes": [],
        "narrative_patterns": [],
        "tone": [],
        "motifs": [],
        "trace": [],
        "error": None,
    }

    graph = get_graph()
    final_state: GraphState = await graph.ainvoke(initial)
    return final_state


def recommend_sync(watched_movie_id: str, *, top_k: int = 5, include_explanations: bool = True) -> dict:
    """Synchronous wrapper — useful for tests and one-off scripts."""
    return asyncio.run(recommend(watched_movie_id, top_k=top_k, include_explanations=include_explanations))


def retrieval_candidates_to_state(candidates: list[RetrievalCandidate]) -> list[dict]:
    """Helper for API serialization if needed."""
    return [
        {
            "movie_id": c.movie_id,
            "movie_title": c.movie_title,
            "score": c.score,
            "n_chunks": len(c.top_chunks),
        }
        for c in candidates
    ]
