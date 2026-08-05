"""LangGraph state schema for the recommendation workflow."""
from __future__ import annotations

from operator import add
from typing import Annotated

from typing_extensions import TypedDict

from app.core.models import Movie, Recommendation, RerankedCandidate, RetrievalCandidate


class GraphState(TypedDict, total=False):
    """All nodes share this dict.

    Fields are all optional (``total=False``) because LangGraph initialises the
    state with only the keys you seed it with; nodes set the rest.

    ``trace`` uses an appending reducer so each node's diagnostic entry is
    preserved instead of being overwritten by the next node.
    """

    # Inputs
    watched_movie: Movie
    top_k: int
    include_explanations: bool
    trace_id: str

    # Intermediate
    watched_summary: str
    watched_chunks_text: str
    themes: list[str]
    narrative_patterns: list[str]
    tone: list[str]
    motifs: list[str]
    query_text: str

    # Retrieval output
    candidates: list[RetrievalCandidate]

    # Rerank output
    reranked: list[RerankedCandidate]

    # Explanation output
    explanations: dict[str, str]

    # Final
    final_recommendations: list[Recommendation]

    # Operational
    error: str | None
    trace: Annotated[list[dict], add]
