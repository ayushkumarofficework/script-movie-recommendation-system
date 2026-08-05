"""High-level retrieval: query embedding + similarity aggregation."""
from __future__ import annotations

from app.config import settings
from app.core import embeddings, vectorstore
from app.core.models import RetrievalCandidate
from app.utils.logging import get_logger


log = get_logger(__name__)


def retrieve_similar(
    query_text: str,
    *,
    watched_movie_id: str | None,
    top_k: int | None = None,
) -> list[RetrievalCandidate]:
    """Embed a query and return ranked retrieval candidates.

    Excludes `watched_movie_id` from results so users don't get back the movie
    they just finished watching.
    """
    if not query_text.strip():
        return []

    vec = embeddings.embed_query(query_text)
    k = top_k if top_k is not None else settings.top_k_retrieval

    candidates = vectorstore.aggregate_by_movie(
        vec,
        exclude_movie_id=watched_movie_id,
        n_results=k,
    )
    log.info("Retrieved %d candidate movies for watched=%s", len(candidates), watched_movie_id)
    return candidates
