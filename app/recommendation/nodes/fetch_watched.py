"""fetch_watched node — load the watched movie's summary from the bank."""
from __future__ import annotations

from app.core.chunking import chunk_to_text
from app.recommendation.state import GraphState


# In this implementation, watched movie summaries are derived from chunks stored
# in Chroma for that movie. The first 20 chunks (most likely early scenes) are
# concatenated into a brief summary suitable for LLM prompts.


class WatchedMovieNotIndexedError(RuntimeError):
    """Raised when the watched movie has no chunks in the vector store.

    This is a user-visible error: the movie is in the bank but has not been
    ingested. The route layer converts this to 404 with a clear message.
    """

    def __init__(self, movie_id: str) -> None:
        super().__init__(
            f"movie '{movie_id}' is in the bank but has no chunks in the vector store. "
            "Run `python -m scripts.run_seed` (or POST /api/v1/ingest) to index its script."
        )
        self.movie_id = movie_id


def fetch_watched_node(state: GraphState) -> GraphState:
    """Populate `watched_summary` and `watched_chunks_text` in the state."""
    from app.core import vectorstore
    from app.utils.logging import get_logger

    log = get_logger("node.fetch_watched")
    movie = state.get("watched_movie")
    if not movie:
        return {"error": "watched_movie missing"}

    try:
        # Read the stored chunks for this movie directly from Chroma.
        coll = vectorstore.get_collection()
        res = coll.get(where={"movie_id": movie.id}, include=["documents"])
        documents = (res.get("documents") or []) if isinstance(res, dict) else []
        if not documents:
            log.warning("No stored chunks found for movie %s", movie.id)

        # Wrap as Chunk-like objects for chunk_to_text.
        from types import SimpleNamespace

        chunks = [SimpleNamespace(text=d) for d in documents]

        # If the watched movie id is well-formed but Chroma has no chunks for it,
        # the graph has no useful input — surface a clear error instead of
        # letting the rest of the pipeline silently produce empty results.
        if not documents:
            raise WatchedMovieNotIndexedError(movie.id)

        summary = chunk_to_text(chunks, max_chunks=20, max_chars=6000)

        return {
            "watched_summary": summary,
            "watched_chunks_text": summary,
            "trace": [{"node": "fetch_watched", "chunks": len(documents)}],
        }
    except WatchedMovieNotIndexedError:
        # Let the route layer translate this to a 404.
        raise
    except Exception as exc:  # noqa: BLE001
        log.exception("fetch_watched failed")
        return {"error": str(exc)}
