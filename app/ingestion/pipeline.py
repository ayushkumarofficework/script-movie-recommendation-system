"""End-to-end ingestion: script → chunks → embeddings → ChromaDB."""
from __future__ import annotations

from pathlib import Path

from app.config import settings
from app.core.chunking import split_script
from app.core.embeddings import embed_documents
from app.core.models import IngestionReport, Movie
from app.core.movie_bank import get_movie_bank
from app.core.vectorstore import upsert_chunks
from app.ingestion import parser
from app.utils.logging import get_logger


log = get_logger(__name__)


def make_movie_id(movie: Movie) -> str:
    """Generate a stable slug if not provided."""
    if movie.id:
        return movie.id
    base = (movie.title or "untitled").lower().strip()
    base = "".join(c if c.isalnum() else "_" for c in base)
    while "__" in base:
        base = base.replace("__", "_")
    return f"{base.strip('_')}_{movie.year or 0}"


def ingest_movie(movie: Movie, script_path: Path) -> IngestionReport:
    """Run the full ingestion pipeline for a single movie.

    1. Parse the script.
    2. Split into scene-aware chunks.
    3. Embed via Cohere.
    4. Upsert into ChromaDB.
    5. Add the movie to the bank.
    """
    movie_id = make_movie_id(movie)
    movie.id = movie_id

    try:
        script = parser.parse_script(script_path)
        # Force the script's movie_id to match the resolved movie.id.
        script.movie_id = movie_id
        scenes, chunks = split_script(
            script.full_text,
            script_id=script.id,
            movie_id=movie_id,
            max_scene_tokens=settings.max_scene_tokens,
            chunk_target_tokens=settings.chunk_target_tokens,
            chunk_overlap_tokens=settings.chunk_overlap_tokens,
        )
        if not chunks:
            return IngestionReport(
                movie_id=movie_id,
                scenes_indexed=0,
                chunks_indexed=0,
                status="error",
                error="No chunks produced; check script content.",
            )

        vectors = embed_documents([c.text for c in chunks])
        upserted = upsert_chunks(
            chunks,
            vectors,
            movie_title=movie.title,
            year=movie.year or 0,
        )

        get_movie_bank().add(movie)

        log.info(
            "Ingested %s: %d scenes, %d chunks",
            movie.title,
            len(scenes),
            upserted,
        )
        return IngestionReport(
            movie_id=movie_id,
            scenes_indexed=len(scenes),
            chunks_indexed=upserted,
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Ingestion failed for %s", movie.title)
        return IngestionReport(
            movie_id=movie_id,
            scenes_indexed=0,
            chunks_indexed=0,
            status="error",
            error=str(exc),
        )
