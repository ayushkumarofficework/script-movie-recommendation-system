"""POST /api/v1/ingest — add a script to the bank."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.core.models import Movie
from app.ingestion.pipeline import ingest_movie
from app.utils.logging import get_logger


log = get_logger("route.ingest")
router = APIRouter(prefix="/api/v1", tags=["ingest"])


@router.post("/ingest", response_model=IngestResponse)
def ingest_route(req: IngestRequest) -> IngestResponse:
    path = Path(req.source_path)
    if not path.exists():
        raise HTTPException(status_code=400, detail=f"source_path not found: {req.source_path}")

    movie = Movie(
        id=req.movie_id or "",
        title=req.title,
        year=req.year,
        director=req.director,
        genres=req.genres,
        runtime_min=req.runtime_min,
        source_uri=None,
    )

    report = ingest_movie(movie, path)
    if report.status != "ok":
        raise HTTPException(status_code=500, detail=report.error or "ingestion failed")

    return IngestResponse(
        movie_id=report.movie_id,
        scenes_indexed=report.scenes_indexed,
        chunks_indexed=report.chunks_indexed,
        status="ok",
    )
