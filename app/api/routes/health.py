"""Health-check endpoint."""
from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.api.schemas import HealthResponse
from app.core import vectorstore


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health and Chroma collection size."""
    try:
        size = vectorstore.count()
        loaded = True
    except Exception:  # noqa: BLE001
        size = 0
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded",
        version=__version__,
        chroma_loaded=loaded,
        collection_size=size,
    )
