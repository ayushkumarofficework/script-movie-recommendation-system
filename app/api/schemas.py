"""Pydantic request/response models for the FastAPI surface."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --- Shared ---
class MovieSummary(BaseModel):
    id: str
    title: str
    year: int
    director: str | None = None
    genres: list[str] = Field(default_factory=list)
    runtime_min: int | None = None


# --- /health ---
class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    chroma_loaded: bool
    collection_size: int


# --- /api/v1/recommend ---
class RecommendRequest(BaseModel):
    watched_movie_id: str = Field(..., description="ID of the just-watched movie (e.g. 'the_matrix_1999')")
    top_k: int = Field(default=5, ge=1, le=20)
    include_explanations: bool = True


class RecommendationItem(BaseModel):
    movie_id: str
    title: str
    year: int
    director: str | None = None
    genres: list[str] = Field(default_factory=list)
    score: float
    themes: list[str] = Field(default_factory=list)
    explanation: str | None = None


class RecommendationResponse(BaseModel):
    watched_movie: MovieSummary
    recommendations: list[RecommendationItem]
    themes: list[str] = Field(default_factory=list)
    trace: list[dict] = Field(
        default_factory=list,
        description="Per-node diagnostics from the LangGraph run (fetch_watched, theme_extract, etc.).",
    )
    trace_id: str


# --- /api/v1/ingest ---
class IngestRequest(BaseModel):
    title: str
    year: int = Field(..., ge=1888)
    director: str | None = None
    genres: list[str] = Field(default_factory=list)
    format: Literal["pdf", "fountain", "plaintext"]
    source_path: str = Field(..., description="Local filesystem path to the script file")
    runtime_min: int | None = None
    movie_id: str | None = Field(default=None, description="Optional explicit ID; auto-derived if omitted")


class IngestResponse(BaseModel):
    movie_id: str
    scenes_indexed: int
    chunks_indexed: int
    status: Literal["ok", "error"]
    error: str | None = None
