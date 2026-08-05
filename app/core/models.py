"""Domain dataclasses representing the core entities of the system."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class Movie:
    """A film in the centralized bank."""

    id: str
    title: str
    year: int
    director: str | None = None
    genres: list[str] = field(default_factory=list)
    runtime_min: int | None = None
    source_uri: str | None = None


@dataclass
class Script:
    """A parsed script document tied to a Movie."""

    id: str
    movie_id: str
    format: Literal["pdf", "fountain", "plaintext"]
    raw_path: Path
    full_text: str
    language: str = "en"


@dataclass
class Scene:
    """A single scene extracted from a script."""

    id: str
    script_id: str
    scene_number: int
    heading: str
    location: str | None
    time_of_day: str | None
    characters: list[str]
    raw_text: str
    start_offset: int
    end_offset: int
    metadata_extra: dict[str, str] = field(default_factory=dict)


@dataclass
class Chunk:
    """An embedding-ready chunk derived from a scene."""

    id: str
    scene_id: str
    movie_id: str
    chunk_index: int
    text: str
    token_count: int
    metadata: dict[str, str | int | float] = field(default_factory=dict)


@dataclass
class RetrievalCandidate:
    """A candidate movie produced by vector retrieval."""

    movie_id: str
    movie_title: str
    score: float
    top_chunks: list[tuple[str, float]] = field(default_factory=list)


@dataclass
class RerankedCandidate:
    """A retrieval candidate that has been LLM-reranked."""

    movie_id: str
    movie_title: str
    base_score: float
    llm_score: float
    llm_reason: str


@dataclass
class Recommendation:
    """A final, user-facing recommendation."""

    movie_id: str
    title: str
    year: int
    score: float
    themes: list[str]
    explanation: str


@dataclass
class IngestionReport:
    """Result of an ingestion run for one movie."""

    movie_id: str
    scenes_indexed: int
    chunks_indexed: int
    status: str
    error: str | None = None
