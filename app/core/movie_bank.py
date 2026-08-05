"""Movie registry — JSON-backed, in-memory cache."""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from app.config import settings
from app.core.models import Movie
from app.utils.logging import get_logger


log = get_logger(__name__)


_MOVIE_FIELDS = {f.name for f in fields(Movie)}


class MovieBank:
    """A simple, process-local registry of movies.

    Backed by a JSON file at settings.seed_manifest_path. Suitable for small
    banks (hundreds of movies); replace with SQLite / Postgres as the bank grows.
    """

    def __init__(self, manifest_path: Path | None = None) -> None:
        self.path = manifest_path or settings.seed_manifest_abs()
        self._movies: dict[str, Movie] = {}
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                for entry in data.get("movies", []):
                    # The seed manifest carries extra metadata (e.g. file_path,
                    # format, notes) that the Movie dataclass doesn't model.
                    # Drop unknown keys so future manifest additions can't break
                    # loading silently.
                    filtered = {k: v for k, v in entry.items() if k in _MOVIE_FIELDS}
                    movie = Movie(**filtered)
                    self._movies[movie.id] = movie
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not load movie bank at %s: %s", self.path, exc)
        self._loaded = True

    def add(self, movie: Movie, *, persist: bool = True) -> None:
        """Add or replace a movie; optionally persist to disk."""
        self._ensure_loaded()
        self._movies[movie.id] = movie
        if persist:
            self._persist()

    def get(self, movie_id: str) -> Movie | None:
        self._ensure_loaded()
        return self._movies.get(movie_id)

    def list(self) -> list[Movie]:
        self._ensure_loaded()
        return sorted(self._movies.values(), key=lambda m: m.title)

    def _persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "movies": [
                {
                    "id": m.id,
                    "title": m.title,
                    "year": m.year,
                    "director": m.director,
                    "genres": m.genres,
                    "runtime_min": m.runtime_min,
                    "source_uri": m.source_uri,
                }
                for m in self._movies.values()
            ]
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


_bank_singleton: MovieBank | None = None


def get_movie_bank() -> MovieBank:
    """Singleton accessor."""
    global _bank_singleton
    if _bank_singleton is None:
        _bank_singleton = MovieBank()
    return _bank_singleton


def reset_movie_bank() -> None:
    """For tests."""
    global _bank_singleton
    _bank_singleton = None
