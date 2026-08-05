"""Seed manifest loader.

Reads the seed manifest at data/seed_manifest.json, ingests any movies whose
script files exist on disk, and writes nothing destructive for missing files.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.core.models import IngestionReport, Movie
from app.ingestion.pipeline import ingest_movie
from app.utils.logging import get_logger


log = get_logger(__name__)


def _resolve_script_path(entry: dict, manifest_path: Path) -> Path | None:
    """Resolve the script file path for a manifest entry.

    Resolution order:
      1. Explicit ``file_path`` field in the manifest (relative to manifest dir).
      2. Convention: ``raw_scripts/{movie_id}.txt`` next to the manifest.
      3. Convention: ``raw_scripts/{slug_without_year}.txt`` for ids like
         ``casablanca_1942`` (strip trailing ``_<year>`` if that file exists).

    Returns None if neither resolves to an existing file.
    """
    raw_scripts_dir = manifest_path.parent / "raw_scripts"

    explicit = entry.get("file_path")
    if explicit:
        candidate = (manifest_path.parent / explicit).resolve()
        if candidate.exists():
            return candidate
        # Fall through to convention-based resolution if the explicit path
        # is stale (e.g. a file was moved or the path was wrong).

    movie_id = entry.get("id") or ""
    if movie_id:
        candidates = [raw_scripts_dir / f"{movie_id}.txt"]
        # Strip a trailing _<year> if present (ids like casablanca_1942).
        for sep in ("_", "-"):
            head, _, tail = movie_id.rpartition(sep)
            if tail.isdigit() and len(tail) == 4:
                candidates.append(raw_scripts_dir / f"{head}.txt")
                break
        for c in candidates:
            if c.exists():
                return c.resolve()

    return None


def load_seed_manifest(path: Path | None = None) -> list[tuple[Movie, Path]]:
    """Load movie entries from the seed manifest file.

    Returns a list of ``(Movie, script_path)`` tuples. Movies whose script is
    missing on disk are skipped with a warning.
    """
    p = path or settings.seed_manifest_abs()
    if not p.exists():
        log.warning("No seed manifest at %s", p)
        return []

    data = json.loads(p.read_text(encoding="utf-8"))
    results: list[tuple[Movie, Path]] = []
    for entry in data.get("movies", []):
        script_abs = _resolve_script_path(entry, p)
        if not script_abs:
            log.warning(
                "Skipping %s: script file not found (looked for file_path "
                "and raw_scripts/%s.txt)",
                entry.get("title"),
                entry.get("id"),
            )
            continue
        results.append(
            (
                Movie(
                    id=entry.get("id") or "",
                    title=entry["title"],
                    year=entry.get("year", 0),
                    director=entry.get("director"),
                    genres=entry.get("genres", []),
                    runtime_min=entry.get("runtime_min"),
                    source_uri=entry.get("source_uri"),
                ),
                script_abs,
            )
        )
    return results


def run_seed(path: Path | None = None) -> list[IngestionReport]:
    """Ingest every movie listed in the manifest. Idempotent."""
    manifest_path = path or settings.seed_manifest_abs()
    movies = load_seed_manifest(manifest_path)

    reports: list[IngestionReport] = []
    for movie, script_abs in movies:
        reports.append(ingest_movie(movie, script_abs))

    log.info(
        "Seed complete: %d movies processed, %d ok",
        len(reports),
        sum(1 for r in reports if r.status == "ok"),
    )
    return reports
