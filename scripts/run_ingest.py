"""CLI to ingest a single script (no server required).

Usage:
    python -m scripts.run_ingest \
        --title "My Movie" --year 2025 --format plaintext \
        --source-path data/raw_scripts/my_movie.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from app.config import settings
from app.core.models import Movie
from app.ingestion.pipeline import ingest_movie
from app.utils.logging import configure_logging, get_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest a single script into the bank.")
    parser.add_argument("--title", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--director", default=None)
    parser.add_argument("--genres", nargs="*", default=[])
    parser.add_argument("--runtime-min", type=int, default=None)
    parser.add_argument("--format", choices=["pdf", "fountain", "plaintext"], required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--movie-id", default=None)
    return parser.parse_args()


def main() -> None:
    configure_logging(settings.log_level)
    log = get_logger("ingest")
    args = _parse_args()
    path = Path(args.source_path)
    if not path.exists():
        log.error("source path not found: %s", path)
        raise SystemExit(1)

    movie = Movie(
        id=args.movie_id or "",
        title=args.title,
        year=args.year,
        director=args.director,
        genres=list(args.genres),
        runtime_min=args.runtime_min,
        source_uri=None,
    )
    report = ingest_movie(movie, path)
    if report.status != "ok":
        log.error("ingest failed: %s", report.error)
        raise SystemExit(1)
    log.info(
        "OK: %s — %d scenes, %d chunks",
        report.movie_id,
        report.scenes_indexed,
        report.chunks_indexed,
    )


if __name__ == "__main__":
    main()
