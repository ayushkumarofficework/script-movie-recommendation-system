"""CLI entrypoint: ingest all movies listed in the seed manifest.

Usage:
    python -m scripts.run_seed
"""
from __future__ import annotations

from app.config import settings
from app.ingestion.seed import run_seed
from app.utils.logging import configure_logging, get_logger


def main() -> None:
    configure_logging(settings.log_level)
    log = get_logger("seed")
    reports = run_seed()
    ok = sum(1 for r in reports if r.status == "ok")
    total_chunks = sum(r.chunks_indexed for r in reports)
    log.info(
        "Indexed %d chunks across %d/%d movies",
        total_chunks,
        ok,
        len(reports),
    )
    for r in reports:
        if r.status != "ok":
            log.warning("- %s (%s): %s", r.movie_id, r.status, r.error)
        else:
            log.info(
                "- %s: %d scenes, %d chunks",
                r.movie_id,
                r.scenes_indexed,
                r.chunks_indexed,
            )


if __name__ == "__main__":
    main()
