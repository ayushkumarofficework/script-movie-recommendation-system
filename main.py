"""Uvicorn entrypoint for the movie recommendation API.

Run with:
    uvicorn main:app --reload --port 8000

Or directly:
    python main.py
"""
from __future__ import annotations

import uvicorn

from app.config import settings


def main() -> None:
    """Start uvicorn with the configured host/port."""
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
