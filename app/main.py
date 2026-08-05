"""FastAPI application factory."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import health, ingest, movies, recommend
from app.config import settings
from app.utils.logging import configure_logging, get_logger


def create_app() -> FastAPI:
    """Build and return the FastAPI app."""
    configure_logging(settings.log_level)
    log = get_logger("app")

    app = FastAPI(
        title="Script-Based Movie Recommendation API",
        version=__version__,
        description="RAG over movie scripts, orchestrated by LangGraph.",
    )

    # CORS — open for local dev.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(movies.router)
    app.include_router(recommend.router)
    app.include_router(ingest.router)

    @app.on_event("startup")
    def _startup() -> None:
        log.info("App started (version=%s, chroma=%s)", __version__, settings.chroma_persist_dir)

    return app


app = create_app()
