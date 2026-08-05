"""POST /api/v1/recommend — run the LangGraph workflow."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    MovieSummary,
    RecommendationItem,
    RecommendationResponse,
    RecommendRequest,
)
from app.core.movie_bank import get_movie_bank
from app.recommendation.nodes.fetch_watched import WatchedMovieNotIndexedError
from app.recommendation.service import recommend
from app.utils.logging import get_logger


log = get_logger("route.recommend")
router = APIRouter(prefix="/api/v1", tags=["recommend"])


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend_route(req: RecommendRequest) -> RecommendationResponse:
    bank = get_movie_bank()
    movie = bank.get(req.watched_movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"movie not found: {req.watched_movie_id}")

    try:
        state = await recommend(
            req.watched_movie_id,
            top_k=req.top_k,
            include_explanations=req.include_explanations,
        )
    except WatchedMovieNotIndexedError as exc:
        # The movie is in the bank but has no chunks in the vector store.
        # This is a user-actionable error, not a 500.
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        log.exception("recommendation failed")
        raise HTTPException(status_code=500, detail=f"recommendation failed: {exc}") from exc

    watched_summary = MovieSummary(
        id=movie.id,
        title=movie.title,
        year=movie.year,
        director=movie.director,
        genres=movie.genres,
        runtime_min=movie.runtime_min,
    )

    items: list[RecommendationItem] = []
    for rec in state.get("final_recommendations", []) or []:
        bank_movie = bank.get(rec.movie_id)
        director = bank_movie.director if bank_movie else None
        genres = bank_movie.genres if bank_movie else []
        items.append(
            RecommendationItem(
                movie_id=rec.movie_id,
                title=rec.title,
                year=rec.year,
                director=director,
                genres=genres,
                score=rec.score,
                themes=rec.themes,
                explanation=rec.explanation if req.include_explanations else None,
            )
        )

    return RecommendationResponse(
        watched_movie=watched_summary,
        recommendations=items,
        themes=list(state.get("themes") or []),
        trace=list(state.get("trace") or []),
        trace_id=str(state.get("trace_id") or ""),
    )
