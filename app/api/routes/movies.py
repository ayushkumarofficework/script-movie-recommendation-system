"""/api/v1/movies endpoints — list and fetch from the MovieBank."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.api.schemas import MovieSummary
from app.core.movie_bank import get_movie_bank


router = APIRouter(prefix="/api/v1/movies", tags=["movies"])


@router.get("", response_model=list[MovieSummary])
def list_movies() -> list[MovieSummary]:
    bank = get_movie_bank()
    return [
        MovieSummary(
            id=m.id,
            title=m.title,
            year=m.year,
            director=m.director,
            genres=m.genres,
            runtime_min=m.runtime_min,
        )
        for m in bank.list()
    ]


@router.get("/{movie_id}", response_model=MovieSummary)
def get_movie(movie_id: str) -> MovieSummary:
    movie = get_movie_bank().get(movie_id)
    if not movie:
        raise HTTPException(status_code=404, detail=f"movie not found: {movie_id}")
    return MovieSummary(
        id=movie.id,
        title=movie.title,
        year=movie.year,
        director=movie.director,
        genres=movie.genres,
        runtime_min=movie.runtime_min,
    )
