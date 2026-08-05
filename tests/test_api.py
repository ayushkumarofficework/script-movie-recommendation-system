"""Tests for the FastAPI surface."""
from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.core import movie_bank
from app.core.models import Movie


@pytest.mark.asyncio
async def test_health_returns_ok(tmp_chroma_dir) -> None:
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert "version" in body


@pytest.mark.asyncio
async def test_movies_list_and_get(tmp_chroma_dir) -> None:
    movie_bank.reset_movie_bank()
    bank = movie_bank.get_movie_bank()
    bank.add(
        Movie(id="m1", title="M1", year=2020, director="D", genres=["sci-fi"]),
        persist=False,
    )

    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r1 = await client.get("/api/v1/movies")
        assert r1.status_code == 200
        assert any(m["id"] == "m1" for m in r1.json())

        r2 = await client.get("/api/v1/movies/m1")
        assert r2.status_code == 200
        assert r2.json()["title"] == "M1"

        r3 = await client.get("/api/v1/movies/does_not_exist")
        assert r3.status_code == 404


@pytest.mark.asyncio
async def test_recommend_returns_404_for_unknown_movie(tmp_chroma_dir) -> None:
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/recommend",
            json={"watched_movie_id": "nope", "top_k": 3},
        )
    assert resp.status_code == 404
