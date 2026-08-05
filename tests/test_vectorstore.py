"""Tests for the ChromaDB wrapper."""
from __future__ import annotations

from app.core import vectorstore
from app.core.models import Chunk


def _chunks(ids: list[str], movie_id: str, texts: list[str]) -> list[Chunk]:
    return [
        Chunk(
            id=cid,
            scene_id=cid,
            movie_id=movie_id,
            chunk_index=i,
            text=t,
            token_count=len(t),
            metadata={"scene_role_hint": "setup"},
        )
        for i, (cid, t) in enumerate(zip(ids, texts))
    ]


def test_upsert_and_count(tmp_chroma_dir) -> None:
    chunks = _chunks(["m1:c0", "m1:c1"], "m1", ["alpha", "beta"])
    embeddings = [[0.1] * 4, [0.2] * 4]
    n = vectorstore.upsert_chunks(chunks, embeddings, movie_title="M1", year=2020)
    assert n == 2
    assert vectorstore.count() == 2


def test_upsert_is_idempotent(tmp_chroma_dir) -> None:
    chunks = _chunks(["m1:c0"], "m1", ["alpha"])
    vectorstore.upsert_chunks(chunks, [[0.1] * 4], movie_title="M1", year=2020)
    vectorstore.upsert_chunks(chunks, [[0.5] * 4], movie_title="M1", year=2020)
    assert vectorstore.count() == 1


def test_aggregate_by_movie_ranks_closest_first(tmp_chroma_dir) -> None:
    # Three movies with two chunks each.
    movie_a = _chunks(["a:c0", "a:c1"], "a", ["alpha alpha", "alpha again"])
    movie_b = _chunks(["b:c0", "b:c1"], "b", ["beta beta", "beta again"])
    movie_c = _chunks(["c:c0", "c:c1"], "c", ["gamma gamma", "gamma again"])
    for mv, mid in [(movie_a, "a"), (movie_b, "b"), (movie_c, "c")]:
        vectorstore.upsert_chunks(
            mv,
            [[0.10] * 4, [0.11] * 4],
            movie_title=mid,
            year=2020,
        )

    # Query closer to "a" than "b"/"c".
    cands = vectorstore.aggregate_by_movie(
        [0.10] * 4,
        exclude_movie_id=None,
        n_results=10,
    )
    assert cands[0].movie_id == "a"
    # All three movies returned.
    assert {c.movie_id for c in cands} == {"a", "b", "c"}


def test_aggregate_by_movie_excludes_watched(tmp_chroma_dir) -> None:
    movie_a = _chunks(["a:c0"], "a", ["alpha"])
    movie_b = _chunks(["b:c0"], "b", ["beta"])
    vectorstore.upsert_chunks(movie_a, [[0.1] * 4], movie_title="a", year=2020)
    vectorstore.upsert_chunks(movie_b, [[0.9] * 4], movie_title="b", year=2020)
    cands = vectorstore.aggregate_by_movie(
        [0.1] * 4,
        exclude_movie_id="a",
        n_results=10,
    )
    assert {c.movie_id for c in cands} == {"b"}
