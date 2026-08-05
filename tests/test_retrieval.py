"""Tests for retrieval.py."""
from __future__ import annotations

from unittest.mock import patch

from app.core import retrieval
from app.core.models import Chunk


def test_retrieve_similar_uses_embed_query_and_aggregates(tmp_chroma_dir) -> None:
    # Seed two movies.
    a = [Chunk(id="a:c0", scene_id="a:c0", movie_id="a", chunk_index=0, text="alpha", token_count=1, metadata={})]
    b = [Chunk(id="b:c0", scene_id="b:c0", movie_id="b", chunk_index=0, text="beta", token_count=1, metadata={})]
    from app.core import vectorstore

    vectorstore.upsert_chunks(a, [[0.1] * 4], movie_title="A", year=2020)
    vectorstore.upsert_chunks(b, [[0.9] * 4], movie_title="B", year=2020)

    with patch("app.core.retrieval.embeddings.embed_query", return_value=[0.1] * 4):
        cands = retrieval.retrieve_similar("query", watched_movie_id=None, top_k=5)
    assert cands[0].movie_id == "a"
    assert all(c.movie_id != "watched" for c in cands)


def test_retrieve_similar_empty_query_returns_empty() -> None:
    assert retrieval.retrieve_similar("", watched_movie_id=None) == []
    assert retrieval.retrieve_similar("   ", watched_movie_id=None) == []
