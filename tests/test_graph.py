"""Tests for the LangGraph recommendation workflow.

These tests monkey-patch the LLM calls so they run without external API access.
"""
from __future__ import annotations

import json
import pytest

from app.core.models import Movie, Chunk
from app.core import vectorstore, movie_bank
from app.recommendation import service, graph


pytestmark = pytest.mark.asyncio


def _seed(tmp_chroma_dir) -> None:
    """Seed two movies in the bank + Chroma."""
    movie_bank.reset_movie_bank()

    # Reset to honour the new CHROMA_PERSIST_DIR.
    from app.config import get_settings
    from app.core import vectorstore as vs

    get_settings.cache_clear()
    vs.reset_caches()

    bank = movie_bank.get_movie_bank()
    bank.add(
        Movie(
            id="matrix",
            title="The Matrix",
            year=1999,
            director="Wachowskis",
            genres=["sci-fi"],
        ),
        persist=False,
    )
    bank.add(
        Movie(
            id="inception",
            title="Inception",
            year=2010,
            director="Nolan",
            genres=["sci-fi"],
        ),
        persist=False,
    )

    matrix_chunks = [
        Chunk(
            id=f"matrix:c{i}",
            scene_id=f"matrix:s{i}",
            movie_id="matrix",
            chunk_index=i,
            text=text,
            token_count=1,
            metadata={"scene_role_hint": "setup"},
        )
        for i, text in enumerate(
            ["simulated reality", "red pill scene", "chosen one hero"]
        )
    ]
    inc_chunks = [
        Chunk(
            id=f"inception:c{i}",
            scene_id=f"inception:s{i}",
            movie_id="inception",
            chunk_index=i,
            text=text,
            token_count=1,
            metadata={"scene_role_hint": "setup"},
        )
        for i, text in enumerate(
            ["dream heist", "memory loop", "spinning top"]
        )
    ]

    # Side-step the embedding step (no Cohere): upsert chunks with deterministic vectors.
    # We rely on the same Chroma call path but bypass embeddings.
    # Use a small dummy vector dimension — Chroma doesn't enforce size.
    vs.upsert_chunks(matrix_chunks, [[float(i)] * 4 for i in range(3)], movie_title="The Matrix", year=1999)
    vs.upsert_chunks(inc_chunks, [[float(i + 100)] * 4 for i in range(3)], movie_title="Inception", year=2010)


async def test_recommend_happy_path(tmp_chroma_dir, monkeypatch) -> None:
    _seed(tmp_chroma_dir)

    # Stub the LLM chat model so we don't need an API key.
    class FakeResp:
        def __init__(self, content: str) -> None:
            self.content = content

    class FakeChat:
        def invoke(self, messages):
            prompt = messages[0].content
            if "extract" in prompt.lower() or "narrative analyst" in prompt.lower():
                return FakeResp(
                    json.dumps(
                        {
                            "themes": ["simulated reality", "identity"],
                            "narrative_patterns": ["chosen one"],
                            "tone": ["philosophical"],
                            "motifs": ["red pill"],
                            "query_text": "a film about simulated reality and a chosen one",
                        }
                    )
                )
            if "0.0-1.0 scale" in prompt.lower():
                return FakeResp(
                    json.dumps(
                        {
                            "rankings": [
                                {
                                    "movie_id": "inception",
                                    "movie_title": "Inception",
                                    "score": 0.8,
                                    "reason": "Thematically similar.",
                                }
                            ]
                        }
                    )
                )
            # explain prompt
            return FakeResp(
                json.dumps(
                    {
                        "explanations": [
                            {
                                "movie_id": "inception",
                                "explanation": "Both probe the boundary between illusion and reality.",
                            }
                        ]
                    }
                )
            )

    monkeypatch.setattr("app.core.llm.get_chat_model", lambda: FakeChat())

    # Also stub embedding so we don't hit Cohere.
    import numpy as np

    def fake_embed_query(text: str):
        # Embedding for "simulated reality" is closer to vector 0 than 100.
        return [0.0] * 4 if "simulated" in text else [200.0] * 4

    monkeypatch.setattr("app.core.retrieval.embeddings.embed_query", fake_embed_query)

    graph.reset_graph_cache()
    state = await service.recommend("matrix", top_k=3, include_explanations=True)

    recs = state.get("final_recommendations") or []
    assert recs, "expected at least one recommendation"
    assert recs[0].movie_id == "inception"
    assert recs[0].explanation
    assert "simulated reality" in recs[0].themes or "identity" in recs[0].themes


async def test_recommend_unknown_movie_raises(tmp_chroma_dir) -> None:
    _seed(tmp_chroma_dir)
    graph.reset_graph_cache()
    with pytest.raises(RuntimeError):
        await service.recommend("does_not_exist", top_k=3)
