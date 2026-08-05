"""Tests for the Cohere embeddings wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core import embeddings


@pytest.fixture
def fake_cohere(monkeypatch):
    """Replace the Cohere client with a mock that returns dummy vectors."""
    calls = {"docs": [], "queries": []}

    def fake_embed(model, texts, input_type):
        if input_type == "search_document":
            calls["docs"].extend(texts)
        elif input_type == "search_query":
            calls["queries"].extend(texts)
        # Return deterministic vectors.
        return [[float(len(t))] * 4 for t in texts]

    client = MagicMock()
    client.embed.side_effect = fake_embed
    monkeypatch.setattr(embeddings, "get_cohere_client", lambda: client)
    return calls


def test_embed_documents_uses_search_document(fake_cohere) -> None:
    out = embeddings.embed_documents(["hello", "world"])
    assert fake_cohere["docs"] == ["hello", "world"]
    assert fake_cohere["queries"] == []
    assert len(out) == 2
    assert all(len(v) == 4 for v in out)


def test_embed_query_uses_search_query(fake_cohere) -> None:
    out = embeddings.embed_query("a question")
    assert fake_cohere["queries"] == ["a question"]
    assert fake_cohere["docs"] == []
    assert len(out) == 4


def test_embedding_batches_over_96(fake_cohere) -> None:
    texts = [f"text {i}" for i in range(200)]
    embeddings.embed_documents(texts)
    # 200 texts / 96 per batch = 3 batches.
    assert len(fake_cohere["docs"]) == 200
