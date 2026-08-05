"""Cohere embeddings wrapper with batching, retries, and asymmetric input types."""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

import cohere
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import settings
from app.utils.logging import get_logger


log = get_logger(__name__)

_MAX_BATCH = 96  # Cohere's hard limit per request.


class CohereEmbeddingError(RuntimeError):
    """Raised when an embedding call fails after all retries."""


@lru_cache(maxsize=1)
def get_cohere_client() -> cohere.Client:
    """Return a cached Cohere client (v5 SDK)."""
    if not settings.cohere_api_key:
        raise CohereEmbeddingError(
            "COHERE_API_KEY is not set. Configure it in .env before ingesting scripts."
        )
    return cohere.Client(api_key=settings.cohere_api_key)


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    reraise=True,
)
def _embed_batch(client: cohere.Client, texts: list[str], input_type: str) -> list[list[float]]:
    """Call Cohere embeddings for one batch."""
    response = client.embed(
        model=settings.cohere_model,
        texts=texts,
        input_type=input_type,
    )
    return response.embeddings


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents (script chunks) using `search_document` input type."""
    return _embed(texts, input_type="search_document")


def embed_query(text: str) -> list[float]:
    """Embed a single query (retrieval) using `search_query` input type."""
    vecs = _embed([text], input_type="search_query")
    return vecs[0]


def _embed(
    texts: list[str],
    *,
    input_type: Literal["search_document", "search_query", "classification", "clustering"],
) -> list[list[float]]:
    """Run embeddings with batching, retries, and explicit input type."""
    if not texts:
        return []

    client = get_cohere_client()
    out: list[list[float]] = []

    for start in range(0, len(texts), _MAX_BATCH):
        batch = texts[start : start + _MAX_BATCH]
        try:
            vecs = _embed_batch(client, batch, input_type=input_type)
        except Exception as exc:
            log.exception("Cohere embedding failed for batch starting at %d: %s", start, exc)
            raise CohereEmbeddingError(str(exc)) from exc
        out.extend(vecs)
    return out
