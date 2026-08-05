"""ChromaDB persistent client wrapper for script chunks."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import settings
from app.core.models import Chunk, RetrievalCandidate
from app.utils.logging import get_logger


log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chroma_client() -> chromadb.api.ClientAPI:
    """Return a cached persistent Chroma client."""
    persist_dir = settings.chroma_persist_abs()
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(persist_dir))


@lru_cache(maxsize=1)
def get_collection() -> Collection:
    """Return (or create) the configured Chroma collection."""
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def reset_caches() -> None:
    """For tests: forget cached clients."""
    get_chroma_client.cache_clear()
    get_collection.cache_clear()


def _sanitize_meta(v: Any) -> Any:
    """Chroma metadata values must be str/int/float/bool. Coerce other types."""
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return str(v)


def upsert_chunks(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    *,
    movie_title: str,
    year: int,
) -> int:
    """Insert or update chunks in the collection. Idempotent via stable IDs."""
    if not chunks:
        return 0
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match in length"
        )

    coll = get_collection()
    ids = [c.id for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = []
    for c in chunks:
        m = {k: _sanitize_meta(v) for k, v in c.metadata.items()}
        m.update(
            {
                "movie_id": c.movie_id,
                "movie_title": movie_title,
                "year": int(year),
            }
        )
        metadatas.append(m)

    # Chroma caps batch upserts at ~41k items but we keep batches manageable.
    BATCH = 5000
    for start in range(0, len(ids), BATCH):
        end = start + BATCH
        coll.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    log.info("Upserted %d chunks for '%s'", len(ids), movie_title)
    return len(ids)


def query(
    query_embedding: list[float],
    *,
    n_results: int = 50,
    where: dict | None = None,
) -> dict:
    """Run a vector query against the collection."""
    coll = get_collection()
    kwargs: dict[str, Any] = {"query_embeddings": [query_embedding], "n_results": n_results}
    if where:
        kwargs["where"] = where
    return coll.query(**kwargs)


def count() -> int:
    """Return number of chunks in the collection."""
    return get_collection().count()


def delete_for_movie(movie_id: str) -> int:
    """Delete all chunks associated with a given movie_id."""
    coll = get_collection()
    # Collect matching ids.
    res = coll.get(where={"movie_id": movie_id}, include=[])
    ids = res.get("ids", []) if isinstance(res, dict) else []
    if not ids:
        return 0
    coll.delete(ids=ids)
    return len(ids)


def aggregate_by_movie(
    query_embedding: list[float],
    *,
    exclude_movie_id: str | None,
    n_results: int = 50,
    top_chunks_per_movie: int = 3,
) -> list[RetrievalCandidate]:
    """Query the collection and group results by movie, ranking by weighted score.

    Cosine distance is converted to cosine similarity (1 - distance), which lives
    in [-1, 1] (clamped to [0, 1] for cosine-on-unit-norm embeddings). Aggregation
    is a weighted sum of per-movie similarities: sum_k similarity_k * 1/(rank+1)
    over the top `top_chunks_per_movie` chunks per movie.
    """
    where: dict | None = None
    if exclude_movie_id:
        where = {"movie_id": {"$ne": exclude_movie_id}}

    res = query(query_embedding, n_results=n_results, where=where)

    ids_list = (res.get("ids") or [[]])[0]
    docs_list = (res.get("documents") or [[]])[0]
    dists_list = (res.get("distances") or [[]])[0]
    metas_list = (res.get("metadatas") or [[]])[0]

    bucket: dict[str, dict] = {}
    for i, (cid, doc, dist, meta) in enumerate(
        zip(ids_list, docs_list, dists_list, metas_list)
    ):
        meta = meta or {}
        movie_id = meta.get("movie_id") or "unknown"
        title = meta.get("movie_title") or movie_id
        entry = bucket.setdefault(
            movie_id,
            {"movie_id": movie_id, "movie_title": title, "weighted_score": 0.0, "chunks": []},
        )
        # Convert cosine distance to cosine similarity, clamped to [0, 1].
        # (For unit-norm embeddings cosine distance is in [0, 2], so similarity
        # = 1 - distance is in [-1, 1]. Negative values are clamped to 0.)
        similarity = max(0.0, 1.0 - float(dist))
        entry["weighted_score"] += similarity * (1.0 / (i + 1))
        if len(entry["chunks"]) < top_chunks_per_movie:
            entry["chunks"].append((doc or "", float(dist)))

    candidates = []
    for entry in bucket.values():
        candidates.append(
            RetrievalCandidate(
                movie_id=entry["movie_id"],
                movie_title=entry["movie_title"],
                score=entry["weighted_score"],
                top_chunks=entry["chunks"],
            )
        )
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
