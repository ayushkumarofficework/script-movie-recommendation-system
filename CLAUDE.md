# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project: Script-Based Movie Recommendation System

A RAG-based movie recommendation engine. When a user finishes watching a movie, the service retrieves thematically/narratively similar films by comparing the **actual scripts** (not metadata like genre/director/cast). Scripts are scene-chunked, embedded with Cohere, stored in ChromaDB; LangGraph orchestrates retrieval, LLM rerank, and explanation generation.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (use source .venv/bin/activate on *nix)
pip install -r requirements.txt
cp .env .env              # then edit: COHERE_API_KEY (required), OPENAI_API_KEY or ANTHROPIC_API_KEY
```

Requires Python 3.11+. `pyproject.toml` pins this via `requires-python`.

## Common commands

```bash
# Seed the ChromaDB bank from data/seed_manifest.json
python -m scripts.run_seed

# Ingest a single script (not in the seed manifest)
python -m scripts.run_ingest --title "Dune" --year 2021 --format plaintext \
    --source-path data/raw_scripts/dune.txt --director "Denis Villeneuve" --genres sci-fi drama

# Run the API (two equivalent ways)
python main.py
uvicorn app.main:app --reload --port 8000

# Tests
pytest -v                       # full suite
pytest tests/test_chunking.py   # one file
pytest -k upsert                # one test by name substring

# Lint
ruff check .
ruff format .
```

After starting the API, Swagger UI is at <http://localhost:8000/docs>.

## Configuration

All knobs live in `app/config.py` (Pydantic `Settings`, loaded from `.env` via `pydantic-settings`). Key tunables:

| Setting | Default | Effect |
|---|---|---|
| `TOP_K_RETRIEVAL` | 50 | Chunks pulled from Chroma |
| `TOP_K_RERANK` | 20 | Candidates the LLM rerank scores |
| `TOP_K_FINAL` | 5 | Final recommendations in API response |
| `CHUNK_TARGET_TOKENS` | 1200 | Target chunk size (sub-chunks at dialogue boundaries) |
| `CHUNK_OVERLAP_TOKENS` | 150 | Sliding-window overlap for oversized dialogue blocks |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model used by LangGraph nodes |

`settings` is exposed as a module-level singleton (`from app.config import settings`) backed by an `lru_cache`. In tests, clear it with `get_settings.cache_clear()` after `monkeypatch.setenv(...)` to pick up new env vars (see `tests/conftest.py::tmp_chroma_dir`).

## High-level architecture

Two pipelines share one ChromaDB collection:

**Ingestion (offline, populates the bank):**
`script file (PDF/Fountain/text)` → `app.ingestion.parser.parse_script` → `app.core.chunking.split_script` (scene-aware, then sub-chunked at dialogue blanks with sliding-window fallback for oversized blocks) → `app.core.embeddings.embed_documents` (Cohere `search_document` input type, batched at 96, retried via tenacity) → `app.core.vectorstore.upsert_chunks` (ChromaDB, cosine HNSW, ids are stable so re-runs are idempotent) → `app.core.movie_bank.MovieBank.add` (JSON-backed registry at `data/seed_manifest.json`).

**Recommendation (online, per request):**
`POST /api/v1/recommend` → `app.api.routes.recommend` → `app.recommendation.service.recommend` → compiled LangGraph `app.recommendation.graph` running nodes in this fixed order:

1. `fetch_watched` — reads up to 20 stored chunks for the watched movie_id directly from Chroma (see `app/recommendation/nodes/fetch_watched.py`).
2. `theme_extract` (LLM) — extracts themes / narrative_patterns / tone / motifs + a `query_text` paragraph; falls back to first 500 chars of summary on JSON parse failure.
3. `retrieve_candidates` — embeds `query_text` (Cohere `search_query`) → `app.core.retrieval.retrieve_similar` → `vectorstore.aggregate_by_movie` (groups by `movie_id`, weighted `sum_k min(distance, 1.0) * 1/(rank+1)`, excludes the watched movie).
4. `rerank` (LLM) — for top-N candidates, scores 0.0–1.0 with a one-line reason; combined score = `0.6 * llm_score + 0.4 * base_score`. Falls back to base similarity if LLM call fails.
5. `explain` (LLM) — 2–3 sentence explanations for the top `TOP_K_FINAL`. Falls back to a generic template if LLM fails.
6. `format_output` — trims to `TOP_K_FINAL`, enriches `year` from `MovieBank`, packages `Recommendation` objects.

The `rerank` node has a **conditional edge** (`app/recommendation/graph.py::_route_after_rerank`): if rerank produced zero candidates, it skips `explain` and goes straight to `format_output`. Each LLM node is wrapped in `try/except` and writes an `error` + `trace` entry on failure so the API never 500s due to a transient LLM error — degraded answers are returned instead.

## Key conventions

- **`GraphState` is `TypedDict(total=False)`** (`app/recommendation/state.py`); nodes return dicts and LangGraph merges them. List/None defaults (`[]`, `{}`) are set explicitly in `service.recommend` because LangGraph won't fill `total=False` keys.
- **Caching is per-process.** `app/core/vectorstore.py` and `app/core/embeddings.py` use module-level `@lru_cache`. Tests reset via `vectorstore.reset_caches()` and `graph.reset_graph_cache()`. The compiled LangGraph is also lazily cached (`app/recommendation/graph.py::_graph`).
- **Chroma metadata sanitization** (`_sanitize_meta` in `vectorstore.py`): only `str | int | float | bool` are stored; lists are joined with commas. Don't put a `list[str]` directly into `Chunk.metadata`.
- **Movie IDs are slug + year** when auto-generated (`pipeline.py::make_movie_id`). Use the explicit `id` from the seed manifest when available.
- **Cohere input types are asymmetric.** Use `search_document` for chunks being indexed and `search_query` for retrieval queries (`app/core/embeddings.py`).
- **CORS is wide-open (`*`)** in `app/main.py` for local dev. Tighten before any deployment.
- **`MovieBank` is JSON-backed** (`app/core/movie_bank.py`), loaded lazily and persisted on every `add(persist=True)`. It is suitable for hundreds of movies; for larger banks replace with SQLite/Postgres.
- **`__init__.py` files are present in every package** but most are empty. `app/__init__.py` exposes `__version__ = "0.1.0"`.

## API surface

- `GET /health` — service status, version, Chroma collection size.
- `GET /api/v1/movies`, `GET /api/v1/movies/{id}` — list/fetch from `MovieBank`.
- `POST /api/v1/recommend` — body `{watched_movie_id, top_k, include_explanations}`; returns watched summary, themes, ranked recommendations with optional explanations, and a `trace_id`.
- `POST /api/v1/ingest` — body `{title, year, format, source_path, ...}`; runs the ingestion pipeline synchronously.

Schemas live in `app/api/schemas.py`; Pydantic v2.

## Testing notes

- `tests/conftest.py::tmp_chroma_dir` points `CHROMA_PERSIST_DIR` and `SEED_MANIFEST_PATH` at a temp dir, clears the settings + chroma caches, and cleans up after. **Every test that touches Chroma must request this fixture.**
- The graph tests (`tests/test_graph.py`) stub both the LLM (`app.core.llm.get_chat_model`) and the Cohere embedder (`app.core.retrieval.embeddings.embed_query`) — they don't require real API keys. Use deterministic 4-dim vectors; Chroma does not enforce a specific dim.
- `pyproject.toml` sets `asyncio_mode = "auto"` for pytest-asyncio; `@pytest.mark.asyncio` is the convention but not strictly required.
- API tests use `httpx.ASGITransport` against the in-process `create_app()` factory.

## Seed data and licensing

`data/raw_scripts/` ships 5 placeholder scripts (Casablanca 1942, It's a Wonderful Life 1946, The Matrix 1999, Inception 2010, Interstellar 2014). Per `data/LICENSE_NOTES.md`, the first two are public-domain; the others are excerpts only. Replace with rights-cleared scripts before any production deployment.

## Out of scope for /init

There is no existing CLAUDE.md, `.cursor/rules/`, `.cursorrules`, or `.github/copilot-instructions.md` in this repo. The `.idea/` directory is a JetBrains IDE project metadata folder — ignore.