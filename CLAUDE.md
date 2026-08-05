---
name: project-claude-md
description: Root guidance file for future Claude Code instances working in the movie-recommendation-system repo.
metadata:
  type: project
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A RAG-based movie recommendation engine that recommends films by comparing their **actual scripts** (not metadata). When a user finishes watching a movie, the service retrieves the most thematically and narratively similar movies and uses an LLM to explain *why*. Scripts are chunked by scene, embedded with Cohere, stored in ChromaDB, retrieved via cosine similarity, then reranked and explained by a LangGraph-orchestrated LLM workflow.

## Common commands

All commands assume the project root and the venv activated (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere).

```bash
# Install deps
pip install -r requirements.txt

# Seed the vector bank from data/seed_manifest.json
python -m scripts.run_seed

# Ingest a single script (no server)
python -m scripts.run_ingest \
    --title "Dune" --year 2021 --format plaintext \
    --source-path data/raw_scripts/dune.txt \
    --director "Denis Villeneuve" --genres sci-fi drama

# Run the API
python main.py                                # production-style
uvicorn app.main:app --reload --port 8000     # dev with autoreload

# Lint
ruff check .

# Tests (all)
pytest -v

# Tests (single file / single test)
pytest -v tests/test_graph.py
pytest -v tests/test_chunking.py::test_specific_name
```

Test config (from `pyproject.toml`): `testpaths = ["tests"]`, `asyncio_mode = "auto"`.

## High-level architecture

The system has two pipelines that share the same vector store.

**Online path** — recommendation request:
```
HTTP POST /api/v1/recommend
  → app/api/routes/recommend.py
  → app/recommendation/service.py (builds GraphState, calls ainvoke)
  → app/recommendation/graph.py (StateGraph)
      fetch_watched → theme_extract (LLM) → retrieve_candidates (Chroma cosine)
                    → rerank (LLM) → [explain (LLM)] → format_output → END
  → RecommendationResponse (Pydantic)
```

**Offline path** — ingestion:
```
script file (PDF / Fountain / plaintext)
  → app/ingestion/parser.py
  → app/core/chunking.py      (scene-aware, sub-chunked at dialogue boundaries)
  → app/core/embeddings.py    (Cohere embed-english-v3.0, search_document input type)
  → app/core/vectorstore.py   (Chroma upsert; collection metadata hnsw:space=cosine)
  → app/core/movie_bank.py    (JSON-backed Movie registry; persist-on-add)
```

The `MovieBank` registry (`data/seed_manifest.json`) is loaded by `MovieBank._ensure_loaded()` lazily on first access; ingestion mutates it via `bank.add(movie)` and persists.

## Key files and what they own

- `app/config.py` — `Settings` (pydantic-settings) + cached `settings` singleton; every tunable (chunk sizes, top_k values, LLM provider/model, paths) lives here.
- `app/main.py` — FastAPI factory; mounts CORS and the four routers (`health`, `movies`, `recommend`, `ingest`).
- `app/core/chunking.py` — `split_script(...)` returns `(scenes, chunks)`. Uses tiktoken `cl100k_base` for token counts. Recognises Fountain-style sluglines first, falls back to `SCENE N` / `ACT N` headers; whole document becomes one scene if no headers found.
- `app/core/embeddings.py` — Cohere wrapper with batching (`_MAX_BATCH = 96`), tenacity retries (4 attempts, exponential backoff 1→20s), and asymmetric `input_type` (documents → `search_document`, queries → `search_query`).
- `app/core/vectorstore.py` — Chroma persistent client. `upsert_chunks(...)` is idempotent (stable chunk ids). `aggregate_by_movie(...)` converts cosine distance to similarity (clamped to [0,1]) and aggregates per-movie via rank-weighted sum `sum similarity_k * 1/(rank+1)`. Uses `{"$ne": ...}` to exclude the watched movie from its own candidates.
- `app/core/movie_bank.py` — JSON-backed registry with `_ensure_loaded` lazy init; the bank file doubles as the seed manifest. `reset_movie_bank()` and `vectorstore.reset_caches()` exist for tests.
- `app/core/llm.py` — `get_chat_model()` returns a LangChain `BaseChatModel` (`ChatOpenAI` or `ChatAnthropic`) based on `settings.llm_provider`; temperature fixed at 0.2.
- `app/recommendation/state.py` — `GraphState` TypedDict (`total=False`) shared by all nodes. `trace` uses an `Annotated[list[dict], add]` reducer so node diagnostics accumulate rather than overwrite.
- `app/recommendation/graph.py` — Compiled StateGraph; conditional edge after `rerank` routes directly to `format_output` if the rerank list is empty (skips the LLM explain step).
- `app/recommendation/nodes/*` — Each node returns a dict of state deltas. LLM nodes have soft-fallback paths: `theme_extract` falls back to the summary's first 500 chars; the README notes `rerank`/`explain` fall back to base similarity / generic template on failure so the API keeps responding.
- `app/ingestion/pipeline.py` — `ingest_movie(movie, script_path)` is the single ingestion entrypoint used by both `scripts.run_seed` and `scripts.run_ingest`.

## Important behavioral details

- Movie IDs are slugified from `<title>_<year>` when not explicitly provided (`pipeline.make_movie_id`). Tests and the seed manifest pin them explicitly.
- The watched movie is **excluded** from its own candidate set via `where={"movie_id": {"$ne": exclude_movie_id}}` in `aggregate_by_movie`.
- Cosine aggregation is rank-weighted: `score = Σ similarity_k / (rank_k + 1)` over the top N results, then sorted descending.
- `fetch_watched_node` raises `WatchedMovieNotIndexedError` when the watched movie is in the bank but has no chunks in Chroma — the API route converts this to HTTP 404 (not 500).
- `recommendation/service.py::recommend()` is async; `recommend_sync()` wraps it with `asyncio.run` for tests/scripts.
- CORS is open (`allow_origins=["*"]`) — fine for local dev, do not ship as-is.
- Chroma persists locally under `./data/chroma/` (gitignored). Re-running `scripts.run_seed` is idempotent because chunk ids are stable.

## Test layout

- `tests/conftest.py` — `tmp_chroma_dir` fixture swaps `CHROMA_PERSIST_DIR` / `SEED_MANIFEST_PATH` to a temp dir and clears the cached settings + chroma clients.
- `tests/test_api.py`, `test_chunking.py`, `test_embeddings.py`, `test_graph.py`, `test_retrieval.py`, `test_vectorstore.py` — unit tests covering each subsystem.
- For graph tests: `app.recommendation.graph.reset_graph_cache()` exists so a fresh compiled graph picks up monkeypatched settings.

## Configuration knobs (see `app/config.py`)

| Setting | Default | Purpose |
|---|---|---|
| `TOP_K_RETRIEVAL` | 50 | Chunks fetched from Chroma |
| `TOP_K_RERANK` | 20 | Candidates the LLM reranks |
| `TOP_K_FINAL` | 5 | Final recommendations in the response |
| `CHUNK_TARGET_TOKENS` | 1200 | Sub-chunk target |
| `CHUNK_OVERLAP_TOKENS` | 150 | Sliding-window overlap for oversized dialogue |
| `MAX_SCENE_TOKENS` | 1500 | If a scene ≤ this, it stays as one chunk |
| `LLM_PROVIDER` | openai | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model used by LangGraph |

## Security note

`.env` is currently tracked in the working tree (per `git status`) and contains a real Cohere API key. `.gitignore` already excludes `.env`, but the file was committed earlier — **rotate the Cohere key** and remove `.env` from git history before any push or PR. Do not commit future secrets; copy from `.env.example` (referenced by the README) instead.