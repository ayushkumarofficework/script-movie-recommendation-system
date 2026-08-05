# Script-Based Movie Recommendation System

A RAG-based movie recommendation engine that compares the **actual scripts** of
films to recommend similar titles. Built for an OTT platform: when a user
finishes watching a movie, this service retrieves the most thematically and
narratively similar movies from a centralized bank and uses an LLM to explain
*why* each recommendation fits.

The recommendation reasoning is **driven by the scripts themselves** — not by
metadata (genre, director, cast). Scripts are chunked by scene, embedded with
Cohere, stored in ChromaDB, and retrieved via cosine similarity. The retrieval
candidates are then re-ranked and explained by a LangGraph-orchestrated LLM.

## Architecture

```
                    ┌────────────────────────────┐
   Watched movie → │  FastAPI /api/v1/recommend  │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │      LangGraph workflow    │
                    │                            │
                    │ fetch_watched              │
                    │   ↓                        │
                    │ theme_extract   (LLM)      │
                    │   ↓                        │
                    │ retrieve_candidates        │ ← ChromaDB cosine
                    │   ↓                        │
                    │ rerank           (LLM)     │
                    │   ↓                        │
                    │ explain          (LLM)     │
                    │   ↓                        │
                    │ format_output              │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                          Ranked recommendations
                          with explanations
```

**Ingestion path** (offline, populates the bank):
```
Script file (PDF / Fountain / .txt)
  → parser  → chunker (scene-aware)
              → Cohere embedder
                → ChromaDB upsert
                  → MovieBank JSON registry
```

## Stack

- **Python** 3.11+
- **FastAPI** for the HTTP API
- **LangGraph** + LangChain for the LLM workflow
- **Cohere** `embed-english-v3.0` (1024-dim, asymmetric)
- **ChromaDB** (persistent, local)
- **PyPDF** + **tiktoken** for parsing and tokenization

## Setup

```bash
# 1. Clone / navigate into the project
cd "C:\Users\abhia\Desktop\projects\movie-recommendation-system"

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env .env
# Edit .env and set COHERE_API_KEY (required) and OPENAI_API_KEY (or ANTHROPIC_API_KEY)
```

## Seed the bank

The repo ships with five placeholder scripts in `data/raw_scripts/` and a
manifest in `data/seed_manifest.json`. Replace them with real scripts (see
`data/LICENSE_NOTES.md`) and then:

```bash
python -m scripts.run_seed
```

Output:
```
Indexed 38 chunks across 5/5 movies
- the_matrix_1999: 8 scenes, 8 chunks
- inception_2010: 10 scenes, 10 chunks
- ...
```

To add a new script without the seed manifest:

```bash
python -m scripts.run_ingest \
  --title "Dune" --year 2021 --format plaintext \
  --source-path data/raw_scripts/dune.txt \
  --director "Denis Villeneuve" --genres sci-fi drama
```

## Run the API

```bash
# Option A: launcher
python main.py

# Option B: uvicorn directly (with autoreload for dev)
uvicorn app.main:app --reload --port 8000
```

Open <http://localhost:8000/docs> for the auto-generated Swagger UI.

## API endpoints

### `GET /health`
```bash
curl http://localhost:8000/health
```
```json
{
  "status": "ok",
  "version": "0.1.0",
  "chroma_loaded": true,
  "collection_size": 38
}
```

### `GET /api/v1/movies`
```bash
curl http://localhost:8000/api/v1/movies
```
```json
[
  {"id": "the_matrix_1999", "title": "The Matrix", "year": 1999, ...},
  {"id": "inception_2010", "title": "Inception", "year": 2010, ...}
]
```

### `POST /api/v1/recommend` (the headline feature)
```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"watched_movie_id": "the_matrix_1999", "top_k": 5, "include_explanations": true}'
```
```json
{
  "watched_movie": {"id": "the_matrix_1999", "title": "The Matrix", "year": 1999, ...},
  "themes": ["simulated reality", "identity", "sacrifice", "chosen one"],
  "recommendations": [
    {
      "movie_id": "inception_2010",
      "title": "Inception",
      "year": 2010,
      "score": 0.81,
      "themes": ["simulated reality", "identity", "sacrifice"],
      "explanation": "Both films probe the boundary between illusion and reality ..."
    }
  ],
  "trace_id": "..."
}
```

### `POST /api/v1/ingest`
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "New Movie",
    "year": 2025,
    "format": "plaintext",
    "source_path": "data/raw_scripts/new_movie.txt"
  }'
```

## Tests

```bash
pytest -v
```

Unit tests cover the chunker, embeddings, vector store, retrieval, LangGraph
graph, and the FastAPI surface.

## Configuration

All settings live in `app/config.py` and are loaded from environment variables
or a `.env` file. See `.env.example` for the full list.

Key knobs:

| Setting | Default | Purpose |
|---|---|---|
| `TOP_K_RETRIEVAL` | 50 | How many chunks to fetch from Chroma |
| `TOP_K_RERANK` | 20 | How many candidates the LLM rerank scores |
| `TOP_K_FINAL` | 5 | Final recommendations in the response |
| `CHUNK_TARGET_TOKENS` | 1200 | Target chunk size (sub-chunked at dialogue boundaries) |
| `CHUNK_OVERLAP_TOKENS` | 150 | Sliding-window overlap for oversized dialogue blocks |
| `LLM_PROVIDER` | openai | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o-mini` | Chat model used by LangGraph |

## How the recommendation works

1. **fetch_watched** — load the first ~20 chunks of the watched movie's script
   from Chroma to build a "summary" for the LLM.
2. **theme_extract** (LLM) — extract structured features: themes, narrative
   patterns, tone, motifs, plus a `query_text` paragraph used for retrieval.
3. **retrieve_candidates** — embed `query_text` with Cohere (`search_query`
   input type) and aggregate cosine-similarity hits by movie, weighted by
   rank.
4. **rerank** (LLM) — for each top-N candidate, ask the LLM to score
   0.0–1.0 thematic similarity and give a one-line reason.
5. **explain** (LLM) — for the top 5 candidates, write a 2-3 sentence
   natural-language explanation citing specific themes.
6. **format_output** — trim to `TOP_K_FINAL`, enrich with metadata from the
   MovieBank, and return.

If the LLM rerank or explain node fails, the graph falls back gracefully
(rerank uses base similarity; explain uses a generic template) so the API
keeps responding.

## License notes

Seed scripts in `data/raw_scripts/` are illustrative excerpts. See
`data/LICENSE_NOTES.md` for the public-domain / copyright status of each
title. **Do not** ship copyrighted full scripts in production — obtain
properly licensed copies first.

## Project layout

```
app/
├── config.py              # settings
├── main.py                # FastAPI app factory
├── api/
│   ├── schemas.py         # Pydantic request/response models
│   └── routes/
│       ├── health.py
│       ├── movies.py
│       ├── recommend.py
│       └── ingest.py
├── core/
│   ├── models.py          # domain dataclasses
│   ├── chunking.py        # scene-aware chunker
│   ├── embeddings.py      # Cohere client
│   ├── vectorstore.py     # ChromaDB wrapper
│   ├── retrieval.py       # similarity aggregation
│   ├── movie_bank.py      # JSON-backed movie registry
│   └── llm.py             # LangChain chat-model factory
├── ingestion/
│   ├── parser.py          # PDF / Fountain / plaintext parsing
│   ├── pipeline.py        # end-to-end ingestion
│   └── seed.py            # manifest loader
├── recommendation/
│   ├── state.py           # GraphState TypedDict
│   ├── graph.py           # StateGraph compilation
│   ├── service.py         # public entrypoint
│   └── nodes/
│       ├── fetch_watched.py
│       ├── theme_extract.py
│       ├── retrieve_candidates.py
│       ├── rerank.py
│       ├── explain.py
│       └── format_output.py
└── utils/
    ├── logging.py
    └── text.py
```
