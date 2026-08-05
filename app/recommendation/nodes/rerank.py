"""rerank node — use the LLM to re-score and explain top retrieval candidates."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.core.models import RerankedCandidate
from app.recommendation.state import GraphState
from app.utils.logging import get_logger


log = get_logger("node.rerank")


_PROMPT = """You are a film recommendation engine. Given the WATCHED movie's themes and
a list of CANDIDATE movies with brief excerpts, score each candidate's
thematic similarity on a 0.0-1.0 scale.

WATCHED themes: {themes}
WATCHED patterns: {patterns}

CANDIDATES (JSON list):
{candidates_json}

Return STRICT JSON: a single object {{"rankings": [{{"movie_id": "...", "movie_title": "...",
"score": 0.0, "reason": "1-2 sentences"}}, ...]}}. Do not include any text outside JSON.
Sort by score descending. Score precisely — only true thematic matches deserve >0.7.
"""


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def rerank_node(state: GraphState) -> GraphState:
    candidates = list(state.get("candidates") or [])
    if not candidates:
        return {"reranked": [], "trace": [{"node": "rerank", "skipped": True}]}

    themes = list(state.get("themes") or [])
    patterns = list(state.get("narrative_patterns") or [])
    top_n = min(settings.top_k_rerank, len(candidates))

    cand_payload = [
        {
            "movie_id": c.movie_id,
            "movie_title": c.movie_title,
            "base_score": round(c.score, 4),
            "excerpt": _truncate(c.top_chunks[0][0] if c.top_chunks else "", 800),
        }
        for c in candidates[:top_n]
    ]

    try:
        llm = get_chat_model()
        resp = llm.invoke(
            [
                HumanMessage(
                    content=_PROMPT.format(
                        themes=", ".join(themes) if themes else "(none extracted)",
                        patterns=", ".join(patterns) if patterns else "(none extracted)",
                        candidates_json=json.dumps(cand_payload, indent=2),
                    )
                )
            ]
        )
        text = (resp.content or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        rankings = data.get("rankings", [])

        # Build a lookup so we can pair LLM output with the original candidate.
        cand_by_id = {c.movie_id: c for c in candidates}

        reranked: list[RerankedCandidate] = []
        for r in rankings:
            mid = r.get("movie_id")
            if mid not in cand_by_id:
                continue
            base = cand_by_id[mid].score
            llm_score = float(r.get("score", 0.0))
            reranked.append(
                RerankedCandidate(
                    movie_id=mid,
                    movie_title=r.get("movie_title") or cand_by_id[mid].movie_title,
                    base_score=base,
                    llm_score=llm_score,
                    llm_reason=str(r.get("reason", "")).strip(),
                )
            )

        # Combined score: 60% LLM + 40% base similarity.
        reranked.sort(key=lambda r: 0.6 * r.llm_score + 0.4 * r.base_score, reverse=True)

        return {
            "reranked": reranked,
            "trace": [{"node": "rerank", "n_reranked": len(reranked)}],
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("rerank failed; falling back to base similarity scores")
        # Fall back to base-similarity ranking.
        reranked = [
            RerankedCandidate(
                movie_id=c.movie_id,
                movie_title=c.movie_title,
                base_score=c.score,
                llm_score=c.score,
                llm_reason="Base similarity fallback (LLM rerank unavailable).",
            )
            for c in candidates[:top_n]
        ]
        return {
            "reranked": reranked,
            "error": f"rerank fallback: {exc}",
            "trace": [{"node": "rerank", "fallback": True}],
        }
