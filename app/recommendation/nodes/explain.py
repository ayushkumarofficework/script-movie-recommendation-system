"""explain node — generate 2-3 sentence explanations for the top-K reranked candidates."""
from __future__ import annotations

from langchain_core.messages import HumanMessage

from app.config import settings
from app.core.llm import get_chat_model
from app.recommendation.state import GraphState
from app.utils.logging import get_logger


log = get_logger("node.explain")


_PROMPT = """You are a film recommendation copywriter.

For each candidate below, write a 2-3 sentence natural-language explanation
explaining why it is recommended because of its thematic similarity to the WATCHED movie.
Reference at least one shared theme or pattern from the WATCHED movie's themes.
Be specific — quote a small phrase from the excerpt if it helps.

WATCHED themes: {themes}
WATCHED patterns: {patterns}

CANDIDATES:
{candidates_block}

Return STRICT JSON: {{"explanations": [{{"movie_id": "...", "explanation": "..."}}, ...]}}.
One entry per candidate, in the same order.
"""


def explain_node(state: GraphState) -> GraphState:
    reranked = list(state.get("reranked") or [])
    if not reranked:
        return {"explanations": {}, "trace": [{"node": "explain", "skipped": True}]}

    themes = list(state.get("themes") or [])
    patterns = list(state.get("narrative_patterns") or [])
    top_k = min(settings.top_k_final, len(reranked))

    # Build a lookup so we can attach the top supporting chunk for each candidate.
    cand_lookup = {c.movie_id: c for c in (state.get("candidates") or [])}
    lines: list[str] = []
    for i, r in enumerate(reranked[:top_k]):
        cand = cand_lookup.get(r.movie_id)
        excerpt = (cand.top_chunks[0][0] if cand and cand.top_chunks else "")[:600]
        lines.append(
            f"{i + 1}. {r.movie_title} (id={r.movie_id})\n"
            f"   base_score={r.base_score:.3f} llm_score={r.llm_score:.3f}\n"
            f"   excerpt: {excerpt}"
        )

    try:
        llm = get_chat_model()
        resp = llm.invoke(
            [
                HumanMessage(
                    content=_PROMPT.format(
                        themes=", ".join(themes) if themes else "(unspecified)",
                        patterns=", ".join(patterns) if patterns else "(unspecified)",
                        candidates_block="\n".join(lines),
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
        import json

        data = json.loads(text)
        explanations = {
            item.get("movie_id"): str(item.get("explanation", "")).strip()
            for item in data.get("explanations", [])
            if item.get("movie_id")
        }
        return {
            "explanations": explanations,
            "trace": [{"node": "explain", "n_explained": len(explanations)}],
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("explain node failed; producing generic explanations")
        watched = state.get("watched_movie")
        watched_title = getattr(watched, "title", None) or (
            watched.get("title") if isinstance(watched, dict) else None
        ) or "the watched movie"
        explanations = {
            r.movie_id: (
                f"Thematically similar to {watched_title} "
                f"based on retrieved script content (LLM explanation unavailable)."
            )
            for r in reranked[:top_k]
        }
        return {
            "explanations": explanations,
            "error": f"explain fallback: {exc}",
            "trace": [{"node": "explain", "fallback": True, "reason": str(exc)}],
        }
