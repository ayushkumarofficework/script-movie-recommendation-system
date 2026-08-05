"""format_output node — assemble the final Recommendation list."""
from __future__ import annotations

from app.config import settings
from app.core.models import Recommendation
from app.recommendation.state import GraphState
from app.utils.logging import get_logger


log = get_logger("node.format_output")


def _combined_score(r) -> float:
    return round(0.6 * r.llm_score + 0.4 * r.base_score, 4)


def format_output_node(state: GraphState) -> GraphState:
    reranked = list(state.get("reranked") or [])
    explanations = dict(state.get("explanations") or {})
    themes = list(state.get("themes") or [])
    include_explanations = bool(state.get("include_explanations", True))

    top_k = min(settings.top_k_final, len(reranked))
    final: list[Recommendation] = []

    for r in reranked[:top_k]:
        explanation = explanations.get(r.movie_id, r.llm_reason or "")
        if not include_explanations:
            explanation = ""
        final.append(
            Recommendation(
                movie_id=r.movie_id,
                title=r.movie_title,
                year=0,  # Filled in below from the bank.
                score=_combined_score(r),
                themes=themes[:5],
                explanation=explanation,
            )
        )

    # Enrich with year from the MovieBank.
    from app.core.movie_bank import get_movie_bank

    bank = get_movie_bank()
    for rec in final:
        m = bank.get(rec.movie_id)
        if m:
            rec.year = m.year

    return {
        "final_recommendations": final,
        "trace": [{"node": "format_output", "n_final": len(final)}],
    }
