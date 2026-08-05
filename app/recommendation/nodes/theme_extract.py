"""theme_extract node — use the LLM to extract themes, narrative patterns, tone, motifs."""
from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from app.core.llm import get_chat_model
from app.recommendation.state import GraphState
from app.utils.logging import get_logger


log = get_logger("node.theme_extract")


_PROMPT = """You are a narrative analyst. Given excerpts from a movie script, extract its
distinctive features so we can retrieve similar films.

Return STRICT JSON with these keys:
- "themes": a list of 3-7 short themes (e.g. "simulated reality", "redemption", "betrayal").
- "narrative_patterns": a list of 3-5 structural patterns (e.g. "heist team", "memory loop", "chosen one").
- "tone": a list of 2-5 mood descriptors (e.g. "philosophical", "tense", "melancholic").
- "motifs": a list of 3-7 recurring visual/dialogue motifs (e.g. "red pill", "top spinning").
- "query_text": a single 1-2 sentence paragraph that, when embedded, would retrieve the most
  thematically similar scripts to this one. Be vivid and concrete.

Do NOT include any text outside the JSON object.

SCRIPT EXCERPTS:
\"\"\"
{summary}
\"\"\"
"""


def theme_extract_node(state: GraphState) -> GraphState:
    summary = state.get("watched_summary", "")
    if not summary.strip():
        return {"error": "no summary to extract themes from"}

    try:
        llm = get_chat_model()
        resp = llm.invoke([HumanMessage(content=_PROMPT.format(summary=summary))])
        text = (resp.content or "").strip()

        # Defensive JSON parse — strip ```json fences if present.
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        data = json.loads(text)
        themes = list(data.get("themes", []))
        patterns = list(data.get("narrative_patterns", []))
        tone = list(data.get("tone", []))
        motifs = list(data.get("motifs", []))
        query_text = str(data.get("query_text", "")).strip() or " ".join(themes)

        return {
            "themes": themes,
            "narrative_patterns": patterns,
            "tone": tone,
            "motifs": motifs,
            "query_text": query_text,
            "trace": [
                {
                    "node": "theme_extract",
                    "themes": themes,
                    "narrative_patterns": patterns,
                    "query_text": query_text[:200],
                }
            ],
        }
    except Exception as exc:  # noqa: BLE001
        log.exception("theme_extract failed; falling back to plain-text query")
        # Soft fallback: use the first 500 chars of the summary as the query.
        fallback = (summary[:500] or "movie script").strip()
        return {
            "themes": [],
            "narrative_patterns": [],
            "tone": [],
            "motifs": [],
            "query_text": fallback,
            "error": f"theme_extract fallback: {exc}",
        }
