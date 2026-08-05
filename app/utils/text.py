"""Text cleaning utilities used by the chunker and ingestion pipeline."""
from __future__ import annotations

import re


# Patterns
_PAGE_NUM = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
_REPEATED_SLUGLINE = re.compile(
    r"^(INT\.|EXT\.|EST\.|INT\/EXT\.|I\/E\.)\s+.+$",
    re.IGNORECASE | re.MULTILINE,
)
_NONPRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MULTI_BLANK = re.compile(r"\n{3,}")
# Character cue: a line of 1-30 uppercase letters/spaces/hyphens/apostrophes/dots, no trailing colon.
_CHAR_CUE = re.compile(r"^(?P<char>[A-Z][A-Z .'\-]{0,29})$")


def clean_script_text(text: str) -> str:
    """Normalize script text for embedding."""
    if not text:
        return ""

    # Strip NULs and other non-printables.
    text = _NONPRINTABLE.sub("", text)

    # Drop page-number-only lines.
    lines = [ln for ln in text.splitlines() if not _PAGE_NUM.match(ln)]
    text = "\n".join(lines)

    # Collapse excessive blank lines.
    text = _MULTI_BLANK.sub("\n\n", text)

    return text.strip()


def extract_characters(scene_text: str) -> list[str]:
    """Return characters who have dialogue in a scene.

    Heuristic: lines that match the character-cue pattern and are
    followed (within ~10 lines) by lowercase dialogue / action lines.
    """
    chars: list[str] = []
    lines = scene_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        m = _CHAR_CUE.match(stripped)
        if not m:
            continue
        candidate = m.group("char").strip()
        # Skip residual sluglines (already caught by the chunker) and overly long strings.
        if len(candidate) < 2 or len(candidate) > 30:
            continue
        # Look ahead for dialogue.
        for j in range(i + 1, min(i + 12, len(lines))):
            nxt = lines[j].strip()
            if not nxt:
                continue
            # If the next non-blank line starts lowercase or with punctuation, it's dialogue.
            if nxt and (nxt[0].islower() or nxt[0] in "(["):
                if candidate not in chars:
                    chars.append(candidate)
                break
            # If it matches another char cue, stop (no dialogue after this one).
            if _CHAR_CUE.match(nxt):
                break
    return chars


def normalize_time_of_day(heading: str) -> str | None:
    """Extract a normalized TIME-OF-DAY token from a scene heading line."""
    if not heading:
        return None
    upper = heading.upper()
    for token in (
        "CONTINUOUS",
        "LATER",
        "MORNING",
        "EVENING",
        "DAWN",
        "DUSK",
        "NIGHT",
        "DAY",
    ):
        # Word-boundary match; allow trailing -DAY etc.
        if re.search(rf"\b{token}\b", upper):
            return token
    return None


def split_location(heading: str) -> str | None:
    """Pull just the location portion from a slugline."""
    if not heading:
        return None
    # Heading pattern: "INT. SOME PLACE - DAY"
    parts = re.split(r"\s+-\s+", heading, maxsplit=1)
    location = parts[0]
    # Strip the leading INT./EXT.
    location = re.sub(r"^(INT\.?|EXT\.?|EST\.?|I/E\.?|INT/EXT\.?)\s+", "", location, flags=re.IGNORECASE)
    return location.strip() or None


def score_role_hint(scene_index: int, total_scenes: int) -> str:
    """Heuristically label a scene by its quartile position in the script."""
    if total_scenes <= 1:
        return "setup"
    pos = scene_index / max(total_scenes - 1, 1)
    if pos < 0.25:
        return "setup"
    if pos < 0.6:
        return "confrontation"
    if pos < 0.85:
        return "climax"
    return "resolution"
