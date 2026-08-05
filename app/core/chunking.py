"""Scene-aware script chunker.

Splits a script into scenes, extracts metadata, and sub-chunks large scenes
into embedding-sized pieces at dialogue boundaries.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import tiktoken

from app.core.models import Chunk, Scene
from app.utils.text import (
    clean_script_text,
    extract_characters,
    normalize_time_of_day,
    score_role_hint,
    split_location,
)


_ENCODER = tiktoken.get_encoding("cl100k_base")

# Primary slugline regex — Fountain-style.
_SLUGLINE_RE = re.compile(
    r"^(INT\.|EXT\.|EST\.|INT/EXT\.|I/E\.)\s+.+$",
    re.IGNORECASE | re.MULTILINE,
)
# Plaintext fallbacks (applied in order; first match wins for a line).
_PLAINTEXT_HEADERS = [
    re.compile(r"^SCENE\s+\d+", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^ACT\s+[IVX]+", re.IGNORECASE | re.MULTILINE),
]

# Dialogue-block separator: blank line.
_BLANK_LINE = re.compile(r"\n\s*\n")


def _token_count(text: str) -> int:
    return len(_ENCODER.encode(text))


def _split_into_scenes(text: str) -> list[tuple[str, int, int]]:
    """Return (scene_text, start_offset, end_offset) tuples for each scene."""
    # Find candidate split points: a slugline or fallback header at the start of a line.
    boundaries: list[int] = []

    for m in _SLUGLINE_RE.finditer(text):
        boundaries.append(m.start())

    if len(boundaries) < 2:
        # Try plaintext fallbacks.
        for pat in _PLAINTEXT_HEADERS:
            cand = [m.start() for m in pat.finditer(text)]
            if len(cand) >= 2:
                boundaries = cand
                break

    if not boundaries:
        # Treat the whole thing as one scene.
        return [(text, 0, len(text))]

    # De-duplicate / sort.
    boundaries = sorted(set(boundaries))

    scenes: list[tuple[str, int, int]] = []
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text:
            scenes.append((chunk_text, start, end))

    # If the text starts before the first boundary, prepend it to the first scene.
    if boundaries and boundaries[0] > 50 and text[: boundaries[0]].strip():
        prefix = text[: boundaries[0]].strip()
        first_text, first_start, first_end = scenes[0]
        scenes[0] = (prefix + "\n\n" + first_text, first_start, first_end)

    return scenes


def _scene_first_line(scene_text: str) -> str:
    """Return the slugline / heading line of a scene block."""
    for line in scene_text.splitlines():
        line = line.strip()
        if line:
            return line
    return ""


def _make_scene(
    scene_text: str,
    scene_index: int,
    script_id: str,
    start: int,
    end: int,
    total_scenes: int,
) -> Scene:
    heading = _scene_first_line(scene_text)
    time_of_day = normalize_time_of_day(heading)
    location = split_location(heading)
    characters = extract_characters(scene_text)
    role_hint = score_role_hint(scene_index, total_scenes)

    scene_id = f"{script_id}:s{scene_index:04d}"

    return Scene(
        id=scene_id,
        script_id=script_id,
        scene_number=scene_index,
        heading=heading,
        location=location,
        time_of_day=time_of_day,
        characters=characters,
        raw_text=scene_text,
        start_offset=start,
        end_offset=end,
        metadata_extra={"scene_role_hint": role_hint},
    )


def _sub_chunk_scene(
    scene: Scene,
    movie_id: str,
    max_tokens: int,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Split a single scene into 1+ Chunks of <= target_tokens."""
    text = scene.raw_text
    if _token_count(text) <= max_tokens:
        # Whole scene is one chunk.
        return [
            Chunk(
                id=f"{scene.id}:c0",
                scene_id=scene.id,
                movie_id=movie_id,
                chunk_index=0,
                text=text,
                token_count=_token_count(text),
                metadata={
                    "scene_number": scene.scene_number,
                    "location": scene.location or "",
                    "time_of_day": scene.time_of_day or "",
                    "characters_csv": ",".join(scene.characters),
                    "scene_role_hint": scene.metadata_extra.get("scene_role_hint", "setup"),
                },
            )
        ]

    # Split at dialogue-block (blank-line) boundaries.
    blocks = [b.strip() for b in _BLANK_LINE.split(text) if b.strip()]

    # If a single block is too large, we'll do a sliding window later.
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = _token_count(block)
        if block_tokens >= target_tokens:
            # Flush current piece first.
            if current:
                pieces.append("\n\n".join(current))
                current = []
                current_tokens = 0
            # Sliding-window split of the oversized block.
            tokens = _ENCODER.encode(block)
            step = max(target_tokens - overlap_tokens, 1)
            for start in range(0, len(tokens), step):
                piece = _ENCODER.decode(tokens[start : start + target_tokens])
                pieces.append(piece)
            continue

        if current_tokens + block_tokens > target_tokens and current:
            pieces.append("\n\n".join(current))
            current = [block]
            current_tokens = block_tokens
        else:
            current.append(block)
            current_tokens += block_tokens

    if current:
        pieces.append("\n\n".join(current))

    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                id=f"{scene.id}:c{idx}",
                scene_id=scene.id,
                movie_id=movie_id,
                chunk_index=idx,
                text=piece,
                token_count=_token_count(piece),
                metadata={
                    "scene_number": scene.scene_number,
                    "location": scene.location or "",
                    "time_of_day": scene.time_of_day or "",
                    "characters_csv": ",".join(scene.characters),
                    "scene_role_hint": scene.metadata_extra.get("scene_role_hint", "setup"),
                },
            )
        )
    return chunks


def split_script(
    script_text: str,
    *,
    script_id: str,
    movie_id: str,
    max_scene_tokens: int = 1500,
    chunk_target_tokens: int = 1200,
    chunk_overlap_tokens: int = 150,
) -> tuple[list[Scene], list[Chunk]]:
    """Parse and chunk a full script.

    Returns a tuple of (scenes, chunks). Chunks reference their scenes.
    """
    text = clean_script_text(script_text)
    raw_scenes = _split_into_scenes(text)

    scenes: list[Scene] = []
    for i, (scene_text, start, end) in enumerate(raw_scenes):
        scenes.append(_make_scene(scene_text, i, script_id, start, end, len(raw_scenes)))

    chunks: list[Chunk] = []
    for scene in scenes:
        chunks.extend(
            _sub_chunk_scene(
                scene,
                movie_id=movie_id,
                max_tokens=max_scene_tokens,
                target_tokens=chunk_target_tokens,
                overlap_tokens=chunk_overlap_tokens,
            )
        )
    return scenes, chunks


def chunk_to_text(chunks: Sequence[Chunk], max_chunks: int = 20, max_chars: int = 6000) -> str:
    """Concatenate up to `max_chunks` chunk texts, truncated to `max_chars` for LLM prompts."""
    parts: list[str] = []
    total = 0
    for c in chunks[:max_chunks]:
        parts.append(c.text.strip())
        total += len(c.text)
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(parts)[:max_chars]
