"""Tests for the scene-aware chunker."""
from __future__ import annotations

from app.core.chunking import split_script


SAMPLE = """INT. APARTMENT - NIGHT

The room is dim. A MAN, 30s, sits at a desk. A WOMAN, also 30s, watches him from the door.

MAN
You said you'd be home by seven.

WOMAN
I was.

She crosses the room. The window behind her is full of stars.

INT. STREET - DAY

A bus stop. The MAN waits, briefcase at his feet. A boy kicks a can.

BOY
Mister, are you a cop?

MAN
(grinning)
No.

BOY
Too bad. They always have donuts.

EXT. ROOFTOP - NIGHT

The MAN stands at the edge. Wind ruffles his coat. He looks down at the city.

MAN (V.O.)
I never learned how to fall properly.

He steps back. He steps forward. He returns to the door, sits down, and waits.

INT. APARTMENT - LATER

The WOMAN is asleep. The MAN watches her from the doorway. He smiles.

MAN
I finally learned how to stay.
"""


def test_split_script_produces_scenes() -> None:
    scenes, chunks = split_script(SAMPLE, script_id="t", movie_id="m")
    # 4 scenes in the sample.
    assert len(scenes) == 4
    # Each scene should have at least one chunk.
    assert len(chunks) >= 4
    # All chunks reference their movie_id.
    assert all(c.movie_id == "m" for c in chunks)


def test_scene_metadata() -> None:
    scenes, _ = split_script(SAMPLE, script_id="t", movie_id="m")
    # First scene should have NIGHT time-of-day.
    s0 = scenes[0]
    assert s0.time_of_day == "NIGHT"
    # Characters extracted from a scene with dialogue.
    assert "MAN" in s0.characters
    assert "WOMAN" in s0.characters


def test_chunk_metadata_includes_role_hint() -> None:
    scenes, chunks = split_script(SAMPLE, script_id="t", movie_id="m")
    # The middle scene should be tagged confrontation/climax.
    role_hints = {c.metadata["scene_role_hint"] for c in chunks}
    assert "setup" in role_hints
    assert "confrontation" in role_hints or "climax" in role_hints


def test_long_scene_is_subchunked() -> None:
    # Build a giant single-block scene.
    big = "INT. WAREHOUSE - NIGHT\n\n" + ("A" * 8000)
    scenes, chunks = split_script(
        big,
        script_id="t",
        movie_id="m",
        max_scene_tokens=20,
        chunk_target_tokens=20,
        chunk_overlap_tokens=5,
    )
    assert len(chunks) >= 2
    assert all(len(c.text) > 0 for c in chunks)
