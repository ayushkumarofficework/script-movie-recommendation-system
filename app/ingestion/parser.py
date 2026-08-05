"""Multi-format script parser.

Dispatches to PDF (via pypdf), Fountain (plain text with .fountain extension),
or plaintext based on file extension / format hint.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pypdf
from pypdf.errors import PdfReadError

from app.core.models import Script
from app.utils.logging import get_logger


log = get_logger(__name__)


def detect_format(path: Path) -> Literal["pdf", "fountain", "plaintext"]:
    """Detect script format by file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in (".fountain", ".spmd"):
        return "fountain"
    return "plaintext"


def parse_pdf(path: Path) -> str:
    """Extract text from a PDF script."""
    text_parts: list[str] = []
    try:
        reader = pypdf.PdfReader(str(path))
        for page in reader.pages:
            page_text = page.extract_text() or ""
            text_parts.append(page_text)
    except PdfReadError as exc:
        log.warning("Failed to read PDF %s: %s", path, exc)
        return ""
    # Use form feed as page-break marker; downstream chunker ignores it.
    return "\f".join(text_parts)


def parse_plaintext(path: Path) -> str:
    """Read a plaintext / Fountain script."""
    return Path(path).read_text(encoding="utf-8", errors="replace")


def parse_script(path: Path, format_hint: Literal["pdf", "fountain", "plaintext"] | None = None) -> Script:
    """Parse a script file and return a Script object.

    `format_hint` overrides extension-based detection.
    """
    fmt = format_hint or detect_format(path)
    path = Path(path)

    if fmt == "pdf":
        text = parse_pdf(path)
    else:
        # Both Fountain and plaintext are read as text.
        text = parse_plaintext(path)

    script_id = path.stem
    movie_id = script_id  # Default: derive from filename.

    return Script(
        id=script_id,
        movie_id=movie_id,
        format=fmt,
        raw_path=path,
        full_text=text,
        language="en",
    )
