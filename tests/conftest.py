"""Shared pytest fixtures."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_chroma_dir(monkeypatch) -> Path:
    """Provide a temporary Chroma persist dir and point settings at it."""
    d = Path(tempfile.mkdtemp(prefix="chroma_test_"))
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(d))
    monkeypatch.setenv("SEED_MANIFEST_PATH", str(d / "manifest.json"))
    # Reset any cached settings/chroma clients.
    from app.config import get_settings
    from app.core import vectorstore

    get_settings.cache_clear()
    vectorstore.reset_caches()
    yield d
    shutil.rmtree(d, ignore_errors=True)
