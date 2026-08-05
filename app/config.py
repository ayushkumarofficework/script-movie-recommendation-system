"""Application settings loaded from environment / .env file."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration. Reads from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Cohere embeddings (required) ---
    cohere_api_key: str = Field(default="")
    cohere_model: str = "embed-english-v3.0"
    embed_dim: int = 1024

    # --- LLM provider ---
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"

    # --- ChromaDB ---
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "movie_scripts"

    # --- API server ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    seed_manifest_path: str = "./data/seed_manifest.json"

    # --- Retrieval / recommendation tuning ---
    top_k_retrieval: int = 50
    top_k_rerank: int = 20
    top_k_final: int = 5

    # --- Chunking ---
    max_scene_tokens: int = 1500
    chunk_target_tokens: int = 1200
    chunk_overlap_tokens: int = 150

    def chroma_persist_abs(self) -> Path:
        """Absolute path to the ChromaDB persistent directory."""
        p = Path(self.chroma_persist_dir)
        return p.resolve() if not p.is_absolute() else p

    def seed_manifest_abs(self) -> Path:
        """Absolute path to the seed manifest file."""
        p = Path(self.seed_manifest_path)
        return p.resolve() if not p.is_absolute() else p


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# Convenience global for ergonomic `from app.config import settings`.
settings = get_settings()
