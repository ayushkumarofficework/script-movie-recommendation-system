"""LangChain chat-model factory for the configured LLM provider."""
from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import settings
from app.utils.logging import get_logger


log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return the configured chat model. Cached."""
    provider = settings.llm_provider

    if provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set; cannot use openai provider.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=0.2,
        )

    if provider == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot use anthropic provider.")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key,
            temperature=0.2,
        )

    raise RuntimeError(f"Unknown LLM provider: {provider}")
