"""LLM client factory with graceful fallback."""

from __future__ import annotations

import logging

from ..config import LLMConfig
from .base import DummyLLMClient, LLMClient
from .deepseek import DeepSeekClient
from .openai_client import OpenAIClient

LOGGER = logging.getLogger(__name__)


def build_client(config: LLMConfig) -> LLMClient:
    provider = config.provider.lower()
    if provider.startswith("deepseek"):
        try:
            return DeepSeekClient(config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("DeepSeek client failed: %s", exc)
    if config.api_key:
        try:
            return OpenAIClient(config)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("OpenAI client failed: %s", exc)
    LOGGER.info("Falling back to dummy LLM client")
    return DummyLLMClient()
