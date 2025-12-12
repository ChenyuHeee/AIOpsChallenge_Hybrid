"""OpenAI-compatible client."""

from __future__ import annotations

import json
import logging
from typing import Any

try:  # pragma: no cover
    from openai import OpenAI
except Exception:  # noqa: BLE001
    OpenAI = None  # type: ignore[assignment]

from ..config import LLMConfig
from .base import LLMClient

LOGGER = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    def __init__(self, config: LLMConfig) -> None:
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed")
        if not config.api_key:
            raise RuntimeError("OpenAI API key missing")
        kwargs: dict[str, Any] = {}
        if config.api_base:
            kwargs["base_url"] = config.api_base
        self.client = OpenAI(api_key=config.api_key, **kwargs)
        self.model = config.model

    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        response = self.client.responses.create(  # type: ignore[call-arg]
            model=self.model,
            input=prompt,
            temperature=temperature,
            max_output_tokens=800,
            response_format={"type": "json_object"},
        )
        try:
            return response.output_text
        except AttributeError:  # pragma: no cover
            chunks = []
            for item in getattr(response, "output", []):
                chunks.append(getattr(item, "content", [{}])[0].get("text", {}).get("value", ""))
            return "".join(chunks)
