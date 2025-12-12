"""Minimal DeepSeek client."""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from ..config import LLMConfig
from .base import LLMClient

LOGGER = logging.getLogger(__name__)


class DeepSeekClient(LLMClient):
    DEFAULT_BASE = "https://api.deepseek.com"

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise RuntimeError("DeepSeek API key missing")
        self.model = config.model or "deepseek-chat"
        self.temperature = config.temperature
        self.timeout = config.timeout
        base = config.api_base or self.DEFAULT_BASE
        self.endpoint = f"{base.rstrip('/')}/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        }

    def complete(self, prompt: str, *, temperature: float | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an RCA expert strictly following SOP procedures."},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": 800,
            "response_format": {"type": "json_object"},
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(self.endpoint, headers=self.headers, json=payload)
        if response.status_code != 200:
            LOGGER.error("DeepSeek error %s: %s", response.status_code, response.text)
            response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            LOGGER.error("Malformed DeepSeek response: %s", json.dumps(data, ensure_ascii=False))
            raise RuntimeError("Malformed DeepSeek response") from exc
