"""LLM abstraction for contest solution."""

from __future__ import annotations

import abc


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:  # pragma: no cover - interface
        raise NotImplementedError


class DummyLLMClient(LLMClient):
    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        return (
            "{"  # JSON-ish fallback
            '"component": "unknown", '
            '"reason": "No live LLM configured", '
            '"analysis_steps": ["Inspect metrics", "Inspect logs", "Inspect traces", "Review consensus"]'
            "}"
        )
