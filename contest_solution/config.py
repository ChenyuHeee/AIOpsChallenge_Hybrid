"""Configuration management for the contest solution."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_ENV_CACHE_LOADED = False


def _load_env_files() -> None:
    """Load environment variables from local .env files exactly once."""

    global _ENV_CACHE_LOADED
    if _ENV_CACHE_LOADED:
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key, value = key.strip(), value.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip("\"\'")
        break
    _ENV_CACHE_LOADED = True


@dataclass(slots=True)
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-chat"
    temperature: float = 0.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    timeout: float = 10.0

    @classmethod
    def from_env(cls) -> "LLMConfig":
        _load_env_files()
        provider = os.getenv("RCA_LLM_PROVIDER", "deepseek")
        model = os.getenv("RCA_LLM_MODEL", "deepseek-chat" if provider.startswith("deepseek") else "gpt-4o-mini")
        api_key = (
            os.getenv("RCA_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        api_base = os.getenv("RCA_LLM_API_BASE")
        temperature = float(os.getenv("RCA_LLM_TEMPERATURE", "0.0"))
        timeout_env = os.getenv("RCA_LLM_TIMEOUT")
        if timeout_env is not None and timeout_env != "":
            timeout = float(timeout_env)
        else:
            # DeepSeek often needs a higher request timeout than OpenAI-compatible defaults.
            timeout = 60.0 if provider.startswith("deepseek") else 10.0
        return cls(provider=provider, model=model, temperature=temperature, api_key=api_key, api_base=api_base, timeout=timeout)


@dataclass(slots=True)
class PipelineConfig:
    max_reason_words: int = 32
    max_observation_words: int = 24
    target_trace_steps: int = 6
    enable_hindsight_memory: bool = True
    memory_path: Path = Path(".aiops_memory.json")
    keyword_bank_path: Optional[Path] = None

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        _load_env_files()
        return cls(
            max_reason_words=int(os.getenv("RCA_MAX_REASON_WORDS", "32")),
            max_observation_words=int(os.getenv("RCA_MAX_OBS_WORDS", "24")),
            target_trace_steps=int(os.getenv("RCA_TARGET_TRACE_STEPS", "6")),
            enable_hindsight_memory=os.getenv("RCA_ENABLE_MEMORY", "1") not in {"0", "false", "False"},
            memory_path=Path(os.getenv("RCA_MEMORY_PATH", ".aiops_memory.json")),
            keyword_bank_path=Path(os.getenv("RCA_KEYWORD_CSV", "")) if os.getenv("RCA_KEYWORD_CSV") else None,
        )


@dataclass(slots=True)
class KnowledgeConfig:
    paper_insights_path: Path
    retrieval_top_k: int = 8
    similarity_threshold: float = 0.4

    @classmethod
    def from_env(cls) -> "KnowledgeConfig":
        _load_env_files()
        default_path = Path(__file__).resolve().parent / "resources" / "paper_insights.json"
        custom_path = Path(os.getenv("RCA_PAPER_INSIGHTS", "")) if os.getenv("RCA_PAPER_INSIGHTS") else default_path
        return cls(paper_insights_path=custom_path)


def load_memory(memory_path: Path) -> Dict[str, Dict[str, List[str]]]:
    if not memory_path.exists():
        return {}
    try:
        return json.loads(memory_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def store_memory(memory_path: Path, memory: Dict[str, Dict[str, List[str]]]) -> None:
    try:
        memory_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
