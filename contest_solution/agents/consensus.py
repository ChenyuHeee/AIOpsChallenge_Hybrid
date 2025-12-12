"""Consensus orchestrator implementing mABC-style weighted voting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

from ..config import load_memory, store_memory
from ..utils.hypothesis import Hypothesis


@dataclass(slots=True)
class ConsensusResult:
    ranked_components: List[Tuple[str, float]]
    supporting_evidence: Dict[str, List[str]]


class ConsensusOrchestrator:
    def __init__(self, memory_path) -> None:
        self.memory_path = memory_path
        self.memory: Dict[str, Any] = load_memory(memory_path)
        self.component_priors = self._load_component_priors()

    def vote(self, case_id: str, hypotheses: Iterable[Hypothesis]) -> ConsensusResult:
        scores: Dict[str, float] = defaultdict(float)
        evidence: Dict[str, List[str]] = defaultdict(list)
        for hypothesis in hypotheses:
            weight = self._specialist_weight(hypothesis.source)
            reinforcement = self._memory_reward(hypothesis.component)
            prior = self._component_prior(hypothesis.component)
            final_score = hypothesis.confidence * weight * reinforcement * prior
            scores[hypothesis.component] += final_score
            for item in hypothesis.evidence:
                evidence[hypothesis.component].append(f"[{item.modality}] {item.summary}")
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            fallback_component = self._top_prior_component()
            ranked = [(fallback_component, self.component_priors.get(fallback_component, 0.0))]
        elif ranked[0][1] <= 0.0:
            fallback_component = self._top_prior_component(preferred=list(scores.keys()))
            ranked.insert(0, (fallback_component, self.component_priors.get(fallback_component, 0.0)))
        self._update_memory(case_id, ranked)
        store_memory(self.memory_path, self.memory)
        return ConsensusResult(ranked_components=ranked, supporting_evidence=evidence)

    def _specialist_weight(self, source: str) -> float:
        weights = {
            "MetricSpecialist": 1.2,
            "LogSpecialist": 1.1,
            "TraceSpecialist": 1.1,
            "GraphSpecialist": 1.0,
            "ReasoningAgent": 1.3,
        }
        return weights.get(source, 1.0)

    def _memory_reward(self, component: str) -> float:
        if component == "unknown":
            return 0.7
        component_memory = self._ensure_dict("component_success")
        success_rate = float(component_memory.get(component, 0.4))
        return 1.0 + 0.6 * success_rate

    def _update_memory(self, case_id: str, ranked: List[Tuple[str, float]]) -> None:
        if "component_success" not in self.memory:
            self.memory["component_success"] = {}
        if not ranked:
            return
        winner, score = ranked[0]
        if winner == "unknown":
            return
        component_memory = self._ensure_dict("component_success")
        baseline = float(component_memory.get(winner, 0.5))
        updated = min(1.0, baseline * 0.9 + (1 if score > 0 else 0) * 0.1)
        component_memory[winner] = updated
        history = self.memory.get("history")
        if not isinstance(history, list):
            history = []
            self.memory["history"] = history
        history.append({"case": case_id, "component": winner, "score": score})

    def _load_component_priors(self) -> Dict[str, float]:
        defaults = {
            "adservice": 1.4,
            "checkoutservice": 1.3,
            "recommendationservice": 1.25,
            "productcatalogservice": 1.2,
            "frontend": 1.15,
            "cartservice": 1.1,
        }
        stored = self.memory.get("component_priors")
        if isinstance(stored, dict):
            for key, value in stored.items():
                try:
                    defaults[key] = float(value)
                except (TypeError, ValueError):
                    continue
        self.memory["component_priors"] = defaults
        return defaults

    def _component_prior(self, component: str) -> float:
        if not component:
            return 0.8
        return self.component_priors.get(component, 1.0)

    def _top_prior_component(self, preferred: List[str] | None = None) -> str:
        candidates = preferred or list(self.component_priors.keys())
        if not candidates:
            return "unknown"
        ranked = sorted(
            [comp for comp in candidates if comp != "unknown"],
            key=lambda comp: self.component_priors.get(comp, 1.0),
            reverse=True,
        )
        if not ranked:
            return "unknown"
        return ranked[0]

    def _ensure_dict(self, key: str) -> Dict[str, Any]:
        value = self.memory.get(key)
        if isinstance(value, dict):
            return value
        value = {}
        self.memory[key] = value
        return value
