"""Consensus orchestrator implementing mABC-style weighted voting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
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
        
        # Filter node candidates to top-2 by score to avoid LLM confusion (too many similar nodes).
        ranked = self._filter_weak_nodes(ranked)
        
        self._append_history(case_id, ranked)
        store_memory(self.memory_path, self.memory)
        return ConsensusResult(ranked_components=ranked, supporting_evidence=evidence)

    def _specialist_weight(self, source: str) -> float:
        # Defaults chosen from empirical trials.
        defaults = {
            "MetricSpecialist": 1.0,
            "LogSpecialist": 1.15,
            "TraceSpecialist": 1.6,
            "GraphSpecialist": 1.0,
            "ReasoningAgent": 1.3,
        }

        # Allow lightweight hyperparameter tuning via env vars (used by CV/search scripts).
        env_overrides = {
            "MetricSpecialist": os.getenv("RCA_WEIGHT_METRIC"),
            "LogSpecialist": os.getenv("RCA_WEIGHT_LOG"),
            "TraceSpecialist": os.getenv("RCA_WEIGHT_TRACE"),
            "GraphSpecialist": os.getenv("RCA_WEIGHT_GRAPH"),
            "ReasoningAgent": os.getenv("RCA_WEIGHT_REASONING"),
        }
        raw = env_overrides.get(source)
        if raw is not None and raw != "":
            try:
                return float(raw)
            except ValueError:
                return defaults.get(source, 1.0)
        return defaults.get(source, 1.0)

    def _memory_reward(self, component: str) -> float:
        # IMPORTANT:
        # We do NOT self-reinforce based on our own predictions (that creates a positive feedback
        # loop that tends to overfit on popular services like checkoutservice/cartservice).
        # Judge-derived feedback can be used, but it must be explicitly enabled because small or
        # biased feedback sets can easily hurt LA.
        if component == "unknown":
            return 0.85

        use_judge_memory = os.getenv("RCA_USE_JUDGE_MEMORY", "0") not in {"0", "false", "False"}

        # Path A (preferred): judge-updated per-component stats.
        if use_judge_memory:
            stats = self.memory.get("component_stats")
            if not isinstance(stats, dict):
                stats = {}
                self.memory["component_stats"] = stats

            entry = stats.get(component)
            if not isinstance(entry, dict):
                return 1.0
            try:
                correct = int(entry.get("correct", 0))
                total = int(entry.get("total", 0))
            except Exception:
                return 1.0

            # Require enough samples before trusting the feedback signal.
            min_total = int(os.getenv("RCA_MEMORY_MIN_TOTAL", "20"))
            if total < min_total:
                return 1.0

            rate = correct / max(1, total)
            # Mild effect: 0.9 (bad) ~ 1.1 (good)
            return 0.9 + 0.2 * rate

        # Path B (legacy): component_success values already accumulated in memory.
        legacy = self.memory.get("component_success")
        if isinstance(legacy, dict) and component in legacy:
            try:
                success_rate = float(legacy.get(component, 0.5))
            except Exception:
                return 1.0
            # Mild effect: 0.9 ~ 1.3 (keeps previous behavior helpful but less dominating)
            return 0.9 + 0.8 * max(0.0, min(1.0, success_rate))

        return 1.0

    def apply_component_feedback(self, case_id: str, predicted: str, gt_component: str) -> None:
        """Update memory using judge-style correctness feedback.

        predicted: submission component (already normalized)
        gt_component: ground truth component string (may include 'a->b')
        """

        predicted = (predicted or "").strip()
        gt_component = (gt_component or "").strip()
        gt_parts = [part.strip() for part in gt_component.replace("->", "+").split("+") if part.strip()]
        is_correct = bool(predicted and gt_parts and predicted in gt_parts)

        stats = self.memory.get("component_stats")
        if not isinstance(stats, dict):
            stats = {}
            self.memory["component_stats"] = stats
        entry = stats.get(predicted)
        if not isinstance(entry, dict):
            entry = {"correct": 0, "total": 0}
            stats[predicted] = entry
        entry["total"] = int(entry.get("total", 0)) + 1
        if is_correct:
            entry["correct"] = int(entry.get("correct", 0)) + 1

        # Keep legacy success-rate mirror for backward compatibility with existing memory files.
        legacy = self.memory.get("component_success")
        if not isinstance(legacy, dict):
            legacy = {}
            self.memory["component_success"] = legacy
        legacy[predicted] = int(entry.get("correct", 0)) / max(1, int(entry.get("total", 0)))

        # Per-case audit trail (useful for debugging but not used at inference time)
        case_fb = self.memory.get("case_feedback")
        if not isinstance(case_fb, dict):
            case_fb = {}
            self.memory["case_feedback"] = case_fb
        case_fb[str(case_id)] = {
            "predicted": predicted,
            "gt": gt_component,
            "correct": is_correct,
        }

    def _append_history(self, case_id: str, ranked: List[Tuple[str, float]]) -> None:
        history = self.memory.get("history")
        if not isinstance(history, list):
            history = []
            self.memory["history"] = history
        if not ranked:
            return
        winner, score = ranked[0]
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
        # Priors were useful early on, but they can heavily bias predictions toward
        # popular services and suppress node/edge-endpoint components.
        # Favor evidence-driven ranking.
        if not component:
            return 0.9
        return 1.0

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
    
    @staticmethod
    def _filter_weak_nodes(ranked: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Keep only top-2 node candidates to avoid LLM picking wrong nodes from many similar ones."""
        node_items = [(c, s) for c, s in ranked if c and (c.startswith("aiops-k8s-") or c.startswith("k8s-master"))]
        non_node_items = [(c, s) for c, s in ranked if not (c and (c.startswith("aiops-k8s-") or c.startswith("k8s-master")))]
        # Keep top-2 nodes by score, drop the rest.
        kept_nodes = node_items[:2]
        # Rebuild ranked list with filtered nodes inserted at their original relative positions.
        result = []
        node_idx = 0
        non_node_idx = 0
        for c, s in ranked:
            is_node = c and (c.startswith("aiops-k8s-") or c.startswith("k8s-master"))
            if is_node:
                if node_idx < len(kept_nodes):
                    result.append(kept_nodes[node_idx])
                    node_idx += 1
            else:
                if non_node_idx < len(non_node_items):
                    result.append(non_node_items[non_node_idx])
                    non_node_idx += 1
        return result
