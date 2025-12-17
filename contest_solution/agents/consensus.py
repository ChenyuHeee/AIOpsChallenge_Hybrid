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

    def vote(self, case_id: str, hypotheses: Iterable[Hypothesis], component_hints: List[str] | None = None) -> ConsensusResult:
        scores: Dict[str, float] = defaultdict(float)
        evidence: Dict[str, List[str]] = defaultdict(list)
        modality_support: Dict[str, set[str]] = defaultdict(set)
        for hypothesis in hypotheses:
            weight = self._specialist_weight(hypothesis.source)
            reinforcement = self._memory_reward(hypothesis.component)
            prior = self._component_prior(hypothesis.component)
            final_score = hypothesis.confidence * weight * reinforcement * prior
            scores[hypothesis.component] += final_score
            for item in hypothesis.evidence:
                evidence[hypothesis.component].append(f"[{item.modality}] {item.summary}")
                modality_support[hypothesis.component].add(str(item.modality))

        # Node-fault aggregation heuristic:
        # If many services show anomalies in the window AND a node candidate has strong
        # metrics/traces evidence, boost the node score. This helps node-root-cause cases
        # where symptoms appear across multiple services.
        self._apply_node_fault_boost(scores, evidence)

        # Optional: prefer components supported by multiple modalities.
        self._apply_multimodal_bonus(scores, modality_support)

        # Optional: boost components explicitly hinted by query keywords (planner hints).
        self._apply_hint_bonus(scores, component_hints)

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

    @staticmethod
    def _apply_hint_bonus(scores: Dict[str, float], component_hints: List[str] | None) -> None:
        if os.getenv("RCA_ENABLE_HINT_BONUS", "0") in {"0", "false", "False"}:
            return
        if not component_hints:
            return
        hints = {str(h).strip().lower() for h in component_hints if str(h).strip()}
        if not hints:
            return

        try:
            bonus = float(os.getenv("RCA_HINT_BONUS", "0.25"))
        except ValueError:
            bonus = 0.25
        bonus = max(0.0, min(1.5, bonus))
        factor = 1.0 + bonus

        for component, base in list(scores.items()):
            if base <= 0:
                continue
            if component and component.lower() in hints:
                scores[component] = base * factor

    @staticmethod
    def _apply_multimodal_bonus(scores: Dict[str, float], modality_support: Dict[str, set[str]]) -> None:
        if os.getenv("RCA_ENABLE_MODALITY_BONUS", "0") in {"0", "false", "False"}:
            return
        try:
            alpha = float(os.getenv("RCA_MODALITY_BONUS_ALPHA", "0.15"))
        except ValueError:
            alpha = 0.15
        alpha = max(0.0, min(0.6, alpha))
        try:
            cap = float(os.getenv("RCA_MODALITY_BONUS_CAP", "1.4"))
        except ValueError:
            cap = 1.4
        cap = max(1.0, min(3.0, cap))

        for component, base in list(scores.items()):
            if base <= 0:
                continue
            modalities = modality_support.get(component) or set()
            # Ignore empty/unknown modality sets.
            m = len([x for x in modalities if x])
            if m <= 1:
                continue
            factor = min(cap, 1.0 + alpha * float(m - 1))
            scores[component] = base * factor

    @staticmethod
    def _is_node(component: str) -> bool:
        return bool(component) and (
            component.startswith("aiops-k8s-") or component.startswith("k8s-master")
        )

    @classmethod
    def _apply_node_fault_boost(cls, scores: Dict[str, float], evidence: Dict[str, List[str]]) -> None:
        # Experimental heuristic. Keep it OFF by default because aggressive node boosting
        # can hurt LA when node evidence is correlated but not causal.
        if os.getenv("RCA_ENABLE_NODE_BOOST", "0") in {"0", "false", "False"}:
            return

        # Count how many non-node components have meaningful scores.
        non_node_scores = [v for k, v in scores.items() if not cls._is_node(k) and k != "unknown" and v > 0]
        if len(non_node_scores) < 3:
            return
        threshold = sorted(non_node_scores, reverse=True)[min(4, len(non_node_scores) - 1)]
        service_pressure = sum(1 for v in non_node_scores if v >= threshold)
        if service_pressure < 3:
            return

        base_boost = float(os.getenv("RCA_NODE_BOOST", "0.35"))
        max_boost = float(os.getenv("RCA_NODE_BOOST_MAX", "0.9"))
        scale = min(1.0, service_pressure / 6.0)
        boost = min(max_boost, base_boost * (1.0 + 0.8 * scale))

        for component in list(scores.keys()):
            if not cls._is_node(component):
                continue
            ev = evidence.get(component) or []
            if len(ev) < 2:
                continue
            modalities = set()
            for line in ev:
                if line.startswith("[metrics]"):
                    modalities.add("metrics")
                elif line.startswith("[traces]"):
                    modalities.add("traces")
            # Only boost when BOTH metrics and traces agree on the node.
            # This keeps precision high; boosting on a single modality was observed to hurt LA.
            if not ("metrics" in modalities and "traces" in modalities):
                continue
            scores[component] *= (1.0 + boost)

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

        base = float(self.component_priors.get(component, 1.0))
        try:
            scale = float(os.getenv("RCA_COMPONENT_PRIOR_SCALE", "1.0"))
        except ValueError:
            scale = 1.0
        scale = max(0.0, min(1.0, scale))
        # scale=1.0 => keep base prior; scale=0.0 => disable (becomes 1.0)
        return 1.0 + (base - 1.0) * scale

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
