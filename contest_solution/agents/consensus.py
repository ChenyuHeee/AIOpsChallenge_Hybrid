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

        # Track strongest node CPU/memory metric evidence per node.
        node_metric_max: Dict[str, float] = defaultdict(float)
        # Track strongest TiDB metric evidence per tidb component (phase2 domain).
        tidb_metric_max: Dict[str, float] = defaultdict(float)
        for hypothesis in hypotheses:
            weight = self._specialist_weight(hypothesis.source)
            # Trace-derived node hints can be noisy; prefer metrics/logs for node identification.
            if self._is_node(hypothesis.component) and hypothesis.source == "TraceSpecialist":
                try:
                    weight = float(os.getenv("RCA_WEIGHT_TRACE_NODE", "0.6"))
                except ValueError:
                    weight = 0.6
            reinforcement = self._memory_reward(hypothesis.component)
            prior = self._component_prior(hypothesis.component)
            final_score = hypothesis.confidence * weight * reinforcement * prior
            scores[hypothesis.component] += final_score
            for item in hypothesis.evidence:
                evidence[hypothesis.component].append(f"[{item.modality}] {item.summary}")
                modality_support[hypothesis.component].add(str(item.modality))

                if (
                    str(item.modality) == "metrics"
                    and self._is_node(hypothesis.component)
                    and ("node_memory" in (item.summary or "").lower() or "node_cpu" in (item.summary or "").lower())
                ):
                    try:
                        node_metric_max[hypothesis.component] = max(node_metric_max[hypothesis.component], float(item.score))
                    except Exception:
                        pass

                if str(item.modality) == "metrics" and self._is_tidb(hypothesis.component):
                    try:
                        tidb_metric_max[hypothesis.component] = max(tidb_metric_max[hypothesis.component], float(item.score))
                    except Exception:
                        pass

            # Optional: seed hinted components into the candidate list.
            # Motivation: hint bonus only affects existing candidates; when evidence is sparse,
            # injecting low-confidence hinted candidates can prevent fallback-to-top-prior collapse.
            self._seed_hints(scores, component_hints)

        # Node-fault aggregation heuristic:
        # If many services show anomalies in the window AND a node candidate has strong
        # metrics/traces evidence, boost the node score. This helps node-root-cause cases
        # where symptoms appear across multiple services.
        self._apply_node_fault_boost(scores, evidence)

        # Optional: strong-node override.
        # If a node has extremely strong node_cpu/node_memory metric evidence, prefer it
        # over service candidates. Useful when true root cause is infra (node fault), while
        # traces/logs only show downstream symptoms on services.
        self._apply_node_strong_metric_override(scores, evidence)

        # Optional: metric-dominance override (safer than blanket node boost).
        # If a node has a very strong CPU/memory metric AND its score is already competitive
        # with the best non-node candidate, promote it to the top.
        self._apply_node_metric_dominance_override(scores, node_metric_max)

        # Optional: TiDB dominance override (phase2) - only when TiDB metric evidence is strong.
        self._apply_tidb_metric_dominance_override(scores, tidb_metric_max)

        # Optional: prefer components supported by multiple modalities.
        self._apply_multimodal_bonus(scores, modality_support)

        # Optional: boost components explicitly hinted by query keywords (planner hints).
        self._apply_hint_bonus(scores, component_hints)

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if not ranked:
            fallback_component = self._fallback_component(component_hints)
            ranked = [(fallback_component, self.component_priors.get(fallback_component, 0.0))]
        elif ranked[0][1] <= 0.0:
            fallback_component = self._fallback_component(component_hints, preferred=list(scores.keys()))
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

    @staticmethod
    def _is_tidb(component: str) -> bool:
        return bool(component) and component.startswith("tidb-")

    @classmethod
    def _apply_tidb_metric_dominance_override(cls, scores: Dict[str, float], tidb_metric_max: Dict[str, float]) -> None:
        if os.getenv("RCA_ENABLE_TIDB_METRIC_DOMINANCE_OVERRIDE", "0") in {"0", "false", "False"}:
            return
        tidb_items = [(c, s) for c, s in scores.items() if cls._is_tidb(c) and s > 0]
        if not tidb_items:
            return

        top_tidb = max(tidb_items, key=lambda x: tidb_metric_max.get(x[0], 0.0))[0]
        top_tidb_score = float(scores.get(top_tidb, 0.0))
        top_tidb_metric = float(tidb_metric_max.get(top_tidb, 0.0))

        try:
            min_metric = float(os.getenv("RCA_TIDB_DOMINANCE_MIN_METRIC", "2.6"))
        except ValueError:
            min_metric = 2.6
        if top_tidb_metric < min_metric:
            return

        non_tidb_best = max((s for c, s in scores.items() if not cls._is_tidb(c)), default=0.0)
        if non_tidb_best <= 0:
            return

        try:
            min_ratio = float(os.getenv("RCA_TIDB_DOMINANCE_MIN_RATIO", "0.7"))
        except ValueError:
            min_ratio = 0.7
        min_ratio = max(0.0, min(1.0, min_ratio))
        if top_tidb_score < non_tidb_best * min_ratio:
            return

        try:
            margin = float(os.getenv("RCA_TIDB_DOMINANCE_MARGIN", "0.02"))
        except ValueError:
            margin = 0.02
        margin = max(0.0, min(0.5, margin))

        scores[top_tidb] = max(top_tidb_score, non_tidb_best * (1.0 + margin))

    @classmethod
    def _apply_node_strong_metric_override(cls, scores: Dict[str, float], evidence: Dict[str, List[str]]) -> None:
        if os.getenv("RCA_ENABLE_NODE_STRONG_OVERRIDE", "0") in {"0", "false", "False"}:
            return
        node_items = [(c, s) for c, s in scores.items() if cls._is_node(c) and s > 0]
        if not node_items:
            return

        top_node, node_score = max(node_items, key=lambda x: x[1])
        ev = evidence.get(top_node) or []
        # Require explicit node CPU/memory metric evidence to keep precision high.
        has_strong_node_metric = any(
            (line.startswith("[metrics]") and ("node_memory" in line.lower() or "node_cpu" in line.lower()))
            for line in ev
        )
        if not has_strong_node_metric:
            return
        try:
            strong_score = float(os.getenv("RCA_NODE_STRONG_SCORE", "6.0"))
        except ValueError:
            strong_score = 6.0
        if node_score < strong_score:
            return

        non_node_best = max((s for c, s in scores.items() if not cls._is_node(c)), default=0.0)
        try:
            margin = float(os.getenv("RCA_NODE_STRONG_MARGIN", "0.08"))
        except ValueError:
            margin = 0.08
        margin = max(0.0, min(0.5, margin))
        target = max(node_score, non_node_best * (1.0 + margin))
        scores[top_node] = target

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
        require_traces = os.getenv("RCA_NODE_BOOST_REQUIRE_TRACES", "1") not in {"0", "false", "False"}
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
                elif line.startswith("[logs]"):
                    modalities.add("logs")
            # Default: require BOTH metrics and traces to agree.
            # For node-focused datasets, allow relaxing to metrics-only via env.
            if require_traces:
                if not ("metrics" in modalities and "traces" in modalities):
                    continue
            else:
                if not modalities:
                    continue
            scores[component] *= (1.0 + boost)

    @classmethod
    def _apply_node_metric_dominance_override(cls, scores: Dict[str, float], node_metric_max: Dict[str, float]) -> None:
        if os.getenv("RCA_ENABLE_NODE_METRIC_DOMINANCE_OVERRIDE", "0") in {"0", "false", "False"}:
            return
        node_items = [(c, s) for c, s in scores.items() if cls._is_node(c) and s > 0]
        if not node_items:
            return

        # Pick the node with the strongest CPU/memory metric evidence.
        top_node = max(node_items, key=lambda x: node_metric_max.get(x[0], 0.0))[0]
        top_node_score = float(scores.get(top_node, 0.0))
        top_node_metric = float(node_metric_max.get(top_node, 0.0))

        try:
            min_metric = float(os.getenv("RCA_NODE_DOMINANCE_MIN_METRIC", "5.2"))
        except ValueError:
            min_metric = 5.2
        if top_node_metric < min_metric:
            return

        non_node_best = max((s for c, s in scores.items() if not cls._is_node(c)), default=0.0)
        if non_node_best <= 0:
            return

        try:
            min_ratio = float(os.getenv("RCA_NODE_DOMINANCE_MIN_RATIO", "0.6"))
        except ValueError:
            min_ratio = 0.6
        min_ratio = max(0.0, min(1.0, min_ratio))
        if top_node_score < non_node_best * min_ratio:
            return

        try:
            margin = float(os.getenv("RCA_NODE_DOMINANCE_MARGIN", "0.03"))
        except ValueError:
            margin = 0.03
        margin = max(0.0, min(0.2, margin))
        scores[top_node] = max(top_node_score, non_node_best * (1.0 + margin))

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
        # IMPORTANT: legacy memory is OFF by default because it is easy to introduce
        # positive-feedback collapse (e.g., adservice dominates predictions -> high success -> more dominance).
        use_legacy = os.getenv("RCA_USE_LEGACY_MEMORY", "0") not in {"0", "false", "False"}
        if use_legacy:
            legacy = self.memory.get("component_success")
            if isinstance(legacy, dict) and component in legacy:
                try:
                    success_rate = float(legacy.get(component, 0.5))
                except Exception:
                    return 1.0
                # Mild effect: 0.9 ~ 1.3
                rate = max(0.0, min(1.0, success_rate))
                return 0.9 + 0.4 * rate

        return 1.0

    @staticmethod
    def _seed_hints(scores: Dict[str, float], component_hints: List[str] | None) -> None:
        if os.getenv("RCA_ENABLE_HINT_SEED", "0") in {"0", "false", "False"}:
            return
        if not component_hints:
            return
        hints = [str(h).strip().lower() for h in component_hints if str(h).strip()]
        if not hints:
            return

        try:
            seed = float(os.getenv("RCA_HINT_SEED_SCORE", "0.15"))
        except ValueError:
            seed = 0.15
        seed = max(0.0, min(2.0, seed))

        # Ensure hinted components appear in candidate list with a small positive score.
        # This score is intentionally weak; real evidence should still dominate.
        for h in hints:
            if h == "service":
                continue
            if scores.get(h, 0.0) > 0.0:
                continue
            scores[h] = max(scores.get(h, 0.0), seed)

    def _fallback_component(self, component_hints: List[str] | None, preferred: List[str] | None = None) -> str:
        prefer_hints = os.getenv("RCA_FALLBACK_PREFER_HINTS", "0") not in {"0", "false", "False"}
        if prefer_hints and component_hints:
            hinted = [str(h).strip().lower() for h in component_hints if str(h).strip()]
            hinted = [h for h in hinted if h and h != "service"]
            if hinted:
                # Choose the highest-prior hinted component (if present in priors);
                # otherwise fall back to generic top prior.
                candidates = [h for h in hinted if h in self.component_priors]
                if candidates:
                    return self._top_prior_component(preferred=candidates)
        return self._top_prior_component(preferred=preferred)

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
