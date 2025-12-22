"""Consensus orchestrator implementing mABC-style weighted voting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import os
import re
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
        self._apply_node_metric_dominance_override(scores, node_metric_max, modality_support)

        # Optional: TiDB dominance override (phase2) - only when TiDB metric evidence is strong.
        self._apply_tidb_metric_dominance_override(scores, tidb_metric_max)

        # Optional: fault-level gate (engineering rule):
        # decide whether this case is node-level vs service-level vs TiDB-level,
        # then penalize other levels to reduce systematic confusion.
        self._apply_fault_level_gate(
            scores,
            evidence,
            modality_support=modality_support,
            node_metric_max=node_metric_max,
            tidb_metric_max=tidb_metric_max,
        )

        # Optional: prefer components supported by multiple modalities.
        self._apply_multimodal_bonus(scores, modality_support)

        # Optional: when replica-specific evidence exists (e.g., logs show adservice-0 errors),
        # prefer the replica over the base service token to avoid systematic LA loss.
        self._prefer_replica_by_evidence(scores, modality_support)

        # Optional: suppress popular-service collapse when evidence is weak.
        # This is OFF by default and should be enabled explicitly in experiments.
        self._apply_popular_component_penalty(scores, modality_support)

        # Optional: boost components explicitly hinted by query keywords (planner hints).
        self._apply_hint_bonus(scores, component_hints)

        # Optional: if the query explicitly mentions a replica pod (xxxservice-0), prefer that
        # replica over the base service token when the base service would otherwise win.
        self._prefer_replica_hints(scores, component_hints)

        # Optional: prefer replica candidates over base services when evidence is pod-specific.
        self._prefer_replica_candidates(scores, modality_support)

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

    @classmethod
    def _prefer_replica_candidates(cls, scores: Dict[str, float], modality_support: Dict[str, set[str]]) -> None:
        if os.getenv("RCA_ENABLE_REPLICA_CANDIDATE_PREFERENCE", "0") in {"0", "false", "False"}:
            return

        try:
            min_ratio = float(os.getenv("RCA_REPLICA_CANDIDATE_MIN_RATIO", "0.75"))
        except ValueError:
            min_ratio = 0.75
        min_ratio = max(0.0, min(1.0, min_ratio))

        try:
            margin = float(os.getenv("RCA_REPLICA_CANDIDATE_MARGIN", "0.01"))
        except ValueError:
            margin = 0.01
        margin = max(0.0, min(0.2, margin))

        require_non_metric = os.getenv("RCA_REPLICA_REQUIRE_NON_METRIC", "1") not in {"0", "false", "False"}

        # For every base service token, check if a replica token exists and is competitive.
        for base, base_score in list(scores.items()):
            if base_score <= 0:
                continue
            if cls._is_node(base) or cls._is_tidb(base) or ("->" in base):
                continue
            # Only consider base tokens (no -<digit> suffix).
            parts = base.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                continue

            # Find best replica candidate for this base.
            best_replica = ""
            best_score = 0.0
            for comp, sc in scores.items():
                if sc <= 0:
                    continue
                if not comp.startswith(base + "-"):
                    continue
                tail = comp.rsplit("-", 1)[-1]
                if not tail.isdigit():
                    continue
                if require_non_metric:
                    mods = modality_support.get(comp) or set()
                    # Require at least one of logs/traces/graph to avoid purely metric-driven noise.
                    if not ({"logs", "traces", "graph"} & set(str(m) for m in mods)):
                        continue
                if float(sc) > best_score:
                    best_replica = comp
                    best_score = float(sc)

            if not best_replica:
                continue

            # Promote replica if it's close enough to the base score.
            if best_score >= float(base_score) * min_ratio:
                scores[best_replica] = max(float(scores.get(best_replica, 0.0)), float(base_score) * (1.0 + margin))

    @staticmethod
    def _prefer_replica_by_evidence(scores: Dict[str, float], modality_support: Dict[str, set[str]]) -> None:
        if os.getenv("RCA_ENABLE_REPLICA_EVIDENCE_PREFERENCE", "0") in {"0", "false", "False"}:
            return

        try:
            min_ratio = float(os.getenv("RCA_REPLICA_EVIDENCE_MIN_RATIO", "0.45"))
        except ValueError:
            min_ratio = 0.45
        min_ratio = max(0.0, min(1.0, min_ratio))

        try:
            min_score = float(os.getenv("RCA_REPLICA_EVIDENCE_MIN_SCORE", "1.8"))
        except ValueError:
            min_score = 1.8
        min_score = max(0.0, min(50.0, min_score))

        try:
            margin = float(os.getenv("RCA_REPLICA_EVIDENCE_MARGIN", "0.01"))
        except ValueError:
            margin = 0.01
        margin = max(0.0, min(0.2, margin))

        # Promote replica when it has logs/traces support and base token exists.
        for replica, replica_score in list(scores.items()):
            if replica_score <= 0:
                continue
            token = (replica or "").strip().lower()
            if not re.fullmatch(r"[a-z0-9_-]*service-\d+", token):
                continue
            base, suffix = token.rsplit("-", 1)
            if not base or not suffix.isdigit():
                continue
            base_score = float(scores.get(base, 0.0))
            if base_score <= 0:
                continue
            mods = modality_support.get(replica) or set()
            if "logs" not in mods and "traces" not in mods:
                continue

            # Only promote when replica is at least somewhat competitive, or has a decent absolute score.
            if float(replica_score) < base_score * min_ratio and float(replica_score) < min_score:
                continue

            if float(replica_score) < base_score:
                scores[replica] = base_score * (1.0 + margin)

    @staticmethod
    def _prefer_replica_hints(scores: Dict[str, float], component_hints: List[str] | None) -> None:
        if os.getenv("RCA_PREFER_REPLICA_HINTS", "0") in {"0", "false", "False"}:
            return
        if not component_hints:
            return
        # Only consider explicit replica tokens like "checkoutservice-2".
        hinted_replicas = []
        for h in component_hints:
            if not isinstance(h, str):
                continue
            token = h.strip().lower()
            if not token:
                continue
            if re.fullmatch(r"[a-z0-9_-]*service-\d+", token):
                hinted_replicas.append(token)
        if not hinted_replicas:
            return
        try:
            margin = float(os.getenv("RCA_REPLICA_HINT_MARGIN", "0.01"))
        except ValueError:
            margin = 0.01
        margin = max(0.0, min(0.2, margin))

        # Promote replica hint if its base is present and competitive.
        for replica in hinted_replicas:
            if replica not in scores:
                continue
            if "-" not in replica:
                continue
            base, suffix = replica.rsplit("-", 1)
            if not suffix.isdigit() or not base:
                continue
            base_score = float(scores.get(base, 0.0))
            replica_score = float(scores.get(replica, 0.0))
            if base_score <= 0:
                continue
            # If base service is stronger, lift replica slightly above it.
            if replica_score < base_score:
                scores[replica] = base_score * (1.0 + margin)

    @staticmethod
    def _apply_popular_component_penalty(scores: Dict[str, float], modality_support: Dict[str, set[str]]) -> None:
        if os.getenv("RCA_ENABLE_POPULAR_COMPONENT_PENALTY", "0") in {"0", "false", "False"}:
            return
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) < 2:
            return
        top_comp, top_score = ranked[0]
        second_score = float(ranked[1][1])
        if top_score <= 0 or second_score <= 0:
            return

        popular_raw = os.getenv(
            "RCA_POPULAR_COMPONENTS",
            "adservice,checkoutservice,cartservice,recommendationservice,emailservice,productcatalogservice,frontend,shippingservice,paymentservice",
        )
        popular = {x.strip().lower() for x in popular_raw.split(",") if x.strip()}
        if not popular:
            return

        top_base = top_comp.rsplit("-", 1)[0] if "-" in top_comp else top_comp
        if top_comp not in popular and top_base not in popular:
            return

        modalities = modality_support.get(top_comp) or set()
        m = len([x for x in modalities if x])
        require_multimodal = os.getenv("RCA_POPULAR_PENALTY_REQUIRE_MULTIMODAL", "1") not in {"0", "false", "False"}
        if require_multimodal and m >= 2:
            return

        try:
            margin = float(os.getenv("RCA_POPULAR_PENALTY_MARGIN", "0.08"))
        except ValueError:
            margin = 0.08
        margin = max(0.0, min(0.5, margin))
        if top_score >= second_score * (1.0 + margin):
            return

        try:
            penalty = float(os.getenv("RCA_POPULAR_PENALTY", "0.25"))
        except ValueError:
            penalty = 0.25
        penalty = max(0.0, min(0.8, penalty))

        scores[top_comp] = float(top_score) * (1.0 - penalty)

    @classmethod
    def _apply_fault_level_gate(
        cls,
        scores: Dict[str, float],
        evidence: Dict[str, List[str]],
        *,
        modality_support: Dict[str, set[str]],
        node_metric_max: Dict[str, float],
        tidb_metric_max: Dict[str, float],
    ) -> None:
        if os.getenv("RCA_ENABLE_FAULT_LEVEL_GATE", "0") in {"0", "false", "False"}:
            return

        node_items = [(c, s) for c, s in scores.items() if cls._is_node(c) and s > 0]
        tidb_items = [(c, s) for c, s in scores.items() if cls._is_tidb(c) and s > 0]
        non_node_best = max((s for c, s in scores.items() if not cls._is_node(c)), default=0.0)
        non_tidb_best = max((s for c, s in scores.items() if not cls._is_tidb(c)), default=0.0)

        top_node = max(node_items, key=lambda x: node_metric_max.get(x[0], 0.0))[0] if node_items else ""
        top_tidb = max(tidb_items, key=lambda x: tidb_metric_max.get(x[0], 0.0))[0] if tidb_items else ""

        top_node_score = float(scores.get(top_node, 0.0)) if top_node else 0.0
        top_node_metric = float(node_metric_max.get(top_node, 0.0)) if top_node else 0.0
        top_tidb_score = float(scores.get(top_tidb, 0.0)) if top_tidb else 0.0
        top_tidb_metric = float(tidb_metric_max.get(top_tidb, 0.0)) if top_tidb else 0.0

        try:
            node_min_metric = float(os.getenv("RCA_FAULT_LEVEL_GATE_NODE_MIN_METRIC", "6.0"))
        except ValueError:
            node_min_metric = 6.0
        try:
            node_min_ratio = float(os.getenv("RCA_FAULT_LEVEL_GATE_NODE_MIN_RATIO", "0.75"))
        except ValueError:
            node_min_ratio = 0.75

        try:
            tidb_min_metric = float(os.getenv("RCA_FAULT_LEVEL_GATE_TIDB_MIN_METRIC", "3.0"))
        except ValueError:
            tidb_min_metric = 3.0
        try:
            tidb_min_ratio = float(os.getenv("RCA_FAULT_LEVEL_GATE_TIDB_MIN_RATIO", "0.70"))
        except ValueError:
            tidb_min_ratio = 0.70

        # Penalty is softer than hard filtering: keeps recall in ambiguous cases.
        try:
            penalty = float(os.getenv("RCA_FAULT_LEVEL_GATE_PENALTY", "0.25"))
        except ValueError:
            penalty = 0.25
        penalty = max(0.0, min(1.0, penalty))

        # Gate decision.
        gate: str | None = None

        # Guardrail: replica-level service evidence (especially from logs) is often more
        # component-specific than node metrics. When such evidence exists, avoid mistakenly
        # gating to node-level based on correlated infra KPIs.
        try:
            block_min_ratio = float(os.getenv("RCA_FAULT_LEVEL_GATE_REPLICA_LOG_BLOCK_RATIO", "0.55"))
        except ValueError:
            block_min_ratio = 0.55
        block_min_ratio = max(0.0, min(1.0, block_min_ratio))

        try:
            block_min_score = float(os.getenv("RCA_FAULT_LEVEL_GATE_REPLICA_LOG_BLOCK_SCORE", "2.2"))
        except ValueError:
            block_min_score = 2.2
        block_min_score = max(0.0, min(50.0, block_min_score))

        def _is_replica_service(token: str) -> bool:
            t = (token or "").strip().lower()
            return bool(t) and re.fullmatch(r"[a-z0-9_-]*service-\d+", t) is not None

        replica_log_best = 0.0
        for comp, s in scores.items():
            if s <= 0:
                continue
            if not _is_replica_service(comp):
                continue
            mods = modality_support.get(comp) or set()
            if "logs" not in mods:
                continue
            replica_log_best = max(replica_log_best, float(s))

        if top_node and top_node_metric >= node_min_metric and non_node_best > 0 and top_node_score >= non_node_best * node_min_ratio:
            # Require explicit node CPU/memory metric evidence for precision.
            ev = evidence.get(top_node) or []
            has_node_metric = any(
                (line.startswith("[metrics]") and ("node_memory" in line.lower() or "node_cpu" in line.lower()))
                for line in ev
            )
            if has_node_metric:
                node_modalities = modality_support.get(top_node) or set()
                # Default: do NOT require traces for node gating, because many node-root cases
                # have decisive node_cpu/node_memory metrics but sparse trace tagging.
                require_traces = os.getenv("RCA_FAULT_LEVEL_GATE_NODE_REQUIRE_TRACES", "0") not in {"0", "false", "False"}
                has_traces = ("traces" in node_modalities)

                # If we have strong replica log evidence, do NOT gate to node unless node also has traces support.
                block_on_replica_logs = os.getenv("RCA_FAULT_LEVEL_GATE_BLOCK_ON_REPLICA_LOG", "1") not in {"0", "false", "False"}
                replica_blocks = block_on_replica_logs and (replica_log_best >= max(block_min_score, non_node_best * block_min_ratio))

                if replica_blocks and not has_traces:
                    gate = None
                elif require_traces and not has_traces:
                    gate = None
                else:
                    gate = "node"

        if gate is None and top_tidb and top_tidb_metric >= tidb_min_metric and non_tidb_best > 0 and top_tidb_score >= non_tidb_best * tidb_min_ratio:
            gate = "tidb"

        if gate is None:
            return

        if gate == "node":
            for c, s in list(scores.items()):
                if s <= 0:
                    continue
                if cls._is_node(c):
                    continue
                scores[c] = s * penalty
            return

        if gate == "tidb":
            for c, s in list(scores.items()):
                if s <= 0:
                    continue
                if cls._is_tidb(c):
                    continue
                scores[c] = s * penalty
            return

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
        # GT 节点命名使用 aiops-k8s-XX；k8s-master* 不在 GT 中，保留会稳定伤害 LA。
        return bool(component) and component.startswith("aiops-k8s-")

    @staticmethod
    def _is_tidb(component: str) -> bool:
        allow_tidb = os.getenv("RCA_ALLOW_TIDB_COMPONENTS", "1") not in {"0", "false", "False"}
        return allow_tidb and bool(component) and component.startswith("tidb-")

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
    def _apply_node_metric_dominance_override(
        cls,
        scores: Dict[str, float],
        node_metric_max: Dict[str, float],
        modality_support: Dict[str, set[str]],
    ) -> None:
        if os.getenv("RCA_ENABLE_NODE_METRIC_DOMINANCE_OVERRIDE", "0") in {"0", "false", "False"}:
            return
        node_items = [(c, s) for c, s in scores.items() if cls._is_node(c) and s > 0]
        if not node_items:
            return

        # Pick the node with the strongest CPU/memory metric evidence.
        # When multiple nodes are close, prefer the one supported by more modalities
        # (e.g., metrics + traces/logs), to avoid selecting the wrong node among neighbors.
        top_metric = max((float(node_metric_max.get(c, 0.0)) for c, _ in node_items), default=0.0)
        if top_metric <= 0:
            return
        try:
            tie_margin = float(os.getenv("RCA_NODE_DOMINANCE_TIE_MARGIN", "0.06"))
        except ValueError:
            tie_margin = 0.06
        tie_margin = max(0.0, min(0.3, tie_margin))
        cutoff = top_metric * (1.0 - tie_margin)
        close_nodes = [c for c, _ in node_items if float(node_metric_max.get(c, 0.0)) >= cutoff]

        def _node_key(c: str) -> tuple[int, float, float]:
            mods = modality_support.get(c) or set()
            m = len([x for x in mods if x])
            return (m, float(node_metric_max.get(c, 0.0)), float(scores.get(c, 0.0)))

        top_node = max(close_nodes, key=_node_key) if close_nodes else max(node_items, key=lambda x: node_metric_max.get(x[0], 0.0))[0]
        top_node_score = float(scores.get(top_node, 0.0))
        top_node_metric = float(node_metric_max.get(top_node, 0.0))

        try:
            min_metric = float(os.getenv("RCA_NODE_DOMINANCE_MIN_METRIC", "5.2"))
        except ValueError:
            min_metric = 5.2
        if top_node_metric < min_metric:
            return

        # Extra safety: require traces support by default. This reduces false positives
        # where node KPIs spike due to downstream service failures.
        # Default OFF: node dominance is already a strong heuristic; requiring traces can
        # reduce recall on true node faults.
        require_traces = os.getenv("RCA_NODE_DOMINANCE_REQUIRE_TRACES", "0") not in {"0", "false", "False"}
        if require_traces:
            mods = modality_support.get(top_node) or set()
            if "traces" not in mods:
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
            scale = float(os.getenv("RCA_COMPONENT_PRIOR_SCALE", "0.0"))
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
