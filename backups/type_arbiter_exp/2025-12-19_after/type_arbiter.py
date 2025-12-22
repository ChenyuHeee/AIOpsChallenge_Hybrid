"""Type arbitration layer to reduce node/service/TiDB systematic confusion.

This module provides two backends:
- rule: deterministic heuristics using consensus evidence/modality signals
- llm: an LLM-based classifier that decides the *type* (node/service/tidb)

It then applies one of the integration modes:
- postcheck: only switch component when predicted type disagrees with current top-1
- pregate: softly reweight candidates by predicted type and rerank
- hardfilter: only allow candidates of predicted type (fallback to original if none)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from ..llm.base import LLMClient
from .consensus import ConsensusResult

LOGGER = logging.getLogger(__name__)


def component_type(component: str) -> str:
    token = (component or "").strip()
    if not token:
        return "unknown"
    if token.startswith("aiops-k8s-"):
        return "node"
    if token.startswith("tidb-"):
        return "tidb"
    return "service"


@dataclass(slots=True)
class TypeDecision:
    decided_type: str  # node | service | tidb | unknown
    confidence: float
    raw_response: str = ""


class TypeArbiterAgent:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    def enabled(self) -> bool:
        return os.getenv("RCA_ENABLE_TYPE_ARBITER", "0") not in {"0", "false", "False"}

    def decide(self, *, uuid: str, query: str, consensus: ConsensusResult) -> TypeDecision:
        backend = (os.getenv("RCA_TYPE_ARBITER_BACKEND", "rule") or "rule").strip().lower()
        if backend == "llm":
            return self._decide_llm(uuid=uuid, query=query, consensus=consensus)
        return self._decide_rule(query=query, consensus=consensus)

    def apply(
        self,
        *,
        ranked_components: List[Tuple[str, float]],
        consensus: ConsensusResult,
        current_component: str,
        decision: TypeDecision,
    ) -> tuple[str, List[Tuple[str, float]]]:
        """Return (chosen_component, reranked_components)."""

        try:
            threshold = float(os.getenv("RCA_TYPE_ARBITER_CONFIDENCE_THRESHOLD", "0.85"))
        except ValueError:
            threshold = 0.85
        threshold = max(0.0, min(1.0, threshold))

        mode = (os.getenv("RCA_TYPE_ARBITER_MODE", "postcheck") or "postcheck").strip().lower()
        if not decision.decided_type or decision.decided_type == "unknown" or decision.confidence < threshold:
            return current_component, ranked_components

        if mode == "hardfilter":
            filtered = [(c, s) for c, s in ranked_components if component_type(c) == decision.decided_type]
            if not filtered:
                return current_component, ranked_components
            return filtered[0][0], filtered

        if mode == "pregate":
            try:
                boost = float(os.getenv("RCA_TYPE_ARBITER_TYPE_BOOST", "0.12"))
            except ValueError:
                boost = 0.12
            try:
                penalty = float(os.getenv("RCA_TYPE_ARBITER_OTHER_PENALTY", "0.05"))
            except ValueError:
                penalty = 0.05
            boost = max(0.0, min(0.8, boost))
            penalty = max(0.0, min(0.8, penalty))
            rescored: List[Tuple[str, float]] = []
            for c, s in ranked_components:
                t = component_type(c)
                if t == decision.decided_type:
                    rescored.append((c, float(s) * (1.0 + boost)))
                else:
                    rescored.append((c, float(s) * (1.0 - penalty)))
            rescored.sort(key=lambda x: x[1], reverse=True)
            return rescored[0][0] if rescored else current_component, rescored

        # default: postcheck
        if component_type(current_component) != decision.decided_type:
            for c, _ in ranked_components:
                if component_type(c) == decision.decided_type:
                    # Only switch when the chosen type has at least some non-metric support,
                    # unless it's an infra-type selected by strong metrics.
                    return c, ranked_components
        return current_component, ranked_components

    def _decide_rule(self, *, query: str, consensus: ConsensusResult) -> TypeDecision:
        ranked = consensus.ranked_components
        if not ranked:
            return TypeDecision(decided_type="unknown", confidence=0.0)

        best_by_type: Dict[str, Tuple[str, float]] = {}
        for c, s in ranked:
            t = component_type(c)
            if t == "unknown":
                continue
            if t not in best_by_type or float(s) > float(best_by_type[t][1]):
                best_by_type[t] = (c, float(s))

        top_c, top_s = ranked[0][0], float(ranked[0][1])
        top_t = component_type(top_c)

        # If only one type exists, accept it.
        if len(best_by_type) == 1:
            return TypeDecision(decided_type=top_t, confidence=0.9)

        # Helper signals
        def has_non_metric_support(comp: str) -> bool:
            mods = set(consensus.modality_support.get(comp) or [])
            return bool({"logs", "traces", "graph"} & mods)

        def has_node_kpi(comp: str) -> bool:
            # Prefer explicit node_cpu/node_memory signals.
            evs = (consensus.supporting_evidence.get(comp) or [])[:8]
            joined = " ".join(evs).lower()
            return "node_cpu" in joined or "node_memory" in joined

        def has_tidb_signal(comp: str) -> bool:
            evs = (consensus.supporting_evidence.get(comp) or [])[:8]
            joined = " ".join(evs).lower()
            return "tikv" in joined or "tidb" in joined or "pd" in joined

        try:
            close_ratio = float(os.getenv("RCA_TYPE_ARBITER_CLOSE_RATIO", "0.85"))
        except ValueError:
            close_ratio = 0.85
        close_ratio = max(0.0, min(1.0, close_ratio))

        # Conservative preference: when close, prefer service if it has logs/traces support.
        service = best_by_type.get("service")
        node = best_by_type.get("node")
        tidb = best_by_type.get("tidb")

        if top_t == "node" and service:
            service_c, service_s = service
            # If node is only barely ahead and service has non-metric support, pick service.
            if float(service_s) >= float(top_s) * close_ratio and has_non_metric_support(service_c):
                return TypeDecision(decided_type="service", confidence=0.78)
            # If node has strong node KPI signals, keep node.
            if has_node_kpi(top_c):
                return TypeDecision(decided_type="node", confidence=0.76)

        if top_t == "tidb" and service:
            service_c, service_s = service
            if float(service_s) >= float(top_s) * close_ratio and has_non_metric_support(service_c):
                return TypeDecision(decided_type="service", confidence=0.76)
            if has_tidb_signal(top_c):
                return TypeDecision(decided_type="tidb", confidence=0.74)

        # If service is top and has non-metric support, accept service.
        if top_t == "service" and has_non_metric_support(top_c):
            return TypeDecision(decided_type="service", confidence=0.72)

        # Fallback: unknown (avoid risky flips)
        return TypeDecision(decided_type="unknown", confidence=0.4)

    def _decide_llm(self, *, uuid: str, query: str, consensus: ConsensusResult) -> TypeDecision:
        require_llm = os.getenv("RCA_REQUIRE_LLM", "1") not in {"0", "false", "False"}
        if os.getenv("RCA_DISABLE_LLM", "0") not in {"", "0", "false", "False"}:
            if require_llm:
                raise RuntimeError("LLM is disabled but RCA_REQUIRE_LLM=1 (type arbiter backend=llm).")
            return TypeDecision(decided_type="unknown", confidence=0.0, raw_response="{}")

        ranked = consensus.ranked_components[:20]
        payload: Dict[str, List[Dict[str, str]]] = {"node": [], "service": [], "tidb": []}
        for c, s in ranked:
            t = component_type(c)
            if t not in payload:
                continue
            ev = "; ".join((consensus.supporting_evidence.get(c) or [])[:3])
            mods = ",".join(consensus.modality_support.get(c) or [])
            payload[t].append({"component": c, "score": f"{float(s):.3f}", "modalities": mods, "evidence": ev})

        prompt = (
            "You are a classifier deciding the *root-cause type* for an incident.\n"
            "Possible types: node, service, tidb, unknown.\n\n"
            f"Case UUID: {uuid}\n"
            f"Incident query: {query}\n\n"
            "Top candidates (grouped by type, with evidence excerpts):\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            "Return STRICT JSON: {\"type\": <one of node|service|tidb|unknown>, \"confidence\": <0..1>, \"reason\": <<=20 words>}\n"
            "Guidance:\n"
            "- Choose node only if node_cpu/node_memory issues dominate and affect many services.\n"
            "- Choose tidb only if DB/storage symptoms dominate (e.g. tikv/pd/tidb latency/errors).\n"
            "- Otherwise prefer service.\n"
        )

        raw = self.client.complete(prompt, temperature=0.0)
        try:
            data = json.loads(self._extract_json(raw))
            t = str(data.get("type", "")).strip().lower()
            conf = float(data.get("confidence", 0.0))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("TypeArbiter LLM parse failed: %s", exc)
            return TypeDecision(decided_type="unknown", confidence=0.0, raw_response=raw)

        if t not in {"node", "service", "tidb", "unknown"}:
            t = "unknown"
        conf = max(0.0, min(1.0, conf))
        return TypeDecision(decided_type=t, confidence=conf, raw_response=raw)

    @staticmethod
    def _extract_json(payload: str) -> str:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        return payload[start : end + 1]
