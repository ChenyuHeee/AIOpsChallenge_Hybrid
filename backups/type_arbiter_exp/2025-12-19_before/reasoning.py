"""Reasoning agent that fuses retrieval, hypotheses, and consensus."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List

from ..config import PipelineConfig
from ..knowledge.insights import PaperInsight
from ..llm.base import LLMClient
from ..utils.hypothesis import Hypothesis

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ReasoningOutput:
    component: str
    reason: str
    steps: List[str]
    raw_response: str


class ReasoningAgent:
    def __init__(self, client: LLMClient, config: PipelineConfig) -> None:
        self.client = client
        self.config = config

    def run(
        self,
        uuid: str,
        query: str,
        plan_steps: List[str],
        insights: List[PaperInsight],
        ranked_components: List[tuple[str, float]],
        hypothesis_bank: Dict[str, List[Hypothesis]],
    ) -> ReasoningOutput:
        require_llm = os.getenv("RCA_REQUIRE_LLM", "1") not in {"0", "false", "False"}
        prompt = self._build_prompt(uuid, query, plan_steps, insights, ranked_components, hypothesis_bank)
        raw = self._call_llm(prompt)
        parsed = self._parse(raw)

        if require_llm:
            # Enforce that LLM produced a usable structured response.
            if not parsed.component or not parsed.reason or not parsed.steps:
                raise RuntimeError(
                    "LLM response is missing required fields (component/reason/analysis_steps). "
                    "Set valid LLM credentials or fix the prompt/model output."
                )
        # Align component token with known candidates (case-insensitive) to avoid token mismatch.
        parsed.component = self._align_component(parsed.component, ranked_components)
        # If LLM is unavailable or returns unknown, fall back to consensus+evidence.
        if not parsed.component or parsed.component.strip().lower() == "unknown":
            parsed = self._fallback(query, ranked_components, hypothesis_bank)
        return ReasoningOutput(component=parsed.component, reason=parsed.reason, steps=parsed.steps, raw_response=raw)

    def _build_prompt(
        self,
        uuid: str,
        query: str,
        plan_steps: List[str],
        insights: List[PaperInsight],
        ranked_components: List[tuple[str, float]],
        hypothesis_bank: Dict[str, List[Hypothesis]],
    ) -> str:
        insight_snippets = "\n".join(insight.to_prompt_segment() for insight in insights)
        
        try:
            topk = int(os.getenv("RCA_LLM_COMPONENT_CANDIDATE_TOPK", "20"))
        except ValueError:
            topk = 20
        topk = max(8, min(40, topk))

        ranked_for_llm = ranked_components[:topk]

        # Separate node vs service candidates to guide LLM attention.
        node_candidates = [(c, s) for c, s in ranked_for_llm if self._is_node_component(c)]
        service_candidates = [(c, s) for c, s in ranked_for_llm if not self._is_node_component(c)]
        
        ranked_summary_lines = []
        if node_candidates:
            ranked_summary_lines.append("Node/Infrastructure candidates (if node_cpu/node_memory anomalies present):")
            for c, s in node_candidates[:5]:
                ranked_summary_lines.append(f"  - {c}: score {s:.3f}")
        if service_candidates:
            ranked_summary_lines.append("Service candidates:")
            for c, s in service_candidates[:8]:
                ranked_summary_lines.append(f"  - {c}: score {s:.3f}")
        ranked_summary = "\n".join(ranked_summary_lines)
        
        # Highlight node KPI evidence at top-level.
        node_kpi_highlights = []
        for comp, hyps in hypothesis_bank.items():
            if not self._is_node_component(comp):
                continue
            for h in hyps:
                for ev in h.evidence[:2]:
                    if "node_cpu" in ev.summary.lower() or "node_memory" in ev.summary.lower():
                        node_kpi_highlights.append(f"[{comp}] {ev.summary}")
        node_kpi_section = ""
        if node_kpi_highlights:
            node_kpi_section = "\n\nNode-level KPI anomalies (infrastructure root causes):\n" + "\n".join(node_kpi_highlights[:6])
        
        evidence_payload: Dict[str, List[Dict[str, str]]] = {}
        for component, hypotheses in hypothesis_bank.items():
            evidence_payload[component] = [
                {
                    "source": hypothesis.source,
                    "confidence": f"{hypothesis.confidence:.2f}",
                    "evidence": "; ".join(f"[{item.modality}] {item.summary}" for item in hypothesis.evidence[:4]),
                }
                for hypothesis in hypotheses
            ]
        
        valid_components = ", ".join([c for c, _ in ranked_for_llm])
        
        prompt = f"""
Case UUID: {uuid}
Incident query: {query}

Standard operating procedure phases:
{plan_steps}

Retrieved expert insights:
{insight_snippets}
{node_kpi_section}

Consensus ranked components:
{ranked_summary}

Evidence per component (JSON):
{json.dumps(evidence_payload, ensure_ascii=False, indent=2)}

Respond with strict JSON having keys component, reason, analysis_steps.
- component MUST be exactly one of: [{valid_components}]. Do NOT invent new names.
- Prefer node candidates (aiops-k8s-*) when node_cpu_usage_rate or node_memory anomalies are dominant and widespread across multiple services.
- reason must be <= {self.config.max_reason_words} words, cite concrete evidence, and include 1-3 exact tokens copied from the Evidence JSON (e.g. metric names or error keywords).
    Put the copied tokens EARLY (within the first 10 words) to maximize keyword matching.
- analysis_steps must list 3-7 short steps aligned with the SOP phases.
"""
        return prompt.strip()
    
    @staticmethod
    def _is_node_component(component: str) -> bool:
        """Detect infrastructure node identifiers."""
        return bool(component and component.startswith("aiops-k8s-"))

    def _call_llm(self, prompt: str) -> str:
        require_llm = os.getenv("RCA_REQUIRE_LLM", "1") not in {"0", "false", "False"}
        if os.getenv("RCA_DISABLE_LLM", "0") not in {"", "0", "false", "False"}:
            if require_llm:
                raise RuntimeError("LLM is disabled via RCA_DISABLE_LLM but RCA_REQUIRE_LLM=1.")
            return "{}"
        try:
            attempts = int(os.getenv("RCA_LLM_ATTEMPTS", "3"))
        except ValueError:
            attempts = 3
        attempts = max(1, attempts)
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return self.client.complete(prompt, temperature=0.0)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                LOGGER.warning("LLM call failed (%d/%d): %s", attempt, attempts, exc)
        if last_error:
            LOGGER.error("LLM repeatedly failed: %s", last_error)
            if require_llm:
                raise RuntimeError(f"LLM repeatedly failed: {last_error}")
            return "{}"
        return "{}"

    def _parse(self, raw: str) -> ReasoningOutput:
        try:
            data = json.loads(self._extract_json(raw))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Failed to parse LLM response: %s", exc)
            return ReasoningOutput(component="", reason="", steps=[], raw_response=raw)
        component = str(data.get("component", "")).strip()
        reason = str(data.get("reason", "")).strip()
        steps_raw = data.get("analysis_steps", [])
        if isinstance(steps_raw, str):
            steps = [steps_raw]
        else:
            steps = [str(step).strip() for step in steps_raw if str(step).strip()]
        steps = steps[: max(3, self.config.target_trace_steps)]
        return ReasoningOutput(component=component, reason=reason, steps=steps, raw_response=raw)

    @staticmethod
    def _align_component(component: str, ranked_components: List[tuple[str, float]]) -> str:
        if not component:
            return component
        component_clean = component.strip()
        if not component_clean:
            return ""

        candidates = [c for c, _ in ranked_components]
        if component_clean in candidates:
            return component_clean

        lowered = component_clean.lower()
        for candidate in candidates:
            if candidate.lower() == lowered:
                return candidate
        return component_clean

    def _fallback(
        self,
        query: str,
        ranked_components: List[tuple[str, float]],
        hypothesis_bank: Dict[str, List[Hypothesis]],
    ) -> ReasoningOutput:
        if ranked_components:
            component = ranked_components[0][0]
        else:
            component = "unknown"
        hypotheses = hypothesis_bank.get(component, [])
        evidence_strings = []
        for hypothesis in hypotheses:
            for item in hypothesis.evidence[:2]:
                evidence_strings.append(f"{item.modality}:{item.summary}")

        prefix = ""
        if os.getenv("RCA_FALLBACK_REASON_INCLUDE_QUERY", "1") not in {"0", "false", "False"}:
            q = (query or "").strip().replace("\n", " ")
            if q:
                prefix = q[:180] + "; "

        reason_body = ", ".join(evidence_strings)
        reason = (prefix + reason_body).strip()[:240]
        if not reason:
            reason = (prefix.strip("; ") or "Insufficient evidence")
        steps = [
            "Scope telemetry per SOP",
            "Analyze highest severity metrics",
            "Inspect correlated logs",
            "Review slow spans",
            f"Confirm {component} as root cause",
        ][: self.config.target_trace_steps]
        return ReasoningOutput(component=component, reason=reason, steps=steps, raw_response="{}")

    @staticmethod
    def _extract_json(payload: str) -> str:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found")
        return payload[start : end + 1]
