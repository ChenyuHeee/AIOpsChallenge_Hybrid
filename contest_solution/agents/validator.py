"""Validator and formatter for final contest submission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from ..config import PipelineConfig
from ..utils.hypothesis import Hypothesis


@dataclass(slots=True)
class SubmissionEntry:
    uuid: str
    component: str
    reason: str
    reasoning_trace: List[dict]

    def to_dict(self) -> dict:
        return {
            "uuid": self.uuid,
            "component": self.component,
            "reason": self.reason,
            "reasoning_trace": self.reasoning_trace,
        }


class SubmissionValidator:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def enforce_limits(self, component: str, reason: str, steps: List[str]) -> tuple[str, str, List[str]]:
        component = component or "unknown"
        words = reason.split()
        if len(words) > self.config.max_reason_words:
            reason = " ".join(words[: self.config.max_reason_words])
        truncated_steps: List[str] = []
        for step in steps[: self.config.target_trace_steps]:
            tokens = step.split()
            if len(tokens) > self.config.max_observation_words:
                truncated_steps.append(" ".join(tokens[: self.config.max_observation_words]))
            else:
                truncated_steps.append(step)
        while len(truncated_steps) < self.config.target_trace_steps:
            truncated_steps.append("Review telemetry status")
        return component, reason, truncated_steps

    def build_trace(self, steps: List[str], component: str, hypothesis_bank: Dict[str, List[Hypothesis]]) -> List[dict]:
        trace = []
        for idx, sentence in enumerate(steps, start=1):
            action = self._infer_action(sentence)
            observation = sentence
            trace.append({"step": idx, "action": action, "observation": observation})
        supporting = hypothesis_bank.get(component, [])
        snippets = []
        for hypothesis in supporting[:2]:
            for evidence in hypothesis.evidence[:2]:
                snippets.append(f"{evidence.modality}:{evidence.summary}")
        if snippets:
            trace.append({"step": len(trace) + 1, "action": "ReferenceEvidence", "observation": "; ".join(snippets[:3])})
        return trace[: self.config.target_trace_steps]

    def _infer_action(self, sentence: str) -> str:
        lowered = sentence.lower()
        if "metric" in lowered or "latency" in lowered:
            return "AnalyzeMetrics"
        if "log" in lowered or "error" in lowered:
            return "AnalyzeLogs"
        if "trace" in lowered or "span" in lowered:
            return "AnalyzeTraces"
        if "graph" in lowered or "chain" in lowered:
            return "AnalyzeGraph"
        if "confirm" in lowered or "root cause" in lowered:
            return "ConfirmRootCause"
        return "Reason"
