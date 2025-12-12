"""Standard operating procedures inspired by Flow-of-Action and RCAEval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class SOPStep:
    name: str
    objective: str
    success_criteria: str
    tools: List[str]


INCIDENT_SOP: List[SOPStep] = [
    SOPStep(
        name="ScopeTelemetry",
        objective="Confirm telemetry coverage and time window alignment.",
        success_criteria="Metrics/logs/traces are loaded or fallback strategies planned.",
        tools=["TelemetryLoader", "WindowAligner"],
    ),
    SOPStep(
        name="SurfaceSignals",
        objective="Identify high-severity anomalies across metrics/logs/traces.",
        success_criteria="Top anomalies ranked with component associations.",
        tools=["MetricAnalyzer", "LogClassifier", "TraceProfiler"],
    ),
    SOPStep(
        name="ConstructGraph",
        objective="Build event chain to capture propagation paths.",
        success_criteria="Graph contains at least one causal chain hypothesis.",
        tools=["EventGraphBuilder"],
    ),
    SOPStep(
        name="GenerateHypotheses",
        objective="Propose candidate root causes using specialist agents.",
        success_criteria="Each hypothesis links components, signals, and narrative reasoning.",
        tools=["MetricSpecialist", "LogSpecialist", "TraceSpecialist", "LLMReasoner"],
    ),
    SOPStep(
        name="RunConsensus",
        objective="Fuse hypotheses via weighted voting with historical priors.",
        success_criteria="Consensus returns ranked components with confidence score.",
        tools=["ConsensusOrchestrator"],
    ),
    SOPStep(
        name="ValidateAndExplain",
        objective="Check consistency, validate against memory, and craft explanation trace.",
        success_criteria="Final submission satisfies competition format and factual constraints.",
        tools=["Validator", "TraceComposer", "MemoryBank"],
    ),
]
