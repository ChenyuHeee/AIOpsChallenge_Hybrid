"""Specialist agents inspired by TAMO and Chain-of-Event."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from ..data.loader import TelemetryFrames
from ..utils.hypothesis import EvidenceItem, Hypothesis


@dataclass(slots=True)
class SpecialistContext:
    keywords: List[str]
    component_hints: List[str]


class MetricSpecialist:
    def run(self, telemetry: TelemetryFrames, ctx: SpecialistContext) -> List[Hypothesis]:
        df = telemetry.metrics
        if df.empty:
            return []
        value_col = self._first_numeric(df, ["value", "gauge", "sum", "count", "sample_value"])
        component_col = self._first(df, ["component", "service", "pod", "k8_pod", "instance"])
        metric_col = self._first(df, ["metric", "metric_name", "__name__", "name"])
        if value_col is None or component_col is None or metric_col is None:
            return []
        anomalies: Dict[str, List[EvidenceItem]] = {}
        for (component, metric), group in df.groupby([component_col, metric_col]):
            numeric = pd.to_numeric(group[value_col], errors="coerce").dropna()
            if numeric.empty:
                continue
            severity = self._zscore(numeric)
            if severity < 1.0 and numeric.mean() <= 0:
                continue
            normalized_component = self._normalize(component)
            evidence = EvidenceItem(
                modality="metrics",
                component=normalized_component,
                summary=f"{metric} spike {numeric.max():.2f} vs avg {numeric.mean():.2f}",
                score=float(severity),
            )
            anomalies.setdefault(normalized_component, []).append(evidence)
        return [
            Hypothesis(
                component=component,
                confidence=sum(item.score for item in evidence) / len(evidence),
                source="MetricSpecialist",
                evidence=evidence,
            )
            for component, evidence in anomalies.items()
        ]

    @staticmethod
    def _zscore(series: pd.Series) -> float:
        mean = series.mean()
        std = series.std()
        if std == 0 or math.isnan(std):
            return 0.0
        return float((series.max() - mean) / std)

    @staticmethod
    def _first(df: pd.DataFrame, candidates: List[str]) -> str | None:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    @staticmethod
    def _first_numeric(df: pd.DataFrame, candidates: List[str]) -> str | None:
        for column in candidates:
            if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                return column
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return numeric_cols[0] if len(numeric_cols) else None

    @staticmethod
    def _normalize(value: str) -> str:
        cleaned = (value or "").strip().lower()
        return cleaned.replace("hipstershop.", "").replace("service=", "").replace("svc-", "")


class LogSpecialist:
    ERROR_KEYWORDS = ("error", "exception", "fail", "timeout", "critical")

    def run(self, telemetry: TelemetryFrames, ctx: SpecialistContext) -> List[Hypothesis]:
        df = telemetry.logs
        if df.empty:
            return []
        message_col = self._first(df, ["message", "log", "body", "msg"])
        level_col = self._first(df, ["level", "severity", "log_level"])
        component_col = self._first(df, ["component", "service", "pod", "hostname"])
        if message_col is None or component_col is None:
            return []
        if level_col is None:
            df["_severity"] = "error"
            level_col = "_severity"
        hypotheses: Dict[str, List[EvidenceItem]] = {}
        keywords = set(ctx.keywords)
        for _, row in df.iterrows():
            message = str(row.get(message_col, ""))
            if not message:
                continue
            lowered = message.lower()
            if not any(word in lowered for word in self.ERROR_KEYWORDS) and not keywords.intersection(lowered.split()):
                continue
            component = self._normalize(row.get(component_col, ""))
            level = str(row.get(level_col, "error"))
            evidence = EvidenceItem(
                modality="logs",
                component=component,
                summary=f"{level} log: {message[:120]}",
                score=1.2 if "error" in level.lower() else 0.8,
            )
            hypotheses.setdefault(component, []).append(evidence)
        return [
            Hypothesis(
                component=component,
                confidence=min(1.5, sum(item.score for item in evidence)),
                source="LogSpecialist",
                evidence=evidence,
            )
            for component, evidence in hypotheses.items()
        ]

    @staticmethod
    def _first(df: pd.DataFrame, candidates: List[str]) -> str | None:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    @staticmethod
    def _normalize(value: str) -> str:
        cleaned = (value or "").strip().lower()
        return cleaned.replace("hipstershop.", "").replace("service=", "").replace("svc-", "")


class TraceSpecialist:
    def run(self, telemetry: TelemetryFrames, ctx: SpecialistContext) -> List[Hypothesis]:
        df = telemetry.traces
        if df.empty:
            return []
        duration_col = self._first(df, ["duration", "durationMs", "latency", "elapsed"])
        component_col = self._first(df, ["service", "serviceName", "process.serviceName", "component"])
        if duration_col is None or component_col is None:
            return []
        df = df.copy()
        df["_duration"] = pd.to_numeric(df[duration_col], errors="coerce")
        df = df.dropna(subset=["_duration"])
        if df.empty:
            return []
        hypotheses: Dict[str, List[EvidenceItem]] = {}
        for component, group in df.groupby(component_col):
            latency_ms = float(group["_duration"].max())
            evidence = EvidenceItem(
                modality="traces",
                component=self._normalize(component),
                summary=f"Trace latency peak {latency_ms:.2f}",
                score=math.log1p(latency_ms / max(group["_duration"].mean(), 1.0)),
            )
            hypotheses.setdefault(evidence.component, []).append(evidence)
        return [
            Hypothesis(
                component=component,
                confidence=sum(item.score for item in evidence) / len(evidence),
                source="TraceSpecialist",
                evidence=evidence,
            )
            for component, evidence in hypotheses.items()
        ]

    @staticmethod
    def _first(df: pd.DataFrame, candidates: List[str]) -> str | None:
        for column in candidates:
            if column in df.columns:
                return column
        return None

    @staticmethod
    def _normalize(value: str) -> str:
        cleaned = (value or "").strip().lower()
        return cleaned.replace("hipstershop.", "").replace("service=", "").replace("svc-", "")


class GraphSpecialist:
    def run(self, telemetry: TelemetryFrames, ctx: SpecialistContext) -> List[Hypothesis]:
        graph = telemetry.event_graph
        hypotheses: List[Hypothesis] = []
        for source, targets in graph.items():
            if not targets:
                continue
            score = 1.0 + 0.2 * len(targets)
            if ctx.component_hints and source not in ctx.component_hints:
                score *= 0.8
            evidence = [
                EvidenceItem(
                    modality="graph",
                    component=source,
                    summary=f"Propagation to {', '.join(targets[:5])}",
                    score=score,
                )
            ]
            hypotheses.append(Hypothesis(component=source, confidence=score, source="GraphSpecialist", evidence=evidence))
        return hypotheses
