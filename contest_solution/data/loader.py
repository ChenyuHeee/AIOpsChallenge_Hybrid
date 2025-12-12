"""Telemetry data loader with event-chain enrichment."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TelemetryFrames:
    metrics: pd.DataFrame
    logs: pd.DataFrame
    traces: pd.DataFrame
    event_graph: Dict[str, List[str]]

    def empty(self) -> bool:
        return self.metrics.empty and self.logs.empty and self.traces.empty


class TelemetryLoader:
    """Combines OpenRCA-style parquet loading with Chain-of-Event style graphs."""

    METRIC_DIR = "metric-parquet"
    LOG_DIR = "log-parquet"
    TRACE_DIR = "trace-parquet"

    def __init__(self, window_padding: timedelta | None = None) -> None:
        self.window_padding = window_padding or timedelta(minutes=45)

    def load(self, telemetry_root: Path, date_hint: str, window: Optional[Tuple[datetime, datetime]]) -> TelemetryFrames:
        day_dir = telemetry_root / date_hint
        if not day_dir.exists():
            LOGGER.warning("Telemetry day directory missing: %s", day_dir)
            day_dir = telemetry_root
        metrics = self._read_dir(day_dir / self.METRIC_DIR, window)
        logs = self._read_dir(day_dir / self.LOG_DIR, window)
        traces = self._read_dir(day_dir / self.TRACE_DIR, window)
        graph = self._build_event_graph(metrics, logs, traces)
        return TelemetryFrames(metrics=metrics, logs=logs, traces=traces, event_graph=graph)

    def _read_dir(self, directory: Path, window: Optional[Tuple[datetime, datetime]]) -> pd.DataFrame:
        if not directory.exists():
            LOGGER.debug("Directory missing: %s", directory)
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for parquet_path in sorted(directory.rglob("*.parquet")):
            try:
                df = pd.read_parquet(parquet_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to read %s: %s", parquet_path, exc)
                continue
            if window is None:
                frames.append(df)
                continue
            filtered = self._filter_by_window(df, window)
            if not filtered.empty:
                frames.append(filtered)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _filter_by_window(self, df: pd.DataFrame, window: Tuple[datetime, datetime]) -> pd.DataFrame:
        start, end = window
        start = self._ensure_utc(start) - self.window_padding
        end = self._ensure_utc(end) + self.window_padding
        for column in ("time", "timestamp", "@timestamp", "startTime", "startTimeMillis"):
            if column not in df.columns:
                continue
            try:
                series = pd.to_datetime(df[column], errors="coerce", utc=True)
            except Exception:  # noqa: BLE001
                continue
            mask = (series >= start) & (series <= end)
            if mask.any():
                return df.loc[mask].reset_index(drop=True)
        return df

    def _build_event_graph(self, metrics: pd.DataFrame, logs: pd.DataFrame, traces: pd.DataFrame) -> Dict[str, List[str]]:
        graph: Dict[str, List[str]] = {}
        edges: List[Tuple[str, str]] = []
        for df, component_col, peer_col in (
            (metrics, self._first(metrics, ["component", "service", "pod", "k8_pod"]), None),
            (logs, self._first(logs, ["component", "service", "pod", "hostname"]), None),
            (traces, self._first(traces, ["service", "serviceName", "process.serviceName"]), "peer.service"),
        ):
            if df.empty or component_col is None:
                continue
            source_series = df[component_col].astype(str)
            if peer_col and peer_col in df.columns:
                target_series = df[peer_col].astype(str)
                edges.extend(zip(source_series, target_series, strict=False))
            operation_col = self._first(df, ["operation", "operationName", "spanName"])
            if operation_col:
                for component, operation in zip(source_series, df[operation_col].astype(str), strict=False):
                    edges.append((component, operation))
        for source, target in edges:
            cleaned_source = self._normalize(source)
            cleaned_target = self._normalize(target)
            if not cleaned_source or not cleaned_target:
                continue
            graph.setdefault(cleaned_source, [])
            if cleaned_target not in graph[cleaned_source]:
                graph[cleaned_source].append(cleaned_target)
        return graph

    @staticmethod
    def _normalize(raw: str) -> str:
        value = (raw or "").strip().lower()
        if not value or value in {"nan", "none", "unknown"}:
            return ""
        return value.replace("hipstershop.", "").replace("service=", "").replace("svc-", "")

    @staticmethod
    def _ensure_utc(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _first(df: pd.DataFrame, candidates: Iterable[str]) -> Optional[str]:
        for name in candidates:
            if name in df.columns:
                return name
        return None
