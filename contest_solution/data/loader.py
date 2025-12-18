"""Telemetry data loader with event-chain enrichment."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import pyarrow.dataset as ds  # type: ignore
except Exception:  # pragma: no cover - optional optimization
    ds = None  # type: ignore

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
        if window_padding is not None:
            self.window_padding = window_padding
            return
        # Smaller padding reduces cross-incident contamination, which improves LA.
        # Can be overridden via env when doing recall-focused runs.
        try:
            minutes = int(os.getenv("RCA_WINDOW_PADDING_MIN", "15"))
        except ValueError:
            minutes = 15
        self.window_padding = timedelta(minutes=max(0, minutes))

    def load(self, telemetry_root: Path, date_hint: str, window: Optional[Tuple[datetime, datetime]]) -> TelemetryFrames:
        day_dir = self._resolve_day_dir(telemetry_root, date_hint, window)
        # When the query doesn't contain a time window (common in this contest metadata),
        # reading the entire day of telemetry can be too large. Apply conservative caps.
        if window is None:
            metrics = self._read_dir(day_dir / self.METRIC_DIR, window, max_files=180, sample_rows=50_000)
            logs = self._read_dir(day_dir / self.LOG_DIR, window, max_files=60, sample_rows=80_000)
            traces = self._read_dir(day_dir / self.TRACE_DIR, window, max_files=6, sample_rows=40_000)
        else:
            metrics = self._read_dir(day_dir / self.METRIC_DIR, window)
            logs = self._read_dir(day_dir / self.LOG_DIR, window)
            traces = self._read_dir(day_dir / self.TRACE_DIR, window)
        graph = self._build_event_graph(metrics, logs, traces)
        return TelemetryFrames(metrics=metrics, logs=logs, traces=traces, event_graph=graph)

    def _resolve_day_dir(
        self,
        telemetry_root: Path,
        date_hint: str,
        window: Optional[Tuple[datetime, datetime]],
    ) -> Path:
        """Resolve the day directory for telemetry.

        In some datasets the CSV `date` is in UTC date, while extracted folders are in local
        date (e.g., UTC+8). When `YYYY-MM-DD` folder is missing, try nearby dates derived
        from the incident window.
        """

        def _as_date_str(value: datetime) -> str:
            return self._ensure_utc(value).date().isoformat()

        candidates: List[str] = []
        if date_hint:
            candidates.append(date_hint)
            try:
                base = datetime.fromisoformat(date_hint).date()
                candidates.append((base + timedelta(days=1)).isoformat())
                candidates.append((base - timedelta(days=1)).isoformat())
            except Exception:  # noqa: BLE001
                pass

        if window is not None:
            start, end = window
            utc_dates = {_as_date_str(start), _as_date_str(end)}
            # Common shift: telemetry folders often follow local date (UTC+8) while query is Z.
            shifted_start = self._ensure_utc(start) + timedelta(hours=8)
            shifted_end = self._ensure_utc(end) + timedelta(hours=8)
            local_dates = {shifted_start.date().isoformat(), shifted_end.date().isoformat()}
            for value in list(local_dates) + list(utc_dates):
                candidates.append(value)
                try:
                    base = datetime.fromisoformat(value).date()
                    candidates.append((base + timedelta(days=1)).isoformat())
                    candidates.append((base - timedelta(days=1)).isoformat())
                except Exception:  # noqa: BLE001
                    continue

        seen: set[str] = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            day_dir = telemetry_root / candidate
            if day_dir.exists():
                return day_dir

        fallback = telemetry_root / date_hint
        LOGGER.warning(
            "Telemetry day directory missing: %s (tried: %s)",
            fallback,
            ", ".join(list(seen)[:8]),
        )
        return telemetry_root

    def _read_dir(
        self,
        directory: Path,
        window: Optional[Tuple[datetime, datetime]],
        max_files: int | None = None,
        sample_rows: int | None = None,
    ) -> pd.DataFrame:
        if not directory.exists():
            LOGGER.debug("Directory missing: %s", directory)
            return pd.DataFrame()
        frames: List[pd.DataFrame] = []
        for idx, parquet_path in enumerate(sorted(directory.rglob("*.parquet"))):
            if max_files is not None and idx >= max_files:
                break
            try:
                if window is not None and directory.name == self.TRACE_DIR:
                    df = self._read_trace_parquet_windowed(parquet_path, window)
                elif window is not None and directory.name in {self.LOG_DIR, self.METRIC_DIR}:
                    df = self._read_time_string_parquet_windowed(parquet_path, window, directory.name)
                else:
                    df = pd.read_parquet(parquet_path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Failed to read %s: %s", parquet_path, exc)
                continue
            if sample_rows is not None and len(df) > sample_rows:
                # Random sample keeps both early/late incident signatures.
                df = df.sample(n=sample_rows, random_state=7).reset_index(drop=True)
            if window is None:
                frames.append(df)
                continue
            filtered = self._filter_by_window(df, window)
            if not filtered.empty:
                frames.append(filtered)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def _read_time_string_parquet_windowed(
        self,
        parquet_path: Path,
        window: Tuple[datetime, datetime],
        directory_name: str,
    ) -> pd.DataFrame:
        """Windowed read for metric/log parquets that store time as ISO strings.

        Using lexical comparisons on ISO8601 strings enables predicate pushdown.
        """

        if ds is None:
            return pd.read_parquet(parquet_path)

        start, end = window
        # Time window alignment:
        # - In this repo's contest data, parquet time strings are ISO8601 with 'Z' and represent UTC.
        # - Day folders may follow local date, but timestamps remain UTC.
        # Therefore we DO NOT shift by default.
        # If you encounter a dataset where query times are local but suffixed with 'Z',
        # set RCA_TIME_SHIFT_HOURS=-8 (or other) to realign.
        try:
            shift_hours = float(os.getenv("RCA_TIME_SHIFT_HOURS", "0"))
        except ValueError:
            shift_hours = 0.0
        start = self._ensure_utc(start) + timedelta(hours=shift_hours) - self.window_padding
        end = self._ensure_utc(end) + timedelta(hours=shift_hours) + self.window_padding
        start_iso = start.isoformat().replace("+00:00", "Z")
        end_iso = end.isoformat().replace("+00:00", "Z")

        dataset = ds.dataset(parquet_path, format="parquet")
        schema_names = {field.name for field in dataset.schema}

        if directory_name == self.LOG_DIR:
            time_candidates = ["@timestamp", "timestamp", "time"]
            preferred = [
                name
                for name in (
                    "@timestamp",
                    "timestamp",
                    "time",
                    "message",
                    "msg",
                    "log",
                    "body",
                    "k8_pod",
                    "k8_node_name",
                    "level",
                    "severity",
                )
                if name in schema_names
            ]
        else:
            time_candidates = ["time", "timestamp", "@timestamp"]
            # Keep all metric KPI columns (many are numeric and needed by MetricSpecialist).
            preferred = None

        time_field = next((c for c in time_candidates if c in schema_names), None)
        if time_field is None:
            return pd.read_parquet(parquet_path)

        filt = (ds.field(time_field) >= start_iso) & (ds.field(time_field) <= end_iso)
        table = dataset.to_table(filter=filt, columns=preferred)
        if table.num_rows > 120_000:
            table = table.slice(0, 120_000)
        return table.to_pandas()

    def _read_trace_parquet_windowed(self, parquet_path: Path, window: Tuple[datetime, datetime]) -> pd.DataFrame:
        """Read trace parquet with predicate pushdown when possible.

        Trace data can be huge; filtering after `pd.read_parquet` may OOM.
        If pyarrow.dataset is available, apply a numeric timestamp filter.
        """

        if ds is None:
            return pd.read_parquet(parquet_path)

        start, end = window
        # Keep aligned with _read_time_string_parquet_windowed (default: no shift).
        try:
            shift_hours = float(os.getenv("RCA_TIME_SHIFT_HOURS", "0"))
        except ValueError:
            shift_hours = 0.0
        start = self._ensure_utc(start) + timedelta(hours=shift_hours) - self.window_padding
        end = self._ensure_utc(end) + timedelta(hours=shift_hours) + self.window_padding

        dataset = ds.dataset(parquet_path, format="parquet")
        schema_names = {field.name for field in dataset.schema}

        # Avoid converting unused nested columns (can be very slow/large).
        preferred_columns = [
            name
            for name in (
                "traceID",
                "spanID",
                "operationName",
                "startTimeMillis",
                "startTime",
                "duration",
                "tags",
                "process",
            )
            if name in schema_names
        ]

        # Prefer millisecond column when present.
        if "startTimeMillis" in schema_names:
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            filt = (ds.field("startTimeMillis") >= start_ms) & (ds.field("startTimeMillis") <= end_ms)
            table = dataset.to_table(filter=filt, columns=preferred_columns or None)
            if table.num_rows > 50_000:
                table = table.slice(0, 50_000)
            return table.to_pandas()

        # Jaeger `startTime` is often microseconds.
        if "startTime" in schema_names:
            start_us = int(start.timestamp() * 1_000_000)
            end_us = int(end.timestamp() * 1_000_000)
            filt = (ds.field("startTime") >= start_us) & (ds.field("startTime") <= end_us)
            table = dataset.to_table(filter=filt, columns=preferred_columns or None)
            if table.num_rows > 50_000:
                table = table.slice(0, 50_000)
            return table.to_pandas()

        return pd.read_parquet(parquet_path)

    def _filter_by_window(self, df: pd.DataFrame, window: Tuple[datetime, datetime]) -> pd.DataFrame:
        start, end = window
        start = self._ensure_utc(start) - self.window_padding
        end = self._ensure_utc(end) + self.window_padding
        for column in ("time", "timestamp", "@timestamp", "startTime", "startTimeMillis"):
            if column not in df.columns:
                continue
            try:
                series = self._to_datetime_utc(df[column], column)
            except Exception:  # noqa: BLE001
                continue
            mask = (series >= start) & (series <= end)
            if mask.any():
                return df.loc[mask].reset_index(drop=True)
        return df

    @staticmethod
    def _to_datetime_utc(series: pd.Series, column_name: str) -> pd.Series:
        """Parse timestamps robustly across common telemetry schemas.

        - ISO strings: use pandas default parsing.
        - Epoch ints:
          - Jaeger `startTime` is typically microseconds.
          - Jaeger `startTimeMillis` is milliseconds.
          - Other numeric columns fall back to heuristic based on magnitude.
        """

        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            unit: str | None = None
            if column_name.endswith("Millis"):
                unit = "ms"
            elif column_name.endswith("Time"):
                unit = "us"
            else:
                max_val = float(numeric.max()) if numeric.notna().any() else 0.0
                if max_val >= 1e14:
                    unit = "us"
                elif max_val >= 1e11:
                    unit = "ms"
                else:
                    unit = "s"
            return pd.to_datetime(numeric, errors="coerce", utc=True, unit=unit)

        return pd.to_datetime(series, errors="coerce", utc=True)

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
