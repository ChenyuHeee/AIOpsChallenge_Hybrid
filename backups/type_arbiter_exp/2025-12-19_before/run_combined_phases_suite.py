#!/usr/bin/env python3
"""Run experiment suite combining phase1 and phase2 into a single report.

This script extends run_final_experiment_suite.py to support multiple ground-truth
files and metadata files, merging results into a single report with per-phase breakdowns.

Usage:
  python tools/run_combined_phases_suite.py \
    --preset la_boost_v9_engineering_rules_plus \
    --suite-name "25_LA扩展_phase1+phase2合并_engineering_rules_v9" \
    --suite-slug la_v9_combined_seed7 \
    --phase1-metadata metadata_phase1.csv \
    --phase1-ground-truth AIOpsChallengeJudge/ground_truth_phase1.jsonl \
    --phase1-dates 2025-06-13 2025-06-07 2025-06-11 2025-06-08 2025-06-10 2025-06-09 2025-06-14 \
    --phase2-metadata metadata_phase2.csv \
    --phase2-ground-truth AIOpsChallengeJudge/ground_truth_phase2.jsonl \
    --phase2-dates 2025-06-27 2025-06-17 2025-06-20 2025-06-18 2025-06-19 2025-06-24 2025-06-28 \
    --telemetry-root /abs/path/data
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

# Import evaluation module (same as run_final_experiment_suite.py)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from contest_solution.main import main as generator_main  # noqa: F401

# Import judge evaluate module
judge_root = Path(__file__).resolve().parents[1].parent / "AIOpsChallengeJudge"
sys.path.insert(0, str(judge_root))
import evaluate  # noqa: E402


@dataclass
class Candidate:
    key: str
    name: str
    env: Dict[str, str]


@dataclass
class PhaseConfig:
    phase_name: str
    metadata: Path
    ground_truth: Path
    dates: List[str]


@dataclass
class CandidateResult:
    candidate: Candidate
    per_phase_per_date: Dict[str, Dict[str, Dict[str, Any]]]  # phase -> date -> metrics


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", type=str, required=True, help="Preset name for candidates")
    p.add_argument("--suite-name", type=str, required=True, help="Suite display name for report")
    p.add_argument("--suite-slug", type=str, required=True, help="Suite slug for output directory")
    p.add_argument(
        "--candidate-keys",
        nargs="+",
        default=None,
        help="Optional: run only the specified candidate keys (e.g., v9_baseline_all)",
    )
    
    p.add_argument("--phase1-metadata", type=Path, required=True)
    p.add_argument("--phase1-ground-truth", type=Path, required=True)
    p.add_argument("--phase1-dates", nargs="+", required=True)
    
    p.add_argument("--phase2-metadata", type=Path, required=True)
    p.add_argument("--phase2-ground-truth", type=Path, required=True)
    p.add_argument("--phase2-dates", nargs="+", required=True)
    
    p.add_argument("--telemetry-root", type=Path, required=True)
    p.add_argument("--output-report", type=Path, default=None)
    return p.parse_args(argv)


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _top_components(submission_path: Path, k: int = 5) -> List[tuple[str, int]]:
    rows = _read_jsonl(submission_path)
    counter = Counter(str(row.get("component", "")).strip() for row in rows)
    return counter.most_common(k)


def _evaluate_with_judge(ground_truth_path: Path, submission_path: Path) -> Dict[str, float]:
    gt_rows = _read_jsonl(ground_truth_path)
    sub_rows = _read_jsonl(submission_path)
    
    gt_index: Dict[str, Dict[str, Any]] = {}
    for row in gt_rows:
        uuid = str(row.get("uuid", "")).strip()
        if uuid:
            gt_index[uuid] = row
    
    sub_index: Dict[str, Dict[str, Any]] = {}
    for row in sub_rows:
        uuid = str(row.get("uuid", "")).strip()
        if uuid:
            sub_index[uuid] = row
    
    gt_uuids = set(gt_index.keys())
    sub_uuids = set(sub_index.keys())
    for uuid in list(gt_uuids - sub_uuids):
        sub_index[uuid] = {"uuid": uuid, "component": "", "reason": "", "reasoning_trace": []}
    
    metrics, _ = evaluate.score_submission(gt_index, sub_index, reason_threshold=0.65)
    return metrics


def _run_generator(telemetry_root: Path, metadata_csv: Path, output_jsonl: Path, env: Dict[str, str]) -> None:
    _ensure_parent(output_jsonl)
    cmd = [
        sys.executable,
        "-m",
        "contest_solution.main",
        "--telemetry-root",
        str(telemetry_root),
        "--metadata",
        str(metadata_csv),
        "--output",
        str(output_jsonl),
    ]
    
    # Inherit the current process environment (API keys, proxy, etc.) and then
    # apply candidate-specific overrides.
    merged_env = dict(os.environ)
    merged_env.update(env)
    merged_env.setdefault("PYTHONUNBUFFERED", "1")
    merged_env.setdefault("TOKENIZERS_PARALLELISM", "false")
    completed = subprocess.run(cmd, env=merged_env, cwd=str(Path(__file__).resolve().parents[1]))
    if completed.returncode != 0:
        raise RuntimeError(f"Generator failed (exit={completed.returncode}): {' '.join(cmd)}")


def build_candidates(preset: str) -> List[Candidate]:
    """Build candidates for the given preset."""
    
    if preset == "la_boost_v9_engineering_rules_plus":
        # v9: v8 baseline + add log/trace baseline contrast capabilities
        base_env = {
            "RCA_COMPONENT_SOURCE": "consensus",
            "RCA_WINDOW_PADDING_MIN": "20",
            "RCA_ENABLE_MODALITY_BONUS": "0",
            "RCA_ENABLE_HINT_BONUS": "0",
            "RCA_COMPONENT_PRIOR_SCALE": "0.0",
            "RCA_USE_LEGACY_MEMORY": "0",
            "RCA_STRIP_REPLICA_SUFFIX": "0",
            # Prefer replica pods (xxxservice-<n>) when logs/traces provide replica-specific evidence.
            "RCA_ENABLE_REPLICA_EVIDENCE_PREFERENCE": "1",
            # Baseline windows
            "RCA_ENABLE_BASELINE_WINDOWS": "1",
            "RCA_BASELINE_GAP_MIN": "10",
            "RCA_BASELINE_LEN_MIN": "10",
            # Fault-level gate
            "RCA_ENABLE_FAULT_LEVEL_GATE": "1",
            "RCA_FAULT_LEVEL_GATE_PENALTY": "0.25",
            "RCA_FAULT_LEVEL_GATE_NODE_MIN_METRIC": "6.0",
            "RCA_FAULT_LEVEL_GATE_NODE_MIN_RATIO": "0.75",
            "RCA_FAULT_LEVEL_GATE_TIDB_MIN_METRIC": "3.0",
            "RCA_FAULT_LEVEL_GATE_TIDB_MIN_RATIO": "0.70",
            # Node/TiDB dominance
            "RCA_ENABLE_NODE_METRIC_DOMINANCE_OVERRIDE": "1",
            "RCA_NODE_DOMINANCE_MIN_METRIC": "5.2",
            "RCA_NODE_DOMINANCE_MIN_RATIO": "0.6",
            "RCA_NODE_DOMINANCE_MARGIN": "0.03",
            "RCA_ENABLE_TIDB_METRIC_DOMINANCE_OVERRIDE": "1",
            "RCA_TIDB_DOMINANCE_MIN_METRIC": "2.6",
            "RCA_TIDB_DOMINANCE_MIN_RATIO": "0.55",
            "RCA_TIDB_DOMINANCE_MARGIN": "0.02",
            "RCA_TIDB_METRIC_SCORE_MULT": "1.8",
            # Competition rule: must use LLM (do not allow algorithm-only runs).
            "RCA_DISABLE_LLM": "0",
            "RCA_REQUIRE_LLM": "1",
            # DeepSeek defaults (can be overridden by user environment or .env).
            "RCA_LLM_PROVIDER": os.getenv("RCA_LLM_PROVIDER", "deepseek"),
            "RCA_LLM_MODEL": os.getenv("RCA_LLM_MODEL", "deepseek-chat"),
            "RCA_LLM_ATTEMPTS": os.getenv("RCA_LLM_ATTEMPTS", "5"),
        }
        
        return [
            Candidate(
                key="v8_baseline_only",
                name="方案A：v8 对照（仅 metrics baseline 对照）",
                env={
                    **base_env,
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "0",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "0",
                },
            ),
            Candidate(
                key="v9_baseline_log",
                name="方案B：v9 补齐 logs baseline（模板聚合对照）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "0",
                },
            ),
            Candidate(
                key="v9_baseline_trace",
                name="方案C：v9 补齐 traces baseline（duration 对照）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "0",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                },
            ),
            Candidate(
                key="v9_baseline_all",
                name="方案D：v9 完整工程规则（metrics + logs + traces baseline 全开）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                },
            ),

            Candidate(
                key="v9_node_trace_multimodal",
                name="方案E：v9 + 节点优先更严格（node dominance 要求 traces + 多模态加成）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_MODALITY_BONUS": "1",
                    "RCA_NODE_DOMINANCE_REQUIRE_TRACES": "1",
                },
            ),

            Candidate(
                key="v9_llm_component_topk",
                name="方案F：v9 + 让 LLM 选 component（TopK 扩大）",
                env={
                    **base_env,
                    "RCA_COMPONENT_SOURCE": "llm",
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                    "RCA_LLM_COMPONENT_CANDIDATE_TOPK": "25",
                },
            ),

            Candidate(
                key="v9_no_priors",
                name="方案G：v9 + 关闭热门服务先验（prior scale=0）",
                env={
                    **base_env,
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_COMPONENT_PRIOR_SCALE": "0.0",
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                },
            ),
            Candidate(
                key="v9_llm_component",
                name="方案E：v9 + LLM 选择 component（候选仍受 consensus top 限制）",
                env={
                    **base_env,
                    # Let LLM pick component among the provided ranked candidates.
                    "RCA_COMPONENT_SOURCE": "llm",
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                },
            ),

            Candidate(
                key="v10_replica_plus_anticollapse",
                name="方案F：v10（replica 优先 + anti-collapse）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                    # Reduce prior bias and suppress popular-service collapse when evidence is weak.
                    "RCA_COMPONENT_PRIOR_SCALE": "0.6",
                    "RCA_ENABLE_POPULAR_COMPONENT_PENALTY": "1",
                    "RCA_POPULAR_PENALTY": "0.25",
                    "RCA_POPULAR_PENALTY_MARGIN": "0.08",
                    "RCA_POPULAR_PENALTY_REQUIRE_MULTIMODAL": "1",
                    # Improve TiDB recall a bit (keep conservative).
                    "RCA_TIDB_MIN_METRIC_SCORE": "0.8",
                },
            ),

            Candidate(
                key="v10_replica_hint_prefer",
                name="方案G：v10（replica hint 优先：query 提到 pod 即强制对齐）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                    # Enable planner hints and make explicit replica hints win over base services.
                    "RCA_ENABLE_HINT_SEED": "1",
                    "RCA_HINT_SEED_SCORE": "0.6",
                    "RCA_ENABLE_HINT_BONUS": "1",
                    "RCA_HINT_BONUS": "0.8",
                    "RCA_PREFER_REPLICA_HINTS": "1",
                    "RCA_REPLICA_HINT_MARGIN": "0.01",
                },
            ),

            Candidate(
                key="v10_logs_full_templates",
                name="方案H：v10（logs 不限 error：用模板频次对照暴露 replica）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                    "RCA_LOG_FILTER_ERRORS_ONLY": "0",
                },
            ),

            Candidate(
                key="v10_replica_candidate_prefer",
                name="方案I：v10（replica 候选优先：pod 证据压过 base service）",
                env={
                    **base_env,
                    "RCA_ENABLE_BASELINE_LOGS": "1",
                    "RCA_ENABLE_BASELINE_TRACES": "1",
                    "RCA_ENABLE_LOG_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_TRACE_BASELINE_CONTRAST": "1",
                    "RCA_ENABLE_REPLICA_CANDIDATE_PREFERENCE": "1",
                    "RCA_REPLICA_CANDIDATE_MIN_RATIO": "0.75",
                    "RCA_REPLICA_CANDIDATE_MARGIN": "0.01",
                    "RCA_REPLICA_REQUIRE_NON_METRIC": "1",
                },
            ),
        ]
    
    raise ValueError(f"Unknown preset: {preset}")


def render_report(
    *,
    suite_name: str,
    phases: List[PhaseConfig],
    candidates: List[Candidate],
    results: List[CandidateResult],
    telemetry_root: Path,
    suite_slug: str,
) -> str:
    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"
    
    def _ok_rows(per_date: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for m in (per_date or {}).values():
            if not isinstance(m, dict):
                continue
            if m.get("_error"):
                continue
            rows.append(m)
        return rows

    def avg_metric(key: str, per_date: Dict[str, Dict[str, Any]]) -> float:
        rows = _ok_rows(per_date)
        if not rows:
            return 0.0
        return sum(float(m.get(key, 0.0) or 0.0) for m in rows) / float(len(rows))
    
    lines: List[str] = []
    lines.append(f"# {suite_name}")
    lines.append("")
    lines.append("目标：在 phase1+phase2 合并数据口径下，比较多种方案对 Final（以及 LA/TA/Explainability/Efficiency）的影响。")
    lines.append("")
    lines.append("## 试验设置")
    for phase in phases:
        lines.append(f"- **{phase.phase_name}**:")
        lines.append(f"  - metadata: `{phase.metadata}`")
        lines.append(f"  - ground truth: `{phase.ground_truth}`")
        lines.append(f"  - dates（按天全量）: {', '.join(phase.dates)}")
    lines.append(f"- telemetry root: `{telemetry_root}`")
    lines.append(f"- suite slug: `{suite_slug}`（对应 outputs/experiments/{suite_slug}/）")
    lines.append("")
    
    lines.append("## 候选方案")
    for c in candidates:
        lines.append(f"- {c.name}（key={c.key}）")
        lines.append(f"  - env: `{json.dumps(c.env, ensure_ascii=False)}`")
    lines.append("")
    
    # Overall summary across all phases
    lines.append("## 结果汇总（全部 phase 合并平均）")
    lines.append("")
    lines.append("| 方案 | 平均LA | 平均TA | 平均Explain | 平均Eff | 平均Final |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        all_dates: Dict[str, Dict[str, float]] = {}
        for phase_metrics in r.per_phase_per_date.values():
            all_dates.update(phase_metrics)
        la = avg_metric("component_accuracy", all_dates)
        ta = avg_metric("reason_accuracy", all_dates)
        ex = avg_metric("explainability", all_dates)
        ef = avg_metric("efficiency", all_dates)
        fi = avg_metric("final_score", all_dates)
        lines.append(f"| {r.candidate.name} | {pct(la)} | {pct(ta)} | {pct(ex)} | {pct(ef)} | {fi:.2f} |")
    
    # Per-phase summary
    for phase in phases:
        lines.append("")
        lines.append(f"## {phase.phase_name} 结果汇总")
        lines.append("")
        lines.append("| 方案 | 平均LA | 平均TA | 平均Explain | 平均Eff | 平均Final |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for r in results:
            per_date = r.per_phase_per_date.get(phase.phase_name, {})
            la = avg_metric("component_accuracy", per_date)
            ta = avg_metric("reason_accuracy", per_date)
            ex = avg_metric("explainability", per_date)
            ef = avg_metric("efficiency", per_date)
            fi = avg_metric("final_score", per_date)
            lines.append(f"| {r.candidate.name} | {pct(la)} | {pct(ta)} | {pct(ex)} | {pct(ef)} | {fi:.2f} |")
    
    # Detailed per-date breakdown
    lines.append("")
    lines.append("## 分 phase 分日期明细")
    lines.append("")
    lines.append("| phase | date | N | 方案key | LA | TA | Explain | Eff | Final |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    
    for phase in phases:
        for date in phase.dates:
            for r in results:
                per_date = r.per_phase_per_date.get(phase.phase_name, {})
                m = per_date.get(date)
                if not m:
                    continue
                if m.get("_error"):
                    continue
                lines.append(
                    "| {phase} | {date} | {n} | {key} | {la} | {ta} | {ex} | {ef} | {fi} |".format(
                        phase=phase.phase_name,
                        date=date,
                        n=m.get("_n", 0),
                        key=r.candidate.key,
                        la=pct(m["component_accuracy"]),
                        ta=pct(m["reason_accuracy"]),
                        ex=pct(m["explainability"]),
                        ef=pct(m["efficiency"]),
                        fi=f"{m['final_score']:.2f}",
                    )
                )
    
    # Component distribution check
    lines.append("")
    lines.append("## 预测 component 分布（Top-5，按 phase）")
    lines.append("")
    lines.append("说明：该表用于快速观察是否出现 'adservice 坍缩'（例如 Top-1 占比异常高）。")
    lines.append("")
    lines.append("| phase | date | 方案key | Top-5 components（component:count） |")
    lines.append("|---|---|---|---|")
    for phase in phases:
        for date in phase.dates:
            for r in results:
                per_date = r.per_phase_per_date.get(phase.phase_name, {})
                m = per_date.get(date)
                if not m:
                    continue
                if m.get("_error"):
                    continue
                top_items = m.get("_top_components") or []
                rendered = "; ".join([f"{c}:{n}" for c, n in top_items])
                lines.append(f"| {phase.phase_name} | {date} | {r.candidate.key} | {rendered} |")

    # Failures
    failures: List[tuple[str, str, str, str]] = []
    for phase in phases:
        for date in phase.dates:
            for r in results:
                per_date = r.per_phase_per_date.get(phase.phase_name, {})
                m = per_date.get(date)
                if not m or not isinstance(m, dict):
                    continue
                err = m.get("_error")
                if err:
                    failures.append((phase.phase_name, date, r.candidate.key, str(err)))
    if failures:
        lines.append("")
        lines.append("## 失败明细")
        lines.append("")
        lines.append("| phase | date | 方案key | error |")
        lines.append("|---|---|---|---|")
        for ph, dt, key, err in failures:
            safe_err = err.replace("\n", " ")[:240]
            lines.append(f"| {ph} | {dt} | {key} | {safe_err} |")
    
    lines.append("")
    lines.append("## 产物路径")
    lines.append(f"- submissions: `outputs/experiments/{suite_slug}/`")
    lines.append("- filtered gt: `tmp/filtered/` (gt_phaseName_YYYY-MM-DD_*.jsonl)")
    lines.append("")
    return "\n".join(lines)


def phase_env_overrides(phase_name: str) -> Dict[str, str]:
    """Phase-specific env overrides.

    Notes:
    - phase1 ground truth does not include TiDB components; disabling TiDB candidates avoids
      occasional cross-domain false positives (e.g., predicting tidb-* for node-fault cases).
    """
    name = (phase_name or "").strip().lower()
    if name == "phase1":
        return {
            "RCA_ALLOW_TIDB_COMPONENTS": "0",
            "RCA_ENABLE_TIDB_METRIC_DOMINANCE_OVERRIDE": "0",
        }
    return {}


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    
    phases = [
        PhaseConfig(
            phase_name="phase1",
            metadata=args.phase1_metadata,
            ground_truth=args.phase1_ground_truth,
            dates=[str(d).strip() for d in args.phase1_dates],
        ),
        PhaseConfig(
            phase_name="phase2",
            metadata=args.phase2_metadata,
            ground_truth=args.phase2_ground_truth,
            dates=[str(d).strip() for d in args.phase2_dates],
        ),
    ]
    
    candidates = build_candidates(args.preset)
    if args.candidate_keys:
        want = {str(k).strip() for k in args.candidate_keys if str(k).strip()}
        candidates = [c for c in candidates if c.key in want]
        if not candidates:
            raise ValueError(f"No candidates matched keys: {sorted(want)}")
    
    output_dir = repo_root / f"outputs/experiments/{args.suite_slug}"
    tmp_dir = repo_root / "tmp/filtered"
    tmp_meta_dir = repo_root / "tmp/experiments"
    tmp_meta_dir.mkdir(parents=True, exist_ok=True)
    
    base_env = os.environ.copy()
    
    results: List[CandidateResult] = []
    
    for candidate in candidates:
        per_phase_per_date: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        for phase in phases:
            df = pd.read_csv(phase.metadata)
            if not {"uuid", "date", "query"}.issubset(df.columns):
                raise ValueError(f"{phase.phase_name} metadata csv must contain columns: uuid,date,query")
            
            gt_all = _read_jsonl(phase.ground_truth)
            per_date: Dict[str, Dict[str, Any]] = {}
            
            for date in phase.dates:
                subset = df[df["date"].astype(str).str.strip() == str(date).strip()].copy()
                if subset.empty:
                    continue
                
                meta_path = tmp_meta_dir / f"metadata_{phase.phase_name}_{date}_n{len(subset)}.csv"
                subset.to_csv(meta_path, index=False)
                
                uuids = set(str(u).strip() for u in subset["uuid"].tolist())
                gt_rows = [row for row in gt_all if str(row.get("uuid", "")).strip() in uuids]
                filtered_gt = tmp_dir / f"gt_{phase.phase_name}_{date}_n{len(gt_rows)}.jsonl"
                _write_jsonl(filtered_gt, gt_rows)
                
                submission_path = output_dir / f"sub_{phase.phase_name}_{date}_n{len(subset)}_{candidate.key}.jsonl"
                
                env = base_env.copy()
                env.update({k: str(v) for k, v in candidate.env.items()})
                env.update(phase_env_overrides(phase.phase_name))
                
                try:
                    _run_generator(args.telemetry_root, meta_path, submission_path, env)
                    raw_metrics = _evaluate_with_judge(filtered_gt, submission_path)
                    metrics: Dict[str, Any] = dict(raw_metrics)
                    metrics["_n"] = float(len(subset))
                    metrics["_top_components"] = _top_components(submission_path, k=5)
                    per_date[str(date).strip()] = metrics
                except Exception as exc:  # noqa: BLE001
                    per_date[str(date).strip()] = {
                        "_n": float(len(subset)),
                        "_error": str(exc),
                    }
                    continue
            
            per_phase_per_date[phase.phase_name] = per_date
        
        results.append(CandidateResult(candidate=candidate, per_phase_per_date=per_phase_per_date))
    
    # Render report
    if args.output_report is None:
        report_path = repo_root / f"docs/对比试验/{args.suite_name}/报告.md"
    else:
        report_path = (repo_root / args.output_report) if not args.output_report.is_absolute() else args.output_report
    
    _ensure_parent(report_path)
    report_path.write_text(
        render_report(
            suite_name=args.suite_name,
            phases=phases,
            candidates=candidates,
            results=results,
            telemetry_root=args.telemetry_root,
            suite_slug=args.suite_slug,
        ),
        encoding="utf-8",
    )
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
