"""Run a systematic Final-optimization experiment suite and write docs in Chinese.

This script runs multiple candidate configurations on selected full-day cases,
then evaluates with the official judge and writes a markdown report.

Outputs:
- submissions: outputs/experiments/<suite_slug>/
- filtered gt: tmp/filtered/
- report: docs/对比试验/<试验名称>/报告.md

Example:
  TOKENIZERS_PARALLELISM=false RCA_LLM_ATTEMPTS=1 RCA_WINDOW_PADDING_MIN=20 \
  python tools/run_final_experiment_suite.py \
    --telemetry-root ../data \
    --metadata ../metadata_phase1.csv \
    --ground-truth ../AIOpsChallengeJudge/ground_truth_phase1.jsonl \
    --dates 2025-06-07 2025-06-08 \
    --suite-name "Final优化网格对比" \
    --suite-slug final_grid_v1

Notes:
- Optimizing Final requires real LLM (deepseek/openai) to keep TA/Explainability high.
- Keep candidate count small to control runtime.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class Candidate:
    key: str
    name: str
    env: Dict[str, str]


@dataclass(frozen=True)
class CandidateResult:
    candidate: Candidate
    per_date: Dict[str, Dict[str, float]]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Final optimization experiment suite")
    p.add_argument("--telemetry-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--dates", nargs="+", required=True)
    p.add_argument("--suite-name", type=str, required=True, help="中文试验名称（用于 docs 文件夹名）")
    p.add_argument("--suite-slug", type=str, required=True, help="ASCII slug（用于 outputs/experiments 目录名）")
    p.add_argument(
        "--preset",
        type=str,
        default="final_grid_v1",
        choices=["final_grid_v1", "bias_fix_v1", "bias_fix_v1_hint_grid"],
        help="候选方案预设：final_grid_v1 / bias_fix_v1 / bias_fix_v1_hint_grid",
    )
    p.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="默认写入 docs/对比试验/<suite-name>/报告.md",
    )
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


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _evaluate_with_judge(ground_truth: Path, submission: Path) -> Dict[str, float]:
    workspace_root = Path(__file__).resolve().parents[2]
    judge_dir = workspace_root / "AIOpsChallengeJudge"
    sys.path.insert(0, str(judge_dir))

    import evaluate  # type: ignore  # noqa: E402

    gt_records = evaluate.load_jsonl(ground_truth)
    sub_records = evaluate.load_jsonl(submission, allow_empty=True)
    gt_index = evaluate.build_index(gt_records)
    sub_index = evaluate.build_index(sub_records)

    gt_uuids = set(gt_index.keys())
    sub_uuids = set(sub_index.keys())

    for uuid in list(sub_uuids - gt_uuids):
        sub_index.pop(uuid, None)

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

    env = dict(env)
    env.setdefault("PYTHONUNBUFFERED", "1")
    completed = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parents[1]))
    if completed.returncode != 0:
        raise RuntimeError(f"Generator failed (exit={completed.returncode}): {' '.join(cmd)}")


def build_candidates(preset: str) -> List[Candidate]:
    # Keep the suite small to control runtime.
    # We ONLY change inference-time hyperparams; no training.
    if preset == "final_grid_v1":
        return [
            Candidate(
                key="base_p20",
                name="方案A：基线（padding=20）",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                },
            ),
            Candidate(
                key="base_p25",
                name="方案B：加大时间窗（padding=25）",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "25",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                },
            ),
            Candidate(
                key="mm_p20",
                name="方案C：多模态一致性加成（padding=20）",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "1",
                    "RCA_MODALITY_BONUS_ALPHA": "0.15",
                    "RCA_MODALITY_BONUS_CAP": "1.4",
                },
            ),
        ]

    if preset == "bias_fix_v1":
        # 针对“预测塌缩到热门组件 / 副本名被剥离 / 先验偏置”做验证。
        # 注意：这些开关默认不会影响基线，只有在本 preset 里显式开启。
        return [
            Candidate(
                key="base_p20",
                name="方案A：基线（padding=20）",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                    "RCA_ENABLE_HINT_BONUS": "0",
                    "RCA_COMPONENT_PRIOR_SCALE": "1.0",
                    "RCA_STRIP_REPLICA_SUFFIX": "1",
                },
            ),
            Candidate(
                key="hint_015",
                name="方案B：开启 hints 温和加成（bonus=0.15）",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                    "RCA_ENABLE_HINT_BONUS": "1",
                    "RCA_HINT_BONUS": "0.15",
                    "RCA_COMPONENT_PRIOR_SCALE": "1.0",
                    "RCA_STRIP_REPLICA_SUFFIX": "1",
                },
            ),
            Candidate(
                key="no_prior",
                name="方案C：关闭 component 先验（scale=0）",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                    "RCA_ENABLE_HINT_BONUS": "0",
                    "RCA_COMPONENT_PRIOR_SCALE": "0.0",
                    "RCA_STRIP_REPLICA_SUFFIX": "1",
                },
            ),
            Candidate(
                key="keep_replica",
                name="方案D：保留副本后缀（不剥离 -0/-1）",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                    "RCA_ENABLE_HINT_BONUS": "0",
                    "RCA_COMPONENT_PRIOR_SCALE": "1.0",
                    "RCA_STRIP_REPLICA_SUFFIX": "0",
                },
            ),
            Candidate(
                key="keep_replica_hint",
                name="方案E：保留副本 + hints 加成",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                    "RCA_ENABLE_HINT_BONUS": "1",
                    "RCA_HINT_BONUS": "0.15",
                    "RCA_COMPONENT_PRIOR_SCALE": "1.0",
                    "RCA_STRIP_REPLICA_SUFFIX": "0",
                },
            ),
            Candidate(
                key="keep_replica_no_prior",
                name="方案F：保留副本 + 关闭先验",
                env={
                    "RCA_COMPONENT_SOURCE": "consensus",
                    "RCA_WINDOW_PADDING_MIN": "20",
                    "RCA_ENABLE_MODALITY_BONUS": "0",
                    "RCA_ENABLE_HINT_BONUS": "0",
                    "RCA_COMPONENT_PRIOR_SCALE": "0.0",
                    "RCA_STRIP_REPLICA_SUFFIX": "0",
                },
            ),
        ]

    if preset == "bias_fix_v1_hint_grid":
        # 仅对 RCA_HINT_BONUS 做网格（其余保持基线一致），用于验证 hints 能否稳定缓解偏置/塌缩。
        base_env = {
            "RCA_COMPONENT_SOURCE": "consensus",
            "RCA_WINDOW_PADDING_MIN": "20",
            "RCA_ENABLE_MODALITY_BONUS": "0",
            "RCA_COMPONENT_PRIOR_SCALE": "1.0",
            "RCA_STRIP_REPLICA_SUFFIX": "1",
        }

        bonuses = [0.05, 0.10, 0.15, 0.20]
        candidates: List[Candidate] = [
            Candidate(
                key="hint_000",
                name="方案A：hints 关闭（对照组）",
                env={
                    **base_env,
                    "RCA_ENABLE_HINT_BONUS": "0",
                },
            )
        ]

        for b in bonuses:
            key = f"hint_{int(round(b * 100)):03d}"
            candidates.append(
                Candidate(
                    key=key,
                    name=f"方案：hints 加成（bonus={b:.2f}）",
                    env={
                        **base_env,
                        "RCA_ENABLE_HINT_BONUS": "1",
                        "RCA_HINT_BONUS": f"{b:.2f}",
                    },
                )
            )

        return candidates

    raise ValueError(f"Unknown preset: {preset}")


def render_report(
    *,
    suite_name: str,
    dates: List[str],
    candidates: List[Candidate],
    results: List[CandidateResult],
    metadata_path: Path,
    ground_truth_path: Path,
    telemetry_root: Path,
    suite_slug: str,
) -> str:
    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    def avg_metric(key: str, per_date: Dict[str, Dict[str, float]]) -> float:
        if not per_date:
            return 0.0
        return sum(m.get(key, 0.0) for m in per_date.values()) / float(len(per_date))

    lines: List[str] = []
    lines.append(f"# {suite_name}")
    lines.append("")
    lines.append("目标：在固定数据口径下，比较多种方案对 Final（以及 LA/TA/Explainability/Efficiency）的影响。")
    lines.append("")
    lines.append("## 试验设置")
    lines.append(f"- metadata: `{metadata_path}`")
    lines.append(f"- ground truth: `{ground_truth_path}`")
    lines.append(f"- telemetry root: `{telemetry_root}`")
    lines.append(f"- dates（按天全量）: {', '.join(dates)}")
    lines.append(f"- suite slug: `{suite_slug}`（对应 outputs/experiments/{suite_slug}/）")
    lines.append("")

    lines.append("## 候选方案")
    for c in candidates:
        lines.append(f"- {c.name}（key={c.key}）")
        lines.append(f"  - env: `{json.dumps(c.env, ensure_ascii=False)}`")
    lines.append("")

    lines.append("## 结果汇总（按日期评测；并给出跨日期平均）")
    lines.append("")
    lines.append("| 方案 | 平均LA | 平均TA | 平均Explain | 平均Eff | 平均Final |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in results:
        la = avg_metric("component_accuracy", r.per_date)
        ta = avg_metric("reason_accuracy", r.per_date)
        ex = avg_metric("explainability", r.per_date)
        ef = avg_metric("efficiency", r.per_date)
        fi = avg_metric("final_score", r.per_date)
        lines.append(f"| {r.candidate.name} | {pct(la)} | {pct(ta)} | {pct(ex)} | {pct(ef)} | {fi:.2f} |")

    lines.append("")
    lines.append("## 分日期明细")
    lines.append("")
    lines.append("| date | N | 方案key | LA | TA | Explain | Eff | Final |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")

    # N is full count for that date in metadata
    for date in dates:
        for r in results:
            m = r.per_date.get(date)
            if not m:
                continue
            lines.append(
                "| {date} | {n} | {key} | {la} | {ta} | {ex} | {ef} | {fi} |".format(
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

    lines.append("")
    lines.append("## 产物路径")
    lines.append(f"- submissions: `outputs/experiments/{suite_slug}/`")
    lines.append("- filtered gt: `tmp/filtered/` (gt_YYYY-MM-DD_*.jsonl)")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    df = pd.read_csv(args.metadata)
    if not {"uuid", "date", "query"}.issubset(df.columns):
        raise ValueError("metadata csv must contain columns: uuid,date,query")

    candidates = build_candidates(args.preset)

    output_dir = repo_root / f"outputs/experiments/{args.suite_slug}"
    tmp_dir = repo_root / "tmp/filtered"
    tmp_meta_dir = repo_root / "tmp/experiments"
    tmp_meta_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()

    results: List[CandidateResult] = []

    # Pre-read GT once for filtering.
    gt_all = _read_jsonl(args.ground_truth)

    for candidate in candidates:
        per_date: Dict[str, Dict[str, float]] = {}
        for date in args.dates:
            subset = df[df["date"].astype(str).str.strip() == str(date).strip()].copy()
            if subset.empty:
                continue

            meta_path = tmp_meta_dir / f"metadata_{date}_n{len(subset)}.csv"
            subset.to_csv(meta_path, index=False)

            uuids = set(str(u).strip() for u in subset["uuid"].tolist())
            gt_rows = [row for row in gt_all if str(row.get("uuid", "")).strip() in uuids]
            filtered_gt = tmp_dir / f"gt_{date}_n{len(gt_rows)}.jsonl"
            _write_jsonl(filtered_gt, gt_rows)

            submission_path = output_dir / f"sub_{date}_n{len(subset)}_{candidate.key}.jsonl"

            env = base_env.copy()
            env.update({k: str(v) for k, v in candidate.env.items()})

            _run_generator(args.telemetry_root, meta_path, submission_path, env)
            metrics = _evaluate_with_judge(filtered_gt, submission_path)
            metrics["_n"] = float(len(subset))
            per_date[str(date).strip()] = metrics

        results.append(CandidateResult(candidate=candidate, per_date=per_date))

    # Default report path (Chinese folder name)
    if args.output_report is None:
        report_path = repo_root / f"docs/对比试验/{args.suite_name}/报告.md"
    else:
        report_path = (repo_root / args.output_report) if not args.output_report.is_absolute() else args.output_report

    _ensure_parent(report_path)
    report_path.write_text(
        render_report(
            suite_name=args.suite_name,
            dates=[str(d).strip() for d in args.dates],
            candidates=candidates,
            results=results,
            metadata_path=args.metadata,
            ground_truth_path=args.ground_truth,
            telemetry_root=args.telemetry_root,
            suite_slug=args.suite_slug,
        ),
        encoding="utf-8",
    )
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
