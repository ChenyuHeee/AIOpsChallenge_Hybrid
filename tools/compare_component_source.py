"""Compare component source strategies on a few dates.

Runs two strategies on the same sampled cases:
- consensus: component from consensus top-1 (LLM only generates reason/trace)
- llm: component from LLM output (fallback to consensus if missing)

Outputs:
- submissions under outputs/experiments/component_source_compare/
- filtered GT under tmp/filtered/
- a markdown report under docs/对比试验/组件来源对比/

Example:
  python tools/compare_component_source.py \
    --telemetry-root ../data \
    --metadata ../metadata_phase1.csv \
    --ground-truth ../AIOpsChallengeJudge/ground_truth_phase1.jsonl \
    --dates 2025-06-07 2025-06-10 2025-06-14 \
    --per-date 12
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
from typing import Any, Dict, Iterable, List, Sequence

import pandas as pd


@dataclass(frozen=True)
class RunResult:
    date: str
    n_cases: int
    metrics_consensus: Dict[str, float]
    metrics_llm: Dict[str, float]
    submission_consensus: Path
    submission_llm: Path
    filtered_gt: Path


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RCA_COMPONENT_SOURCE strategies")
    p.add_argument("--telemetry-root", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--dates", nargs="*", default=None, help="Explicit dates like 2025-06-07 2025-06-10 ...")
    p.add_argument("--per-date", type=int, default=12, help="Sample N cases per date; use 0 to run all cases for the date")
    p.add_argument("--max-dates", type=int, default=3, help="Pick at most K dates when --dates is omitted")
    p.add_argument("--seed", type=int, default=7, help="Sampling seed")
    p.add_argument(
        "--output-md",
        type=Path,
        default=Path("docs/对比试验/组件来源对比/报告.md"),
        help="Markdown report output path (relative to repo root by default)",
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
    # Import judge module from sibling folder.
    workspace_root = Path(__file__).resolve().parents[2]
    judge_dir = workspace_root / "AIOpsChallengeJudge"
    sys.path.insert(0, str(judge_dir))

    import evaluate  # type: ignore  # noqa: E402

    gt_records = evaluate.load_jsonl(ground_truth)
    sub_records = evaluate.load_jsonl(submission, allow_empty=True)

    gt_index = evaluate.build_index(gt_records)
    sub_index = evaluate.build_index(sub_records)

    # Align uuid sets same as CLI (ignore extra, assume blank for missing)
    gt_uuids = set(gt_index.keys())
    sub_uuids = set(sub_index.keys())

    for uuid in list(sub_uuids - gt_uuids):
        sub_index.pop(uuid, None)

    for uuid in list(gt_uuids - sub_uuids):
        sub_index[uuid] = {"uuid": uuid, "component": "", "reason": "", "reasoning_trace": []}

    metrics, _ = evaluate.score_submission(gt_index, sub_index, reason_threshold=0.65)
    return metrics


def _pick_dates(df: pd.DataFrame, telemetry_root: Path, max_dates: int) -> List[str]:
    # Pick most frequent dates that also exist in telemetry root.
    counts = Counter(str(d).strip() for d in df["date"].tolist())
    picked: List[str] = []
    for date, _ in counts.most_common():
        if (telemetry_root / date).exists():
            picked.append(date)
        if len(picked) >= max_dates:
            break
    return picked


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


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    df = pd.read_csv(args.metadata)
    if not {"uuid", "date", "query"}.issubset(df.columns):
        raise ValueError("metadata csv must contain columns: uuid,date,query")

    dates: List[str]
    if args.dates:
        dates = [str(d).strip() for d in args.dates if str(d).strip()]
    else:
        dates = _pick_dates(df, args.telemetry_root, max_dates=args.max_dates)

    if not dates:
        raise ValueError("No dates selected; provide --dates or ensure telemetry-root/date folders exist")

    output_dir = repo_root / "outputs/experiments/component_source_compare"
    tmp_dir = repo_root / "tmp/filtered"
    tmp_meta_dir = repo_root / "tmp/experiments"
    tmp_meta_dir.mkdir(parents=True, exist_ok=True)

    base_env = os.environ.copy()

    results: List[RunResult] = []
    for date in dates:
        subset = df[df["date"].astype(str).str.strip() == date].copy()
        if subset.empty:
            continue
        if args.per_date and args.per_date > 0:
            subset = subset.sample(n=min(args.per_date, len(subset)), random_state=args.seed)

        meta_path = tmp_meta_dir / f"metadata_{date}_n{len(subset)}.csv"
        subset.to_csv(meta_path, index=False)

        uuids = set(str(u).strip() for u in subset["uuid"].tolist())
        gt_rows = [row for row in _read_jsonl(args.ground_truth) if str(row.get("uuid", "")).strip() in uuids]
        filtered_gt = tmp_dir / f"gt_{date}_n{len(gt_rows)}.jsonl"
        _write_jsonl(filtered_gt, gt_rows)

        sub_consensus = output_dir / f"sub_{date}_n{len(subset)}_consensus.jsonl"
        sub_llm = output_dir / f"sub_{date}_n{len(subset)}_llm.jsonl"

        env_consensus = base_env.copy()
        env_consensus["RCA_COMPONENT_SOURCE"] = "consensus"
        _run_generator(args.telemetry_root, meta_path, sub_consensus, env_consensus)

        env_llm = base_env.copy()
        env_llm["RCA_COMPONENT_SOURCE"] = "llm"
        _run_generator(args.telemetry_root, meta_path, sub_llm, env_llm)

        m_consensus = _evaluate_with_judge(filtered_gt, sub_consensus)
        m_llm = _evaluate_with_judge(filtered_gt, sub_llm)

        results.append(
            RunResult(
                date=date,
                n_cases=len(subset),
                metrics_consensus=m_consensus,
                metrics_llm=m_llm,
                submission_consensus=sub_consensus,
                submission_llm=sub_llm,
                filtered_gt=filtered_gt,
            )
        )

    _ensure_parent(args.output_md)
    args.output_md.write_text(render_report(results, args), encoding="utf-8")
    print(f"Wrote report to {args.output_md}")
    return 0


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(results: List[RunResult], args: argparse.Namespace) -> str:
    lines: List[str] = []
    lines.append("# component 来源对比试验")
    lines.append("")
    lines.append("对比两种策略：")
    lines.append("- consensus：component=共识层 Top-1（LLM 仅生成 reason/trace）")
    lines.append("- llm：component=LLM 输出（为空则回退共识）")
    lines.append("")
    lines.append("## 试验设置")
    lines.append(f"- metadata: `{args.metadata}`")
    lines.append(f"- ground truth: `{args.ground_truth}`")
    lines.append(f"- telemetry root: `{args.telemetry_root}`")
    lines.append(f"- dates: {', '.join(r.date for r in results) if results else '(none)' }")
    lines.append(f"- per-date sampling: {args.per_date}")
    lines.append("")

    if not results:
        lines.append("未产出结果：可能是日期筛选为空或运行失败。")
        lines.append("")
        return "\n".join(lines)

    lines.append("## 结果汇总（每个日期独立评测）")
    lines.append("")
    lines.append("| date | N | strategy | LA(component) | TA(reason) | Explain | Eff | Final |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")

    for r in results:
        for name, m in (("consensus", r.metrics_consensus), ("llm", r.metrics_llm)):
            lines.append(
                "| {date} | {n} | {name} | {la} | {ta} | {ex} | {ef} | {fi} |".format(
                    date=r.date,
                    n=r.n_cases,
                    name=name,
                    la=_pct(m["component_accuracy"]),
                    ta=_pct(m["reason_accuracy"]),
                    ex=_pct(m["explainability"]),
                    ef=_pct(m["efficiency"]),
                    fi=f"{m['final_score']:.2f}",
                )
            )

    lines.append("")
    lines.append("## 产物路径")
    lines.append("- submissions: `outputs/experiments/component_source_compare/`")
    lines.append("- filtered gt: `tmp/filtered/` (gt_YYYY-MM-DD_*.jsonl)")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
