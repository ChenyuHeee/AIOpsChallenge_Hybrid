"""Analyze LA error points from filtered ground-truth and submissions.

This script is designed to work with outputs produced by tools/run_final_experiment_suite.py.
It produces:
- Overall LA
- Confusion summary (top wrong predictions)
- Breakdown by GT type: node/service/edge/replica/other
- Case list for misses with a short evidence preview

Usage example:
  python tools/analyze_la_errors.py \
    --ground-truth AIOpsChallengeJudge/ground_truth_phase1.jsonl \
    --metadata metadata_phase1.csv \
    --telemetry-root /abs/path/data \
    --suite-slug la_ext_p1_7d_node_metric_dominance \
    --date 2025-06-08 \
    --candidate node_dominance_on_keep_replica

You can also pass multiple dates; metrics are aggregated.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--suite-slug", type=str, required=True)
    p.add_argument("--date", nargs="+", required=True)
    p.add_argument("--candidate", type=str, required=True)
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _norm_component(value: str) -> str:
    return str(value or "").strip()


def _gt_parts(gt_component: str) -> List[str]:
    raw = _norm_component(gt_component)
    if not raw:
        return []
    parts = [p.strip() for p in raw.replace("->", "+").split("+") if p.strip()]
    return parts


def _classify_component(component: str) -> str:
    c = _norm_component(component)
    if not c:
        return "empty"
    if c.startswith("aiops-k8s-") or c.startswith("k8s-master"):
        return "node"
    if ":" in c:
        return "edge_or_instance"
    # Heuristic: replica suffix looks like "service-xxxxx" (hash-ish)
    if re.search(r"-[0-9a-f]{5,}$", c):
        return "replica"
    return "service_or_other"


@dataclass
class MissRow:
    uuid: str
    date: str
    gt: str
    pred: str
    gt_type: str
    pred_type: str


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root or Path(__file__).resolve().parents[1]

    gt_all = _read_jsonl(args.ground_truth)
    gt_index = {str(r.get("uuid", "")).strip(): r for r in gt_all}

    md = pd.read_csv(args.metadata)
    md["uuid"] = md["uuid"].astype(str).str.strip()
    md["date"] = md["date"].astype(str).str.strip()

    misses: List[MissRow] = []
    total = 0
    correct = 0

    confusions = Counter()
    gt_type_total = Counter()
    gt_type_correct = Counter()

    for date in args.date:
        subset = md[md["date"] == str(date).strip()].copy()
        if subset.empty:
            continue
        sub_path = repo_root / f"outputs/experiments/{args.suite_slug}/sub_{date}_n{len(subset)}_{args.candidate}.jsonl"
        if not sub_path.exists():
            raise SystemExit(f"submission not found: {sub_path}")
        sub_rows = _read_jsonl(sub_path)
        sub_index = {str(r.get("uuid", "")).strip(): r for r in sub_rows}

        for uuid in subset["uuid"].tolist():
            uuid = str(uuid).strip()
            gt_row = gt_index.get(uuid) or {}
            sub_row = sub_index.get(uuid) or {}
            gt_comp = _norm_component(gt_row.get("component", ""))
            pred = _norm_component(sub_row.get("component", ""))

            parts = _gt_parts(gt_comp)
            is_ok = bool(pred and parts and pred in parts)

            total += 1
            gt_type = _classify_component(parts[0] if parts else gt_comp)
            pred_type = _classify_component(pred)
            gt_type_total[gt_type] += 1
            if is_ok:
                correct += 1
                gt_type_correct[gt_type] += 1
            else:
                misses.append(
                    MissRow(
                        uuid=uuid,
                        date=str(date).strip(),
                        gt=gt_comp,
                        pred=pred,
                        gt_type=gt_type,
                        pred_type=pred_type,
                    )
                )
                confusions[(parts[0] if parts else gt_comp, pred)] += 1

    la = correct / max(1, total)

    top_conf = confusions.most_common(args.topk)

    # Render report
    lines: List[str] = []
    lines.append(f"# LA丢分点分析：{args.suite_slug} / {args.candidate}")
    lines.append("")
    lines.append(f"- dates: {', '.join([str(d).strip() for d in args.date])}")
    lines.append(f"- total cases: {total}")
    lines.append(f"- LA: {la*100:.2f}% ({correct}/{total})")
    lines.append("")

    lines.append("## GT类型分布与命中")
    lines.append("")
    lines.append("| GT类型 | 样本数 | 命中数 | 命中率 |")
    lines.append("|---|---:|---:|---:|")
    for t, n in gt_type_total.most_common():
        hit = int(gt_type_correct.get(t, 0))
        rate = hit / max(1, int(n))
        lines.append(f"| {t} | {n} | {hit} | {rate*100:.2f}% |")
    lines.append("")

    lines.append(f"## Top-{args.topk} 混淆（GT -> Pred）")
    lines.append("")
    lines.append("| # | GT | Pred | count |")
    lines.append("|---:|---|---|---:|")
    for i, ((gt, pred), cnt) in enumerate(top_conf, 1):
        lines.append(f"| {i} | {gt or '(empty)'} | {pred or '(empty)'} | {cnt} |")
    lines.append("")

    lines.append("## 失败样本（uuid / gt / pred / 类型）")
    lines.append("")
    lines.append("| date | uuid | gt | pred | gt_type | pred_type |")
    lines.append("|---|---|---|---|---|---|")
    for row in misses:
        lines.append(f"| {row.date} | {row.uuid} | {row.gt or '(empty)'} | {row.pred or '(empty)'} | {row.gt_type} | {row.pred_type} |")

    report = "\n".join(lines) + "\n"

    if args.output is None:
        out_path = repo_root / f"docs/对比试验/LA丢分点分析_{args.suite_slug}_{args.candidate}/报告.md"
    else:
        out_path = (repo_root / args.output) if not args.output.is_absolute() else args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
