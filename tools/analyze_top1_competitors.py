#!/usr/bin/env python3
"""Analyze "top1 competitors" when GT is present in consensus topK but not selected.

Designed for Experiment 37 outputs (JSONL with optional debug.consensus_topk).

Key questions answered:
- When GT is in topK but output != GT, who is the top1 competitor?
- Is the miss mostly a kind mismatch (GT node/tidb/service vs top1 kind)?
- Which modality combos tend to win at top1?

Outputs Markdown to stdout or a file.

Usage:
  python tools/analyze_top1_competitors.py \
    --exp-dir outputs/experiments/37_evidence_probe_hybrid_p1p2_6d \
    --candidate e0_fusion_baseline --topk 80 --topn 15
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_gt_index(gt_path: Path) -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for r in _read_jsonl(gt_path):
        u = str(r.get("uuid", "")).strip()
        if u:
            idx[u] = r
    return idx


def _gt_parts(gt_comp: str) -> list[str]:
    gt_comp = (gt_comp or "").strip()
    # judge semantics: A->B means either end is acceptable
    return [x.strip() for x in gt_comp.replace("->", "+").split("+") if x.strip()]


def _kind(component: str) -> str:
    c = (component or "").strip()
    if c.startswith("aiops-k8s-"):
        return "node"
    if c.startswith("tidb-"):
        return "tidb"
    return "service"


def _modal_sig(mods: Iterable[str]) -> str:
    # stable signature for counting
    s = sorted({m.strip() for m in mods if m and m.strip()})
    return "+".join(s) if s else "(none)"


@dataclass(frozen=True)
class CaseRecord:
    phase: str
    date: str
    uuid: str
    gt_parts: tuple[str, ...]
    gt_kind: str
    pred: str
    pred_kind: str
    gt_best_rank: int
    gt_best_modal_sig: str
    top1: str
    top1_kind: str
    top1_modal_sig: str


def analyze_candidate(
    exp_dir: Path,
    candidate: str,
    topk: int,
    gt_phase1: Path,
    gt_phase2: Path,
) -> tuple[list[CaseRecord], dict[str, Any]]:
    p1 = _load_gt_index(gt_phase1)
    p2 = _load_gt_index(gt_phase2)

    pat = re.compile(r"sub_(phase\d)_(\d{4}-\d{2}-\d{2})_n\d+_(.+)\.jsonl$")

    files: list[Path] = []
    for p in sorted(exp_dir.glob(f"sub_phase*_2025-*_n*_{candidate}.jsonl")):
        if pat.search(p.name):
            files.append(p)

    total_N = 0
    hit = 0
    gt_in_topk = 0
    miss_but_gt_in_topk = 0

    records: list[CaseRecord] = []

    for p in files:
        m = pat.search(p.name)
        assert m
        phase, date, _cand = m.group(1), m.group(2), m.group(3)
        gt_index = p1 if phase == "phase1" else p2

        rows = _read_jsonl(p)
        for r in rows:
            u = str(r.get("uuid", "")).strip()
            if not u or u not in gt_index:
                continue
            total_N += 1

            gt = _gt_parts(gt_index[u].get("component", ""))
            gt_t = tuple(gt)
            gt_k = _kind(gt[0]) if gt else "service"

            pred = (r.get("component", "") or "").strip()
            pred_k = _kind(pred) if pred else "service"
            is_hit = bool(pred and pred in gt)
            if is_hit:
                hit += 1

            dbg = r.get("debug") or {}
            top = (dbg.get("consensus_topk") or [])[:topk]
            if not top:
                continue

            # locate GT best rank
            best_rank = -1
            best_modal_sig = "(missing)"
            for i, item in enumerate(top, start=1):
                c = (item.get("component", "") or "").strip()
                if c in gt:
                    best_rank = i
                    best_modal_sig = _modal_sig(item.get("modalities") or [])
                    break

            if best_rank != -1:
                gt_in_topk += 1

            # focus on the most actionable bucket
            if (best_rank != -1) and (not is_hit):
                miss_but_gt_in_topk += 1
                top1_item = top[0]
                top1 = (top1_item.get("component", "") or "").strip()
                top1_k = _kind(top1)
                top1_modal_sig = _modal_sig(top1_item.get("modalities") or [])
                records.append(
                    CaseRecord(
                        phase=phase,
                        date=date,
                        uuid=u,
                        gt_parts=gt_t,
                        gt_kind=gt_k,
                        pred=pred,
                        pred_kind=pred_k,
                        gt_best_rank=best_rank,
                        gt_best_modal_sig=best_modal_sig,
                        top1=top1,
                        top1_kind=top1_k,
                        top1_modal_sig=top1_modal_sig,
                    )
                )

    summary = {
        "candidate": candidate,
        "N": total_N,
        "LA%": 0.0 if total_N == 0 else 100.0 * hit / total_N,
        "GT_in_topk%": 0.0 if total_N == 0 else 100.0 * gt_in_topk / total_N,
        "miss_but_gt_in_topk": miss_but_gt_in_topk,
    }
    return records, summary


def render_markdown(
    candidate: str,
    topk: int,
    records: list[CaseRecord],
    summary: dict[str, Any],
    topn: int,
) -> str:
    lines: list[str] = []
    lines.append(f"## Top1 竞争对手分析（candidate={candidate}；topK={topk}）")
    lines.append("")
    lines.append(
        f"- N={summary['N']}, LA={summary['LA%']:.2f}%, GT-in-topK={summary['GT_in_topk%']:.2f}%\n"
        f"- 重点样本：GT 在 topK 但未命中（miss_but_gt_in_topk）= {summary['miss_but_gt_in_topk']}"
    )
    lines.append("")

    if not records:
        lines.append("（无样本满足：GT in topK 且未命中）")
        lines.append("")
        return "\n".join(lines)

    top1_counter = Counter(r.top1 for r in records)
    kind_pair_counter = Counter((r.gt_kind, r.top1_kind) for r in records)
    top1_mod_counter = Counter(r.top1_modal_sig for r in records)
    gt_best_mod_counter = Counter(r.gt_best_modal_sig for r in records)

    # top1 competitors
    lines.append(f"### Top1 最常见竞争对手（Top {topn}）")
    lines.append("")
    lines.append("| 排名 | top1 组件 | 次数 | 占比 |")
    lines.append("|---:|---|---:|---:|")
    total = len(records)
    for i, (comp, cnt) in enumerate(top1_counter.most_common(topn), start=1):
        lines.append(f"| {i} | {comp} | {cnt} | {100.0*cnt/total:.2f}% |")
    lines.append("")

    # kind mismatch
    lines.append("### GT 类型 vs top1 类型（计数）")
    lines.append("")
    lines.append("| GT_kind | top1_kind | 次数 | 占比 |")
    lines.append("|---|---|---:|---:|")
    for (gk, tk), cnt in kind_pair_counter.most_common():
        lines.append(f"| {gk} | {tk} | {cnt} | {100.0*cnt/total:.2f}% |")
    lines.append("")

    # modality signatures
    lines.append(f"### top1 的模态组合（Top {min(topn, 10)}）")
    lines.append("")
    lines.append("| 排名 | top1_modalities | 次数 | 占比 |")
    lines.append("|---:|---|---:|---:|")
    for i, (sig, cnt) in enumerate(top1_mod_counter.most_common(min(topn, 10)), start=1):
        lines.append(f"| {i} | {sig} | {cnt} | {100.0*cnt/total:.2f}% |")
    lines.append("")

    lines.append(f"### GT（best-rank 命中项）的模态组合（Top {min(topn, 10)}）")
    lines.append("")
    lines.append("| 排名 | gt_best_modalities | 次数 | 占比 |")
    lines.append("|---:|---|---:|---:|")
    for i, (sig, cnt) in enumerate(gt_best_mod_counter.most_common(min(topn, 10)), start=1):
        lines.append(f"| {i} | {sig} | {cnt} | {100.0*cnt/total:.2f}% |")
    lines.append("")

    # examples
    lines.append("### 代表性样本（前 12 条，按 gt_best_rank 由小到大）")
    lines.append("")
    lines.append("| date | uuid | GT_parts | GT_kind | top1 | top1_kind | top1_mod | gt_best_rank | gt_best_mod |")
    lines.append("|---|---|---|---|---|---|---|---:|---|")
    for r in sorted(records, key=lambda x: (x.gt_best_rank, x.date, x.uuid))[:12]:
        gt_disp = "+".join(r.gt_parts)
        lines.append(
            f"| {r.date} | {r.uuid} | {gt_disp} | {r.gt_kind} | {r.top1} | {r.top1_kind} | {r.top1_modal_sig} | {r.gt_best_rank} | {r.gt_best_modal_sig} |"
        )
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--candidate", type=str, default="e0_fusion_baseline")
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument("--topn", type=int, default=15)
    ap.add_argument(
        "--gt-phase1",
        type=Path,
        default=Path("../AIOpsChallengeJudge/ground_truth_phase1.jsonl"),
    )
    ap.add_argument(
        "--gt-phase2",
        type=Path,
        default=Path("../AIOpsChallengeJudge/ground_truth_phase2.jsonl"),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    records, summary = analyze_candidate(
        exp_dir=args.exp_dir,
        candidate=args.candidate,
        topk=args.topk,
        gt_phase1=args.gt_phase1,
        gt_phase2=args.gt_phase2,
    )
    md = render_markdown(
        candidate=args.candidate,
        topk=args.topk,
        records=records,
        summary=summary,
        topn=args.topn,
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md, encoding="utf-8")
    else:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
