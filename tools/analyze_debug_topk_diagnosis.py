#!/usr/bin/env python3
"""Diagnose why LA is zero/low using debug.consensus_topk.

This script reads submission JSONL produced by run_type_arbiter_groups_shared_suite.py
(with RCA_OUTPUT_DEBUG=1) and compares each prediction against judge semantics.

Key questions answered:
- Does the ground-truth component (or any side of A->B) appear in consensus_topk?
- If yes, what is its best rank and score?
- If not, is it likely a recall issue (specialists/consensus never surfaced it)?
- If present but not selected, was it overridden by type arbiter / component_source?

Output: a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _gt_parts(comp: str) -> List[str]:
    # Match judge semantics described in repo docs: split A->B into parts.
    # Also accept already-combined forms using '+'.
    return [x.strip() for x in comp.replace("->", "+").split("+") if x.strip()]


def _find_submission(exp_root: Path, phase: str, date: str, cand_key: str) -> Path:
    matches = sorted(exp_root.glob(f"sub_{phase}_{date}_n*_{cand_key}.jsonl"))
    if not matches:
        matches = sorted(exp_root.glob(f"sub_{phase}_{date}_*_{cand_key}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No submission JSONL found for phase={phase} date={date} cand={cand_key} under {exp_root}")
    return matches[0]


@dataclass(frozen=True)
class CaseDiag:
    uuid: str
    gt: str
    gt_parts: Tuple[str, ...]
    pred: str
    hit: bool
    gt_in_topk: bool
    best_rank: Optional[int]
    best_score: Optional[float]
    best_node_metric_max: Optional[float]
    best_tidb_metric_max: Optional[float]
    best_modalities: Tuple[str, ...]
    top1: Optional[str]
    top1_score: Optional[float]
    top1_node_metric_max: Optional[float]
    top1_tidb_metric_max: Optional[float]
    top1_modalities: Tuple[str, ...]
    chosen_component: Optional[str]
    component_source: str
    arbiter_enabled: bool


def _extract_case_diag(row: Dict[str, Any], gt_comp: str) -> CaseDiag:
    uuid = str(row.get("uuid", "")).strip()
    pred = str(row.get("component", "")).strip()
    parts = tuple(_gt_parts(gt_comp))
    hit = pred in parts

    debug = row.get("debug") or {}
    topk = debug.get("consensus_topk") or []
    top_items = [x for x in topk if isinstance(x, dict)]
    top_components = [str(x.get("component", "")).strip() for x in top_items]

    gt_in_topk = any(p in top_components for p in parts)
    best_rank = None
    best_score = None
    best_node_metric_max = None
    best_tidb_metric_max = None
    best_modalities: Tuple[str, ...] = ()
    if gt_in_topk:
        best = None
        best_r = None
        for p in parts:
            try:
                idx = top_components.index(p)
            except ValueError:
                continue
            r = idx + 1
            if best_r is None or r < best_r:
                best_r = r
                best = top_items[idx]
        best_rank = best_r
        if best is not None:
            try:
                best_score = float(best.get("score", 0.0))
            except (TypeError, ValueError):
                best_score = None
            try:
                best_node_metric_max = float(best.get("node_metric_max", 0.0))
            except (TypeError, ValueError):
                best_node_metric_max = None
            try:
                best_tidb_metric_max = float(best.get("tidb_metric_max", 0.0))
            except (TypeError, ValueError):
                best_tidb_metric_max = None
            mods = best.get("modalities")
            if isinstance(mods, list):
                best_modalities = tuple(str(m) for m in mods)

    top1 = top_components[0] if top_components else None
    top1_score = None
    top1_node_metric_max = None
    top1_tidb_metric_max = None
    top1_modalities: Tuple[str, ...] = ()
    if top_items:
        it0 = top_items[0]
        try:
            top1_score = float(it0.get("score", 0.0))
        except (TypeError, ValueError):
            top1_score = None
        try:
            top1_node_metric_max = float(it0.get("node_metric_max", 0.0))
        except (TypeError, ValueError):
            top1_node_metric_max = None
        try:
            top1_tidb_metric_max = float(it0.get("tidb_metric_max", 0.0))
        except (TypeError, ValueError):
            top1_tidb_metric_max = None
        mods = it0.get("modalities")
        if isinstance(mods, list):
            top1_modalities = tuple(str(m) for m in mods)

    type_arbiter = debug.get("type_arbiter") or {}
    chosen_component = None
    if isinstance(type_arbiter, dict):
        chosen_component = type_arbiter.get("chosen_component")
    component_source = str(debug.get("component_source", "consensus")).strip().lower() or "consensus"

    arbiter_enabled = bool(type_arbiter.get("enabled")) if isinstance(type_arbiter, dict) else False

    return CaseDiag(
        uuid=uuid,
        gt=gt_comp,
        gt_parts=parts,
        pred=pred,
        hit=hit,
        gt_in_topk=gt_in_topk,
        best_rank=best_rank,
        best_score=best_score,
        best_node_metric_max=best_node_metric_max,
        best_tidb_metric_max=best_tidb_metric_max,
        best_modalities=best_modalities,
        top1=top1,
        top1_score=top1_score,
        top1_node_metric_max=top1_node_metric_max,
        top1_tidb_metric_max=top1_tidb_metric_max,
        top1_modalities=top1_modalities,
        chosen_component=str(chosen_component).strip() if chosen_component else None,
        component_source=component_source,
        arbiter_enabled=arbiter_enabled,
    )


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-root", type=Path, required=True)
    ap.add_argument("--phase", type=str, required=True, choices=["phase1", "phase2"])
    ap.add_argument("--date", type=str, required=True)
    ap.add_argument("--cand-key", type=str, required=True)
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)

    sub_path = _find_submission(args.exp_root, args.phase, args.date, args.cand_key)
    sub_rows = _read_jsonl(sub_path)

    gt_rows = _read_jsonl(args.ground_truth)
    gt_index = {str(r.get("uuid", "")).strip(): r for r in gt_rows if str(r.get("uuid", "")).strip()}

    diags: List[CaseDiag] = []
    missing_gt = 0
    for r in sub_rows:
        u = str(r.get("uuid", "")).strip()
        if not u or u not in gt_index:
            missing_gt += 1
            continue
        gt_comp = str(gt_index[u].get("component", "")).strip()
        diags.append(_extract_case_diag(r, gt_comp))

    n = len(diags)
    hits = sum(1 for d in diags if d.hit)
    la = (hits / n) if n else 0.0

    gt_in_topk = sum(1 for d in diags if d.gt_in_topk)
    gt_in_topk_rate = (gt_in_topk / n) if n else 0.0

    # If present in topk but not hit: diagnose whether top1 or chosen_component would have hit.
    present_but_miss = [d for d in diags if d.gt_in_topk and not d.hit]
    top1_would_hit = sum(1 for d in present_but_miss if d.top1 and d.top1 in d.gt_parts)
    chosen_would_hit = sum(
        1 for d in present_but_miss if d.chosen_component and d.chosen_component in d.gt_parts
    )

    # Rank distribution
    rank_counter = Counter(d.best_rank for d in diags if d.best_rank is not None)

    # Evidence stats for GT when present
    present = [d for d in diags if d.gt_in_topk and d.best_rank is not None]
    gt_node_metric_pos = sum(1 for d in present if (d.best_node_metric_max or 0.0) > 0.0)
    gt_tidb_metric_pos = sum(1 for d in present if (d.best_tidb_metric_max or 0.0) > 0.0)
    top1_node_metric_pos = sum(1 for d in diags if (d.top1_node_metric_max or 0.0) > 0.0)
    top1_tidb_metric_pos = sum(1 for d in diags if (d.top1_tidb_metric_max or 0.0) > 0.0)

    # Top confusions
    pair_counter = Counter((d.gt, d.pred) for d in diags if not d.hit)
    top_pairs = pair_counter.most_common(10)

    # Category breakdown
    node_re = re.compile(r"^(aiops-k8s-\d+|tidb-[a-z-]+-\d+)$")
    cat = Counter()
    for d in diags:
        if d.hit:
            cat["HIT"] += 1
            continue
        gt_is_node = any(node_re.match(p) for p in d.gt_parts)
        pred_is_node = bool(node_re.match(d.pred))
        if gt_is_node and not pred_is_node:
            cat["GT_NODE__PRED_SERVICE"] += 1
        elif (not gt_is_node) and pred_is_node:
            cat["GT_SERVICE__PRED_NODE"] += 1
        else:
            cat["SAME_KIND_BUT_WRONG"] += 1

    # Compose report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append(f"# Debug 诊断：{args.phase} {args.date}（{args.cand_key}）\n")
    lines.append(f"- submission: `{sub_path}`")
    lines.append(f"- ground truth: `{args.ground_truth}`")
    lines.append(f"- N={n}，LA={_pct(la)}（{hits}/{n}）")
    lines.append(f"- GT 出现在 debug.topK 的比例：{_pct(gt_in_topk_rate)}（{gt_in_topk}/{n}）\n")

    lines.append("## 关键结论（用于判断瓶颈在哪）\n")
    if gt_in_topk == 0 and n > 0:
        lines.append("- **几乎纯召回问题**：GT 在 topK 里从未出现（或极少出现），继续调仲裁/阈值通常无效，需要改召回/候选注入。\n")
    elif gt_in_topk_rate > 0.5 and hits == 0:
        lines.append(
            "- **候选存在但决策未命中**：GT 经常在 topK 里出现，但最终 prediction 从未选中；需要看排序/过滤/仲裁路径（如 node 过滤、dominance 逻辑、component_source 等）。\n"
        )

    lines.append("## 命中/错因分类\n")
    lines.append("| 类别 | 计数 |")
    lines.append("|---|---:|")
    for k in ["HIT", "GT_NODE__PRED_SERVICE", "GT_SERVICE__PRED_NODE", "SAME_KIND_BUT_WRONG"]:
        lines.append(f"| {k} | {int(cat.get(k, 0))} |")
    lines.append("")

    lines.append("## GT 在 topK 中的名次分布（best_rank）\n")
    if rank_counter:
        lines.append("| 名次 | 数量 |")
        lines.append("|---:|---:|")
        for r, c in sorted(rank_counter.items(), key=lambda x: x[0]):
            lines.append(f"| {r} | {c} |")
        lines.append("")
    else:
        lines.append("- 本日无任何样本 GT 出现在 topK。\n")

    if present_but_miss:
        lines.append("## ‘GT 在 topK 但仍未命中’的路径提示\n")
        lines.append(
            f"- 这类样本数：{len(present_but_miss)}；其中 top1 本可命中：{top1_would_hit}；chosen_component 本可命中：{chosen_would_hit}\n"
        )
        lines.append(
            "解释：如果 top1 本可命中但最终没选到，优先检查 `component_source` 是否是 `llm` 以及 LLM 是否把 component 改错；如果 chosen_component 本可命中但最终没命中，优先检查 validator/后处理是否改写（通常不应）。\n"
        )

    lines.append("## 证据强度概览（基于 debug 字段）\n")
    if present:
        lines.append(
            f"- 在 ‘GT 出现在 topK’ 的 {len(present)} 个样本里：GT 的 node_metric_max>0 有 {gt_node_metric_pos}；GT 的 tidb_metric_max>0 有 {gt_tidb_metric_pos}"
        )
    else:
        lines.append("- 本日无样本 GT 出现在 topK，无法从 debug 统计 GT 的证据强度。")
    lines.append(
        f"- 对所有样本：top1 的 node_metric_max>0 有 {top1_node_metric_pos}；top1 的 tidb_metric_max>0 有 {top1_tidb_metric_pos}\n"
    )

    lines.append("## Top 混淆对（GT -> Pred）\n")
    lines.append("| 次数 | GT | Pred |")
    lines.append("|---:|---|---|")
    for (gt, pred), c in top_pairs:
        lines.append(f"| {c} | {gt} | {pred} |")
    lines.append("")

    if missing_gt:
        lines.append(f"- 注意：submission 中有 {missing_gt} 条 uuid 不在 GT（已忽略）。\n")

    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
