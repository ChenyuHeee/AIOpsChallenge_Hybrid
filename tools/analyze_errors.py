"""误差分析：统计 component/TA 失分的主要模式（不依赖训练）。

用法示例：
  /Users/hechenyu/projects/AIOPS/.venv/bin/python tools/analyze_errors.py \
    --ground-truth ../AIOpsChallengeJudge/ground_truth_phase1.jsonl \
    --submission outputs/experiments/final_grid_v1/sub_2025-06-07_n24_base_p20.jsonl \
    --topk 20

输出：
- 总体 LA/TA（按 judge 规则：component 对 edge 以端点命中判定）
- Top-K 预测最多的 component
- Top-K GT 最多的 component
- Top-K “GT→预测”混淆对（帮助定位系统性偏差）
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze component/reason errors")
    p.add_argument("--ground-truth", "-g", type=Path, required=True)
    p.add_argument("--submission", "-s", type=Path, required=True)
    p.add_argument("--topk", type=int, default=20)
    return p.parse_args(argv)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_index(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        u = str(r.get("uuid", "")).strip()
        if not u:
            continue
        idx[u] = r
    return idx


def component_is_correct(pred: str, gt: str) -> bool:
    pred = (pred or "").strip()
    gt = (gt or "").strip()
    if not pred or not gt:
        return False
    parts = [p.strip() for p in gt.replace("->", "+").split("+") if p.strip()]
    return pred in parts


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    gt_rows = read_jsonl(args.ground_truth)
    sub_rows = read_jsonl(args.submission)
    gt = build_index(gt_rows)
    sub = build_index(sub_rows)

    if len(gt) > max(50, len(sub) * 2):
        print(
            "[提示] ground truth 的 uuid 数量远大于 submission。\n"
            "      如果你是在分析某个日期/子集的输出，请先把 ground truth 过滤到相同 uuid 集合（例如使用 tmp/filtered/gt_YYYY-MM-DD_nXX.jsonl）。"
        )

    uuids = sorted(set(gt.keys()))

    correct = 0
    total = 0

    pred_counter: Counter[str] = Counter()
    gt_counter: Counter[str] = Counter()
    confusion: Counter[Tuple[str, str]] = Counter()

    for u in uuids:
        g = gt[u]
        s = sub.get(u, {})
        gt_comp = str(g.get("component", "") or g.get("root_cause", "") or "").strip()
        pred_comp = str(s.get("component", "")).strip()

        # For stats, count GT edge as raw string; confusion uses raw GT.
        gt_counter[gt_comp] += 1
        pred_counter[pred_comp] += 1

        total += 1
        if component_is_correct(pred_comp, gt_comp):
            correct += 1
        else:
            confusion[(gt_comp, pred_comp)] += 1

    la = correct / max(1, total)
    print(f"Total cases: {total}")
    print(f"Component accuracy (LA): {la * 100:.2f}%")
    print("")

    print(f"Top-{args.topk} predicted components:")
    for comp, c in pred_counter.most_common(args.topk):
        print(f"  {c:4d}  {comp}")
    print("")

    print(f"Top-{args.topk} GT components:")
    for comp, c in gt_counter.most_common(args.topk):
        print(f"  {c:4d}  {comp}")
    print("")

    print(f"Top-{args.topk} confusions (GT -> Pred):")
    for (gt_comp, pred_comp), c in confusion.most_common(args.topk):
        print(f"  {c:4d}  {gt_comp}  ->  {pred_comp}")

    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
