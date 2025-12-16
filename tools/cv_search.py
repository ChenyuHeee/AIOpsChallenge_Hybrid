"""Lightweight k-fold CV search for LA (component accuracy).

This is intentionally simple and limited-scope: it is meant to validate whether small
hyperparameter tweaks (window padding, specialist weights) improve LA on held-out folds.

How it works:
- Reads metadata CSV (uuid/date/query).
- Selects first N cases (or all if --limit not set).
- Splits uuids into K folds (round-robin after a deterministic shuffle).
- For each candidate config:
  - Runs pipeline on validation fold uuids (LLM disabled).
  - Evaluates LA via judge, filtering GT to the same uuids.

NOTE: This can be slow on large N; start with N=24/48.

Example:
  python AIOpsChallenge_Hybrid/tools/cv_search.py \
    --telemetry-root data \
    --metadata metadata_2025-06-07.csv \
    --ground-truth AIOpsChallengeJudge/ground_truth_phase12.jsonl \
    --k 3 --limit 24
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

# Allow running as a script by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass(frozen=True)
class Candidate:
    padding_min: int
    w_metric: float
    w_log: float
    w_trace: float
    w_graph: float


def _load_jsonl_index(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            uuid = str(obj.get("uuid", "")).strip()
            if uuid:
                out[uuid] = obj
    return out


def _write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    payload = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(payload) + "\n", encoding="utf-8")


def _split_kfold(uuids: List[str], k: int, seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    uuids = list(uuids)
    rng.shuffle(uuids)
    folds: List[List[str]] = [[] for _ in range(k)]
    for i, u in enumerate(uuids):
        folds[i % k].append(u)
    return folds


def _filter_gt(gt_all: Path, uuids: Sequence[str], out_path: Path) -> None:
    want = set(uuids)
    kept = []
    with gt_all.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if str(obj.get("uuid", "")).strip() in want:
                kept.append(obj)
    _write_jsonl(kept, out_path)


def _filter_metadata(metadata: Path, uuids: Sequence[str], out_path: Path) -> None:
    want = set(uuids)
    df = pd.read_csv(metadata)
    df = df[df["uuid"].astype(str).str.strip().isin(want)]
    df.to_csv(out_path, index=False)


def _run_pipeline(telemetry_root: Path, metadata_csv: Path, out_submission: Path) -> None:
    cmd = [
        os.environ.get("PYTHON", "python"),
        "-m",
        "AIOpsChallenge_Hybrid.contest_solution.main",
        "--telemetry-root",
        str(telemetry_root),
        "--metadata",
        str(metadata_csv),
        "--output",
        str(out_submission),
    ]
    subprocess.check_call(cmd)


def _judge_la(judge_py: Path, gt: Path, sub: Path) -> float:
    out = subprocess.check_output([
        os.environ.get("PYTHON", "python"),
        str(judge_py),
        "-g",
        str(gt),
        "-s",
        str(sub),
    ], text=True)
    for line in out.splitlines():
        if "Component Accuracy" in line:
            value = line.split(":", 1)[1].strip().rstrip("%")
            return float(value) / 100.0
    raise RuntimeError("Failed to parse LA from judge output")


def main() -> int:
    parser = argparse.ArgumentParser(description="K-fold CV search (LA) for Hybrid")
    parser.add_argument("--telemetry-root", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--judge", type=Path, default=Path("AIOpsChallengeJudge/evaluate.py"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workdir", type=Path, default=Path("AIOpsChallenge_Hybrid/tmp_cv"))
    parser.add_argument("--paddings", type=str, default="5,10,15")
    parser.add_argument("--trace-weights", type=str, default="1.6")
    parser.add_argument("--graph-weights", type=str, default="1.0")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata)
    uuids_all = [str(u).strip() for u in df["uuid"].tolist() if str(u).strip()]
    uuids = uuids_all[: args.limit] if args.limit else uuids_all
    folds = _split_kfold(uuids, args.k, args.seed)

    def _parse_floats(value: str) -> List[float]:
        items = [v.strip() for v in (value or "").split(",") if v.strip()]
        out: List[float] = []
        for item in items:
            out.append(float(item))
        return out

    paddings = [int(v.strip()) for v in (args.paddings or "").split(",") if v.strip()]
    trace_weights = _parse_floats(args.trace_weights)
    graph_weights = _parse_floats(args.graph_weights)

    if not paddings or not trace_weights or not graph_weights:
        raise SystemExit("Empty candidate grid: check --paddings/--trace-weights/--graph-weights")

    candidates: List[Candidate] = []
    for pad in paddings:
        for w_trace in trace_weights:
            for w_graph in graph_weights:
                candidates.append(Candidate(padding_min=pad, w_metric=1.0, w_log=1.15, w_trace=w_trace, w_graph=w_graph))

    best: Tuple[float, Candidate] | None = None

    for cand in candidates:
        # Configure env for this candidate
        os.environ["RCA_LLM_PROVIDER"] = "dummy"
        os.environ["RCA_WINDOW_PADDING_MIN"] = str(cand.padding_min)
        os.environ["RCA_WEIGHT_METRIC"] = str(cand.w_metric)
        os.environ["RCA_WEIGHT_LOG"] = str(cand.w_log)
        os.environ["RCA_WEIGHT_TRACE"] = str(cand.w_trace)
        os.environ["RCA_WEIGHT_GRAPH"] = str(cand.w_graph)
        # Ensure no memory leakage across folds
        os.environ["RCA_ENABLE_MEMORY"] = "0"

        fold_scores: List[float] = []
        for i, val_uuids in enumerate(folds):
            fold_dir = args.workdir / f"cand_pad{cand.padding_min}_tr{cand.w_trace}_gr{cand.w_graph}_fold{i}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            meta_fold = fold_dir / "metadata.csv"
            sub_fold = fold_dir / "submission.jsonl"
            gt_fold = fold_dir / "gt.jsonl"

            _filter_metadata(args.metadata, val_uuids, meta_fold)
            _filter_gt(args.ground_truth, val_uuids, gt_fold)
            _run_pipeline(args.telemetry_root, meta_fold, sub_fold)
            la = _judge_la(args.judge, gt_fold, sub_fold)
            fold_scores.append(la)

        avg = sum(fold_scores) / max(1, len(fold_scores))
        print(f"cand={cand}  LA={avg:.3f}  folds={[round(s,3) for s in fold_scores]}")
        if best is None or avg > best[0]:
            best = (avg, cand)

    if best:
        avg, cand = best
        print("\nBEST")
        print(f"LA={avg:.3f}  cand={cand}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
