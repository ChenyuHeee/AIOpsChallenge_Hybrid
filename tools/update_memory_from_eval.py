"""Update Hybrid memory using judge ground-truth feedback.

This script reads a submission JSONL + ground-truth JSONL and writes feedback into the
memory file used by the Hybrid pipeline (default: AIOpsChallenge_Hybrid/.aiops_memory.json).

Goal: improve LA by discouraging components that are consistently wrong, without creating
self-reinforcing loops.

Usage:
  python AIOpsChallenge_Hybrid/tools/update_memory_from_eval.py \
    --memory AIOpsChallenge_Hybrid/.aiops_memory.json \
    --ground-truth AIOpsChallengeJudge/ground_truth_phase12.jsonl \
    --submission AIOpsChallenge_Hybrid/submissions_2025-06-07_pad15_24.jsonl

Tip:
- For apples-to-apples on subsets, pass ground truth filtered to the same UUIDs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

# Allow running as a script (python path/to/file.py) by ensuring repo root is on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from AIOpsChallenge_Hybrid.contest_solution.agents.consensus import ConsensusOrchestrator


def _load_index(path: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            uuid = str(obj.get("uuid", "")).strip()
            if not uuid:
                continue
            index[uuid] = obj
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description="Update Hybrid memory from judge evaluation data")
    parser.add_argument("--memory", type=Path, default=Path("AIOpsChallenge_Hybrid/.aiops_memory.json"))
    parser.add_argument("--ground-truth", "-g", type=Path, required=True)
    parser.add_argument("--submission", "-s", type=Path, required=True)
    args = parser.parse_args()

    gt = _load_index(args.ground_truth)
    sub = _load_index(args.submission)

    uuids = sorted(set(gt.keys()) & set(sub.keys()))
    if not uuids:
        raise SystemExit("No overlapping UUIDs between ground truth and submission")

    orchestrator = ConsensusOrchestrator(args.memory)

    correct = 0
    for u in uuids:
        pred = str(sub[u].get("component", "")).strip()
        gt_component = str(gt[u].get("component", "")).strip()
        orchestrator.apply_component_feedback(u, pred, gt_component)

        gt_parts = [part.strip() for part in gt_component.replace("->", "+").split("+") if part.strip()]
        if pred and gt_parts and pred in gt_parts:
            correct += 1

    # Persist
    from AIOpsChallenge_Hybrid.contest_solution.config import store_memory

    store_memory(args.memory, orchestrator.memory)

    print(f"Updated memory: {args.memory}")
    print(f"Samples: {len(uuids)}  component_correct: {correct}/{len(uuids)} ({correct/len(uuids):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
