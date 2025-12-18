"""Sample N dates from metadata that also exist in telemetry root.

Usage:
  python tools/sample_days.py --metadata metadata_phase1.csv --telemetry-root /abs/path/data --n 7

Outputs JSON to stdout:
  {"dates": [...], "missing_in_telemetry": [...], "counts": {date: ncases}}

Deterministic: uses stable sorting and optional --seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--telemetry-root", type=Path, required=True)
    p.add_argument("--n", type=int, default=7)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--prefer-full-days",
        action="store_true",
        help="Prefer dates with >=24 cases when sampling.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.metadata)
    if "date" not in df.columns:
        raise SystemExit("metadata missing date column")

    counts = df["date"].astype(str).str.strip().value_counts().to_dict()
    all_dates = sorted(counts.keys())

    existing = {p.name for p in args.telemetry_root.iterdir() if p.is_dir() and p.name[0].isdigit()}
    candidates = [d for d in all_dates if d in existing]
    missing = [d for d in all_dates if d not in existing]

    rng = random.Random(args.seed)

    if args.prefer_full_days:
        full = [d for d in candidates if int(counts.get(d, 0)) >= 24]
        rest = [d for d in candidates if d not in full]
        rng.shuffle(full)
        rng.shuffle(rest)
        picked = (full + rest)[: args.n]
    else:
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        picked = shuffled[: args.n]

    out = {
        "dates": picked,
        "missing_in_telemetry": missing,
        "counts": {d: int(counts.get(d, 0)) for d in picked},
        "available_dates": candidates,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
