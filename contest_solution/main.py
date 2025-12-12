"""CLI entry point for the contest solution."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from .config import KnowledgeConfig, LLMConfig, PipelineConfig
from .orchestrator import ContestOrchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Contest RCA submission generator")
    parser.add_argument("--telemetry-root", type=Path, required=True, help="Root directory containing telemetry folders")
    parser.add_argument("--metadata", type=Path, required=True, help="CSV mapping uuid,date,query")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSONL submission file")
    parser.add_argument("--limit", type=int, help="Process only the first N cases")
    parser.add_argument("--dry-run", action="store_true", help="Print submission entries without writing file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    orchestrator = ContestOrchestrator(
        args.telemetry_root,
        llm_config=LLMConfig.from_env(),
        pipeline_config=PipelineConfig.from_env(),
        knowledge_config=KnowledgeConfig.from_env(),
    )
    cases = list(orchestrator.iter_cases(args.metadata))
    if args.limit is not None:
        cases = cases[: args.limit]
    submissions = []
    for case in tqdm(cases, desc="Running RCA cases"):
        submissions.append(orchestrator.run_case(case))
    if args.dry_run:
        for entry in submissions:
            print(entry.to_dict())
    else:
        orchestrator.write(submissions, args.output)
        print(f"Wrote {len(submissions)} entries to {args.output}")


if __name__ == "__main__":  # pragma: no cover
    main()
