# AIOpsChallenge_Hybrid

Hybrid root-cause analysis pipeline derived from the Beta solution. This repo contains the runnable submission code (`contest_solution`) plus sample outputs for different phases.

## What this repo has
- `contest_solution/`: full pipeline (planner, specialists, consensus, reasoning, validator)
- `submissions_phase1.jsonl`, `submissions_phase2.jsonl`, `submissions_2025-06-07.jsonl`: example outputs
- `ground_truth_sample.jsonl`, `submissions_sample.jsonl`: small samples for quick checks
- `.env`: LLM credentials and endpoints (not tracked)

## Quick start
```bash
# Install deps (Python 3.10+ recommended)
pip install -r contest_solution/requirements.txt  # if you extracted a separate requirements file
# or reuse project venv

# Run full pipeline
python -m contest_solution.main \
  --telemetry-root /path/to/telemetry \
  --metadata /path/to/metadata_phase1.csv \
  --output submissions.jsonl

# Optional flags
# --limit N      : process first N cases only
# --dry-run      : print records instead of writing to file
```

## Algorithm (high level)
1) Load telemetry window per case; build lightweight event graph when signals exist.
2) Planner (Flow-of-Action) sets scope and retrieves paper insights for prompts.
3) Specialists (metrics, logs, traces/graph) generate hypotheses with scores.
4) Consensus combines mABC-style voting with priors and memory to rank components.
5) Reasoning LLM produces component + reason + reasoning_trace; heuristic fallback if LLM fails.
6) Validator enforces format/length limits and writes JSONL for the evaluator.

## Configuration knobs
- `.env`: `OPENAI_API_KEY`, base URL, model name, concurrency/timeouts.
- `contest_solution/config.py`: toggle experts, priors, memory strength, max steps/lengths.
- `contest_solution/resources/paper_insights.json`: RAG knowledge base for prompts.
- `contest_solution/agents/consensus.py`: adjust prior weights and memory reward/penalty.

## Baseline scores (new judge)
- Phase1: Component 14.69%, Reason 59.24%, Efficiency 81.87%, Explainability 14.67%, Final 39.23
- Phase2: Component 2.08%, Reason 43.23%, Efficiency 81.87%, Explainability 17.23%, Final 28.04

## Notes
- Keep `.env` out of version control; set env vars or export before running.
- If telemetry days are missing, pipeline falls back to reasoning with empty signals.
