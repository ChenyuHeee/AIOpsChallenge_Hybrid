#!/usr/bin/env python3
"""Run a shared-hypotheses experiment suite (fast multi-candidate evaluation).

Why this script exists
- The standard suite runner invokes the generator once per candidate (subprocess),
  which re-runs telemetry loading and specialists repeatedly.
- We run specialists ONCE per case and then evaluate multiple candidates on the same hypotheses.
- For evidence/weight ablations (e.g. preset evidence_probe_v1), use
    --recompute-consensus-per-candidate so RCA_WEIGHT_* and rule toggles affect ranking.

Candidates are provided by preset (see tools/run_combined_phases_suite.py build_candidates).

Usage:
  python tools/run_type_arbiter_groups_shared_suite.py \
    --suite-name "36_type_arbiter_5groups_p1p2_7d" \
    --suite-slug type_arbiter_5groups_p1p2_7d \
    --preset type_arbiter_groups_v1 \
    --phase1-metadata ../metadata_phase1.csv \
    --phase1-ground-truth ../AIOpsChallengeJudge/ground_truth_phase1.jsonl \
    --phase1-dates 2025-06-13 2025-06-07 2025-06-11 2025-06-08 2025-06-10 2025-06-09 2025-06-14 \
    --phase2-metadata ../metadata_phase2.csv \
    --phase2-ground-truth ../AIOpsChallengeJudge/ground_truth_phase2.jsonl \
    --phase2-dates 2025-06-27 2025-06-17 2025-06-20 2025-06-18 2025-06-19 2025-06-24 2025-06-28 \
    --telemetry-root ../data
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

import pandas as pd

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from contest_solution.orchestrator import CaseMetadata, ContestOrchestrator  # noqa: E402

# Import judge evaluate module
judge_root = Path(__file__).resolve().parents[1].parent / "AIOpsChallengeJudge"
sys.path.insert(0, str(judge_root))
import evaluate  # pyright: ignore[reportMissingImports]  # noqa: E402

# Reuse the candidate preset builder from the existing suite runner.
import run_combined_phases_suite as suite_runner  # noqa: E402

Candidate = suite_runner.Candidate
PhaseConfig = suite_runner.PhaseConfig
build_candidates = suite_runner.build_candidates


@dataclass
class CandidateResult:
    candidate: Candidate
    per_phase_per_date: Dict[str, Dict[str, Dict[str, Any]]]


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", type=str, required=True)
    p.add_argument("--suite-name", type=str, required=True)
    p.add_argument("--suite-slug", type=str, required=True)
    p.add_argument("--candidate-keys", nargs="+", default=None)

    p.add_argument("--phase1-metadata", type=Path, required=True)
    p.add_argument("--phase1-ground-truth", type=Path, required=True)
    p.add_argument("--phase1-dates", nargs="+", required=True)

    p.add_argument("--phase2-metadata", type=Path, required=True)
    p.add_argument("--phase2-ground-truth", type=Path, required=True)
    p.add_argument("--phase2-dates", nargs="+", required=True)

    p.add_argument("--telemetry-root", type=Path, required=True)
    p.add_argument("--output-report", type=Path, default=None)

    # For evidence/weight ablations: run specialists once, but recompute consensus under each
    # candidate env so RCA_WEIGHT_* and rule toggles take effect.
    p.add_argument(
        "--recompute-consensus-per-candidate",
        action="store_true",
        help="Recompute consensus vote per candidate env (still reuses shared specialist hypotheses).",
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


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def _evaluate_with_judge(ground_truth_path: Path, submission_path: Path) -> Dict[str, float]:
    """Evaluate a submission JSONL against judge.

    IMPORTANT: When evaluating per-(phase,date) slices, we must score ONLY on the uuids
    present in that slice. Scoring against the full phase ground truth and filling
    missing uuids with blank predictions will artificially drive LA/TA towards zero.
    """

    gt_rows = _read_jsonl(ground_truth_path)
    sub_rows = _read_jsonl(submission_path)

    # Build submission index first to know which uuids to score.
    sub_index: Dict[str, Dict[str, Any]] = {}
    for row in sub_rows:
        uuid = str(row.get("uuid", "")).strip()
        if uuid:
            sub_index[uuid] = row

    if not sub_index:
        # Judge divides by total_samples; avoid ZeroDivisionError on empty slices.
        return {
            "component_accuracy": 0.0,
            "reason_accuracy": 0.0,
            "explainability": 0.0,
            "efficiency": 0.0,
            "final_score": 0.0,
        }

    uuids = set(sub_index.keys())
    gt_index: Dict[str, Dict[str, Any]] = {}
    for row in gt_rows:
        uuid = str(row.get("uuid", "")).strip()
        if uuid and uuid in uuids:
            gt_index[uuid] = row

    # If any submission uuid is not found in GT (shouldn't happen), drop it.
    missing_in_gt = sorted(uuids - set(gt_index.keys()))
    for uuid in missing_in_gt:
        sub_index.pop(uuid, None)

    metrics, _ = evaluate.score_submission(gt_index, sub_index, reason_threshold=0.65)
    return metrics


@contextmanager
def _env(overrides: Dict[str, str]) -> Iterator[None]:
    old = os.environ.copy()
    try:
        os.environ.update({k: str(v) for k, v in overrides.items()})
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


def _phase_env_overrides(phase_name: str) -> Dict[str, str]:
    # Mirror run_combined_phases_suite behavior: avoid TiDB tokens in phase1.
    if phase_name == "phase1":
        return {
            "RCA_ALLOW_TIDB_COMPONENTS": "0",
            "RCA_ENABLE_TIDB_METRIC_DOMINANCE_OVERRIDE": "0",
        }
    return {
        "RCA_ALLOW_TIDB_COMPONENTS": "1",
    }


def _iter_cases_for_date(metadata_path: Path, date: str) -> Iterator[CaseMetadata]:
    df = pd.read_csv(metadata_path)
    df = df[df["date"].astype(str).str.strip() == str(date).strip()]
    for _, row in df.iterrows():
        yield CaseMetadata.from_row(row)


def _run_all_candidates_for_case(
    orchestrator: ContestOrchestrator,
    *,
    case: CaseMetadata,
    candidates: List[Candidate],
    phase_name: str,
    recompute_consensus_per_candidate: bool,
) -> Dict[str, Dict[str, Any]]:
    """Return per-candidate entry dicts for this case."""

    plan = orchestrator.planner.build_plan(case.uuid, case.query)
    telemetry = orchestrator.loader.load(orchestrator.telemetry_root, case.date, plan.time_window)
    # Safe construction (matches ContestOrchestrator.run_case)
    from contest_solution.agents.specialists import SpecialistContext  # local import

    ctx = SpecialistContext(keywords=plan.keywords, component_hints=plan.component_hints)
    hypotheses = orchestrator._run_specialists(telemetry, ctx)  # noqa: SLF001
    hypothesis_bank = orchestrator._group_by_component(hypotheses)  # noqa: SLF001

    # Default behavior (original): compute consensus once.
    base_consensus: Any = None
    if not recompute_consensus_per_candidate:
        base_consensus = orchestrator.consensus.vote(case.uuid, hypotheses, component_hints=plan.component_hints)

    out: Dict[str, Dict[str, Any]] = {}

    for cand in candidates:
        merged_env = {}
        merged_env.update(_phase_env_overrides(phase_name))
        merged_env.update(cand.env)

        with _env(merged_env):
            # For evidence ablations we must recompute consensus under this env.
            if recompute_consensus_per_candidate:
                consensus_result = orchestrator.consensus.vote(case.uuid, hypotheses, component_hints=plan.component_hints)
            else:
                assert base_consensus is not None
                consensus_result = base_consensus

            # Start from consensus top-1
            consensus_component = (
                consensus_result.ranked_components[0][0] if consensus_result.ranked_components else "unknown"
            )

            chosen_component = consensus_component
            ranked_for_reasoning = list(consensus_result.ranked_components)
            decision_obj = None

            if orchestrator.type_arbiter.enabled():
                decision_obj = orchestrator.type_arbiter.decide(
                    uuid=case.uuid, query=case.query, consensus=consensus_result
                )
                chosen_component, ranked_for_reasoning = orchestrator.type_arbiter.apply(
                    ranked_components=ranked_for_reasoning,
                    consensus=consensus_result,
                    current_component=chosen_component,
                    decision=decision_obj,
                )
                if ranked_for_reasoning and ranked_for_reasoning[0][0] != chosen_component:
                    for i, (c, _) in enumerate(ranked_for_reasoning):
                        if c == chosen_component:
                            ranked_for_reasoning.insert(0, ranked_for_reasoning.pop(i))
                            break

            reasoning = orchestrator.reasoning.run(
                uuid=case.uuid,
                query=case.query,
                plan_steps=plan.sop_steps,
                insights=plan.insights,
                ranked_components=ranked_for_reasoning,
                hypothesis_bank=hypothesis_bank,
            )

            component_source = os.getenv("RCA_COMPONENT_SOURCE", "consensus").strip().lower() or "consensus"
            if component_source == "llm":
                final_component = reasoning.component or chosen_component
            else:
                final_component = chosen_component

            component, reason, steps = orchestrator.validator.enforce_limits(
                final_component, reasoning.reason, reasoning.steps
            )
            reason = orchestrator.validator.enrich_reason(reason, component, hypothesis_bank)
            trace = orchestrator.validator.build_trace(steps, component, hypothesis_bank)

            out[cand.key] = {
                "uuid": case.uuid,
                "component": component,
                "reason": reason,
                "reasoning_trace": trace,
            }

            if os.getenv("RCA_OUTPUT_DEBUG", "0") not in {"0", "false", "False"}:
                try:
                    debug_topk = int(os.getenv("RCA_DEBUG_TOPK", "20"))
                except ValueError:
                    debug_topk = 20
                debug_topk = max(1, min(debug_topk, 200))
                # Keep debug compact; enough to answer "did the correct candidate exist?"
                debug_payload: Dict[str, Any] = {
                    "consensus_topk": [
                        {
                            "component": c,
                            "score": float(s),
                            "node_metric_max": float(consensus_result.node_metric_max.get(c, 0.0)),
                            "tidb_metric_max": float(consensus_result.tidb_metric_max.get(c, 0.0)),
                            "modalities": list(consensus_result.modality_support.get(c) or []),
                        }
                        for c, s in list(consensus_result.ranked_components)[:debug_topk]
                    ],
                    "type_arbiter": {
                        "enabled": orchestrator.type_arbiter.enabled(),
                        "backend": os.getenv("RCA_TYPE_ARBITER_BACKEND", "llm"),
                        "mode": os.getenv("RCA_TYPE_ARBITER_MODE", "postcheck"),
                        "threshold": os.getenv("RCA_TYPE_ARBITER_CONFIDENCE_THRESHOLD", "0.85"),
                    },
                    "component_source": component_source,
                    "final_component_before_validator": final_component,
                    "reasoning_component": reasoning.component,
                }
                if orchestrator.type_arbiter.enabled():
                    if decision_obj is not None:
                        debug_payload["type_arbiter"]["decision"] = {
                            "decided_type": getattr(decision_obj, "decided_type", ""),
                            "confidence": float(getattr(decision_obj, "confidence", 0.0)),
                            "raw_response": getattr(decision_obj, "raw_response", "")[:2000],
                        }
                    debug_payload["type_arbiter"]["chosen_component"] = chosen_component
                out[cand.key]["debug"] = debug_payload

    return out


def _write_report(
    *,
    report_path: Path,
    suite_name: str,
    suite_slug: str,
    exp_root: Path,
    phases: List[PhaseConfig],
    candidates: List[Candidate],
    results: List[CandidateResult],
) -> None:
    _ensure_parent(report_path)

    def pct(x: float) -> str:
        return f"{x * 100:.2f}%"

    def _aggregate_from_samples(samples: List[Any]) -> Dict[str, float]:
        total = len(samples)
        component_hits = sum(1 for s in samples if getattr(s, "component_correct", False))
        reason_hits_total = sum(1 for s in samples if getattr(s, "reason_correct", False))
        path_lengths = [int(getattr(s, "step_count", 0)) for s in samples if getattr(s, "component_correct", False)]

        efficiency = 0.0
        if path_lengths:
            apl = sum(path_lengths) / len(path_lengths)
            efficiency = math.exp(-(apl - 5.0) / 5.0)
            efficiency = min(max(efficiency, 0.0), 1.0)

        total_evidence_hits = sum(int(getattr(s, "evidence_hit", 0)) for s in samples)
        total_evidence_points = sum(int(getattr(s, "evidence_total", 0)) for s in samples)
        explainability = (total_evidence_hits / total_evidence_points) if total_evidence_points else 0.0

        la = (component_hits / total) if total else 0.0
        ta = (reason_hits_total / total) if total else 0.0
        final_score = 100.0 * (0.40 * la + 0.40 * ta + 0.10 * efficiency + 0.10 * explainability)

        return {
            "component_accuracy": la,
            "reason_accuracy": ta,
            "efficiency": efficiency,
            "explainability": explainability,
            "final_score": final_score,
            "N": float(total),
        }

    def _score_candidate_phase(*, phase: PhaseConfig, cand_key: str) -> tuple[Dict[str, float], List[Any]]:
        # Concatenate all per-date submissions for this candidate within the phase.
        all_rows: List[Dict[str, Any]] = []
        for d in phase.dates:
            matches = list(exp_root.glob(f"sub_{phase.phase_name}_{d}_*_{cand_key}.jsonl"))
            if not matches:
                continue
            all_rows.extend(_read_jsonl(matches[0]))

        gt_rows = _read_jsonl(Path(phase.ground_truth))
        gt_index = {str(r.get("uuid", "")).strip(): r for r in gt_rows if str(r.get("uuid", "")).strip()}
        sub_index = {str(r.get("uuid", "")).strip(): r for r in all_rows if str(r.get("uuid", "")).strip()}

        # Score only on uuids present in the concatenated submission.
        gt_sub = {u: gt_index[u] for u in sub_index.keys() if u in gt_index}
        for u in list(sub_index.keys()):
            if u not in gt_sub:
                sub_index.pop(u, None)

        metrics, per_sample = evaluate.score_submission(gt_sub, sub_index, reason_threshold=0.65)
        metrics = dict(metrics)
        metrics["N"] = float(len(sub_index))
        return metrics, per_sample

    def _analyze_component_mismatches(*, phase: PhaseConfig, cand_key: str) -> Dict[str, Any]:
        """Return mismatch diagnostics for LA, based on exact-match judge semantics.

        This does NOT change the score; it explains why LA is low.
        """

        def _gt_parts(comp: str) -> List[str]:
            return [x.strip() for x in comp.replace("->", "+").split("+") if x.strip()]

        def _strip_pod_ordinal(name: str) -> str:
            # Very conservative: common cases like adservice-0
            import re

            return re.sub(r"-\d+$", "", name)

        def _strip_rs_hash(name: str) -> str:
            import re

            return re.sub(r"-[0-9a-f]{6,}$", "", name)

        def _norm_combo(name: str) -> str:
            n = name.strip()
            n = _strip_pod_ordinal(n)
            n = _strip_rs_hash(n)
            return n

        import re
        import collections

        node_re = re.compile(r"^(aiops-k8s-\d+|tidb-[a-z-]+-\d+)$")

        all_rows: List[Dict[str, Any]] = []
        for d in phase.dates:
            matches = list(exp_root.glob(f"sub_{phase.phase_name}_{d}_n*_{cand_key}.jsonl"))
            if not matches:
                matches = list(exp_root.glob(f"sub_{phase.phase_name}_{d}_*_{cand_key}.jsonl"))
            if matches:
                all_rows.extend(_read_jsonl(matches[0]))

        gt_rows = _read_jsonl(Path(phase.ground_truth))
        gt_index = {str(r.get("uuid", "")).strip(): r for r in gt_rows if str(r.get("uuid", "")).strip()}

        categories = collections.Counter()
        pair_counts = collections.Counter()

        hits_raw = 0
        hits_norm = 0
        total = 0

        suffix_fixable_strip0 = 0
        suffix_fixable_add0 = 0

        for r in all_rows:
            u = str(r.get("uuid", "")).strip()
            if not u or u not in gt_index:
                continue
            gt = gt_index[u]
            gt_comp = str(gt.get("component", "")).strip()
            parts = _gt_parts(gt_comp)
            pred = str(r.get("component", "")).strip()
            total += 1

            if pred in parts:
                hits_raw += 1
                categories["HIT"] += 1
                continue

            pair_counts[(gt_comp, pred)] += 1

            gt_is_node = any(node_re.match(p) for p in parts)
            pred_is_node = bool(node_re.match(pred))
            if gt_is_node and not pred_is_node:
                categories["GT_NODE__PRED_SERVICE"] += 1
            elif (not gt_is_node) and pred_is_node:
                categories["GT_SERVICE__PRED_NODE"] += 1
            else:
                categories["SAME_KIND_BUT_WRONG"] += 1

            # Suffix mismatch diagnostics (not used in scoring; just explanation)
            for p in parts:
                if p.endswith("-0") and pred == p[:-2]:
                    suffix_fixable_add0 += 1
                    categories["SUFFIX_GT_-0_PRED_BASE"] += 1
                    break
            if pred.endswith("-0") and any(p == pred[:-2] for p in parts):
                suffix_fixable_strip0 += 1
                categories["SUFFIX_PRED_-0_GT_BASE"] += 1

            pred_norm = _norm_combo(pred)
            if pred_norm in parts:
                hits_norm += 1

        top_pairs = [
            {"count": c, "gt": g, "pred": p}
            for (g, p), c in pair_counts.most_common(10)
        ]

        la_raw = (hits_raw / total) if total else 0.0
        la_upper_norm = ((hits_raw + hits_norm) / total) if total else 0.0

        return {
            "N": total,
            "hits_raw": hits_raw,
            "la_raw": la_raw,
            "la_upper_norm": la_upper_norm,
            "categories": dict(categories),
            "top_pairs": top_pairs,
            "suffix_fixable_add0": suffix_fixable_add0,
            "suffix_fixable_strip0": suffix_fixable_strip0,
        }

    lines: List[str] = []
    lines.append(f"# {suite_name}\n")
    lines.append("目标：比较5组（无LLM/规则仲裁/LLM三种集成方式）对 LA/TA/Explain/Eff/Final 的影响。\n")
    lines.append(
        "说明：评测中 GT 的 `A->B` 会被拆成 `[A,B]`，命中任一端即算 component_correct；但仍要求 **完全相等**（例如 `frontend-0` 不等于 `frontend`）。\n"
    )

    lines.append("## 试验设置")
    for ph in phases:
        lines.append(f"- **{ph.phase_name}**: metadata={ph.metadata}; gt={ph.ground_truth}; dates={', '.join(ph.dates)}")
    lines.append(f"- suite slug: `{suite_slug}`（对应 outputs/experiments/{suite_slug}/）\n")

    lines.append("## 候选方案")
    for c in candidates:
        lines.append(f"- {c.name}（key={c.key}）")
    lines.append("")

    # Exact aggregates (match judge semantics)
    per_phase_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    per_phase_samples: Dict[str, Dict[str, List[Any]]] = {}

    for ph in phases:
        per_phase_metrics[ph.phase_name] = {}
        per_phase_samples[ph.phase_name] = {}
        for cand in candidates:
            m, samples = _score_candidate_phase(phase=ph, cand_key=cand.key)
            per_phase_metrics[ph.phase_name][cand.key] = m
            per_phase_samples[ph.phase_name][cand.key] = samples

    lines.append("## 结果汇总（全部 phase 合并；严格按 judge 累计计算）\n")
    lines.append("| 方案 | 样本数N | LA | TA | Explain | Eff | Final |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cand in candidates:
        combined_samples: List[Any] = []
        for ph in phases:
            combined_samples.extend(per_phase_samples[ph.phase_name].get(cand.key, []))
        m = _aggregate_from_samples(combined_samples)
        lines.append(
            f"| {cand.name} | {int(m['N'])} | {pct(float(m['component_accuracy']))} | {pct(float(m['reason_accuracy']))} | {pct(float(m['explainability']))} | {pct(float(m['efficiency']))} | {float(m['final_score']):.2f} |"
        )

    lines.append("")

    for ph in phases:
        lines.append(f"## {ph.phase_name} 结果汇总（该 phase 合并；严格按 judge 累计计算）\n")
        lines.append("| 方案 | 样本数N | LA | TA | Explain | Eff | Final |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for cand in candidates:
            m = per_phase_metrics[ph.phase_name].get(cand.key) or {}
            lines.append(
                "| {name} | {n:d} | {la} | {ta} | {ex} | {ef} | {fi:.2f} |".format(
                    name=cand.name,
                    n=int(float(m.get("N", 0.0))),
                    la=pct(float(m.get("component_accuracy", 0.0))),
                    ta=pct(float(m.get("reason_accuracy", 0.0))),
                    ex=pct(float(m.get("explainability", 0.0))),
                    ef=pct(float(m.get("efficiency", 0.0))),
                    fi=float(m.get("final_score", 0.0)),
                )
            )
        lines.append("")

    # Diagnostics for LA (best candidate only, to keep report compact)
    best_cand = None
    best_score = -1.0
    combined_metrics_by_key: Dict[str, Dict[str, float]] = {}
    for cand in candidates:
        combined_samples: List[Any] = []
        for ph in phases:
            combined_samples.extend(per_phase_samples[ph.phase_name].get(cand.key, []))
        m = _aggregate_from_samples(combined_samples)
        combined_metrics_by_key[cand.key] = m
        if float(m.get("final_score", 0.0)) > best_score:
            best_score = float(m.get("final_score", 0.0))
            best_cand = cand

    if best_cand is not None:
        lines.append("## LA 丢分点分析（组件命中）\n")
        lines.append(
            f"为避免报告过长，本节仅对 **Final 最优** 的方案（{best_cand.name}，key={best_cand.key}）做错因分解。\n"
        )

        combined_diag = {
            "categories": {},
            "top_pairs": [],
            "N": 0,
            "hits_raw": 0,
            "la_raw": 0.0,
            "la_upper_norm": 0.0,
            "suffix_fixable_add0": 0,
            "suffix_fixable_strip0": 0,
        }
        # Merge phase-level diagnostics
        for ph in phases:
            d = _analyze_component_mismatches(phase=ph, cand_key=best_cand.key)
            combined_diag["N"] += int(d.get("N", 0))
            combined_diag["hits_raw"] += int(d.get("hits_raw", 0))
            combined_diag["suffix_fixable_add0"] += int(d.get("suffix_fixable_add0", 0))
            combined_diag["suffix_fixable_strip0"] += int(d.get("suffix_fixable_strip0", 0))
            for k, v in (d.get("categories") or {}).items():
                combined_diag["categories"][k] = int(combined_diag["categories"].get(k, 0)) + int(v)
            combined_diag["top_pairs"].extend(d.get("top_pairs") or [])

        # Dedup/merge top_pairs by (gt,pred)
        pair_map: Dict[tuple[str, str], int] = {}
        for item in combined_diag["top_pairs"]:
            gt = str(item.get("gt", ""))
            pred = str(item.get("pred", ""))
            c = int(item.get("count", 0))
            if not gt and not pred:
                continue
            pair_map[(gt, pred)] = pair_map.get((gt, pred), 0) + c
        top_pairs = sorted(pair_map.items(), key=lambda x: x[1], reverse=True)[:10]

        n_total = max(1, int(combined_diag["N"]))
        la_raw = float(combined_diag["hits_raw"]) / n_total
        # We can only estimate the suffix-normalization upper bound reliably per-phase;
        # here we approximate by recomputing per-phase and weighted-averaging.
        la_upper_num = 0.0
        for ph in phases:
            d = _analyze_component_mismatches(phase=ph, cand_key=best_cand.key)
            la_upper_num += float(d.get("la_upper_norm", 0.0)) * float(d.get("N", 0))
        la_upper = la_upper_num / float(combined_diag["N"] or 1)

        lines.append(f"- 样本数 N={combined_diag['N']}，原始 LA={pct(la_raw)}（命中 {combined_diag['hits_raw']}/{combined_diag['N']}）")
        lines.append(
            "- 命名后缀归一化（仅用于解释，不改变 judge 规则）可带来的理论上限：约 "
            + pct(la_upper)
        )
        lines.append(
            f"- 纯后缀错配计数：GT 为 `*-0` 但预测 base（可选 pod 才能修复）={combined_diag['suffix_fixable_add0']}；预测为 `*-0` 但 GT 为 base（可通过 strip 修复）={combined_diag['suffix_fixable_strip0']}\n"
        )

        cat = combined_diag["categories"]
        lines.append("| 类别 | 计数 | 解释 |")
        lines.append("|---|---:|---|")
        lines.append(f"| HIT | {int(cat.get('HIT', 0))} | 预测 component 与 GT parts 任一端完全相等 |")
        lines.append(f"| GT_NODE__PRED_SERVICE | {int(cat.get('GT_NODE__PRED_SERVICE', 0))} | GT 是 node/TiDB，预测成服务 |")
        lines.append(f"| GT_SERVICE__PRED_NODE | {int(cat.get('GT_SERVICE__PRED_NODE', 0))} | GT 是服务，预测成 node/TiDB |")
        lines.append(f"| SAME_KIND_BUT_WRONG | {int(cat.get('SAME_KIND_BUT_WRONG', 0))} | 同类型但选错对象（服务/节点内互相混淆） |")
        lines.append("")

        lines.append("Top 混淆对（GT -> Pred，按出现次数）")
        lines.append("| 次数 | GT | Pred |")
        lines.append("|---:|---|---|")
        for (gt, pred), c in top_pairs:
            lines.append(f"| {c} | {gt} | {pred} |")
        lines.append("")

    # Per-date details
    lines.append("## 分 phase 分日期明细\n")
    lines.append("| phase | date | N | 方案key | LA | TA | Explain | Eff | Final |")
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|")
    for ph in phases:
        for d in ph.dates:
            for cr in results:
                m = cr.per_phase_per_date.get(ph.phase_name, {}).get(d)
                if not m:
                    continue
                lines.append(
                    "| {phase} | {date} | {n:.0f} | {key} | {la} | {ta} | {ex} | {ef} | {fi:.2f} |".format(
                        phase=ph.phase_name,
                        date=d,
                        n=float(m.get("N", 0.0)),
                        key=cr.candidate.key,
                        la=pct(float(m.get("component_accuracy", 0.0))),
                        ta=pct(float(m.get("reason_accuracy", 0.0))),
                        ex=pct(float(m.get("explainability", 0.0))),
                        ef=pct(float(m.get("efficiency", 0.0))),
                        fi=float(m.get("final_score", 0.0)),
                    )
                )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    phases = [
        PhaseConfig("phase1", args.phase1_metadata, args.phase1_ground_truth, list(args.phase1_dates)),
        PhaseConfig("phase2", args.phase2_metadata, args.phase2_ground_truth, list(args.phase2_dates)),
    ]

    candidates = build_candidates(args.preset)
    if args.candidate_keys:
        wanted = set(args.candidate_keys)
        candidates = [c for c in candidates if c.key in wanted]

    suite_slug = args.suite_slug
    exp_root = Path(__file__).resolve().parents[1] / "outputs" / "experiments" / suite_slug
    docs_root = Path(__file__).resolve().parents[1] / "docs" / "对比试验" / args.suite_name
    report_path = args.output_report or (docs_root / "报告.md")

    orchestrator = ContestOrchestrator(telemetry_root=args.telemetry_root)

    # Stream per phase/date to keep memory bounded and allow partial progress.
    results: List[CandidateResult] = [
        CandidateResult(candidate=c, per_phase_per_date={ph.phase_name: {} for ph in phases}) for c in candidates
    ]
    results_by_key = {r.candidate.key: r for r in results}

    for ph in phases:
        for d in ph.dates:
            print(f"[RUN] {ph.phase_name} {d} ...")
            entries_by_candidate: Dict[str, List[Dict[str, Any]]] = {c.key: [] for c in candidates}

            cases = list(_iter_cases_for_date(ph.metadata, d))
            print(f"  cases: {len(cases)}")
            for i, case in enumerate(cases, start=1):
                if i % 5 == 0 or i == 1 or i == len(cases):
                    print(f"  - case {i}/{len(cases)}: {case.uuid}")
                cand_entries = _run_all_candidates_for_case(
                    orchestrator,
                    case=case,
                    candidates=candidates,
                    phase_name=ph.phase_name,
                    recompute_consensus_per_candidate=bool(args.recompute_consensus_per_candidate),
                )
                for cand_key, entry in cand_entries.items():
                    entries_by_candidate[cand_key].append(entry)

            # Write and score per candidate for this (phase, date)
            for cand in candidates:
                entries = entries_by_candidate[cand.key]
                out_jsonl = exp_root / f"sub_{ph.phase_name}_{d}_n{len(entries)}_{cand.key}.jsonl"
                _write_jsonl(out_jsonl, entries)
                metrics = _evaluate_with_judge(ph.ground_truth, out_jsonl) if entries else {
                    "component_accuracy": 0.0,
                    "reason_accuracy": 0.0,
                    "explainability": 0.0,
                    "efficiency": 0.0,
                    "final_score": 0.0,
                }
                metrics = dict(metrics)
                metrics["N"] = float(len(entries))
                results_by_key[cand.key].per_phase_per_date[ph.phase_name][d] = metrics

                la = float(metrics.get("component_accuracy", 0.0)) * 100.0
                ta = float(metrics.get("reason_accuracy", 0.0)) * 100.0
                final = float(metrics.get("final_score", 0.0))
                print(f"  [SCORE] {cand.key} LA={la:.2f}% TA={ta:.2f}% Final={final:.2f}")

    _write_report(
        report_path=report_path,
        suite_name=args.suite_name,
        suite_slug=suite_slug,
        exp_root=exp_root,
        phases=phases,
        candidates=candidates,
        results=results,
    )

    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
