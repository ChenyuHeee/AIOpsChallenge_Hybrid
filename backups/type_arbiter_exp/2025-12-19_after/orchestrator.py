"""Contest orchestrator blending Polaris, OpenRCA, and downloaded resources."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List

import pandas as pd

from .agents.consensus import ConsensusOrchestrator
from .agents.planner import PlannerAgent
from .agents.reasoning import ReasoningAgent
from .agents.specialists import GraphSpecialist, LogSpecialist, MetricSpecialist, SpecialistContext, TraceSpecialist
from .agents.type_arbiter import TypeArbiterAgent
from .agents.validator import SubmissionEntry, SubmissionValidator
from .config import KnowledgeConfig, LLMConfig, PipelineConfig
from .data.loader import TelemetryLoader
from .knowledge.insights import InsightRepository
from .llm.factory import build_client
from .utils.hypothesis import Hypothesis

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CaseMetadata:
    uuid: str
    date: str
    query: str

    @classmethod
    def from_row(cls, row: pd.Series) -> "CaseMetadata":
        return cls(uuid=str(row["uuid"]).strip(), date=str(row["date"]).strip(), query=str(row["query"]).strip())


class ContestOrchestrator:
    def __init__(
        self,
        telemetry_root: Path,
        *,
        llm_config: LLMConfig | None = None,
        pipeline_config: PipelineConfig | None = None,
        knowledge_config: KnowledgeConfig | None = None,
    ) -> None:
        self.telemetry_root = telemetry_root
        self.pipeline_config = pipeline_config or PipelineConfig.from_env()
        knowledge_config = knowledge_config or KnowledgeConfig.from_env()
        self.loader = TelemetryLoader()
        self.insight_repo = InsightRepository(knowledge_config.paper_insights_path)
        self.planner = PlannerAgent(self.insight_repo)
        self.metric_specialist = MetricSpecialist()
        self.log_specialist = LogSpecialist()
        self.trace_specialist = TraceSpecialist()
        self.graph_specialist = GraphSpecialist()
        self.consensus = ConsensusOrchestrator(self.pipeline_config.memory_path)
        llm_client = build_client(llm_config or LLMConfig.from_env())
        self.reasoning = ReasoningAgent(llm_client, self.pipeline_config)
        self.type_arbiter = TypeArbiterAgent(llm_client)
        self.validator = SubmissionValidator(self.pipeline_config)

    def iter_cases(self, metadata_path: Path) -> Iterator[CaseMetadata]:
        df = pd.read_csv(metadata_path)
        for _, row in df.iterrows():
            yield CaseMetadata.from_row(row)

    def run_case(self, case: CaseMetadata) -> SubmissionEntry:
        plan = self.planner.build_plan(case.uuid, case.query)
        telemetry = self.loader.load(self.telemetry_root, case.date, plan.time_window)
        ctx = SpecialistContext(keywords=plan.keywords, component_hints=plan.component_hints)
        hypotheses = self._run_specialists(telemetry, ctx)
        consensus_result = self.consensus.vote(case.uuid, hypotheses, component_hints=plan.component_hints)
        # Strategy: keep component selection evidence-driven (consensus top-1) to preserve LA.
        # Use LLM primarily to improve TA via better reason/trace generation.
        consensus_component = (
            consensus_result.ranked_components[0][0] if consensus_result.ranked_components else "unknown"
        )
        hypothesis_bank = self._group_by_component(hypotheses)

        # Optional: type arbitration to reduce systematic node/service/TiDB confusion.
        chosen_component = consensus_component
        ranked_for_reasoning = list(consensus_result.ranked_components)
        if self.type_arbiter.enabled():
            decision = self.type_arbiter.decide(uuid=case.uuid, query=case.query, consensus=consensus_result)
            chosen_component, ranked_for_reasoning = self.type_arbiter.apply(
                ranked_components=ranked_for_reasoning,
                consensus=consensus_result,
                current_component=chosen_component,
                decision=decision,
            )
            # Keep the chosen component at top to nudge LLM reason generation (even when component_source=consensus).
            if ranked_for_reasoning and ranked_for_reasoning[0][0] != chosen_component:
                for i, (c, s) in enumerate(ranked_for_reasoning):
                    if c == chosen_component:
                        ranked_for_reasoning.insert(0, ranked_for_reasoning.pop(i))
                        break

        reasoning = self.reasoning.run(
            uuid=case.uuid,
            query=case.query,
            plan_steps=plan.sop_steps,
            insights=plan.insights,
            ranked_components=ranked_for_reasoning,
            hypothesis_bank=hypothesis_bank,
        )

        component_source = os.getenv("RCA_COMPONENT_SOURCE", "consensus").strip().lower() or "consensus"
        if component_source == "llm":
            chosen_component = reasoning.component or chosen_component
        else:
            chosen_component = chosen_component

        component, reason, steps = self.validator.enforce_limits(chosen_component, reasoning.reason, reasoning.steps)
        reason = self.validator.enrich_reason(reason, component, hypothesis_bank)
        trace = self.validator.build_trace(steps, component, hypothesis_bank)
        LOGGER.debug("Case %s => component %s", case.uuid, component)
        return SubmissionEntry(uuid=case.uuid, component=component, reason=reason, reasoning_trace=trace)

    def run_cases(self, cases: Iterable[CaseMetadata]) -> List[SubmissionEntry]:
        return [self.run_case(case) for case in cases]

    def write(self, entries: Iterable[SubmissionEntry], output_path: Path) -> None:
        payload = [json.dumps(entry.to_dict(), ensure_ascii=False) for entry in entries]
        output_path.write_text("\n".join(payload) + "\n", encoding="utf-8")

    def _run_specialists(self, telemetry, ctx: SpecialistContext) -> List[Hypothesis]:
        collected: List[Hypothesis] = []
        for specialist in (
            self.metric_specialist,
            self.log_specialist,
            self.trace_specialist,
            self.graph_specialist,
        ):
            try:
                outputs = specialist.run(telemetry, ctx)
            except Exception as exc:  # noqa: BLE001
                LOGGER.error("Specialist %s failed: %s", specialist.__class__.__name__, exc)
                outputs = []
            collected.extend(outputs)
        return collected

    def _group_by_component(self, hypotheses: List[Hypothesis]):
        bank: dict[str, List[Hypothesis]] = {}
        for hypothesis in hypotheses:
            bank.setdefault(hypothesis.component, []).append(hypothesis)
        return bank
