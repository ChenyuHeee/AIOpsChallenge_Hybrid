"""Planner agent combining SOP guidance and paper insights."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from ..knowledge.insights import InsightRepository, PaperInsight
from ..sop import INCIDENT_SOP


@dataclass(slots=True)
class CasePlan:
    uuid: str
    query: str
    keywords: List[str]
    component_hints: List[str]
    time_window: Optional[Tuple[datetime, datetime]]
    sop_steps: List[str]
    insights: List[PaperInsight]


class PlannerAgent:
    ISO_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")

    def __init__(self, insight_repo: InsightRepository) -> None:
        self.insight_repo = insight_repo

    def build_plan(self, uuid: str, query: str) -> CasePlan:
        keywords = self._extract_keywords(query)
        hints = [
            kw
            for kw in keywords
            if (
                ("service" in kw)
                or kw.endswith("pod")
                or kw.endswith("db")
                or kw.startswith("aiops-k8s-")
                or kw.startswith("k8s-master")
            )
        ]
        window = self._extract_time_window(query)
        sop_steps = [step.name for step in INCIDENT_SOP]
        matched_insights = self.insight_repo.relevant(query, limit=4)
        return CasePlan(
            uuid=uuid,
            query=query,
            keywords=keywords,
            component_hints=hints,
            time_window=window,
            sop_steps=sop_steps,
            insights=matched_insights,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_.-]+", query)
        return [token.lower() for token in tokens if len(token) > 2]

    def _extract_time_window(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        matches = self.ISO_PATTERN.findall(query)
        if len(matches) < 2:
            return None
        start = datetime.fromisoformat(matches[0].replace("Z", "+00:00"))
        end = datetime.fromisoformat(matches[1].replace("Z", "+00:00"))
        if end < start:
            start, end = end, start
        return start, end
