"""Shared dataclasses for hypothesis generation and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(slots=True)
class EvidenceItem:
    modality: str
    component: str
    summary: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class Hypothesis:
    component: str
    confidence: float
    source: str
    evidence: List[EvidenceItem]

    def merge(self, other: "Hypothesis") -> "Hypothesis":
        total_weight = self.confidence + other.confidence
        if total_weight == 0:
            weight_a = weight_b = 0.5
        else:
            weight_a = self.confidence / total_weight
            weight_b = other.confidence / total_weight
        merged_evidence = self.evidence + other.evidence
        return Hypothesis(
            component=self.component,
            confidence=max(self.confidence, other.confidence) + 0.1,
            source=f"{self.source}+{other.source}",
            evidence=merged_evidence,
        )
