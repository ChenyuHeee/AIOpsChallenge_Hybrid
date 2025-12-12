"""Knowledge base utilities synthesising insights from downloaded resources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass(slots=True)
class PaperInsight:
    paper_id: str
    title: str
    focus: List[str]
    takeaways: List[str]

    def matches(self, query: str) -> bool:
        lowered = query.lower()
        return any(keyword in lowered for keyword in self.focus) or any(keyword in lowered for keyword in self.takeaways)

    def to_prompt_segment(self) -> str:
        focus = ", ".join(self.focus)
        bullets = "\n".join(f"- {point}" for point in self.takeaways[:4])
        return f"Paper: {self.title}. Focus: {focus}. Key takeaways:\n{bullets}"


class InsightRepository:
    def __init__(self, json_path: Path) -> None:
        self.json_path = json_path
        self._insights = self._load(json_path)

    def _load(self, json_path: Path) -> Dict[str, PaperInsight]:
        if not json_path.exists():
            return {}
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        insights: Dict[str, PaperInsight] = {}
        for key, payload in raw.items():
            insights[key] = PaperInsight(
                paper_id=key,
                title=payload.get("title", key),
                focus=payload.get("focus", []),
                takeaways=payload.get("takeaways", []),
            )
        return insights

    def relevant(self, query: str, limit: int = 3) -> List[PaperInsight]:
        matches = [insight for insight in self._insights.values() if insight.matches(query)]
        if not matches:
            matches = list(self._insights.values())
        return matches[:limit]

    def iter_all(self) -> Iterable[PaperInsight]:
        return self._insights.values()
