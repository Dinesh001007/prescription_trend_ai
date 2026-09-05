"""Best-effort online learner for columns the local schema mapper cannot name."""

import os
import re
from typing import Any, Dict, Optional

import requests

from utils.data_profiling import CANONICAL_FIELDS
from utils.db import get_column_meaning, save_column_meaning


class ColumnMeaningAgent:
    """Looks up unknown column names and stores reusable, confidence-scored memory."""

    def __init__(self, timeout: float = 3.0):
        self.timeout = timeout

    @staticmethod
    def signature(column_name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "_", column_name.lower()).strip("_")

    def resolve(self, column_name: str, profile: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        signature = self.signature(column_name)
        learned = get_column_meaning(signature)
        if learned and float(learned["confidence"]) >= 0.80:
            return learned

        if os.getenv("COLUMN_MEANING_ONLINE_LOOKUP", "true").lower() not in {"1", "true", "yes"}:
            return None

        evidence = self._search(column_name)
        if not evidence:
            return None
        result = self._analyze(column_name, profile, evidence)
        if not result or result["canonical"] == "OTHER" or result["confidence"] < 0.80:
            return None

        save_column_meaning(signature, column_name, source="online", **result)
        return get_column_meaning(signature)

    def teach(self, column_name: str, canonical: str, profile: Optional[Dict[str, Any]] = None) -> None:
        canonical = canonical.upper()
        if canonical not in CANONICAL_FIELDS and canonical != "OTHER":
            raise ValueError(f"Unsupported canonical field: {canonical}")
        save_column_meaning(
            self.signature(column_name), column_name, canonical, 1.0,
            meaning=f"User-confirmed meaning: {CANONICAL_FIELDS.get(canonical, {}).get('description', 'Unclassified field')}.",
            usage=f"Use this field as {canonical} in downstream clinical analysis.",
            source="feedback",
        )

    def _search(self, column_name: str) -> str:
        try:
            response = requests.get(
                "https://api.duckduckgo.com/",
                params={"q": f"healthcare dataset column {column_name} meaning", "format": "json", "no_html": 1},
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
            parts = [payload.get("AbstractText", "")]
            for topic in payload.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict):
                    parts.append(topic.get("Text", ""))
            return " ".join(part for part in parts if part)
        except (requests.RequestException, ValueError, TypeError):
            return ""

    def _analyze(self, column_name: str, profile: Dict[str, Any], evidence: str) -> Optional[Dict[str, Any]]:
        text = f"{column_name} {evidence}".lower()
        scores = {}
        for canonical, info in CANONICAL_FIELDS.items():
            scores[canonical] = sum(1 for keyword in info["keywords"] if keyword.replace("_", " ") in text)
        canonical, score = max(scores.items(), key=lambda item: item[1])
        if score < 1:
            return None
        confidence = min(0.94, 0.80 + score * 0.04)
        return {
            "canonical": canonical,
            "confidence": confidence,
            "meaning": CANONICAL_FIELDS[canonical]["description"],
            "usage": f"Use '{column_name}' as {canonical} for downstream analysis; verify this mapping before clinical decisions.",
        }