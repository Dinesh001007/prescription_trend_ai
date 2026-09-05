"""
Base Tool Contract Module
Defines standard interface, input/output schemas, evaluation criteria, and normalized results.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd

EXTENDED_CLINICAL_FEATURES = {
    "MENOPAUSE_STATUS",
    "TUMOR_GRADE",
    "LYMPH_NODE_COUNT",
    "PROGESTERONE_RECEPTOR",
    "HORMONE_THERAPY",
    "RELAPSE_FREE_SURVIVAL_TIME",
}


class BaseMLTool(ABC):
    """
    Standardized ML Tool Contract.
    Every tool implements dynamic candidate model competition, objective mathematical scoring,
    and returns a strictly normalized result structure.
    """

    def __init__(self, name: str, purpose: str):
        self.name = name
        self.purpose = purpose
        self.required_semantic_fields: List[str] = []
        self.optional_fields: List[str] = []
        self.candidate_models: List[str] = []
        self.evaluation_metrics: List[str] = []

    @abstractmethod
    def run(self, df, canonical_map: Dict[str, str], **kwargs) -> Dict[str, Any]:
        """
        Executes candidate model competition and returns normalized result dictionary.
        """
        pass

    def create_normalized_result(
        self,
        model_name: str,
        status: str,
        inputs: List[str],
        metrics: Dict[str, Any],
        findings: List[str],
        warnings: List[str],
        evidence: List[str],
        figures: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        duration_ms: float = 0.0,
        leaderboard: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Standard Output Contract conforming to Section 8 & 9 of the architecture plan.
        """
        return {
            "tool": self.name,
            "model": model_name,
            "status": status,
            "inputs": inputs,
            "metrics": metrics,
            "findings": findings,
            "summary": summary or ("\n\n".join([f"• {f}" for f in findings]) if findings else "Analysis completed successfully."),
            "figures": figures or [],
            "warnings": warnings,
            "evidence": evidence,
            "data": data or {},
            "execution_time_ms": round(duration_ms, 2),
            "leaderboard": leaderboard or []
        }

    def create_unavailable_result(self, reason: str, missing_fields: List[str]) -> Dict[str, Any]:
        """Graceful degradation result when prerequisites are missing."""
        return {
            "tool": self.name,
            "model": "None (Unavailable)",
            "status": "unavailable",
            "inputs": [],
            "metrics": {},
            "findings": [],
            "summary": f"Tool '{self.name}' unavailable: {reason}",
            "figures": [],
            "warnings": [f"Tool '{self.name}' skipped: {reason}"],
            "evidence": [f"Missing required canonical fields: {missing_fields}"],
            "data": {},
            "execution_time_ms": 0.0,
            "leaderboard": []
        }

    @staticmethod
    def resolve_feature_columns(
        df: pd.DataFrame,
        canonical_map: Dict[str, str],
        allowed_canonical_fields: List[str],
        max_features: int = 6,
    ) -> List[str]:
        """Resolve known features plus safe unknown numeric/categorical features."""
        selected = [
            source for source, canonical in canonical_map.items()
            if (
                canonical in allowed_canonical_fields
                or canonical in EXTENDED_CLINICAL_FEATURES
            ) and source in df.columns
        ]
        for source, canonical in canonical_map.items():
            if canonical == "OTHER" and source in df.columns and BaseMLTool.is_safe_dynamic_feature(df[source]):
                if source not in selected:
                    selected.append(source)
        return selected[:max_features]

    @staticmethod
    def is_safe_dynamic_feature(series: pd.Series) -> bool:
        """Exclude dates, IDs, free text, and constant fields from generic ML inputs."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return False
        clean = series.dropna()
        if clean.empty or series.nunique(dropna=True) <= 1:
            return False
        name = str(series.name or "").lower()
        if any(token in name for token in ("id", "uuid", "guid", "mrn", "ref", "key")):
            return False
        if pd.api.types.is_numeric_dtype(series):
            return True
        cardinality = clean.nunique()
        average_length = clean.astype(str).str.len().mean()
        return 2 <= cardinality <= min(50, max(10, int(len(series) * 0.2))) and average_length <= 40

    @staticmethod
    def resolve_group_column(df: pd.DataFrame, canonical_map: Dict[str, str]) -> Optional[str]:
        """Find a canonical or safe low-cardinality grouping field for pattern mining."""
        for source, canonical in canonical_map.items():
            if canonical == "PATIENT_ID" and source in df.columns:
                return source
        for source, canonical in canonical_map.items():
            if canonical == "OTHER" and source in df.columns and BaseMLTool.is_safe_dynamic_feature(df[source]):
                return source
        return None
