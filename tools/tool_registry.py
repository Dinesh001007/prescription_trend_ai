"""
Tool Registry Module
Maintains dynamic catalogue of all ML analysis tools and validates capability contracts.
"""

from typing import Dict, Any, List, Optional
from tools.base_tool import BaseMLTool
from tools.cohort_tool import CohortTool
from tools.trend_tool import TrendTool
from tools.anomaly_tool import AnomalyTool
from tools.risk_tool import RiskTool
from tools.pattern_tool import PatternTool


class ToolRegistry:
    """
    Central dynamic tool registry for the AI Agent architecture.
    """

    def __init__(self):
        self._tools: Dict[str, BaseMLTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        self.register(CohortTool())
        self.register(TrendTool())
        self.register(AnomalyTool())
        self.register(RiskTool())
        self.register(PatternTool())

    def register(self, tool: BaseMLTool):
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[BaseMLTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "purpose": t.purpose,
                "required_fields": t.required_semantic_fields,
                "candidate_models": t.candidate_models,
                "evaluation_metrics": t.evaluation_metrics
            }
            for t in self._tools.values()
        ]

    def execute_tool(self, tool_name: str, df, canonical_map: Dict[str, str], **kwargs) -> Dict[str, Any]:
        tool = self.get_tool(tool_name)
        if not tool:
            return {
                "tool": tool_name,
                "model": "Unknown",
                "status": "error",
                "inputs": [],
                "metrics": {},
                "findings": [],
                "warnings": [f"Tool '{tool_name}' not found in registry."],
                "evidence": [],
                "data": {},
                "execution_time_ms": 0.0,
                "leaderboard": []
            }
        return tool.run(df, canonical_map, **kwargs)
