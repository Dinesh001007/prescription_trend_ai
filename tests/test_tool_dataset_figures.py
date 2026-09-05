import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from tools.trend_tool import TrendTool
from tools.cohort_tool import CohortTool
from tools.anomaly_tool import AnomalyTool
from tools.risk_tool import RiskTool
from tools.pattern_tool import PatternTool

class TestToolDatasetFigures(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n = 80
        start_dt = datetime(2024, 1, 1)
        dates = [start_dt + timedelta(days=int(i * 1.5)) for i in range(n)]
        
        self.df = pd.DataFrame({
            "subject_id": [f"P_{i:03d}" for i in range(n)],
            "medication": np.random.choice(["Metformin", "Atorvastatin", "Lisinopril", "Amlodipine"], n),
            "date": dates,
            "age": np.random.randint(25, 80, n),
            "qty": np.random.randint(10, 100, n),
            "dosage": [f"{np.random.choice([10, 20, 50])}mg" for _ in range(n)],
            "adverse_risk": np.random.choice([0, 1], p=[0.7, 0.3], size=n)
        })
        self.canonical_map = {
            "subject_id": "PATIENT_ID",
            "medication": "DRUG",
            "date": "DATE",
            "age": "AGE",
            "qty": "QUANTITY",
            "dosage": "DOSAGE",
            "adverse_risk": "RISK_SCORE"
        }

    def test_figures_are_dataset_specific(self):
        tools = [
            ("trend", TrendTool()),
            ("cohort", CohortTool()),
            ("anomaly", AnomalyTool()),
            ("risk", RiskTool()),
            ("pattern", PatternTool())
        ]
        
        for name, tool in tools:
            res = tool.run(self.df, self.canonical_map)
            self.assertEqual(res["status"], "success")
            figures = res.get("figures", [])
            self.assertGreater(len(figures), 0, f"Tool {name} produced no figures")
            
            # Verify no figure is a tool performance comparison
            for title, fig in figures:
                self.assertNotIn("Model Evaluation", title, f"Tool {name} still has tool evaluation chart in figures")
                self.assertNotIn("Model Performance", title, f"Tool {name} still has tool performance chart in figures")
                self.assertNotIn("Algorithm Comparison", title, f"Tool {name} still has algorithm comparison chart in figures")
                self.assertNotIn("Separation Contrast", title, f"Tool {name} still has separation score chart in figures")

if __name__ == "__main__":
    unittest.main()
