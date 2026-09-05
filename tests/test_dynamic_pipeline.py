"""
Comprehensive Test Suite for Dynamic AI-Agent Pipeline
Tests arbitrary dataset schemas, model competition, capability matrix, and orchestrator.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from utils.data_profiling import DatasetProfiler
from utils.data_profiling import SemanticMapper
from utils.core_pipeline import CapabilityMatrix
from tools.cohort_tool import CohortTool
from tools.trend_tool import TrendTool
from tools.anomaly_tool import AnomalyTool
from tools.risk_tool import RiskTool
from tools.pattern_tool import PatternTool
from utils.core_pipeline import AgentOrchestrator
from utils.llm_core import AIReasoner


class FakeColumnMeaningAgent:
    def __init__(self):
        self.taught = []

    def resolve(self, column_name, profile):
        if column_name == "custom_measure":
            return {
                "canonical": "DRUG",
                "confidence": 0.91,
                "meaning": "A code identifying a pharmaceutical treatment.",
                "usage": "Use it to group prescriptions by treatment.",
            }
        return None

    def teach(self, column_name, canonical, profile=None):
        self.taught.append((column_name, canonical))


class TestDynamicAIAgentPipeline(unittest.TestCase):

    def setUp(self):
        self.profiler = DatasetProfiler()
        self.mapper = SemanticMapper()
        self.cap_eval = CapabilityMatrix()
        self.orchestrator = AgentOrchestrator()
        self.reasoner = AIReasoner()

        # Create Synthetic Dataset with Non-Standard Column Names
        np.random.seed(42)
        n = 120
        start_dt = datetime(2024, 1, 1)
        dates = [start_dt + timedelta(days=int(i * 1.5)) for i in range(n)]
        
        self.synth_df = pd.DataFrame({
            "subject_ref_code": [f"SUBJ_{i:04d}" for i in range(n)],
            "prescribed_pharmaceutical": np.random.choice(["Metformin", "Atorvastatin", "Lisinopril", "Amlodipine", "Warfarin"], n),
            "dispensed_timestamp": dates,
            "patient_chronological_age": np.random.randint(22, 88, n),
            "biological_sex_type": np.random.choice(["Male", "Female"], n),
            "dispensing_units_qty": np.random.exponential(scale=30, size=n).astype(int) + 1,
            "dose_milligrams": [f"{np.random.choice([5, 10, 20, 50, 500])}mg" for _ in range(n)],
            "adverse_risk_event": np.random.choice([0, 1], p=[0.75, 0.25], size=n)
        })

    def test_1_dataset_profiler(self):
        """Test profiling arbitrary dataset without fixed schema."""
        profile = self.profiler.profile_dataframe(self.synth_df, "synth_prescription_data.csv")
        self.assertEqual(profile["row_count"], 120)
        self.assertEqual(profile["column_count"], 8)
        self.assertGreater(profile["data_quality_score"], 80)
        self.assertTrue("dispensed_timestamp" in profile["columns"])
        self.assertTrue(profile["columns"]["dispensed_timestamp"]["is_date"])
        self.assertTrue(profile["columns"]["patient_chronological_age"]["is_numeric"])

    def test_2_semantic_schema_mapping(self):
        """Test 3-layer semantic mapping on non-standard column names."""
        profile = self.profiler.profile_dataframe(self.synth_df)
        mapping_res = self.mapper.map_columns(self.synth_df, profile["columns"], use_llm=False)
        c_map = mapping_res["canonical_mapping"]

        self.assertEqual(c_map.get("prescribed_pharmaceutical"), "DRUG")
        self.assertEqual(c_map.get("dispensed_timestamp"), "DATE")
        self.assertEqual(c_map.get("subject_ref_code"), "PATIENT_ID")
        self.assertEqual(c_map.get("patient_chronological_age"), "AGE")
        self.assertEqual(c_map.get("biological_sex_type"), "GENDER")
        self.assertEqual(c_map.get("dispensing_units_qty"), "QUANTITY")
        self.assertEqual(c_map.get("dose_milligrams"), "DOSAGE")
        self.assertEqual(c_map.get("adverse_risk_event"), "RISK_SCORE")

    def test_2b_learned_column_meaning_replaces_other(self):
        """A persisted agent result is used before the OTHER fallback."""
        df = pd.DataFrame({"custom_measure": ["RX", "TX", "RX"], "mystery": ["x", "y", "z"]})
        profile = self.profiler.profile_dataframe(df)
        agent = FakeColumnMeaningAgent()
        mapping_res = SemanticMapper(meaning_agent=agent).map_columns(
            df, profile["columns"], use_llm=False
        )

        self.assertEqual(mapping_res["canonical_mapping"]["custom_measure"], "DRUG")
        self.assertEqual(mapping_res["mapping_details"]["custom_measure"]["layer"], "Learned Column Meaning Agent")
        self.assertEqual(mapping_res["canonical_mapping"]["mystery"], "OTHER")

    def test_2e_numeric_non_age_columns_are_not_mapped_as_age(self):
        """Numeric values alone must not turn grade or node counts into AGE."""
        df = pd.DataFrame({"grade": [1, 2, 3, 2], "nodes": [10, 12, 14, 11]})
        profile = self.profiler.profile_dataframe(df)
        mapper = SemanticMapper(meaning_agent=FakeColumnMeaningAgent())
        result = mapper.map_columns(df, profile["columns"], use_llm=False)

        self.assertNotEqual(result["canonical_mapping"]["grade"], "AGE")
        self.assertNotEqual(result["canonical_mapping"]["nodes"], "AGE")

    def test_2f_common_oncology_abbreviations_are_mapped(self):
        """Known oncology abbreviations map locally without online lookup."""
        df = pd.DataFrame({
            "meno": [0, 1], "grade": [1, 2], "nodes": [2, 4],
            "pgr": [0, 1], "hormon": [0, 1], "rfstime": [12, 24],
        })
        profile = self.profiler.profile_dataframe(df)
        result = SemanticMapper(meaning_agent=FakeColumnMeaningAgent()).map_columns(
            df, profile["columns"], use_llm=False
        )

        self.assertEqual(result["canonical_mapping"]["meno"], "MENOPAUSE_STATUS")
        self.assertEqual(result["canonical_mapping"]["grade"], "TUMOR_GRADE")
        self.assertEqual(result["canonical_mapping"]["nodes"], "LYMPH_NODE_COUNT")
        self.assertEqual(result["canonical_mapping"]["pgr"], "PROGESTERONE_RECEPTOR")
        self.assertEqual(result["canonical_mapping"]["hormon"], "HORMONE_THERAPY")
        self.assertEqual(result["canonical_mapping"]["rfstime"], "RELAPSE_FREE_SURVIVAL_TIME")

    def test_2g_new_clinical_concepts_reach_tools(self):
        """Newly mapped clinical concepts are included in generic tool feature matrices."""
        df = self.synth_df.copy()
        df["meno"] = np.random.choice([0, 1], len(df))
        df["grade"] = np.random.choice([1, 2, 3], len(df))
        df["nodes"] = np.random.randint(0, 8, len(df))
        df["rfstime"] = np.arange(len(df)) + 1
        canonical_map = {
            "meno": "MENOPAUSE_STATUS",
            "grade": "TUMOR_GRADE",
            "nodes": "LYMPH_NODE_COUNT",
            "rfstime": "RELAPSE_FREE_SURVIVAL_TIME",
        }

        anomaly_result = AnomalyTool().run(df, canonical_map)
        cohort_result = CohortTool().run(df, canonical_map)
        risk_result = RiskTool().run(df, canonical_map)
        trend_result = TrendTool().run(
            df.assign(event_date=pd.date_range("2024-01-01", periods=len(df))),
            {**canonical_map, "event_date": "DATE"},
        )

        self.assertIn("meno", anomaly_result["inputs"])
        self.assertIn("grade", cohort_result["inputs"])
        self.assertIn("nodes", risk_result["inputs"])
        self.assertIn("rfstime", trend_result["inputs"])

    def test_2d_unknown_features_are_available_to_other_tools(self):
        """Safe unknown fields flow through anomaly, risk, pattern, and trend tools."""
        df = self.synth_df.copy()
        df["care_setting"] = np.random.choice(["outpatient", "inpatient", "emergency"], len(df))
        df["volume_measure"] = np.arange(len(df)) + 1
        canonical_map = {
            "dispensed_timestamp": "DATE",
            "prescribed_pharmaceutical": "DRUG",
            "subject_ref_code": "PATIENT_ID",
            "care_setting": "OTHER",
            "volume_measure": "OTHER",
        }

        anomaly_result = AnomalyTool().run(df, canonical_map)
        risk_result = RiskTool().run(df, canonical_map)
        pattern_result = PatternTool().run(df, canonical_map)
        trend_result = TrendTool().run(df, canonical_map)

        self.assertIn("care_setting", anomaly_result["inputs"])
        self.assertIn("care_setting", risk_result["inputs"])
        self.assertEqual(pattern_result["status"], "success")
        self.assertIn("volume_measure", trend_result["inputs"])

    def test_3_capability_matrix_and_graceful_degradation(self):
        """Test capability evaluation and graceful degradation on missing columns."""
        profile = self.profiler.profile_dataframe(self.synth_df)
        mapping_res = self.mapper.map_columns(self.synth_df, profile["columns"], use_llm=False)
        caps = self.cap_eval.evaluate_capabilities(mapping_res["canonical_mapping"], profile)

        self.assertTrue(caps["capabilities"]["trend"]["feasible"])
        self.assertTrue(caps["capabilities"]["cohort"]["feasible"])
        self.assertTrue(caps["capabilities"]["risk"]["feasible"])
        self.assertTrue(caps["capabilities"]["anomaly"]["feasible"])
        self.assertTrue(caps["capabilities"]["pattern"]["feasible"])

        # Test degradation with truncated dataset lacking DATE
        df_no_date = self.synth_df.drop(columns=["dispensed_timestamp"])
        prof_no_date = self.profiler.profile_dataframe(df_no_date)
        map_no_date = self.mapper.map_columns(df_no_date, prof_no_date["columns"], use_llm=False)
        caps_no_date = self.cap_eval.evaluate_capabilities(map_no_date["canonical_mapping"], prof_no_date)
        
        self.assertFalse(caps_no_date["capabilities"]["trend"]["feasible"])
        self.assertIn("Missing", caps_no_date["capabilities"]["trend"]["reason"])

    def test_4_cohort_tool_model_competition(self):
        """Test dynamic clustering model competition (KMeans vs DBSCAN vs Agglomerative)."""
        profile = self.profiler.profile_dataframe(self.synth_df)
        mapping_res = self.mapper.map_columns(self.synth_df, profile["columns"], use_llm=False)
        
        cohort_tool = CohortTool()
        result = cohort_tool.run(self.synth_df, mapping_res["canonical_mapping"])
        
        self.assertEqual(result["status"], "success")
        self.assertIn(result["model"], ["KMeans", "DBSCAN", "AgglomerativeClustering", "GaussianMixture"])
        self.assertIn("silhouette_score", result["metrics"])
        self.assertGreater(len(result["leaderboard"]), 1)
        self.assertTrue(any(item["is_winner"] for item in result["leaderboard"]))

    def test_5_trend_tool_holdout_validation(self):
        """Test longitudinal forecasting competition on holdout set."""
        profile = self.profiler.profile_dataframe(self.synth_df)
        mapping_res = self.mapper.map_columns(self.synth_df, profile["columns"], use_llm=False)
        
        trend_tool = TrendTool()
        result = trend_tool.run(self.synth_df, mapping_res["canonical_mapping"])
        
        self.assertEqual(result["status"], "success")
        self.assertIn(result["model"], ["Prophet", "ExponentialSmoothing (ETS)", "LinearTrend", "AutoRegressive (ARIMA)"])
        self.assertIn("rmse", result["metrics"])
        self.assertGreater(len(result["leaderboard"]), 0)

    def test_6_anomaly_tool_separation_evaluation(self):
        """Test anomaly tool model selection (IsolationForest, LOF, OneClassSVM)."""
        profile = self.profiler.profile_dataframe(self.synth_df)
        mapping_res = self.mapper.map_columns(self.synth_df, profile["columns"], use_llm=False)
        
        anomaly_tool = AnomalyTool()
        result = anomaly_tool.run(self.synth_df, mapping_res["canonical_mapping"])
        
        self.assertEqual(result["status"], "success")
        self.assertGreaterEqual(result["metrics"]["anomaly_count"], 0)
        self.assertIn("separation_score", result["metrics"])

    def test_7_supervised_vs_unsupervised_risk_routing(self):
        """Test supervised ML when target exists vs unsupervised composite index when target absent."""
        profile = self.profiler.profile_dataframe(self.synth_df)
        mapping_res = self.mapper.map_columns(self.synth_df, profile["columns"], use_llm=False)
        
        risk_tool = RiskTool()
        # Case A: Supervised (target exists)
        res_sup = risk_tool.run(self.synth_df, mapping_res["canonical_mapping"])
        self.assertEqual(res_sup["status"], "success")
        self.assertEqual(res_sup["metrics"]["mode"], "supervised")
        self.assertIn(res_sup["model"], ["XGBoost", "RandomForest", "GradientBoosting", "LogisticRegression"])

        # Case B: Unsupervised (target removed)
        df_no_target = self.synth_df.drop(columns=["adverse_risk_event"])
        map_no_target = {k: v for k, v in mapping_res["canonical_mapping"].items() if k != "adverse_risk_event"}
        res_unsup = risk_tool.run(df_no_target, map_no_target)
        self.assertEqual(res_unsup["status"], "success")
        self.assertEqual(res_unsup["metrics"]["mode"], "unsupervised_composite")

    def test_8_orchestrator_concurrent_execution_and_synthesis(self):
        """Test concurrent multi-tool execution and AI reasoning synthesis."""
        profile = self.profiler.profile_dataframe(self.synth_df)
        mapping_res = self.mapper.map_columns(self.synth_df, profile["columns"], use_llm=False)
        c_map = mapping_res["canonical_mapping"]
        caps = self.cap_eval.evaluate_capabilities(c_map, profile)

        plan = self.orchestrator.plan_execution("Evaluate full prescription trend, risk, and anomaly patterns", caps)
        self.assertGreaterEqual(len(plan["selected_tools"]), 3)

        events_logged = []
        def on_progress(event, data):
            events_logged.append(event)

        exec_res = self.orchestrator.execute_plan(self.synth_df, c_map, plan, progress_callback=on_progress)
        self.assertEqual(exec_res["executed_tool_count"], len(plan["selected_tools"]))
        self.assertIn("tool_started", events_logged)
        self.assertIn("tool_completed", events_logged)

        # Test Evidence-Grounded AI Reasoning Synthesis
        synthesis = self.reasoner._generate_offline_synthesis(
            query="Evaluate full prescription trend",
            metrics_digest=[{"tool": "risk", "winner_model": "XGBoost", "metrics": {"ROC-AUC": 0.85}}],
            evidence_digest=["Cohort 0 (344 patients): Mean Age 68.4 yrs"],
            warnings_digest=[],
            dataset_profile=profile
        )
        self.assertIn("Executive Clinical Summary", synthesis)
        self.assertIn("Key Machine Learning Findings", synthesis)


if __name__ == "__main__":
    unittest.main()
