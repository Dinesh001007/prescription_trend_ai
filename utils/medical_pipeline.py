import pandas as pd
import numpy as np
import time
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from utils.medical_ontology import SemanticCategory, MEDICAL_ONTOLOGY
from utils.schema_analyzer import SchemaAnalyzer, ColumnType

logger = logging.getLogger(__name__)

class MedicalDataPipeline:
    """
    Strict 5-step pipeline for analyzing unknown medical datasets.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_mapping = {}
        self.preprocessed_df = None
        self.analysis_results = {}
        self.pipeline_stats = {
            "start_time": time.time(),
            "steps_completed": []
        }
        self.schema_analyzer = SchemaAnalyzer()

    # ==================================================
    # STEP 1: ADVANCED COLUMN UNDERSTANDING
    # ==================================================
    def step1_column_understanding(self) -> Dict[str, Any]:
        """
        Dynamically infer meaning using medical ontology.
        """
        logger.info("Step 1: Advanced Column Understanding")
        mapping_table = []
        
        for col in self.df.columns:
            # 1. Profiling
            series = self.df[col]
            dtype = str(series.dtype)
            unique_count = series.nunique()
            null_count = series.isnull().sum()
            sample_values = series.dropna().head(3).tolist()
            
            # 2. Semantic Mapping
            category, confidence, reasoning = self._map_to_category(col, series)
            
            self.column_mapping[col] = {
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning,
                "dtype": dtype
            }
            
            mapping_table.append({
                "Column Name": col,
                "Data Type": dtype,
                "Assigned Category": category.value,
                "Confidence Score": round(confidence, 2),
                "Reasoning": reasoning
            })
            
        self.pipeline_stats["steps_completed"].append("STEP 1: COLUMN UNDERSTANDING")
        return pd.DataFrame(mapping_table)

    def _map_to_category(self, col_name: str, series: pd.Series) -> Tuple[SemanticCategory, float, str]:
        """
        Logic for semantic category mapping.
        """
        col_lower = col_name.lower()
        best_cat = SemanticCategory.UNKNOWN
        max_score = 0.0
        reasoning = "No reliable mapping possible."

        # Strategy 1: Keyword Matching
        for cat, info in MEDICAL_ONTOLOGY.items():
            score = 0.0
            matched_keywords = []
            for kw in info["keywords"]:
                if kw in col_lower:
                    score += 0.5
                    matched_keywords.append(kw)
            
            if score > max_score:
                max_score = score
                best_cat = cat
                reasoning = f"Matched keywords: {', '.join(matched_keywords)}"

        # Strategy 2: Value Range / Domain Heuristics
        if pd.api.types.is_numeric_dtype(series):
            # BP check
            if "bp" in col_lower or "systolic" in col_lower or "diastolic" in col_lower:
                if series.median() > 60 and series.median() < 200:
                    best_cat = SemanticCategory.VITAL_SIGNS
                    max_score = max(max_score, 0.9)
                    reasoning = "Numeric range (60-200) and name suggest blood pressure."
            
            # Heart rate check
            if "hr" in col_lower or "pulse" in col_lower:
                if series.median() > 40 and series.median() < 220:
                    best_cat = SemanticCategory.VITAL_SIGNS
                    max_score = max(max_score, 0.9)
                    reasoning = "Numeric range (40-220) and name suggest heart rate."

            # Age check
            if "age" in col_lower:
                if series.max() < 120:
                    best_cat = SemanticCategory.DEMOGRAPHICS
                    max_score = max(max_score, 0.95)
                    reasoning = "Numeric range (0-120) and name suggest patient age."

        # Strategy 3: Formatting (Dates)
        if best_cat == SemanticCategory.UNKNOWN:
            try:
                if self.schema_analyzer._is_datetime_column(series, col_name):
                    best_cat = SemanticCategory.TEMPORAL
                    max_score = 0.8
                    reasoning = "Values conform to datetime patterns."
            except:
                pass

        # Final adjustments
        if max_score > 1.0: max_score = 1.0
        if max_score < 0.3:
            best_cat = SemanticCategory.UNKNOWN
            reasoning = "Low confidence in categorical/numeric mapping."

        return best_cat, max_score, reasoning

    # ==================================================
    # STEP 2: DATA PREPROCESSING
    # ==================================================
    def step2_preprocessing(self) -> Dict[str, Any]:
        """
        Handle missing values, normalize, and encode.
        """
        logger.info("Step 2: Data Preprocessing")
        df_clean = self.df.copy()
        preprocessing_log = []

        for col in df_clean.columns:
            series = df_clean[col]
            cat = self.column_mapping[col]["category"]
            
            # Handle missing values
            if series.isnull().any():
                if pd.api.types.is_numeric_dtype(series):
                    df_clean[col] = series.fillna(series.median())
                    preprocessing_log.append(f"Imputed '{col}' with median.")
                else:
                    df_clean[col] = series.fillna(series.mode()[0] if not series.mode().empty else "UNKNOWN")
                    preprocessing_log.append(f"Imputed '{col}' with mode.")

            # Normalize numeric features (except IDs and dates)
            if pd.api.types.is_numeric_dtype(series) and cat not in [SemanticCategory.IDENTIFIERS, SemanticCategory.TEMPORAL]:
                scaler = StandardScaler()
                df_clean[col] = scaler.fit_transform(df_clean[[col]])
                preprocessing_log.append(f"Normalized '{col}'.")

            # Encode categorical features
            if not pd.api.types.is_numeric_dtype(series) and cat not in [SemanticCategory.CLINICAL_NOTES, SemanticCategory.TEMPORAL]:
                le = LabelEncoder()
                df_clean[col] = le.fit_transform(df_clean[col].astype(str))
                preprocessing_log.append(f"Encoded '{col}'.")

        self.preprocessed_df = df_clean
        self.pipeline_stats["steps_completed"].append("STEP 2: PREPROCESSING")
        return {"preprocessing_log": preprocessing_log}

    # ==================================================
    # STEP 3: AGENT EXECUTION (Orchestration)
    # ==================================================
    def step3_agent_execution(self, agents_config: Dict[str, bool]) -> Dict[str, Any]:
        """
        Execute specialized agents using mapped semantic categories.
        """
        logger.info("Step 3: Agent Execution")
        results = {}
        
        # Helper to get columns by category
        def get_cols(category: SemanticCategory):
            return [col for col, info in self.column_mapping.items() if info["category"] == category]

        # 1. RiskAgent
        if agents_config.get("risk", True):
            from agents.risk_agent_improved import run_risk_agent_improved
            # Map column categories to the agent's expected simplified col_map
            simplified_map = {}
            for col, info in self.column_mapping.items():
                cat = info["category"]
                if cat == SemanticCategory.IDENTIFIERS: simplified_map[col] = "patient_id"
                elif cat == SemanticCategory.MEDICATIONS: simplified_map[col] = "drug_name"
                elif cat == SemanticCategory.TEMPORAL: simplified_map[col] = "date"
                elif cat == SemanticCategory.DEMOGRAPHICS and "age" in col.lower(): simplified_map[col] = "age"
                elif cat == SemanticCategory.DIAGNOSIS: simplified_map[col] = "diagnosis"
                else: simplified_map[col] = "other"
            
            results["risk"] = run_risk_agent_improved(self.df, simplified_map)

        # 2. CohortAgent
        if agents_config.get("cohort", True):
            from agents.cohort_agent_advanced import run_cohort_agent_advanced
            results["cohort"] = run_cohort_agent_advanced(self.df, simplified_map)

        # 3. AnomalyAgent
        if agents_config.get("anomaly", True):
            from agents.anomaly_agent_improved import run_anomaly_agent_improved
            results["anomaly"] = run_anomaly_agent_improved(self.df, simplified_map)

        # 4. TrendAgent
        temporal_cols = get_cols(SemanticCategory.TEMPORAL)
        if agents_config.get("trend", True) and temporal_cols:
            from agents.trend_agent import run_trend_agent
            results["trend"] = run_trend_agent(self.df, simplified_map)

        # 5. PatternAgent
        if agents_config.get("pattern", True):
            from agents.pattern_agent import run_pattern_agent
            results["pattern"] = run_pattern_agent(self.df, simplified_map)

        self.analysis_results = results
        self.pipeline_stats["steps_completed"].append("STEP 3: AGENT EXECUTION")
        return results

    # ==================================================
    # STEP 4: PERFORMANCE & QUALITY EVALUATION
    # ==================================================
    def step4_evaluation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Unsupervised evaluation and multi-agent statistical validation.
        """
        logger.info("Step 4: Evaluation")
        eval_metrics = {}
        
        # Analysis Confidence Score calculation
        completeness = 1.0 - (self.df.isnull().sum().sum() / self.df.size)
        mapping_confidence = np.mean([info["confidence"] for info in self.column_mapping.values()])
        
        analysis_confidence = (completeness * 0.4) + (mapping_confidence * 0.6)
        
        eval_metrics["Analysis Confidence Score"] = round(analysis_confidence, 2)
        eval_metrics["Data Completeness"] = round(completeness, 2)
        
        # Multi-agent Statistical Validation
        try:
            from utils.agent_performance_validator import validate_agent_performance
            validation_results = validate_agent_performance(results)
            eval_metrics["statistical_validation"] = validation_results
            logger.info("Multi-agent statistical validation completed")
        except Exception as e:
            logger.error(f"Statistical validation failed: {e}")
            eval_metrics["statistical_validation"] = None
            
        eval_metrics["Ontology Match Rate"] = round(len([c for c in self.column_mapping.values() if c["category"] != SemanticCategory.UNKNOWN]) / len(self.column_mapping), 2)
        
        self.pipeline_stats["steps_completed"].append("STEP 4: EVALUATION")
        return eval_metrics

    # ==================================================
    # STEP 5: FINAL OUTPUT GENERATION
    # ==================================================
    def step5_final_report(self) -> Dict[str, Any]:
        """
        Structured final report.
        """
        return {
            "summary": {
                "rows": len(self.df),
                "cols": len(self.df.columns),
                "steps": self.pipeline_stats["steps_completed"]
            },
            "mapping": self.column_mapping,
            "results": self.analysis_results
        }
