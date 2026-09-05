"""
Anomaly Detection Tool with Dynamic Model Selection
Competes IsolationForest, LocalOutlierFactor (LOF), OneClassSVM, and Statistical IQR/ZScore.
Identifies abnormal prescribing dosages, outlier frequencies, and clinical anomalies.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from tools.base_tool import BaseMLTool


class AnomalyTool(BaseMLTool):
    def __init__(self):
        super().__init__(name="anomaly", purpose="Clinical Outlier & Abnormal Prescribing Detection")
        self.required_semantic_fields = ["AGE", "QUANTITY", "DOSAGE", "RISK_SCORE"]
        self.candidate_models = ["IsolationForest", "LocalOutlierFactor (LOF)", "OneClassSVM", "Statistical IQR-Envelope"]
        self.evaluation_metrics = ["anomaly_count", "contamination_rate", "anomaly_separation_score", "stability_score"]

    def run(self, df: pd.DataFrame, canonical_map: Dict[str, str], **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Gather numerical / cohort features
        features = self.resolve_feature_columns(
            df, canonical_map, ["AGE", "QUANTITY", "DOSAGE", "RISK_SCORE"], max_features=6
        )

        # Fallback to numerical columns when no mapped feature is usable.
        if len(features) < 1:
            features = df.select_dtypes(include=[np.number]).columns.tolist()[:6]

        if len(features) < 1 or len(df) < 8:
            return self.create_unavailable_result("Insufficient numeric dimensions or records (< 8) for anomaly detection.", ["Numeric columns"])

        working_df = df[features].copy()
        for col in working_df.columns:
            if working_df[col].dtype == object or str(working_df[col].dtype) == "category":
                # Try extracting numeric part (e.g. '20mg' -> 20.0)
                extracted_nums = working_df[col].astype(str).str.extract(r"(\d+\.?\d*)")[0]
                num_conv = pd.to_numeric(extracted_nums, errors="coerce")
                if num_conv.notna().sum() > len(working_df) * 0.5:
                    working_df[col] = num_conv.fillna(num_conv.median() if num_conv.notna().sum() > 0 else 0)
                else:
                    working_df[col] = pd.factorize(working_df[col].astype(str))[0]
            else:
                num_conv = pd.to_numeric(working_df[col], errors="coerce")
                working_df[col] = num_conv.fillna(num_conv.median() if num_conv.notna().sum() > 0 else 0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(working_df)
        n_samples = len(X_scaled)
        contamination = 0.05

        candidates = []

        # --- Candidate A: Isolation Forest ---
        try:
            iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
            preds_iso = iso.fit_predict(X_scaled)
            scores_iso = -iso.decision_function(X_scaled)
            candidates.append(self._evaluate_anomaly("IsolationForest", preds_iso, scores_iso))
        except Exception as e:
            candidates.append({"model": "IsolationForest", "valid": False, "error": str(e)})

        # --- Candidate B: Local Outlier Factor (LOF) ---
        try:
            n_neighbors = min(20, max(3, n_samples // 4))
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            preds_lof = lof.fit_predict(X_scaled)
            scores_lof = -lof.negative_outlier_factor_
            candidates.append(self._evaluate_anomaly("LocalOutlierFactor (LOF)", preds_lof, scores_lof))
        except Exception as e:
            candidates.append({"model": "LocalOutlierFactor (LOF)", "valid": False, "error": str(e)})

        # --- Candidate C: One-Class SVM ---
        try:
            oc_svm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
            preds_svm = oc_svm.fit_predict(X_scaled)
            scores_svm = -oc_svm.decision_function(X_scaled).flatten()
            candidates.append(self._evaluate_anomaly("OneClassSVM", preds_svm, scores_svm))
        except Exception as e:
            candidates.append({"model": "OneClassSVM", "valid": False, "error": str(e)})

        # --- Candidate D: Statistical IQR / Z-Score Envelope ---
        try:
            z_scores = np.abs(X_scaled)
            max_z = np.max(z_scores, axis=1)
            preds_iqr = np.where(max_z > 2.8, -1, 1)
            candidates.append(self._evaluate_anomaly("Statistical IQR-Envelope", preds_iqr, max_z))
        except Exception as e:
            candidates.append({"model": "Statistical IQR-Envelope", "valid": False, "error": str(e)})

        # 2. Select Winner: Best Anomaly Separation Score
        valid_candidates = [c for c in candidates if c.get("valid", False)]
        if not valid_candidates:
            # Fallback
            winner = candidates[0]
            preds = np.ones(n_samples)
        else:
            valid_candidates.sort(key=lambda x: x.get("separation_score", 0.0), reverse=True)
            winner = valid_candidates[0]
            preds = winner["preds"]

        anomaly_indices = np.where(preds == -1)[0].tolist()
        anomaly_count = len(anomaly_indices)
        anomaly_rate_pct = round((anomaly_count / max(n_samples, 1)) * 100, 2)

        duration = (time.time() - start_time) * 1000

        findings = [
            f"Detected {anomaly_count} statistical prescription anomalies ({anomaly_rate_pct}% of total records).",
            f"Top contributing risk dimensions: {', '.join(features[:3])}.",
            f"Anomaly separation clarity scored at {winner.get('separation_score', 0.0):.2f} standard deviations."
        ]

        evidence = [
            f"Model Selection Winner: {winner['model']} selected for highest outlier separation contrast.",
            f"Candidate comparison: " + ", ".join([f"{c['model']} (Found {c.get('anomaly_count', 0)} anomalies)" for c in valid_candidates]),
            f"Evaluated features: {', '.join(features)}."
        ]

        warnings = []
        if anomaly_rate_pct > 15:
            warnings.append(f"Elevated outlier density ({anomaly_rate_pct}% flagged). Review data collection pipelines for calibration errors.")

        leaderboard = [
            {
                "model": c["model"],
                "valid": c.get("valid", False),
                "anomaly_count": c.get("anomaly_count", None),
                "separation_score": c.get("separation_score", None),
                "is_winner": c["model"] == winner["model"]
            }
            for c in candidates
        ]

        # --- Build Interactive Plotly Figures ---
        figures = []
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.decomposition import PCA

            # Figure 1: 2D Anomaly PCA Projection
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_scaled)
                x_title = f"Component 1 ({int(pca.explained_variance_ratio_[0]*100)}% var)"
                y_title = f"Component 2 ({int(pca.explained_variance_ratio_[1]*100)}% var)"
            else:
                coords = X_scaled[:, :2]
                x_title = features[0]
                y_title = features[1] if len(features) > 1 else "Index"

            status_labels = ["⚠️ Flagged Anomaly" if p == -1 else "Normal Prescribing" for p in preds]
            anom_plot_df = pd.DataFrame({
                "x": coords[:, 0],
                "y": coords[:, 1],
                "Status": status_labels
            })
            for col in features[:3]:
                anom_plot_df[col] = df[col].values[:len(anom_plot_df)]

            fig_scatter = px.scatter(
                anom_plot_df, x="x", y="y", color="Status",
                hover_data=features[:3],
                title=f"🔍 Prescription Anomaly & Outlier Distribution ({winner['model']})",
                template="plotly_dark",
                color_discrete_map={"Normal Prescribing": "#00E5BE", "⚠️ Flagged Anomaly": "#EF4444"}
            )
            fig_scatter.update_layout(
                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                xaxis_title=x_title, yaxis_title=y_title,
                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            figures.append(("🔍 Prescription Anomaly Distribution", fig_scatter))

            # Figure 2: Anomaly Ratio Breakdown
            anom_ratio_df = pd.DataFrame([
                {"Category": "Normal Prescriptions", "Count": n_samples - anomaly_count},
                {"Category": "Flagged Outliers", "Count": anomaly_count}
            ])
            fig_ratio = px.pie(
                anom_ratio_df, names="Category", values="Count",
                hole=0.45,
                title="⚠️ Outlier vs Normal Cohort Proportions",
                template="plotly_dark",
                color_discrete_map={"Normal Prescriptions": "#00E5BE", "Flagged Outliers": "#EF4444"}
            )
            fig_ratio.update_layout(
                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
            )
            figures.append(("⚠️ Outlier Proportion Breakdown", fig_ratio))

            # Figure 3: Clinical Feature Comparison (Normal vs Flagged Outliers) (Dataset)
            try:
                comp_df = working_df.copy()
                comp_df["Status"] = ["⚠️ Flagged Anomaly" if p == -1 else "Normal Prescribing" for p in preds]
                num_comp_cols = [c for c in features[:4] if pd.api.types.is_numeric_dtype(comp_df[c])]
                if num_comp_cols:
                    comp_means = comp_df.groupby("Status")[num_comp_cols].mean().reset_index()
                    comp_melted = pd.melt(comp_means, id_vars=["Status"], value_vars=num_comp_cols, var_name="Clinical Metric", value_name="Mean Value")

                    fig_comp = px.bar(
                        comp_melted, x="Clinical Metric", y="Mean Value", color="Status",
                        barmode="group",
                        title="📊 Clinical Metric Deviation: Normal vs Flagged Anomalies",
                        template="plotly_dark",
                        color_discrete_map={"Normal Prescribing": "#00E5BE", "⚠️ Flagged Anomaly": "#EF4444"}
                    )
                    fig_comp.update_layout(
                        paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                        xaxis_title="Clinical Dimension", yaxis_title="Mean Value in Dataset",
                        font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                    )
                    figures.append(("📊 Anomaly Feature Deviation Profile", fig_comp))
            except Exception:
                pass

            # Figure 4: Flagged Anomalies by Drug (Dataset)
            try:
                drug_cols = [src for src, can in canonical_map.items() if can == "DRUG" and src in df.columns]
                if drug_cols:
                    d_col = drug_cols[0]
                    anom_drug_df = pd.DataFrame({
                        "Drug": df[d_col].astype(str).str.title(),
                        "Is_Anomaly": preds == -1
                    })
                    anom_by_drug = anom_drug_df[anom_drug_df["Is_Anomaly"]]["Drug"].value_counts().head(8).reset_index()
                    anom_by_drug.columns = ["Medication", "Anomaly Count"]
                    if not anom_by_drug.empty:
                        fig_drug_anom = px.bar(
                            anom_by_drug, x="Medication", y="Anomaly Count",
                            title="💊 Flagged Outlier Prescriptions by Medication",
                            template="plotly_dark",
                            color="Anomaly Count",
                            color_continuous_scale="Reds"
                        )
                        fig_drug_anom.update_layout(
                            paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                            xaxis_title="Medication", yaxis_title="Flagged Anomaly Count",
                            font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                        )
                        figures.append(("💊 Anomalies by Therapeutic Agent", fig_drug_anom))
            except Exception:
                pass
        except Exception:
            pass

        # Top 5 Anomaly Samples for UI preview
        anomaly_records = []
        for idx in anomaly_indices[:5]:
            rec = {col: str(df.iloc[idx][col]) for col in features[:4]}
            rec["Anomaly_Status"] = "⚠️ Flagged Outlier"
            anomaly_records.append(rec)

        return self.create_normalized_result(
            model_name=winner["model"],
            status="success",
            inputs=features,
            metrics={
                "anomaly_count": anomaly_count,
                "anomaly_rate_pct": anomaly_rate_pct,
                "separation_score": winner.get("separation_score", 0.0),
                "evaluated_records": n_samples
            },
            findings=findings,
            warnings=warnings,
            evidence=evidence,
            figures=figures,
            data={
                "anomaly_count": anomaly_count,
                "anomaly_rate_pct": anomaly_rate_pct,
                "anomaly_records": anomaly_records,
                "features": features
            },
            duration_ms=duration,
            leaderboard=leaderboard
        )

    def _evaluate_anomaly(self, name: str, preds: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
        anom_mask = preds == -1
        anom_count = int(np.sum(anom_mask))
        n_samples = len(preds)
        
        # Calculate separation contrast between normal vs outlier score distributions
        if 0 < anom_count < n_samples:
            mean_anom = np.mean(scores[anom_mask])
            mean_norm = np.mean(scores[~anom_mask])
            std_norm = np.std(scores[~anom_mask]) + 1e-4
            separation = float(round((mean_anom - mean_norm) / std_norm, 3))
        else:
            separation = 0.0

        return {
            "model": name,
            "valid": True,
            "anomaly_count": anom_count,
            "separation_score": separation,
            "preds": preds,
            "scores": scores
        }
