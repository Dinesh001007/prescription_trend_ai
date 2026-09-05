"""
Clinical Risk Tool with Dynamic Model Selection
Uses supervised algorithms (XGBoost, RandomForest, GradientBoosting, LogisticRegression)
when a valid target exists. Strictly degrades gracefully to unsupervised multi-factor scoring
when no target exists (never fabricates fake labels).
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from tools.base_tool import BaseMLTool


class RiskTool(BaseMLTool):
    def __init__(self):
        super().__init__(name="risk", purpose="Supervised & Unsupervised Clinical Risk Stratification")
        self.required_semantic_fields = ["AGE", "DOSAGE", "QUANTITY", "DRUG", "DIAGNOSIS"]
        self.optional_fields = ["RISK_SCORE"]
        self.candidate_models = ["XGBoost", "RandomForest", "GradientBoosting", "LogisticRegression", "CompositeRiskIndex"]
        self.evaluation_metrics = ["roc_auc", "f1_score", "accuracy", "precision", "recall"]

    def run(self, df: pd.DataFrame, canonical_map: Dict[str, str], **kwargs) -> Dict[str, Any]:
        start_time = time.time()

        # 1. Check for target label
        target_cols = [src for src, can in canonical_map.items() if can == "RISK_SCORE" and src in df.columns]
        target_col = target_cols[0] if target_cols else None

        # Feature columns
        feature_cols = self.resolve_feature_columns(
            df,
            canonical_map,
            ["AGE", "DOSAGE", "QUANTITY", "DRUG", "DIAGNOSIS", "GENDER", "REGION"],
            max_features=6,
        )

        if not feature_cols:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:6]

        if len(feature_cols) < 1 or len(df) < 10:
            return self.create_unavailable_result("Insufficient features or records (< 10) for risk stratification.", ["Clinical feature columns"])

        # Prepare X matrix
        X_df = df[feature_cols].copy()
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype) == "category":
                extracted_nums = X_df[col].astype(str).str.extract(r"(\d+\.?\d*)")[0]
                num_conv = pd.to_numeric(extracted_nums, errors="coerce")
                if num_conv.notna().sum() > len(X_df) * 0.5:
                    X_df[col] = num_conv.fillna(num_conv.median() if num_conv.notna().sum() > 0 else 0)
                else:
                    X_df[col] = pd.factorize(X_df[col].astype(str))[0]
            else:
                num_conv = pd.to_numeric(X_df[col], errors="coerce")
                X_df[col] = num_conv.fillna(num_conv.median() if num_conv.notna().sum() > 0 else 0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_df)
        n_samples = len(df)

        # --- SCENARIO A: Supervised Risk Prediction (Target Exists) ---
        if target_col and df[target_col].nunique(dropna=True) >= 2 and n_samples >= 20:
            y_raw = df[target_col].dropna()
            # Binarize target if needed
            if pd.api.types.is_numeric_dtype(y_raw):
                threshold = y_raw.median()
                y = (df[target_col].fillna(threshold) > threshold).astype(int).values
            else:
                y = pd.factorize(df[target_col].astype(str))[0]
                y = (y > 0).astype(int)

            if len(np.unique(y)) < 2:
                # Target has only 1 class, fallback to unsupervised
                return self._run_unsupervised_risk(df, X_scaled, feature_cols, start_time, reason="Target label had zero class variance.")

            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y if np.min(np.bincount(y)) >= 2 else None)

            candidates = []

            # Model 1: XGBoost
            try:
                model_xgb = xgb.XGBClassifier(n_estimators=50, max_depth=3, eval_metric="logloss", random_state=42)
                model_xgb.fit(X_train, y_train)
                candidates.append(self._evaluate_classifier("XGBoost", model_xgb, X_test, y_test))
            except Exception as e:
                candidates.append({"model": "XGBoost", "valid": False, "error": str(e)})

            # Model 2: Random Forest
            try:
                model_rf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
                model_rf.fit(X_train, y_train)
                candidates.append(self._evaluate_classifier("RandomForest", model_rf, X_test, y_test))
            except Exception as e:
                candidates.append({"model": "RandomForest", "valid": False, "error": str(e)})

            # Model 3: Gradient Boosting
            try:
                model_gb = GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42)
                model_gb.fit(X_train, y_train)
                candidates.append(self._evaluate_classifier("GradientBoosting", model_gb, X_test, y_test))
            except Exception as e:
                candidates.append({"model": "GradientBoosting", "valid": False, "error": str(e)})

            # Model 4: Logistic Regression
            try:
                model_lr = LogisticRegression(random_state=42)
                model_lr.fit(X_train, y_train)
                candidates.append(self._evaluate_classifier("LogisticRegression", model_lr, X_test, y_test))
            except Exception as e:
                candidates.append({"model": "LogisticRegression", "valid": False, "error": str(e)})

            valid_candidates = [c for c in candidates if c.get("valid", False)]
            if not valid_candidates:
                return self._run_unsupervised_risk(df, X_scaled, feature_cols, start_time, reason="Supervised classifiers failed to converge.")

            # Winner selected by ROC-AUC and F1 Score
            valid_candidates.sort(key=lambda x: (x.get("roc_auc", 0.0) + x.get("f1_score", 0.0)), reverse=True)
            winner = valid_candidates[0]

            duration = (time.time() - start_time) * 1000

            findings = [
                f"Supervised risk prediction executed with target variable '{target_col}'.",
                f"Winning model '{winner['model']}' achieved ROC-AUC of {winner.get('roc_auc', 0.0):.3f} and F1-score of {winner.get('f1_score', 0.0):.3f}.",
                f"High-risk patient proportion in cohort: {round(np.mean(y)*100, 1)}%."
            ]

            evidence = [
                f"Model Selection Winner: {winner['model']} based on multi-metric test-set holdout evaluation.",
                "Algorithm Leaderboard: " + ", ".join([f"{c['model']} (ROC-AUC: {c.get('roc_auc', 0.0):.3f}, Acc: {c.get('accuracy', 0.0):.2f})" for c in valid_candidates]),
                f"Features used: {', '.join(feature_cols)}."
            ]

            leaderboard = [
                {
                    "model": c["model"],
                    "valid": c.get("valid", False),
                    "roc_auc": c.get("roc_auc", None),
                    "f1_score": c.get("f1_score", None),
                    "accuracy": c.get("accuracy", None),
                    "is_winner": c["model"] == winner["model"]
                }
                for c in candidates
            ]

            # --- Build Supervised Plotly Figures ---
            figures = []
            try:
                import plotly.express as px
                import plotly.graph_objects as go

                # Figure 1: High vs Low Risk Distribution Donut
                risk_counts_df = pd.DataFrame([
                    {"Risk Category": "Low / Standard Risk", "Count": int(len(y) - np.sum(y))},
                    {"Risk Category": "High Risk Cohort", "Count": int(np.sum(y))}
                ])
                fig_pie = px.pie(
                    risk_counts_df, names="Risk Category", values="Count",
                    hole=0.45,
                    title="⚠️ Clinical Patient Risk Stratification",
                    template="plotly_dark",
                    color_discrete_map={"Low / Standard Risk": "#00E5BE", "High Risk Cohort": "#EF4444"}
                )
                fig_pie.update_layout(
                    paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                    font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                )
                figures.append(("⚠️ Clinical Risk Stratification", fig_pie))

                # Figure 2: Risk by Age Bracket / Demographics (Dataset)
                try:
                    age_cols = [src for src, can in canonical_map.items() if can == "AGE" and src in df.columns]
                    if age_cols:
                        age_col = age_cols[0]
                        age_vals = pd.to_numeric(df[age_col], errors="coerce").fillna(50)
                        bins = [0, 30, 50, 70, 120]
                        labels = ["<30 yrs", "30-50 yrs", "51-70 yrs", ">70 yrs"]
                        age_groups = pd.cut(age_vals, bins=bins, labels=labels, right=False)
                        age_risk_df = pd.DataFrame({
                            "Age Group": age_groups,
                            "Risk Category": ["High Risk" if val == 1 else "Low Risk" for val in y]
                        })
                        age_risk_counts = age_risk_df.groupby(["Age Group", "Risk Category"], observed=False).size().reset_index(name="Patient Count")
                        fig_age_risk = px.bar(
                            age_risk_counts, x="Age Group", y="Patient Count", color="Risk Category",
                            barmode="group",
                            title="👥 Patient Clinical Risk Stratification by Age Group",
                            template="plotly_dark",
                            color_discrete_map={"Low Risk": "#00E5BE", "High Risk": "#EF4444"}
                        )
                        fig_age_risk.update_layout(
                            paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                            font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                        )
                        figures.append(("👥 Risk Distribution by Age Group", fig_age_risk))
                except Exception:
                    pass

                # Figure 3: High-Risk Rate by Drug (Dataset)
                try:
                    drug_cols = [src for src, can in canonical_map.items() if can == "DRUG" and src in df.columns]
                    if drug_cols:
                        d_col = drug_cols[0]
                        drug_risk_df = pd.DataFrame({
                            "Drug": df[d_col].astype(str).str.title(),
                            "High_Risk": y
                        })
                        top_d = drug_risk_df["Drug"].value_counts().head(8).index.tolist()
                        drug_risk_sub = drug_risk_df[drug_risk_df["Drug"].isin(top_d)]
                        drug_risk_summary = drug_risk_sub.groupby("Drug")["High_Risk"].agg(["count", "sum"]).reset_index()
                        drug_risk_summary["High_Risk_Pct"] = round((drug_risk_summary["sum"] / drug_risk_summary["count"]) * 100, 1)
                        drug_risk_summary.columns = ["Medication", "Total Prescriptions", "High Risk Count", "High Risk Rate (%)"]
                        
                        fig_drug_risk = px.bar(
                            drug_risk_summary, x="Medication", y="High Risk Rate (%)",
                            color="High Risk Rate (%)",
                            title="💊 High-Risk Prescription Rate by Therapeutic Agent",
                            template="plotly_dark",
                            color_continuous_scale="Reds"
                        )
                        fig_drug_risk.update_layout(
                            paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                            xaxis_title="Medication", yaxis_title="High Risk Proportion (%)",
                            font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                        )
                        figures.append(("💊 High-Risk Rate by Medication", fig_drug_risk))
                except Exception:
                    pass
            except Exception:
                pass

            duration = (time.time() - start_time) * 1000

            return self.create_normalized_result(
                model_name=winner["model"],
                status="success",
                inputs=feature_cols + [target_col],
                metrics={
                    "mode": "supervised",
                    "roc_auc": winner.get("roc_auc", 0.0),
                    "f1_score": winner.get("f1_score", 0.0),
                    "accuracy": winner.get("accuracy", 0.0),
                    "precision": winner.get("precision", 0.0),
                    "recall": winner.get("recall", 0.0),
                    "high_risk_rate_pct": round(float(np.mean(y) * 100), 2)
                },
                findings=findings,
                warnings=[],
                evidence=evidence,
                figures=figures,
                data={
                    "mode": "supervised",
                    "target_column": target_col,
                    "feature_columns": feature_cols,
                    "high_risk_count": int(np.sum(y)),
                    "total_count": len(y)
                },
                duration_ms=duration,
                leaderboard=leaderboard
            )

        # --- SCENARIO B: Unsupervised Composite Risk Index (No Target) ---
        return self._run_unsupervised_risk(df, X_scaled, feature_cols, start_time, reason="No explicit ground-truth target label present in dataset.")

    def _evaluate_classifier(self, name: str, model, X_test, y_test) -> Dict[str, Any]:
        preds = model.predict(X_test)
        try:
            probs = model.predict_proba(X_test)[:, 1]
            auc = float(round(roc_auc_score(y_test, probs), 4))
        except Exception:
            auc = float(round(accuracy_score(y_test, preds), 4))

        f1 = float(round(f1_score(y_test, preds, zero_division=0), 4))
        acc = float(round(accuracy_score(y_test, preds), 4))
        prec = float(round(precision_score(y_test, preds, zero_division=0), 4))
        rec = float(round(recall_score(y_test, preds, zero_division=0), 4))

        return {
            "model": name,
            "valid": True,
            "roc_auc": auc,
            "f1_score": f1,
            "accuracy": acc,
            "precision": prec,
            "recall": rec
        }

    def _run_unsupervised_risk(self, df: pd.DataFrame, X_scaled: np.ndarray, feature_cols: List[str], start_time: float, reason: str) -> Dict[str, Any]:
        """Calculates normalized multi-factor statistical risk index."""
        z_magnitudes = np.linalg.norm(X_scaled, axis=1)
        min_m, max_m = np.min(z_magnitudes), np.max(z_magnitudes)
        risk_scores = ((z_magnitudes - min_m) / max(max_m - min_m, 1e-5)) * 100

        high_risk_mask = risk_scores > 70
        moderate_risk_mask = (risk_scores >= 40) & (risk_scores <= 70)
        low_risk_mask = risk_scores < 40

        high_cnt = int(np.sum(high_risk_mask))
        mod_cnt = int(np.sum(moderate_risk_mask))
        low_cnt = int(np.sum(low_risk_mask))

        # --- Build Unsupervised Plotly Figures ---
        figures = []
        try:
            import plotly.express as px
            import plotly.graph_objects as go

            # Figure 1: Composite Risk Tier Distribution Donut
            tier_df = pd.DataFrame([
                {"Risk Tier": "Low Risk (<40)", "Count": low_cnt},
                {"Risk Tier": "Moderate Risk (40-70)", "Count": mod_cnt},
                {"Risk Tier": "High Risk (>70)", "Count": high_cnt}
            ])
            fig_pie = px.pie(
                tier_df, names="Risk Tier", values="Count",
                hole=0.45,
                title="⚠️ Multi-Factor Composite Clinical Risk Tiers",
                template="plotly_dark",
                color_discrete_sequence=["#00E5BE", "#F59E0B", "#EF4444"]
            )
            fig_pie.update_layout(
                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
            )
            figures.append(("⚠️ Multi-Factor Risk Tiers", fig_pie))

            # Figure 2: Risk Score Histogram
            fig_hist = px.histogram(
                pd.DataFrame({"Risk Score (0-100)": risk_scores}),
                x="Risk Score (0-100)",
                nbins=20,
                title="📊 Cohort Composite Risk Score Distribution",
                template="plotly_dark",
                color_discrete_sequence=["#0A84FF"]
            )
            fig_hist.update_layout(
                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                xaxis_title="Calculated Risk Index", yaxis_title="Patient Frequency",
                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
            )
            figures.append(("📊 Risk Index Distribution", fig_hist))

            # Figure 3: Risk Score Distribution across Age Groups (Dataset)
            try:
                age_cols = [src for src, can in canonical_map.items() if can == "AGE" and src in df.columns]
                if age_cols:
                    age_col = age_cols[0]
                    age_vals = pd.to_numeric(df[age_col], errors="coerce").fillna(50)
                    bins = [0, 30, 50, 70, 120]
                    labels = ["<30 yrs", "30-50 yrs", "51-70 yrs", ">70 yrs"]
                    age_groups = pd.cut(age_vals, bins=bins, labels=labels, right=False)
                    age_risk_df = pd.DataFrame({
                        "Age Group": age_groups,
                        "Risk Score": risk_scores
                    })
                    fig_age_box = px.box(
                        age_risk_df, x="Age Group", y="Risk Score",
                        title="👥 Clinical Risk Index Distribution Across Age Groups",
                        template="plotly_dark",
                        color="Age Group",
                        color_discrete_sequence=["#00E5BE", "#0A84FF", "#F59E0B", "#EF4444"]
                    )
                    fig_age_box.update_layout(
                        paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                        xaxis_title="Age Group", yaxis_title="Composite Risk Index (0-100)",
                        font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                        showlegend=False
                    )
                    figures.append(("👥 Risk Distribution by Age Group", fig_age_box))
            except Exception:
                pass

            # Figure 4: Average Risk by Drug (Dataset)
            try:
                drug_cols = [src for src, can in canonical_map.items() if can == "DRUG" and src in df.columns]
                if drug_cols:
                    d_col = drug_cols[0]
                    drug_risk_df = pd.DataFrame({
                        "Drug": df[d_col].astype(str).str.title(),
                        "Risk_Score": risk_scores
                    })
                    top_d = drug_risk_df["Drug"].value_counts().head(8).index.tolist()
                    drug_risk_sub = drug_risk_df[drug_risk_df["Drug"].isin(top_d)]
                    drug_mean_risk = drug_risk_sub.groupby("Drug")["Risk_Score"].mean().reset_index()
                    drug_mean_risk.columns = ["Medication", "Mean Risk Score"]
                    drug_mean_risk = drug_mean_risk.sort_values("Mean Risk Score", ascending=False)
                    
                    fig_drug_risk = px.bar(
                        drug_mean_risk, x="Medication", y="Mean Risk Score",
                        color="Mean Risk Score",
                        title="💊 Mean Clinical Risk Score by Therapeutic Agent",
                        template="plotly_dark",
                        color_continuous_scale="Reds"
                    )
                    fig_drug_risk.update_layout(
                        paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                        xaxis_title="Medication", yaxis_title="Mean Risk Score (0-100)",
                        font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                    )
                    figures.append(("💊 Mean Risk Score by Medication", fig_drug_risk))
            except Exception:
                pass
        except Exception:
            pass

        duration = (time.time() - start_time) * 1000

        findings = [
            f"Unsupervised Composite Risk Stratification calculated across {len(feature_cols)} clinical dimensions.",
            f"Risk Tier Breakdown: High Risk: {high_cnt} ({round(high_cnt/len(df)*100, 1)}%), Moderate Risk: {mod_cnt} ({round(mod_cnt/len(df)*100, 1)}%), Low Risk: {low_cnt} ({round(low_cnt/len(df)*100, 1)}%).",
            f"Mean cohort risk index: {np.mean(risk_scores):.1f} / 100."
        ]

        evidence = [
            f"Model Selection: Multi-Factor Clinical Risk Scorer (Unsupervised). {reason}",
            f"Dimension vectors: {', '.join(feature_cols)}."
        ]

        return self.create_normalized_result(
            model_name="Multi-Factor Risk Index (Unsupervised)",
            status="success",
            inputs=feature_cols,
            metrics={
                "mode": "unsupervised_composite",
                "mean_risk_index": float(round(np.mean(risk_scores), 2)),
                "high_risk_count": high_cnt,
                "moderate_risk_count": mod_cnt,
                "low_risk_count": low_cnt,
                "evaluated_records": len(df)
            },
            findings=findings,
            warnings=[f"Notice: {reason} Generated composite multi-factor risk index rather than supervised probability."],
            evidence=evidence,
            figures=figures,
            data={
                "mode": "unsupervised_composite",
                "high_risk_count": high_cnt,
                "moderate_risk_count": mod_cnt,
                "low_risk_count": low_cnt,
                "feature_columns": feature_cols
            },
            duration_ms=duration,
            leaderboard=[
                {"model": "Multi-Factor Risk Index", "valid": True, "type": "Unsupervised", "is_winner": True}
            ]
        )
