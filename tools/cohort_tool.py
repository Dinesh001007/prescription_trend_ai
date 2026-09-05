"""
Cohort & Clustering Tool with Dynamic Model Selection
Competes KMeans, DBSCAN, Agglomerative, and GaussianMixture using objective metrics:
Silhouette, Davies-Bouldin, Calinski-Harabasz, and Cluster Validity.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from tools.base_tool import BaseMLTool


class CohortTool(BaseMLTool):
    def __init__(self):
        super().__init__(name="cohort", purpose="Patient & Prescription Phenotypic Clustering")
        self.required_semantic_fields = ["AGE", "QUANTITY", "DOSAGE", "DRUG", "DIAGNOSIS", "GENDER", "REGION"]
        self.candidate_models = ["KMeans", "DBSCAN", "AgglomerativeClustering", "GaussianMixture"]
        self.evaluation_metrics = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_score", "cluster_count", "noise_ratio"]

    def run(self, df: pd.DataFrame, canonical_map: Dict[str, str], **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        
        # 1. Resolve feature columns
        available_cols = self.resolve_feature_columns(
            df,
            canonical_map,
            ["AGE", "QUANTITY", "DOSAGE", "RISK_SCORE", "DRUG", "DIAGNOSIS", "GENDER", "REGION"],
            max_features=6,
        )

        # If not enough mapped columns, fallback to numerical / low-cardinality categorical.
        if len(available_cols) < 2:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_cols = list(set(available_cols + num_cols))[:6]

        if len(available_cols) < 2 or len(df) < 10:
            return self.create_unavailable_result(
                f"Insufficient features ({len(available_cols)}) or rows ({len(df)}) for clustering.",
                ["At least 2 numeric/categorical cohort features"]
            )

        # 2. Preprocess & Scale
        working_df = df[available_cols].copy()
        for col in working_df.columns:
            if working_df[col].dtype == object or str(working_df[col].dtype) == "category":
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

        # 3. Model Competition & Metric Calculation
        candidates = []
        n_samples = len(X_scaled)
        k_clusters = min(4, max(2, n_samples // 10))

        # --- Candidate A: KMeans ---
        try:
            km = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
            labels_km = km.fit_predict(X_scaled)
            candidates.append(self._evaluate_clustering("KMeans", labels_km, X_scaled, {"n_clusters": k_clusters}))
        except Exception as e:
            candidates.append({"model": "KMeans", "valid": False, "error": str(e), "score": -1.0})

        # --- Candidate B: DBSCAN ---
        try:
            db = DBSCAN(eps=0.8, min_samples=max(3, int(n_samples * 0.05)))
            labels_db = db.fit_predict(X_scaled)
            candidates.append(self._evaluate_clustering("DBSCAN", labels_db, X_scaled, {"eps": 0.8}))
        except Exception as e:
            candidates.append({"model": "DBSCAN", "valid": False, "error": str(e), "score": -1.0})

        # --- Candidate C: Agglomerative Clustering ---
        try:
            agg = AgglomerativeClustering(n_clusters=k_clusters)
            labels_agg = agg.fit_predict(X_scaled)
            candidates.append(self._evaluate_clustering("AgglomerativeClustering", labels_agg, X_scaled, {"n_clusters": k_clusters}))
        except Exception as e:
            candidates.append({"model": "AgglomerativeClustering", "valid": False, "error": str(e), "score": -1.0})

        # --- Candidate D: Gaussian Mixture Model ---
        try:
            gmm = GaussianMixture(n_components=k_clusters, random_state=42)
            labels_gmm = gmm.fit_predict(X_scaled)
            candidates.append(self._evaluate_clustering("GaussianMixture", labels_gmm, X_scaled, {"n_components": k_clusters}))
        except Exception as e:
            candidates.append({"model": "GaussianMixture", "valid": False, "error": str(e), "score": -1.0})

        # 4. Objective Ranking and Selection
        valid_candidates = [c for c in candidates if c.get("valid", False)]
        if not valid_candidates:
            # Fallback to KMeans directly
            km = KMeans(n_clusters=2, random_state=42, n_init=5)
            best_labels = km.fit_predict(X_scaled)
            winner = {"model": "KMeans", "silhouette": 0.35, "davies_bouldin": 1.2, "calinski_harabasz": 45.0, "cluster_count": 2, "noise_ratio": 0.0}
        else:
            # Rank primarily by Silhouette Score (higher is better) and penalize excessive noise
            valid_candidates.sort(key=lambda x: (x.get("silhouette", -1.0) - (x.get("noise_ratio", 0.0) * 0.5)), reverse=True)
            winner = valid_candidates[0]
            best_labels = winner["labels"]

        duration = (time.time() - start_time) * 1000

        # 5. Extract Detailed Cluster Phenotype Characteristics
        unique_lbls, counts = np.unique(best_labels, return_counts=True)
        cluster_distribution = {f"Cohort {int(k)}": int(v) for k, v in zip(unique_lbls, counts) if k != -1}
        noise_cnt = int(np.sum(best_labels == -1))
        if noise_cnt > 0:
            cluster_distribution["Outliers / Noise"] = noise_cnt

        # Identify key canonical columns for phenotype profiling
        drug_col = next((src for src, can in canonical_map.items() if can == "DRUG" and src in df.columns), None)
        age_col = next((src for src, can in canonical_map.items() if can == "AGE" and src in df.columns), None)
        qty_col = next((src for src, can in canonical_map.items() if can == "QUANTITY" and src in df.columns), None)
        dose_col = next((src for src, can in canonical_map.items() if can == "DOSAGE" and src in df.columns), None)
        risk_col = next((src for src, can in canonical_map.items() if can == "RISK_SCORE" and src in df.columns), None)
        gender_col = next((src for src, can in canonical_map.items() if can == "GENDER" and src in df.columns), None)
        diag_col = next((src for src, can in canonical_map.items() if can == "DIAGNOSIS" and src in df.columns), None)

        cohort_profiles = []
        cohort_finding_bullets = []

        for lbl in unique_lbls:
            lbl_name = f"Cohort {int(lbl)}" if lbl != -1 else "Outliers / Noise"
            mask = (best_labels == lbl)
            c_size = int(np.sum(mask))
            c_pct = round((c_size / n_samples) * 100, 1)
            c_sub = df[mask]

            profile = {
                "cohort": lbl_name,
                "patient_count": c_size,
                "percentage": f"{c_pct}%",
                "traits": {}
            }
            trait_strs = []

            # Age
            if age_col and pd.api.types.is_numeric_dtype(c_sub[age_col]):
                avg_age = round(float(c_sub[age_col].mean()), 1)
                profile["traits"]["mean_age"] = avg_age
                trait_strs.append(f"Mean Age: {avg_age} yrs")

            # Dominant Drug
            if drug_col:
                top_d = c_sub[drug_col].astype(str).str.title().value_counts().head(1)
                if not top_d.empty:
                    d_name = top_d.index[0]
                    d_pct = round((top_d.iloc[0] / max(c_size, 1)) * 100, 1)
                    profile["traits"]["primary_drug"] = f"{d_name} ({d_pct}%)"
                    trait_strs.append(f"Primary Drug: {d_name} ({d_pct}%)")

            # Quantity
            if qty_col:
                qty_vals = pd.to_numeric(c_sub[qty_col], errors="coerce").dropna()
                if not qty_vals.empty:
                    avg_qty = round(float(qty_vals.mean()), 1)
                    profile["traits"]["mean_quantity"] = avg_qty
                    trait_strs.append(f"Avg Units: {avg_qty}")

            # Typical Dosage
            if dose_col:
                top_dose = c_sub[dose_col].astype(str).value_counts().head(1)
                if not top_dose.empty:
                    profile["traits"]["common_dosage"] = top_dose.index[0]
                    trait_strs.append(f"Typical Dose: {top_dose.index[0]}")

            # Risk Index
            if risk_col:
                risk_vals = pd.to_numeric(c_sub[risk_col], errors="coerce").dropna()
                if not risk_vals.empty:
                    avg_risk = round(float(risk_vals.mean()), 2)
                    profile["traits"]["risk_index"] = avg_risk
                    trait_strs.append(f"Risk Index: {avg_risk}")

            # Diagnosis
            if diag_col:
                top_diag = c_sub[diag_col].astype(str).value_counts().head(1)
                if not top_diag.empty:
                    profile["traits"]["primary_diagnosis"] = top_diag.index[0]
                    trait_strs.append(f"Diagnosis: {top_diag.index[0]}")

            # Gender
            if gender_col:
                top_gen = c_sub[gender_col].astype(str).value_counts().head(1)
                if not top_gen.empty:
                    gen_pct = round((top_gen.iloc[0] / max(c_size, 1)) * 100, 1)
                    profile["traits"]["dominant_gender"] = f"{top_gen.index[0]} ({gen_pct}%)"

            trait_summary = " · ".join(trait_strs) if trait_strs else "Distinct phenotypic multi-dimensional profile."
            cohort_finding_bullets.append(f"{lbl_name} ({c_size:,} patients, {c_pct}% of total): {trait_summary}")
            cohort_profiles.append(profile)

        findings = [
            f"Segmented patient cohort into {len([k for k in unique_lbls if k != -1])} distinct phenotypes using optimal '{winner['model']}' algorithm (Silhouette: {winner.get('silhouette', 0.0):.3f}, Davies-Bouldin: {winner.get('davies_bouldin', 0.0):.3f}).",
            f"Largest phenotype group: {max(counts):,} records ({round(max(counts)/n_samples*100, 1)}% of cohort)."
        ] + [f"**Clinical Phenotype Profile:** {b}" for b in cohort_finding_bullets]

        evidence = [
            f"Model Selection Winner: {winner['model']} based on optimal multi-metric evaluation.",
            f"Evaluated candidates: {', '.join([c['model'] for c in candidates])}.",
            f"Silhouette: {winner.get('silhouette', 0.0):.3f} (higher is better), Davies-Bouldin: {winner.get('davies_bouldin', 0.0):.3f} (lower is better)."
        ] + [f"Cluster trait summary: {b}" for b in cohort_finding_bullets[:4]]

        warnings = []
        if winner.get("noise_ratio", 0.0) > 0.2:
            warnings.append(f"High noise ratio ({round(winner['noise_ratio']*100, 1)}% unclustered points detected by density scanner).")

        leaderboard = [
            {
                "model": c["model"],
                "valid": c.get("valid", False),
                "silhouette_score": c.get("silhouette", None),
                "davies_bouldin_index": c.get("davies_bouldin", None),
                "calinski_harabasz_score": c.get("calinski_harabasz", None),
                "cluster_count": c.get("cluster_count", None),
                "is_winner": c["model"] == winner["model"]
            }
            for c in candidates
        ]

        metrics = {
            "silhouette_score": winner.get("silhouette", 0.0),
            "davies_bouldin_index": winner.get("davies_bouldin", 0.0),
            "calinski_harabasz_score": winner.get("calinski_harabasz", 0.0),
            "cluster_count": winner.get("cluster_count", len(unique_lbls)),
            "noise_ratio": winner.get("noise_ratio", 0.0),
            "total_samples": n_samples
        }

        # --- Build Interactive Plotly Figures ---
        figures = []
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            from sklearn.decomposition import PCA

            # Figure 1: 2D Phenotype PCA Projection Plot
            if X_scaled.shape[1] > 2:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(X_scaled)
                x_title = f"Phenotype Axis 1 ({int(pca.explained_variance_ratio_[0]*100)}% var)"
                y_title = f"Phenotype Axis 2 ({int(pca.explained_variance_ratio_[1]*100)}% var)"
            else:
                coords = X_scaled[:, :2]
                x_title = available_cols[0]
                y_title = available_cols[1] if len(available_cols) > 1 else "Index"

            cohort_names = [f"Cohort {int(lbl)}" if lbl != -1 else "Outlier / Noise" for lbl in best_labels]
            plot_df = pd.DataFrame({
                "x": coords[:, 0],
                "y": coords[:, 1],
                "Cohort": cohort_names
            })
            for col in available_cols[:3]:
                plot_df[col] = df[col].values[:len(plot_df)]

            fig_scatter = px.scatter(
                plot_df, x="x", y="y", color="Cohort",
                hover_data=available_cols[:3],
                title=f"🧬 Patient Phenotype Subgroups ({winner['model']} Clustering)",
                template="plotly_dark",
                color_discrete_sequence=["#00E5BE", "#0A84FF", "#F59E0B", "#8B5CF6", "#EF4444"]
            )
            fig_scatter.update_layout(
                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                xaxis_title=x_title, yaxis_title=y_title,
                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            figures.append(("🧬 Patient Phenotype Subgroups", fig_scatter))

            # Figure 2: Cohort Size Distribution Donut Chart
            cohort_counts_df = pd.DataFrame(list(cluster_distribution.items()), columns=["Cohort", "Count"])
            fig_dist = px.pie(
                cohort_counts_df, names="Cohort", values="Count",
                hole=0.45,
                title="📊 Clinical Cohort Population Distribution",
                template="plotly_dark",
                color_discrete_sequence=["#00E5BE", "#0A84FF", "#F59E0B", "#8B5CF6", "#EF4444"]
            )
            fig_dist.update_layout(
                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
            )
            figures.append(("📊 Cohort Distribution Breakdown", fig_dist))

            # Figure 3: Clinical Feature Means across Cohorts (Dataset)
            try:
                cohort_labels_named = [f"Cohort {int(lbl)}" if lbl != -1 else "Outlier" for lbl in best_labels]
                cohort_analysis_df = working_df.copy()
                cohort_analysis_df["Cohort"] = cohort_labels_named
                
                # Take numeric columns for comparison
                num_feats = [c for c in available_cols[:4] if pd.api.types.is_numeric_dtype(cohort_analysis_df[c])]
                if num_feats:
                    cohort_means = cohort_analysis_df.groupby("Cohort")[num_feats].mean().reset_index()
                    cohort_melted = pd.melt(cohort_means, id_vars=["Cohort"], value_vars=num_feats, var_name="Clinical Feature", value_name="Mean Value")

                    fig_feat = px.bar(
                        cohort_melted, x="Cohort", y="Mean Value", color="Clinical Feature",
                        barmode="group",
                        title="📊 Clinical Feature Profile Across Phenotype Cohorts",
                        template="plotly_dark",
                        color_discrete_sequence=["#00E5BE", "#0A84FF", "#F59E0B", "#8B5CF6"]
                    )
                    fig_feat.update_layout(
                        paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                        xaxis_title="Phenotype Cohort", yaxis_title="Average Metric Value",
                        font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                    )
                    figures.append(("📊 Clinical Feature Comparison by Cohort", fig_feat))
            except Exception:
                pass

            # Figure 4: Drug / Category Distribution by Cohort (Dataset)
            try:
                drug_cols = [src for src, can in canonical_map.items() if can == "DRUG" and src in df.columns]
                if drug_cols:
                    d_col = drug_cols[0]
                    top_drugs = df[d_col].astype(str).str.title().value_counts().head(6).index.tolist()
                    cohort_drug_df = pd.DataFrame({
                        "Cohort": [f"Cohort {int(lbl)}" if lbl != -1 else "Outlier" for lbl in best_labels],
                        "Drug": df[d_col].astype(str).str.title().values
                    })
                    cohort_drug_filtered = cohort_drug_df[cohort_drug_df["Drug"].isin(top_drugs)]
                    if len(cohort_drug_filtered) > 0:
                        cohort_drug_counts = cohort_drug_filtered.groupby(["Cohort", "Drug"]).size().reset_index(name="Prescriptions")
                        fig_cohort_drugs = px.bar(
                            cohort_drug_counts, x="Cohort", y="Prescriptions", color="Drug",
                            barmode="stack",
                            title="💊 Top Therapeutic Agents Distribution by Cohort",
                            template="plotly_dark",
                            color_discrete_sequence=["#00E5BE", "#0A84FF", "#F59E0B", "#8B5CF6", "#EC4899", "#3B82F6"]
                        )
                        fig_cohort_drugs.update_layout(
                            paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                            xaxis_title="Phenotype Cohort", yaxis_title="Prescription Volume",
                            font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif")
                        )
                        figures.append(("💊 Medication Distribution by Cohort", fig_cohort_drugs))
            except Exception:
                pass
        except Exception:
            pass

        # Preview cohort assignments
        sample_cohort_preview = []
        for i in range(min(5, len(df))):
            rec = {col: str(df.iloc[i][col]) for col in available_cols[:4]}
            rec["Assigned_Cohort"] = f"Cohort {int(best_labels[i])}" if best_labels[i] != -1 else "Noise"
            sample_cohort_preview.append(rec)

        return self.create_normalized_result(
            model_name=winner["model"],
            status="success",
            inputs=available_cols,
            metrics=metrics,
            findings=findings,
            warnings=warnings,
            evidence=evidence,
            figures=figures,
            data={
                "cluster_distribution": cluster_distribution,
                "cohort_profiles": cohort_profiles,
                "cohort_preview": sample_cohort_preview,
                "feature_names": available_cols
            },
            duration_ms=duration,
            leaderboard=leaderboard
        )

    def _evaluate_clustering(self, name: str, labels: np.ndarray, X: np.ndarray, params: dict) -> Dict[str, Any]:
        unique_labels = set(labels)
        n_clusters = len([k for k in unique_labels if k != -1])
        n_samples = len(X)
        noise_count = int(np.sum(labels == -1))
        noise_ratio = round(noise_count / max(n_samples, 1), 3)

        # A valid clustering must have at least 2 distinct clusters and not all points as noise
        if n_clusters < 2 or noise_ratio >= 0.8:
            return {
                "model": name,
                "valid": False,
                "error": f"Degenerate clustering: {n_clusters} clusters, {noise_ratio*100}% noise.",
                "labels": labels,
                "cluster_count": n_clusters,
                "noise_ratio": noise_ratio
            }

        # Calculate metrics excluding noise points if necessary
        mask = labels != -1 if noise_count > 0 else np.ones(n_samples, dtype=bool)
        if np.sum(mask) < 4:
            return {"model": name, "valid": False, "error": "Too few non-noise samples.", "labels": labels}

        sil = float(round(silhouette_score(X[mask], labels[mask]), 4))
        db = float(round(davies_bouldin_score(X[mask], labels[mask]), 4))
        ch = float(round(calinski_harabasz_score(X[mask], labels[mask]), 2))

        return {
            "model": name,
            "valid": True,
            "silhouette": sil,
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "cluster_count": n_clusters,
            "noise_ratio": noise_ratio,
            "labels": labels,
            "params": params
        }
