import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import time


def run_cohort_agent(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Cohort Agent: Uses KMeans to find patient/prescription cohorts.
    """
    start_time = time.perf_counter()
    result = {"status": "ok", "figures": [], "summary": "", "cohort_df": None, "metrics": {}}

    # Build feature matrix from all usable columns
    feature_cols = [
        col for col, cat in col_map.items()
        if cat not in ["patient_id", "date"] and col in df.columns
    ]

    if len(feature_cols) < 2:
        result["status"] = "insufficient_columns"
        result["summary"] = "Not enough columns for cohort clustering."
        return result

    X = df[feature_cols].copy()

    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("unknown"))
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal k (2–6)
    n_clusters = min(4, max(2, len(df) // 20))

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        # Silhouette score
        if len(X_scaled) > 1000:
            indices = np.random.choice(len(X_scaled), 1000, replace=False)
            sil_score = silhouette_score(X_scaled[indices], labels[indices])
        else:
            sil_score = silhouette_score(X_scaled, labels)

        cohort_df = df.copy()
        cohort_df["__cohort"] = [f"Cohort {l+1}" for l in labels]
        result["cohort_df"] = cohort_df

        # PCA for 2D visualization
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        pca_df["Cohort"] = cohort_df["__cohort"].values

        # Tooltip: add drug or id column if available
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        if drug_col:
            pca_df["Drug"] = df[drug_col].values

        fig_scatter = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cohort",
            hover_data=["Drug"] if drug_col else None,
            title=f"Patient/Prescription Cohorts (KMeans, k={n_clusters})",
            template="plotly_dark",
            color_discrete_sequence=["#00C9A7", "#FF6B6B", "#FFC300", "#6C63FF", "#00B4D8"],
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
        )
        result["figures"].append(("Cohort Clusters", fig_scatter))

        # Cohort size distribution
        cohort_counts = cohort_df["__cohort"].value_counts().reset_index()
        cohort_counts.columns = ["Cohort", "Count"]
        fig_pie = px.pie(
            cohort_counts,
            names="Cohort",
            values="Count",
            title="Cohort Size Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#00C9A7", "#FF6B6B", "#FFC300", "#6C63FF", "#00B4D8"],
            hole=0.4,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
        )
        result["figures"].append(("Cohort Sizes", fig_pie))

        # Cohort profile: mean values per cohort
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            X_with_cohort = X.copy()
            X_with_cohort["__cohort"] = cohort_df["__cohort"]
            cohort_profile = X_with_cohort.groupby("__cohort")[numeric_cols[:6]].mean().reset_index()
            melted = cohort_profile.melt(id_vars="__cohort", var_name="Feature", value_name="Mean Value")
            fig_profile = px.bar(
                melted,
                x="Feature",
                y="Mean Value",
                color="__cohort",
                barmode="group",
                title="Cohort Feature Profiles",
                template="plotly_dark",
                color_discrete_sequence=["#00C9A7", "#FF6B6B", "#FFC300", "#6C63FF", "#00B4D8"],
            )
            fig_profile.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAF0",
                xaxis_title="Feature",
                yaxis_title="Mean Value",
            )
            result["figures"].append(("Cohort Profiles", fig_profile))

        # Drug distribution per cohort if drug column present
        if drug_col:
            top_drugs = df[drug_col].value_counts().head(8).index.tolist()
            drug_cohort = cohort_df[cohort_df[drug_col].isin(top_drugs)]
            drug_cohort_count = drug_cohort.groupby([drug_col, "__cohort"]).size().reset_index(name="Count")
            fig_drug_cohort = px.bar(
                drug_cohort_count,
                x=drug_col,
                y="Count",
                color="__cohort",
                barmode="stack",
                title="Drug Distribution Across Cohorts",
                template="plotly_dark",
                color_discrete_sequence=["#00C9A7", "#FF6B6B", "#FFC300", "#6C63FF", "#00B4D8"],
            )
            fig_drug_cohort.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAF0",
            )
            result["figures"].append(("Drug Distribution by Cohort", fig_drug_cohort))

        sizes = cohort_df["__cohort"].value_counts().to_dict()
        result["summary"] = (
            f"Cohort analysis identified {n_clusters} distinct groups.\n"
            f"Cohort sizes: {sizes}.\n"
            f"Clustering performed on {len(feature_cols)} features using KMeans.\n"
            f"PCA explains {sum(pca.explained_variance_ratio_)*100:.1f}% of variance in 2D projection."
        )

        # Performance metrics
        duration = (time.perf_counter() - start_time) * 1000
        result["metrics"] = {
            "Silhouette": f"{sil_score:.3f}",
            "Clusters": f"{n_clusters}",
            "Variance": f"{sum(pca.explained_variance_ratio_)*100:.1f}%",
            "Execution": f"{duration:.1f}ms",
            "Model": "KMeans",
            "Iterations": f"{kmeans.n_iter_}"
        }

    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Cohort agent error: {str(e)}"

    return result
