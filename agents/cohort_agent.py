import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import time


def generate_cohort_names(X, labels, col_map, feature_cols):
    """
    Generate meaningful cohort names based on the characteristics of each cluster.
    """
    cohort_names = []
    X_with_labels = X.copy()
    X_with_labels["__cluster"] = labels
    
    for cluster_id in range(len(np.unique(labels))):
        cluster_data = X_with_labels[X_with_labels["__cluster"] == cluster_id]
        cluster_characteristics = []
        
        # Analyze each feature for this cluster
        for col in feature_cols:
            if col in cluster_data.columns:
                cluster_values = cluster_data[col]
                overall_values = X[col]
                
                # For numeric columns: check if significantly higher or lower than overall
                if pd.api.types.is_numeric_dtype(cluster_values):
                    cluster_mean = cluster_values.mean()
                    overall_mean = overall_values.mean()
                    overall_std = overall_values.std()
                    
                    if overall_std > 0:
                        z_score = (cluster_mean - overall_mean) / overall_std
                        if abs(z_score) > 0.5:  # Significant difference
                            category = col_map.get(col, col)
                            if z_score > 0:
                                cluster_characteristics.append(f"High {category}")
                            else:
                                cluster_characteristics.append(f"Low {category}")
                
                # For categorical columns: find dominant category
                else:
                    value_counts = cluster_values.value_counts()
                    if len(value_counts) > 0:
                        dominant_val = value_counts.index[0]
                        dominant_pct = value_counts.iloc[0] / len(cluster_values)
                        
                        if dominant_pct > 0.6:  # 60% or more of cluster has this value
                            category = col_map.get(col, col)
                            cluster_characteristics.append(f"{category}: {dominant_val}")
        
        # Generate meaningful name based on characteristics
        if cluster_characteristics:
            # Take top 2-3 characteristics for the name
            top_characteristics = cluster_characteristics[:3]
            if len(top_characteristics) == 1:
                name = f"{top_characteristics[0]} Group"
            elif len(top_characteristics) == 2:
                name = f"{top_characteristics[0]} & {top_characteristics[1]} Cohort"
            else:
                name = f"{top_characteristics[0]}, {top_characteristics[1]} Group"
        else:
            # Fallback: use cluster size
            cluster_size = len(cluster_data)
            total_size = len(X)
            percentage = (cluster_size / total_size) * 100
            
            if percentage > 40:
                name = "Large Cohort"
            elif percentage > 20:
                name = "Medium Cohort"
            else:
                name = "Small Cohort"
        
        cohort_names.append(name)
    
    return cohort_names


def generate_cohort_explanations(X, labels, col_map, feature_cols, cohort_names):
    """
    Generate detailed explanations of what defines each cohort.
    """
    explanations = []
    X_with_labels = X.copy()
    X_with_labels["__cluster"] = labels
    
    for cluster_id, cohort_name in enumerate(cohort_names):
        cluster_data = X_with_labels[X_with_labels["__cluster"] == cluster_id]
        cluster_size = len(cluster_data)
        total_size = len(X)
        percentage = (cluster_size / total_size) * 100
        
        explanation_parts = [f"• **{cohort_name}** ({cluster_size} records, {percentage:.1f}%):"]
        
        # Analyze each feature for this cluster
        for col in feature_cols:
            if col in cluster_data.columns:
                cluster_values = cluster_data[col]
                overall_values = X[col]
                category = col_map.get(col, col)
                
                # For numeric columns: provide detailed statistics
                if pd.api.types.is_numeric_dtype(cluster_values):
                    cluster_mean = cluster_values.mean()
                    cluster_median = cluster_values.median()
                    overall_mean = overall_values.mean()
                    overall_std = overall_values.std()
                    
                    if overall_std > 0:
                        z_score = (cluster_mean - overall_mean) / overall_std
                        if abs(z_score) > 0.3:  # Less strict threshold for explanations
                            direction = "higher" if z_score > 0 else "lower"
                            explanation_parts.append(f"  - {category.title()}: {direction} than average (mean: {cluster_mean:.2f} vs {overall_mean:.2f})")
                
                # For categorical columns: show dominant values
                else:
                    value_counts = cluster_values.value_counts()
                    if len(value_counts) > 0:
                        dominant_val = value_counts.index[0]
                        dominant_pct = value_counts.iloc[0] / len(cluster_data)
                        
                        if dominant_pct > 0.4:  # 40% or more for explanations
                            explanation_parts.append(f"  - {category.title()}: Primarily '{dominant_val}' ({dominant_pct*100:.1f}% of cohort)")
                            
                        # Show second most common if significant
                        if len(value_counts) > 1:
                            second_val = value_counts.index[1]
                            second_pct = value_counts.iloc[1] / len(cluster_data)
                            if second_pct > 0.25:  # 25% or more
                                explanation_parts.append(f"  - {category.title()}: Also '{second_val}' ({second_pct*100:.1f}%)")
        
        if len(explanation_parts) > 1:  # More than just the header
            explanations.extend(explanation_parts)
        else:
            explanations.append(f"• **{cohort_name}** ({cluster_size} records, {percentage:.1f}%): No distinctive characteristics identified")
    
    return "\n".join(explanations)


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

    # Feature selection for better clustering
    # Remove low-variance features and highly correlated features
    feature_selector = []
    for i, col in enumerate(X.columns):
        col_data = X[col]
        # Skip features with very low variance
        if col_data.var() < 1e-6:
            continue
        feature_selector.append(i)
    
    if len(feature_selector) < 2:
        result["status"] = "insufficient_columns"
        result["summary"] = "Not enough meaningful features for cohort clustering."
        return result
    
    X_filtered = X.iloc[:, feature_selector]
    X_scaled_filtered = scaler.fit_transform(X_filtered)
    
    # Determine optimal k dynamically (2-5 groups max)
    max_clusters = min(5, max(2, len(df) // 20))  # More conservative clustering
    silhouette_scores = []
    
    # Test different numbers of clusters with multiple random seeds
    for k in range(2, min(max_clusters + 1, 6)):
        k_silhouettes = []
        for seed in [42, 123, 456]:  # Multiple seeds for stability
            kmeans_test = KMeans(n_clusters=k, random_state=seed, n_init=20, max_iter=300)
            labels_test = kmeans_test.fit_predict(X_scaled_filtered)
            
            if len(set(labels_test)) > 1:  # Ensure we have multiple clusters
                if len(X_scaled_filtered) > 1000:
                    indices = np.random.choice(len(X_scaled_filtered), min(1000, len(X_scaled_filtered)), replace=False)
                    sil_score = silhouette_score(X_scaled_filtered[indices], labels_test[indices])
                else:
                    sil_score = silhouette_score(X_scaled_filtered, labels_test)
                k_silhouettes.append(sil_score)
        
        if k_silhouettes:
            avg_silhouette = np.mean(k_silhouettes)
            silhouette_scores.append((k, avg_silhouette))
    
    # Choose best k based on silhouette score, but ensure minimum quality
    if silhouette_scores:
        best_k, best_silhouette = max(silhouette_scores, key=lambda x: x[1])
        
        # If silhouette is too low, try reducing to 2 clusters
        if best_silhouette < 0.1 and best_k > 2:
            best_k = 2
            # Recalculate with 2 clusters
            kmeans_test = KMeans(n_clusters=2, random_state=42, n_init=20, max_iter=300)
            labels_test = kmeans_test.fit_predict(X_scaled_filtered)
            best_silhouette = silhouette_score(X_scaled_filtered, labels_test)
        
        n_clusters = best_k
    else:
        n_clusters = 2
        best_silhouette = 0.0

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20, max_iter=300)
        labels = kmeans.fit_predict(X_scaled_filtered)

        # Silhouette score
        if len(X_scaled_filtered) > 1000:
            indices = np.random.choice(len(X_scaled_filtered), min(1000, len(X_scaled_filtered)), replace=False)
            sil_score = silhouette_score(X_scaled_filtered[indices], labels[indices])
        else:
            sil_score = silhouette_score(X_scaled_filtered, labels)

        cohort_df = df.copy()
        
        # Generate meaningful cohort names based on characteristics (use filtered features)
        filtered_feature_cols = [feature_cols[i] for i in feature_selector]
        cohort_names = generate_cohort_names(X_filtered, labels, col_map, filtered_feature_cols)
        cohort_df["__cohort"] = [cohort_names[l] for l in labels]
        result["cohort_df"] = cohort_df

        # PCA for 2D visualization (use filtered data for consistency)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_scaled_filtered)
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
        
        # Generate detailed cohort explanations
        cohort_explanations = generate_cohort_explanations(X, labels, col_map, feature_cols, cohort_names)
        
        result["summary"] = (
            f"Cohort analysis identified {n_clusters} distinct groups using optimal silhouette scoring.\n"
            f"Tested {len(silhouette_scores)} cluster configurations; selected {n_clusters} clusters (silhouette: {sil_score:.3f}).\n"
            f"Cohort sizes: {sizes}.\n\n"
            f"**Cohort Characteristics:**\n{cohort_explanations}\n\n"
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

