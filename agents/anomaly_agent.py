import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler
import plotly.express as px
import plotly.graph_objects as go


def run_anomaly_agent(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Anomaly Agent: Uses Isolation Forest to detect unusual prescriptions.
    """
    result = {"status": "ok", "figures": [], "summary": "", "anomaly_df": None}

    feature_cols = [
        col for col, cat in col_map.items()
        if cat not in ["patient_id", "date"] and col in df.columns
    ]

    if len(feature_cols) < 1:
        result["status"] = "insufficient_columns"
        result["summary"] = "Not enough columns for anomaly detection."
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

    try:
        iso = IsolationForest(contamination=0.1, random_state=42, n_estimators=100)
        preds = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)

        anomaly_df = df.copy()
        anomaly_df["__anomaly"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
        anomaly_df["__anomaly_score"] = scores
        result["anomaly_df"] = anomaly_df

        n_anomalies = (preds == -1).sum()

        # Anomaly score distribution
        fig_score = go.Figure()
        normal_scores = scores[preds == 1]
        anomaly_scores = scores[preds == -1]

        fig_score.add_trace(go.Histogram(
            x=normal_scores, name="Normal", marker_color="#00C9A7", opacity=0.75, nbinsx=30
        ))
        fig_score.add_trace(go.Histogram(
            x=anomaly_scores, name="Anomaly", marker_color="#FF6B6B", opacity=0.85, nbinsx=20
        ))
        fig_score.update_layout(
            title="Anomaly Score Distribution",
            xaxis_title="Isolation Forest Score (lower = more anomalous)",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
        )
        result["figures"].append(("Anomaly Score Distribution", fig_score))

        # Numeric scatter: first two numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            fig_2d = px.scatter(
                anomaly_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color="__anomaly",
                title=f"Anomaly Detection: {numeric_cols[0]} vs {numeric_cols[1]}",
                color_discrete_map={"Normal": "#00C9A7", "Anomaly": "#FF6B6B"},
                template="plotly_dark",
                opacity=0.7,
            )
            fig_2d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAF0",
            )
            result["figures"].append(("Anomaly Scatter", fig_2d))

        # Anomalies per drug if drug column exists
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        if drug_col:
            drug_anomaly = anomaly_df.groupby([drug_col, "__anomaly"]).size().reset_index(name="Count")
            top_drugs = anomaly_df[drug_col].value_counts().head(12).index
            drug_anomaly = drug_anomaly[drug_anomaly[drug_col].isin(top_drugs)]
            fig_drug = px.bar(
                drug_anomaly,
                x=drug_col,
                y="Count",
                color="__anomaly",
                barmode="stack",
                title="Anomalies by Drug",
                color_discrete_map={"Normal": "#00C9A7", "Anomaly": "#FF6B6B"},
                template="plotly_dark",
            )
            fig_drug.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAF0",
            )
            result["figures"].append(("Anomalies by Drug", fig_drug))

        # Region-level anomaly heatmap
        region_col = next((c for c, cat in col_map.items() if cat == "region" and c in df.columns), None)
        if region_col and drug_col:
            heat_data = anomaly_df[anomaly_df["__anomaly"] == "Anomaly"].groupby([region_col, drug_col]).size().reset_index(name="Anomaly Count")
            if not heat_data.empty:
                pivot = heat_data.pivot(index=region_col, columns=drug_col, values="Anomaly Count").fillna(0)
                fig_heat = px.imshow(
                    pivot,
                    title="Anomaly Heatmap: Region vs Drug",
                    color_continuous_scale="Reds",
                    template="plotly_dark",
                    aspect="auto",
                )
                fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF0")
                result["figures"].append(("Anomaly Heatmap", fig_heat))

        result["summary"] = (
            f"Anomaly detection completed on {len(df)} records.\n"
            f"Anomalies detected: {n_anomalies} ({100*n_anomalies/len(df):.1f}% of dataset).\n"
            f"Isolation Forest used {len(feature_cols)} features with 10% contamination threshold.\n"
            f"Anomalous records may represent unusual dosage, rare co-prescriptions, or data errors."
        )

    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Anomaly agent error: {str(e)}"

    return result