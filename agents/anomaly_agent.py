import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go


def run_anomaly_agent_v2(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Improved Anomaly Agent:
    - Proper encoding (OneHot instead of LabelEncoder)
    - Dynamic contamination
    - More realistic anomaly detection
    """

    result = {"status": "ok", "figures": [], "summary": "", "anomaly_df": None}

    # -------------------------------
    # 1. Feature Selection
    # -------------------------------
    feature_cols = [
        col for col, cat in col_map.items()
        if cat not in ["patient_id", "date"] and col in df.columns
    ]

    if len(feature_cols) < 2:
        result["status"] = "insufficient_columns"
        result["summary"] = "Need at least 2 columns for anomaly detection."
        return result

    X = df[feature_cols].copy()

    # -------------------------------
    # 2. Separate column types
    # -------------------------------
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # -------------------------------
    # 3. Preprocessing pipeline
    # -------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # -------------------------------
    # 4. Dynamic contamination
    # -------------------------------
    # Estimate anomaly fraction based on dataset size
    contamination = min(0.05, max(0.01, 50 / len(df)))
    # Example: ~1–5%

    # -------------------------------
    # 5. Isolation Forest pipeline
    # -------------------------------
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("iso", IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
            max_samples="auto"
        ))
    ])

    try:
        model.fit(X)

        # Get predictions
        preds = model.named_steps["iso"].predict(
            model.named_steps["preprocessor"].transform(X)
        )
        scores = model.named_steps["iso"].decision_function(
            model.named_steps["preprocessor"].transform(X)
        )

        anomaly_df = df.copy()
        anomaly_df["__anomaly"] = np.where(preds == -1, "Anomaly", "Normal")
        anomaly_df["__score"] = scores

        result["anomaly_df"] = anomaly_df

        n_anomalies = (preds == -1).sum()

        # -------------------------------
        # 6. Visualization
        # -------------------------------

        # Score distribution
        fig1 = px.histogram(
            anomaly_df,
            x="__score",
            color="__anomaly",
            title="Anomaly Score Distribution",
            opacity=0.7
        )
        result["figures"].append(("Score Distribution", fig1))

        # Scatter (first 2 numeric cols if exist)
        if len(numeric_cols) >= 2:
            fig2 = px.scatter(
                anomaly_df,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color="__anomaly",
                title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
            )
            result["figures"].append(("Scatter Plot", fig2))

        # Drug analysis
        drug_col = next((c for c, cat in col_map.items()
                         if cat == "drug_name" and c in df.columns), None)

        if drug_col:
            drug_counts = anomaly_df.groupby([drug_col, "__anomaly"]).size().reset_index(name="Count")

            fig3 = px.bar(
                drug_counts,
                x=drug_col,
                y="Count",
                color="__anomaly",
                barmode="stack",
                title="Anomalies by Drug"
            )
            result["figures"].append(("Drug Analysis", fig3))

        # -------------------------------
        # 7. Summary
        # -------------------------------
        result["summary"] = (
            f"Analyzed {len(df)} records using Isolation Forest.\n"
            f"Detected {n_anomalies} anomalies ({100*n_anomalies/len(df):.2f}%).\n"
            f"Dynamic contamination used: {contamination:.3f}.\n\n"
            f"Model uses proper encoding (OneHot + Scaling), so anomalies are more realistic.\n"
            f"Detected anomalies may represent rare drug combinations, unusual dosages, or true clinical outliers."
        )

    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Error: {str(e)}"

    return result