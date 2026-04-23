import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import plotly.express as px


def run_risk_agent_v2(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Improved Risk Agent:
    - Proper encoding (OneHot)
    - Better synthetic target
    - Handles imbalance
    - Uses percentile threshold
    """

    result = {"status": "ok", "figures": [], "summary": "", "risk_df": None}

    # -------------------------------
    # 1. Feature selection
    # -------------------------------
    feature_cols = []
    target_col = None

    for col, cat in col_map.items():
        if cat == "risk_score" and col in df.columns:
            target_col = col
        elif cat not in ["patient_id", "date"] and col in df.columns:
            feature_cols.append(col)

    if len(feature_cols) < 2:
        result["status"] = "insufficient_columns"
        result["summary"] = "Not enough features."
        return result

    X = df[feature_cols].copy()

    # -------------------------------
    # 2. Column split
    # -------------------------------
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # -------------------------------
    # 3. Preprocessing
    # -------------------------------
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ])

    # -------------------------------
    # 4. Target creation (better)
    # -------------------------------
    if target_col is None:
        # Risk heuristic (better than raw z-score)
        score = 0

        if "dosage" in df.columns:
            score += pd.to_numeric(df["dosage"], errors="coerce").fillna(0)

        if "frequency" in df.columns:
            score += df["frequency"].astype(str).str.len()

        # Normalize
        score = (score - score.mean()) / (score.std() + 1e-9)

        # Top 25% as high risk
        y = (score > np.quantile(score, 0.75)).astype(int)
    else:
        y_raw = df[target_col]
        if y_raw.dtype == object:
            y = pd.factorize(y_raw)[0]
        else:
            y = (pd.to_numeric(y_raw, errors="coerce") > y_raw.median()).astype(int)

    # -------------------------------
    # 5. Train/Test
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle imbalance
    scale_pos_weight = (len(y_train) - sum(y_train)) / (sum(y_train) + 1e-9)

    model = Pipeline([
        ("prep", preprocessor),
        ("xgb", XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        ))
    ])

    try:
        model.fit(X_train, y_train)

        # Predictions
        risk_proba = model.predict_proba(X)[:, 1]

        # Dynamic threshold (top 25%)
        threshold = np.quantile(risk_proba, 0.75)
        risk_labels = np.where(risk_proba >= threshold, "High Risk", "Low Risk")

        risk_df = df.copy()
        risk_df["__risk_score"] = risk_proba
        risk_df["__risk_label"] = risk_labels

        result["risk_df"] = risk_df

        # -------------------------------
        # 6. Evaluation
        # -------------------------------
        test_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, test_pred)

        # -------------------------------
        # 7. Visualization
        # -------------------------------
        fig = px.histogram(
            risk_df,
            x="__risk_score",
            color="__risk_label",
            title="Risk Score Distribution"
        )
        result["figures"].append(("Risk Distribution", fig))

        # -------------------------------
        # 8. Summary
        # -------------------------------
        high_risk_count = (risk_labels == "High Risk").sum()

        result["summary"] = (
            f"Analyzed {len(df)} records.\n"
            f"High-risk cases: {high_risk_count} ({100*high_risk_count/len(df):.1f}%).\n"
            f"Model AUC: {auc:.3f}.\n"
            f"Dynamic threshold used (top 25%).\n"
            f"Proper encoding + imbalance handling applied.\n"
        )

    except Exception as e:
        result["status"] = "error"
        result["summary"] = str(e)

    return result