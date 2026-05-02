import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go
import time


def run_risk_agent(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Risk Agent: Uses XGBoost to identify high-risk prescriptions or patients.
    Works with any numeric/categorical columns available.
    Returns: risk scores, feature importances, visualizations.
    """
    start_time = time.perf_counter()
    result = {"status": "ok", "figures": [], "summary": "", "risk_df": None, "metrics": {}}

    # Select usable columns
    feature_cols = []
    target_col = None

    # Check for explicit risk score column
    for col, cat in col_map.items():
        if cat == "risk_score" and col in df.columns:
            target_col = col
        elif cat not in ["patient_id", "date"] and col in df.columns:
            feature_cols.append(col)

    if len(feature_cols) < 2:
        result["status"] = "insufficient_columns"
        result["summary"] = "Not enough feature columns for risk analysis."
        return result

    # Build feature matrix
    X = df[feature_cols].copy()
    encoders = {}

    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("unknown"))
            encoders[col] = le
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)

    # Create synthetic target if no risk column exists
    if target_col is None:
        # Use anomaly-style heuristic: flag top 20% by combined z-score as high risk
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.shape[1] == 0:
            result["status"] = "no_numeric"
            result["summary"] = "No numeric columns found for risk scoring."
            return result
        z_scores = (numeric_X - numeric_X.mean()) / (numeric_X.std() + 1e-9)
        combined_score = z_scores.abs().mean(axis=1)
        y = (combined_score > combined_score.quantile(0.80)).astype(int)
    else:
        raw_target = df[target_col].copy()
        if raw_target.dtype == object:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(raw_target.astype(str).fillna("unknown")))
        else:
            y = pd.to_numeric(raw_target, errors="coerce").fillna(0)
            median_val = y.median()
            y = (y > median_val).astype(int)

    # Align indices
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Train XGBoost
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss", verbosity=0)
        model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        # Predict risk scores on full dataset
        risk_proba = model.predict_proba(X)[:, 1]
        risk_labels = ["High Risk" if p > 0.5 else "Low Risk" for p in risk_proba]

        risk_df = df.copy()
        risk_df["__risk_score"] = risk_proba
        risk_df["__risk_label"] = risk_labels
        result["risk_df"] = risk_df

        # Feature importances
        importances = model.feature_importances_
        feat_imp = pd.DataFrame({"Feature": feature_cols, "Importance": importances})
        feat_imp = feat_imp.sort_values("Importance", ascending=True).tail(10)

        fig_imp = px.bar(
            feat_imp,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Risk Factors (Feature Importance)",
            color="Importance",
            color_continuous_scale="Reds",
            template="plotly_dark",
        )
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
            showlegend=False,
        )
        result["figures"].append(("Risk Factor Importance", fig_imp))

        # Risk distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=risk_proba,
            nbinsx=30,
            name="Risk Score Distribution",
            marker_color="#FF6B6B",
            opacity=0.85,
        ))
        fig_dist.update_layout(
            title="Risk Score Distribution",
            xaxis_title="Risk Probability",
            yaxis_title="Count",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
        )
        result["figures"].append(("Risk Score Distribution", fig_dist))

        # Risk heatmap if region column exists
        region_col = next((c for c, cat in col_map.items() if cat == "region" and c in df.columns), None)
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)

        if region_col and drug_col:
            heatmap_data = risk_df.groupby([region_col, drug_col])["__risk_score"].mean().reset_index()
            pivot = heatmap_data.pivot(index=region_col, columns=drug_col, values="__risk_score").fillna(0)
            fig_heat = px.imshow(
                pivot,
                title="Risk Heatmap: Region vs Drug",
                color_continuous_scale="RdYlGn_r",
                template="plotly_dark",
                aspect="auto",
            )
            fig_heat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAF0",
            )
            result["figures"].append(("Risk Heatmap", fig_heat))
        elif drug_col:
            drug_risk = risk_df.groupby(drug_col)["__risk_score"].mean().sort_values(ascending=False).head(15)
            fig_drug_risk = px.bar(
                x=drug_risk.index,
                y=drug_risk.values,
                title="Average Risk Score by Drug",
                labels={"x": "Drug", "y": "Avg Risk Score"},
                color=drug_risk.values,
                color_continuous_scale="RdYlGn_r",
                template="plotly_dark",
            )
            fig_drug_risk.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#E8EAF0",
                showlegend=False,
            )
            result["figures"].append(("Drug Risk Scores", fig_drug_risk))

        high_risk_count = sum(1 for p in risk_proba if p > 0.5)
        result["summary"] = (
            f"Risk Analysis completed on {len(df)} records.\n"
            f"High-risk prescriptions identified: {high_risk_count} ({100*high_risk_count/len(df):.1f}%).\n"
            f"Top risk factors: {', '.join(feat_imp.tail(3)['Feature'].tolist())}.\n"
            f"Model trained with XGBoost on {len(feature_cols)} features."
        )

        # Performance metrics
        duration = (time.perf_counter() - start_time) * 1000
        result["metrics"] = {
            "Accuracy": f"{acc*100:.1f}%",
            "Precision": f"{prec*100:.1f}%",
            "Recall": f"{rec*100:.1f}%",
            "High Risk": f"{high_risk_count}",
            "Execution": f"{duration:.1f}ms",
            "Model": "XGBoost"
        }

    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Risk agent error: {str(e)}"

    return result
