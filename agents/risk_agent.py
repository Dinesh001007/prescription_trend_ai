import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import plotly.express as px
import plotly.graph_objects as go
import time


class RiskMLP(nn.Module):
    def __init__(self, input_dim):
        super(RiskMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def run_risk_agent(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Risk Agent: Uses Deep Learning (PyTorch MLP) to identify high-risk prescriptions.
    """
    start_time = time.perf_counter()
    result = {"status": "ok", "figures": [], "summary": "", "risk_df": None, "metrics": {}}

    # Select usable columns
    feature_cols = []
    target_col = None

    for col, cat in col_map.items():
        if cat == "risk_score" and col in df.columns:
            target_col = col
        elif cat not in ["patient_id", "date"] and col in df.columns:
            feature_cols.append(col)

    if len(feature_cols) < 2:
        result["status"] = "insufficient_columns"
        result["summary"] = "Not enough feature columns for risk analysis."
        return result

    # Preprocessing
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "category":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str).fillna("unknown"))
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)

    # Scaling is crucial for Deep Learning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if target_col is None:
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.shape[1] == 0:
            result["status"] = "no_numeric"
            result["summary"] = "No numeric columns found for risk scoring."
            return result
        z_scores = (numeric_X - numeric_X.mean()) / (numeric_X.std() + 1e-9)
        combined_score = z_scores.abs().mean(axis=1)
        y = (combined_score > combined_score.quantile(0.80)).astype(int).values
    else:
        raw_target = df[target_col].copy()
        if raw_target.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(raw_target.astype(str).fillna("unknown"))
        else:
            y = pd.to_numeric(raw_target, errors="coerce").fillna(0)
            median_val = y.median()
            y = (y > median_val).astype(int).values

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to Tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).view(-1, 1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).view(-1, 1)

    # Initialize Model
    model = RiskMLP(X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Training Loop
    model.train()
    epochs = 50
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        y_pred = (test_outputs > 0.5).float().numpy()
        
        full_outputs = model(torch.FloatTensor(X_scaled))
        risk_proba = full_outputs.view(-1).numpy()

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    # Prepare Results
    risk_labels = ["High Risk" if p > 0.5 else "Low Risk" for p in risk_proba]
    risk_df = df.copy()
    risk_df["__risk_score"] = risk_proba
    risk_df["__risk_label"] = risk_labels
    result["risk_df"] = risk_df

    # Feature Importance (Proxy for MLP: using input weights)
    # Note: Not as direct as XGBoost, but using first layer mean absolute weights
    weights = torch.abs(model.net[0].weight).mean(dim=0).detach().numpy()
    feat_imp = pd.DataFrame({"Feature": feature_cols, "Importance": weights})
    feat_imp = feat_imp.sort_values("Importance", ascending=True).tail(10)

    fig_imp = px.bar(
        feat_imp, x="Importance", y="Feature", orientation="h",
        title="Clinical Feature Weights (Deep Learning)",
        color="Importance", color_continuous_scale="Reds", template="plotly_dark",
    )
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF0", showlegend=False)
    result["figures"].append(("Model Weights", fig_imp))

    # Risk Distribution
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=risk_proba, nbinsx=30, name="Risk Probability", marker_color="#FF6B6B", opacity=0.85))
    fig_dist.update_layout(
        title="Neural Risk Score Distribution", xaxis_title="Risk Probability", yaxis_title="Count",
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF0",
    )
    result["figures"].append(("Risk Distribution", fig_dist))

    # Risk Heatmap
    region_col = next((c for c, cat in col_map.items() if cat == "region" and c in df.columns), None)
    drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)

    if region_col and drug_col:
        heatmap_data = risk_df.groupby([region_col, drug_col])["__risk_score"].mean().reset_index()
        pivot = heatmap_data.pivot(index=region_col, columns=drug_col, values="__risk_score").fillna(0)
        fig_heat = px.imshow(pivot, title="Neural Risk Heatmap", color_continuous_scale="RdYlGn_r", template="plotly_dark", aspect="auto")
        fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF0")
        result["figures"].append(("Risk Heatmap", fig_heat))

    high_risk_count = sum(1 for p in risk_proba if p > 0.5)
    result["summary"] = (
        f"Deep Learning Risk Analysis (MLP) completed.\n"
        f"Neural network converged over {epochs} epochs.\n"
        f"High-risk prescriptions detected: {high_risk_count} ({100*high_risk_count/len(df):.1f}%).\n"
        f"Primary features identified: {', '.join(feat_imp.tail(3)['Feature'].tolist())}."
    )

    duration = (time.perf_counter() - start_time) * 1000
    result["metrics"] = {
        "Accuracy": f"{acc*100:.1f}%",
        "Precision": f"{prec*100:.1f}%",
        "Recall": f"{rec*100:.1f}%",
        "Loss (BCE)": f"{loss.item():.4f}",
        "Execution": f"{duration:.1f}ms",
        "Model": "PyTorch MLP"
    }

    return result
