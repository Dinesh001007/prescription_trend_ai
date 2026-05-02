import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
import time


class AnomalyAE(nn.Module):
    def __init__(self, input_dim):
        super(AnomalyAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8), # Latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def run_anomaly_agent(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Anomaly Agent: Uses Deep Learning (PyTorch Autoencoder) to detect unusual prescriptions.
    """
    start_time = time.perf_counter()
    result = {"status": "ok", "figures": [], "summary": "", "anomaly_df": None, "metrics": {}}

    feature_cols = [
        col for col, cat in col_map.items()
        if cat not in ["patient_id", "date"] and col in df.columns
    ]

    if len(feature_cols) < 2:
        result["status"] = "insufficient_columns"
        result["summary"] = "Not enough columns for deep anomaly detection."
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
    X_tensor = torch.FloatTensor(X_scaled)

    try:
        # Initialize and Train Autoencoder
        input_dim = X_scaled.shape[1]
        model = AnomalyAE(input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        model.train()
        epochs = 60
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, X_tensor)
            loss.backward()
            optimizer.step()

        # Detection: Compute Reconstruction Error
        model.eval()
        with torch.no_grad():
            reconstructed = model(X_tensor)
            # MSE per sample
            errors = torch.mean((X_tensor - reconstructed)**2, dim=1).numpy()

        # Heuristic threshold: Top 10% errors are anomalies
        threshold = np.percentile(errors, 90)
        preds = np.where(errors > threshold, -1, 1)

        # Silhouette score (unsupervised quality metric)
        if len(X_scaled) > 1000:
            indices = np.random.choice(len(X_scaled), 1000, replace=False)
            sil_score = silhouette_score(X_scaled[indices], preds[indices])
        else:
            sil_score = silhouette_score(X_scaled, preds)

        anomaly_df = df.copy()
        anomaly_df["__anomaly"] = ["Anomaly" if p == -1 else "Normal" for p in preds]
        anomaly_df["__anomaly_score"] = errors
        result["anomaly_df"] = anomaly_df

        n_anomalies = (preds == -1).sum()

        # Anomaly score distribution
        fig_score = go.Figure()
        fig_score.add_trace(go.Histogram(
            x=errors[preds == 1], name="Normal (Low Error)", marker_color="#00C9A7", opacity=0.75, nbinsx=30
        ))
        fig_score.add_trace(go.Histogram(
            x=errors[preds == -1], name="Anomaly (High Error)", marker_color="#FF6B6B", opacity=0.85, nbinsx=20
        ))
        fig_score.update_layout(
            title="Autoencoder Reconstruction Error Distribution",
            xaxis_title="MSE Reconstruction Error",
            yaxis_title="Count",
            barmode="overlay",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
        )
        result["figures"].append(("Reconstruction Error", fig_score))

        # Latent Space Visualization (Encoder output)
        with torch.no_grad():
            latent = model.encoder(X_tensor).numpy()
        
        if latent.shape[1] >= 2:
            fig_latent = px.scatter(
                x=latent[:, 0], y=latent[:, 1],
                color=anomaly_df["__anomaly"],
                title="Deep Latent Space Projection (Encoder)",
                labels={"x": "Latent Dim 1", "y": "Latent Dim 2"},
                color_discrete_map={"Normal": "#00C9A7", "Anomaly": "#FF6B6B"},
                template="plotly_dark",
                opacity=0.7,
            )
            fig_latent.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF0")
            result["figures"].append(("Latent Space", fig_latent))

        # Region-level heatmap
        region_col = next((c for c, cat in col_map.items() if cat == "region" and c in df.columns), None)
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        if region_col and drug_col:
            heat_data = anomaly_df[anomaly_df["__anomaly"] == "Anomaly"].groupby([region_col, drug_col]).size().reset_index(name="Anomaly Count")
            if not heat_data.empty:
                pivot = heat_data.pivot(index=region_col, columns=drug_col, values="Anomaly Count").fillna(0)
                fig_heat = px.imshow(pivot, title="Neural Anomaly Heatmap", color_continuous_scale="Reds", template="plotly_dark", aspect="auto")
                fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#E8EAF0")
                result["figures"].append(("Anomaly Heatmap", fig_heat))

        result["summary"] = (
            f"Neural Anomaly Detection (Autoencoder) completed.\n"
            f"Detected {n_anomalies} anomalies based on high reconstruction error.\n"
            f"Model learned a {latent.shape[1]}-dimensional latent representation of clinical patterns.\n"
            f"Anomalies represent records that deviate significantly from the learned 'normal' baseline."
        )

        duration = (time.perf_counter() - start_time) * 1000
        result["metrics"] = {
            "Silhouette": f"{sil_score:.3f}",
            "Final Loss": f"{loss.item():.6f}",
            "Anomalies": f"{n_anomalies}",
            "Anomaly %": f"{100*n_anomalies/len(df):.1f}%",
            "Execution": f"{duration:.1f}ms",
            "Model": "Deep Autoencoder"
        }

    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Anomaly agent error: {str(e)}"

    return result
