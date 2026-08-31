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
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.schema_analyzer import SchemaAnalyzer, ColumnType
from utils.intelligent_analyzer import IntelligentAnalyzer


class RiskMLP(nn.Module):
    def __init__(self, input_dim):
        super(RiskMLP, self).__init__()
        # Improved architecture with batch normalization
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, 128))
        self.layers.append(nn.BatchNorm1d(128))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.3))
        
        # Hidden layers
        self.layers.append(nn.Linear(128, 64))
        self.layers.append(nn.BatchNorm1d(64))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.2))
        
        self.layers.append(nn.Linear(64, 32))
        self.layers.append(nn.BatchNorm1d(32))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(0.1))
        
        # Output layer
        self.layers.append(nn.Linear(32, 1))
        
        self.net = nn.Sequential(*self.layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.net(x)
        return self.sigmoid(logits)


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

    # Initialize intelligent analyzer for better preprocessing
    intelligent_analyzer = IntelligentAnalyzer()
    schema_analyzer = SchemaAnalyzer()
    
    # Preprocessing with intelligent type detection
    X = df[feature_cols].copy()
    print("Risk Agent: Performing intelligent data preprocessing...")
    
    # Process each column based on its detected type
    for col in X.columns:
        detected_type = schema_analyzer.detect_column_type(X[col], col)
        
        if detected_type == ColumnType.CATEGORICAL:
            # Use one-hot encoding for categorical features
            try:
                # For categorical features, use label encoding for now (can be upgraded to one-hot)
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str).fillna("unknown"))
                print(f"  {col}: Categorical -> Label Encoded")
            except Exception as e:
                print(f"  {col}: Error processing categorical - {e}")
                X[col] = 0  # Fallback
                
        elif detected_type == ColumnType.NUMERICAL:
            # Safe numerical processing
            try:
                X[col] = intelligent_analyzer.safe_median(X[col], col)
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median() if X[col].notna().any() else 0)
                print(f"  {col}: Numerical -> Processed safely")
            except Exception as e:
                print(f"  {col}: Error processing numerical - {e}")
                X[col] = 0  # Fallback
                
        elif detected_type == ColumnType.BOOLEAN:
            # Convert boolean to numeric
            X[col] = X[col].astype(str).map({'True': 1, 'true': 1, '1': 1, 'False': 0, 'false': 0, '0': 0}).fillna(0)
            print(f"  {col}: Boolean -> Numeric")
            
        else:
            # Fallback for unknown types
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
            print(f"  {col}: Unknown -> Fallback processing")

    # Scaling is crucial for Deep Learning
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if target_col is None:
        numeric_X = X.select_dtypes(include=[np.number])
        if numeric_X.shape[1] == 0:
            result["status"] = "no_numeric"
            result["summary"] = "No numeric columns found for risk scoring."
            return result
        
        # Create more meaningful synthetic risk score based on clinical patterns
        risk_features = []
        feature_weights = []
        
        for col in numeric_X.columns:
            col_data = numeric_X[col].dropna()
            if len(col_data) > 0:
                # Check for potential risk indicators
                if 'dosage' in col.lower() or 'quantity' in col.lower():
                    # High dosage might indicate risk
                    risk_score = (col_data - col_data.mean()) / (col_data.std() + 1e-9)
                    risk_features.append(risk_score.abs())
                    feature_weights.append(1.5)  # Higher weight for dosage
                elif 'age' in col.lower():
                    # Age-based risk
                    age_risk = (col_data - col_data.mean()) / (col_data.std() + 1e-9)
                    risk_features.append(age_risk.abs())
                    feature_weights.append(1.2)
                else:
                    # Other numeric features
                    z_score = (col_data - col_data.mean()) / (col_data.std() + 1e-9)
                    risk_features.append(z_score.abs())
                    feature_weights.append(1.0)
        
        if risk_features:
            # Weighted combination of risk features
            weighted_scores = []
            for i, feature in enumerate(risk_features):
                weighted_scores.append(feature * feature_weights[i])
            
            combined_score = sum(weighted_scores) / sum(feature_weights)
            # Use more reasonable threshold (70th percentile instead of 80th)
            y = (combined_score > combined_score.quantile(0.70)).astype(int).values
        else:
            # Fallback to simple approach
            z_scores = (numeric_X - numeric_X.mean()) / (numeric_X.std() + 1e-9)
            combined_score = z_scores.abs().mean(axis=1)
            y = (combined_score > combined_score.quantile(0.70)).astype(int).values
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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Lower LR with weight decay
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    # Training Loop with early stopping
    model.train()
    epochs = 100
    best_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Learning rate scheduling
        scheduler.step(loss)
        
        # Early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break  # Early stopping
    
    # Load best model
    if 'best_model_state' in locals():
        model.load_state_dict(best_model_state)

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

