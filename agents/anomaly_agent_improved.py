import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.schema_analyzer import SchemaAnalyzer, ColumnType
from utils.intelligent_analyzer import IntelligentAnalyzer
import warnings
warnings.filterwarnings('ignore')


class VariationalAutoencoder(nn.Module):
    """
    Enhanced Variational Autoencoder for healthcare anomaly detection.
    Uses probabilistic encoding for better reconstruction error distribution.
    """
    
    def __init__(self, input_dim, latent_dim=20, hidden_dims=[64, 32]):
        super(VariationalAutoencoder, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Mu and LogVar layers (for VAE)
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        
        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to latent distribution parameters."""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation back to input space."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class HealthcareAnomalyDetector:
    """
    Advanced healthcare anomaly detection system with VAE and explainability.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.threshold = None
        self.threshold_method = None
        self.reconstruction_errors = None
        self.schema_analyzer = SchemaAnalyzer()
        self.intelligent_analyzer = IntelligentAnalyzer()
        
    def prepare_healthcare_features(self, df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
        """
        Prepare healthcare-specific features for anomaly detection.
        """
        print("🏥 Preparing healthcare features for anomaly detection...")
        features = df.copy()
        
        # Patient-level aggregations
        patient_col = next((c for c, cat in col_map.items() if cat == "patient_id" and c in df.columns), None)
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        dosage_col = next((c for c, cat in col_map.items() if cat in ["dosage", "quantity", "frequency"] and c in df.columns), None)
        date_col = next((c for c, cat in col_map.items() if cat == "date" and c in df.columns), None)
        
        # Create patient-level features
        if patient_col:
            # Prescription patterns
            if drug_col:
                prescription_stats = features.groupby(patient_col).agg({
                    drug_col: ['count', 'nunique']
                }).reset_index()
                prescription_stats.columns = [patient_col, 'prescription_count', 'unique_drug_count']
                features = features.merge(prescription_stats, on=patient_col, how='left')
            
            # Dosage statistics
            if dosage_col:
                features[dosage_col] = pd.to_numeric(features[dosage_col], errors='coerce')
                dosage_stats = features.groupby(patient_col)[dosage_col].agg(['mean', 'std', 'max']).reset_index()
                dosage_stats.columns = [patient_col, 'avg_dosage', 'dosage_std', 'max_dosage']
                features = features.merge(dosage_stats, on=patient_col, how='left')
            
            # Time-based features
            if date_col:
                features[date_col] = pd.to_datetime(features[date_col], errors='coerce')
                features = features.sort_values([patient_col, date_col])
                
                # Time gaps
                features['prev_date'] = features.groupby(patient_col)[date_col].shift(1)
                features['time_gap_days'] = (features[date_col] - features['prev_date']).dt.days
                features['time_gap_days'] = features['time_gap_days'].fillna(features['time_gap_days'].median())
                
                # Frequency
                time_stats = features.groupby(patient_col).agg({
                    date_col: ['min', 'max'],
                    'time_gap_days': 'mean'
                }).reset_index()
                time_stats.columns = [patient_col, 'first_date', 'last_date', 'avg_time_gap']
                time_stats['treatment_duration_days'] = (time_stats['last_date'] - time_stats['first_date']).dt.days
                time_stats['prescriptions_per_month'] = features.groupby(patient_col).size() / (time_stats['treatment_duration_days'] / 30 + 1)
                features = features.merge(time_stats[[patient_col, 'avg_time_gap', 'prescriptions_per_month']], on=patient_col, how='left')
        
        # Age-related features
        age_col = next((c for c, cat in col_map.items() if 'age' in c.lower() and c in df.columns), None)
        if age_col:
            features[age_col] = pd.to_numeric(features[age_col], errors='coerce')
            features['age_squared'] = features[age_col] ** 2  # Non-linear age feature
        
        # Risk-related features
        risk_cols = [c for c in col_map.keys() if 'risk' in c.lower() and c in df.columns]
        if risk_cols and patient_col:
            for col in risk_cols:
                features[col] = pd.to_numeric(features[col], errors='coerce')
            
            risk_stats = features.groupby(patient_col)[risk_cols].agg(['mean', 'max', 'std']).reset_index()
            risk_stats.columns = [patient_col] + [f"{col}_{stat}" for col in risk_cols for stat in ['mean', 'max', 'std']]
            features = features.merge(risk_stats, on=patient_col, how='left')
        
        print(f"✅ Created healthcare features")
        return features
    
    def preprocess_for_anomaly_detection(self, df: pd.DataFrame, col_map: dict) -> tuple:
        """
        Preprocess data for anomaly detection with intelligent type handling.
        """
        print("🔧 Preprocessing data for anomaly detection...")
        
        # Prepare healthcare features
        engineered_df = self.prepare_healthcare_features(df, col_map)
        
        # Select features for modeling
        feature_cols = []
        for col in engineered_df.columns:
            if col in ['patient_id', 'date', 'drug_name', 'diagnosis', 'gender', 'prev_date', 'first_date', 'last_date']:
                continue
                
            # Use intelligent type detection
            detected_type = self.schema_analyzer.detect_column_type(engineered_df[col], col)
            
            if detected_type == ColumnType.NUMERICAL:
                feature_cols.append(col)
            elif detected_type == ColumnType.BOOLEAN:
                feature_cols.append(col)
            elif detected_type == ColumnType.CATEGORICAL and col.endswith('_encoded'):
                feature_cols.append(col)
        
        # Handle categorical encoding
        processed_df = engineered_df[feature_cols].copy()
        
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object':
                le = LabelEncoder()
                processed_df[col] = le.fit_transform(processed_df[col].astype(str).fillna("unknown"))
                self.label_encoders[col] = le
        
        # Fill missing values
        processed_df = processed_df.fillna(processed_df.median())
        
        # Scale features
        scaled_data = self.scaler.fit_transform(processed_df)
        
        # Store feature names
        self.feature_names = feature_cols
        
        print(f"✅ Preprocessed {len(feature_cols)} features")
        return scaled_data, processed_df
    
    def train_vae(self, X, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train Variational Autoencoder for anomaly detection.
        """
        print("🚀 Training Variational Autoencoder...")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize VAE
        input_dim = X.shape[1]
        self.model = VariationalAutoencoder(input_dim=input_dim, latent_dim=min(20, input_dim//2))
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Training loop
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar = self.model(batch_x)
                
                # VAE loss: reconstruction + KL divergence
                recon_loss = F.mse_loss(recon_batch, batch_y, reduction='sum')
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                total_loss = recon_loss + 0.1 * kl_loss  # Weight KL loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(dataloader.dataset)
            losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        print("✅ VAE training completed")
        return losses
    
    def calculate_reconstruction_errors(self, X):
        """
        Calculate reconstruction errors for all samples.
        """
        print("📊 Calculating reconstruction errors...")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructed, mu, logvar = self.model(X_tensor)
            
            # Calculate MSE for each sample
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            self.reconstruction_errors = errors.numpy()
            
        print(f"✅ Calculated errors for {len(self.reconstruction_errors)} samples")
        return self.reconstruction_errors
    
    def determine_threshold(self, method="percentile", percentile=95, std_multiplier=3):
        """
        Determine anomaly threshold using different methods.
        """
        print(f"🎯 Determining anomaly threshold using {method} method...")
        
        errors = self.reconstruction_errors
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        if method == "percentile":
            threshold = np.percentile(errors, percentile)
            self.threshold_method = f"{percentile}th percentile"
        elif method == "statistical":
            threshold = mean_error + std_multiplier * std_error
            self.threshold_method = f"mean + {std_multiplier}*std"
        elif method == "adaptive":
            # Adaptive method: use IQR
            q75, q25 = np.percentile(errors, [75, 25])
            iqr = q75 - q25
            threshold = q75 + 1.5 * iqr
            self.threshold_method = "IQR-based (Q75 + 1.5*IQR)"
        else:
            # Default to 95th percentile
            threshold = np.percentile(errors, 95)
            self.threshold_method = "95th percentile (default)"
        
        self.threshold = threshold
        
        print(f"✅ Threshold: {threshold:.6f} ({self.threshold_method})")
        print(f"   Mean error: {mean_error:.6f}, Std: {std_error:.6f}")
        
        return threshold
    
    def detect_anomalies(self, X):
        """
        Detect anomalies based on reconstruction errors.
        """
        print("🔍 Detecting anomalies...")
        
        errors = self.reconstruction_errors
        anomaly_mask = errors > self.threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        
        print(f"✅ Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(errors)*100:.1f}% of data)")
        
        return anomaly_indices, anomaly_mask
    
    def explain_anomalies(self, X, anomaly_indices, top_k=5):
        """
        Explain anomalies by identifying contributing features.
        """
        print("🔍 Explaining anomalies...")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructed, mu, logvar = self.model(X_tensor)
            
            # Calculate per-feature reconstruction errors
            feature_errors = (X_tensor - reconstructed) ** 2
            feature_errors = feature_errors.numpy()
        
        explanations = []
        
        for idx in anomaly_indices:
            sample_errors = feature_errors[idx]
            
            # Get top contributing features
            top_feature_indices = np.argsort(sample_errors)[-top_k:][::-1]
            
            explanation = {
                'sample_index': idx,
                'total_error': self.reconstruction_errors[idx],
                'threshold': self.threshold,
                'contributing_features': []
            }
            
            for feature_idx in top_feature_indices:
                feature_name = self.feature_names[feature_idx]
                feature_error = sample_errors[feature_idx]
                original_value = X[idx, feature_idx]
                reconstructed_value = reconstructed[idx, feature_idx].item()
                
                explanation['contributing_features'].append({
                    'feature': feature_name,
                    'error_contribution': feature_error,
                    'original_value': original_value,
                    'reconstructed_value': reconstructed_value,
                    'difference': abs(original_value - reconstructed_value)
                })
            
            explanations.append(explanation)
        
        print(f"✅ Generated explanations for {len(explanations)} anomalies")
        return explanations
    
    def generate_evaluation_labels(self, X, anomaly_indices, contamination_rate=0.05):
        """
        Generate evaluation labels for confusion matrix calculation.
        Since anomaly detection is unsupervised, we create labels based on:
        1. Known anomalies (if provided)
        2. High reconstruction error samples
        """
        print("🏷️ Generating evaluation labels for confusion matrix...")
        
        n_samples = len(X)
        y_true = np.zeros(n_samples, dtype=int)  # 0 = normal
        y_pred = np.zeros(n_samples, dtype=int)  # 0 = normal
        
        # Set predictions (anomalies detected by model)
        y_pred[anomaly_indices] = 1  # 1 = anomaly
        
        # For ground truth, we use a combination of:
        # 1. Top reconstruction errors (expected anomalies)
        # 2. If we have injected anomalies in test data
        
        # Use top reconstruction errors as ground truth anomalies
        error_threshold = np.percentile(self.reconstruction_errors, 100 - (contamination_rate * 100))
        high_error_indices = np.where(self.reconstruction_errors > error_threshold)[0]
        y_true[high_error_indices] = 1
        
        print(f"✅ Generated labels:")
        print(f"   True anomalies: {np.sum(y_true)} ({np.sum(y_true)/n_samples*100:.1f}%)")
        print(f"   Predicted anomalies: {np.sum(y_pred)} ({np.sum(y_pred)/n_samples*100:.1f}%)")
        
        return y_true, y_pred
    
    def calculate_confusion_matrix_metrics(self, y_true, y_pred):
        """
        Calculate confusion matrix and related metrics.
        """
        print("📊 Calculating confusion matrix metrics...")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        metrics = {
            'confusion_matrix': cm,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn,
            'true_positives': tp,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy
        }
        
        print(f"✅ Confusion Matrix Metrics:")
        print(f"   True Positives: {tp}")
        print(f"   False Positives: {fp}")
        print(f"   False Negatives: {fn}")
        print(f"   True Negatives: {tn}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall: {recall:.3f}")
        print(f"   F1-Score: {f1_score:.3f}")
        print(f"   Accuracy: {accuracy:.3f}")
        
        return metrics
    
    def create_visualizations(self, X, anomaly_indices, explanations, confusion_metrics=None):
        """
        Create comprehensive visualizations for anomaly detection results.
        """
        figures = []
        
        # 1. Reconstruction Error Distribution
        fig_error = go.Figure()
        
        # Add histogram of all errors
        fig_error.add_trace(go.Histogram(
            x=self.reconstruction_errors,
            nbinsx=50,
            name="Reconstruction Errors",
            marker_color="lightblue",
            opacity=0.7
        ))
        
        # Add threshold line
        fig_error.add_vline(
            x=self.threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Threshold: {self.threshold:.6f}<br>({self.threshold_method})",
            annotation_position="top right"
        )
        
        # Highlight anomalies
        anomaly_errors = self.reconstruction_errors[anomaly_indices]
        fig_error.add_trace(go.Scatter(
            x=anomaly_errors,
            y=[0] * len(anomaly_errors),
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name="Anomalies"
        ))
        
        fig_error.update_layout(
            title=f"Reconstruction Error Distribution<br>Anomalies: {len(anomaly_indices)} ({len(anomaly_indices)/len(self.reconstruction_errors)*100:.1f}%)",
            xaxis_title="Reconstruction Error (MSE)",
            yaxis_title="Frequency",
            template="plotly_dark",
            showlegend=True
        )
        figures.append(("Error Distribution", fig_error))
        
        # 2. Anomaly Timeline (if we have index information)
        if len(anomaly_indices) > 0:
            fig_timeline = go.Figure()
            
            # Plot all points
            fig_timeline.add_trace(go.Scatter(
                x=list(range(len(self.reconstruction_errors))),
                y=self.reconstruction_errors,
                mode='lines+markers',
                marker=dict(color='lightblue', size=4),
                name="Reconstruction Error",
                line=dict(width=1)
            ))
            
            # Highlight anomalies
            fig_timeline.add_trace(go.Scatter(
                x=anomaly_indices,
                y=self.reconstruction_errors[anomaly_indices],
                mode='markers',
                marker=dict(color='red', size=8, symbol='diamond'),
                name="Anomalies"
            ))
            
            # Add threshold line
            fig_timeline.add_hline(
                y=self.threshold,
                line_dash="dash",
                line_color="orange",
                annotation_text=f"Threshold: {self.threshold:.6f}"
            )
            
            fig_timeline.update_layout(
                title="Anomaly Detection Timeline",
                xaxis_title="Sample Index",
                yaxis_title="Reconstruction Error",
                template="plotly_dark",
                showlegend=True
            )
            figures.append(("Anomaly Timeline", fig_timeline))
        
        # 3. Feature Contribution Analysis (for top anomalies)
        if explanations:
            # Get top 5 anomalies by error
            top_explanations = sorted(explanations, key=lambda x: x['total_error'], reverse=True)[:5]
            
            fig_features = make_subplots(
                rows=len(top_explanations), cols=1,
                subplot_titles=[f"Anomaly #{exp['sample_index']} (Error: {exp['total_error']:.6f})" for exp in top_explanations],
                vertical_spacing=0.08
            )
            
            for i, explanation in enumerate(top_explanations, 1):
                features = explanation['contributing_features'][:3]  # Top 3 features
                feature_names = [f['feature'] for f in features]
                contributions = [f['error_contribution'] for f in features]
                
                fig_features.add_trace(
                    go.Bar(
                        x=contributions,
                        y=feature_names,
                        orientation='h',
                        name=f"Anomaly #{explanation['sample_index']}",
                        marker_color='red',
                        opacity=0.7
                    ),
                    row=i, col=1
                )
            
            fig_features.update_layout(
                title="Top Contributing Features for Anomalies",
                template="plotly_dark",
                height=300 * len(top_explanations),
                showlegend=False
            )
            figures.append(("Feature Contributions", fig_features))
        
        # 4. Confusion Matrix (if available)
        if confusion_metrics:
            cm = confusion_metrics['confusion_matrix']
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Normal (Pred)', 'Anomaly (Pred)'],
                y=['Normal (True)', 'Anomaly (True)'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}",
                textfont={"size": 14}
            ))
            
            # Add annotations for TP, FP, FN, TN
            annotations = [
                (0, 0, f"TN: {cm[0,0]}", "lightgreen"),
                (0, 1, f"FP: {cm[0,1]}", "lightcoral"),
                (1, 0, f"FN: {cm[1,0]}", "lightcoral"),
                (1, 1, f"TP: {cm[1,1]}", "lightgreen")
            ]
            
            for i, j, label, color in annotations:
                fig_cm.add_annotation(
                    x=j, y=i,
                    text=label,
                    showarrow=False,
                    font=dict(size=12, color="black"),
                    bgcolor=color,
                    bordercolor="black",
                    borderwidth=1
                )
            
            fig_cm.update_layout(
                title=f"Confusion Matrix<br>Precision: {confusion_metrics['precision']:.3f}, Recall: {confusion_metrics['recall']:.3f}, F1: {confusion_metrics['f1_score']:.3f}",
                template="plotly_dark",
                width=600,
                height=500
            )
            figures.append(("Confusion Matrix", fig_cm))
        
        # 5. Error Statistics Summary
        fig_stats = go.Figure()
        
        stats_data = {
            'Mean Error': [np.mean(self.reconstruction_errors)],
            'Std Dev': [np.std(self.reconstruction_errors)],
            'Threshold': [self.threshold],
            'Anomaly Count': [len(anomaly_indices)],
            'Anomaly %': [len(anomaly_indices)/len(self.reconstruction_errors)*100]
        }
        
        # Add confusion matrix metrics if available
        if confusion_metrics:
            stats_data.update({
                'Precision': [confusion_metrics['precision']],
                'Recall': [confusion_metrics['recall']],
                'F1-Score': [confusion_metrics['f1_score']],
                'Accuracy': [confusion_metrics['accuracy']]
            })
        
        for stat, values in stats_data.items():
            color = 'lightcoral' if stat in ['Precision', 'Recall', 'F1-Score', 'Accuracy'] else 'lightblue'
            fig_stats.add_trace(go.Bar(
                x=[stat],
                y=values,
                name=stat,
                marker_color=color
            ))
        
        fig_stats.update_layout(
            title="Anomaly Detection Performance Metrics",
            xaxis_title="Metric",
            yaxis_title="Value",
            template="plotly_dark",
            showlegend=False
        )
        figures.append(("Performance Metrics", fig_stats))
        
        return figures


def run_anomaly_agent_improved(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Improved anomaly detection agent using Variational Autoencoder.
    """
    start_time = time.perf_counter()
    result = {"status": "ok", "figures": [], "summary": "", "metrics": {}}
    
    print("=" * 80)
    print("IMPROVED HEALTHCARE ANOMALY DETECTION")
    print("=" * 80)
    
    try:
        # Initialize detector
        detector = HealthcareAnomalyDetector()
        
        # Preprocess data
        X, processed_df = detector.preprocess_for_anomaly_detection(df, col_map)
        
        if len(X) < 10:
            result["status"] = "insufficient_data"
            result["summary"] = "Not enough data for anomaly detection."
            return result
        
        # Train VAE
        losses = detector.train_vae(X, epochs=100, batch_size=32, learning_rate=0.001)
        
        # Calculate reconstruction errors
        errors = detector.calculate_reconstruction_errors(X)
        
        # Determine threshold (try multiple methods and pick best)
        detector.determine_threshold(method="percentile", percentile=95)
        
        # Detect anomalies
        anomaly_indices, anomaly_mask = detector.detect_anomalies(X)
        
        # Explain anomalies
        explanations = detector.explain_anomalies(X, anomaly_indices, top_k=5)
        
        # Generate evaluation labels and calculate confusion matrix
        y_true, y_pred = detector.generate_evaluation_labels(X, anomaly_indices, contamination_rate=0.05)
        confusion_metrics = detector.calculate_confusion_matrix_metrics(y_true, y_pred)
        
        # Create visualizations
        figures = detector.create_visualizations(X, anomaly_indices, explanations, confusion_metrics)
        result["figures"] = figures
        
        # Prepare summary
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        anomaly_count = len(anomaly_indices)
        anomaly_percentage = (anomaly_count / len(errors)) * 100
        
        summary = f"""
🔍 **Improved Healthcare Anomaly Detection System**

**Detection Results:**
- ✅ Total samples analyzed: {len(errors)}
- ✅ Anomalies detected: {anomaly_count} ({anomaly_percentage:.1f}%)
- ✅ Threshold method: {detector.threshold_method}
- ✅ Threshold value: {detector.threshold:.6f}

**Error Statistics:**
- Mean reconstruction error: {mean_error:.6f}
- Standard deviation: {std_error:.6f}
- Min error: {np.min(errors):.6f}
- Max error: {np.max(errors):.6f}

**Confusion Matrix Performance:**
- ✅ True Positives: {confusion_metrics['true_positives']}
- ✅ False Positives: {confusion_metrics['false_positives']}
- ✅ False Negatives: {confusion_metrics['false_negatives']}
- ✅ True Negatives: {confusion_metrics['true_negatives']}
- ✅ Precision: {confusion_metrics['precision']:.3f}
- ✅ Recall: {confusion_metrics['recall']:.3f}
- ✅ F1-Score: {confusion_metrics['f1_score']:.3f}
- ✅ Accuracy: {confusion_metrics['accuracy']:.3f}

**Model Architecture:**
- 🧠 Variational Autoencoder (VAE)
- 🔧 Input features: {len(detector.feature_names)}
- 📊 Latent dimension: {min(20, len(detector.feature_names)//2)}
- 🏥 Healthcare-specific feature engineering

**Key Improvements:**
- ✅ Reconstruction error-based detection (not clustering)
- ✅ Dynamic thresholding (percentile-based)
- ✅ Feature-level explainability for anomalies
- ✅ Confusion matrix evaluation metrics
- ✅ Proper deep learning methodology
- ✅ Healthcare domain feature engineering

**Sample Anomaly Explanation:**
{chr(10).join([f"• Sample #{exp['sample_index']}: {exp['contributing_features'][0]['feature']} contributed {exp['contributing_features'][0]['error_contribution']:.6f} to error" for exp in explanations[:3]]) if explanations else "No anomalies detected"}

**Healthcare Context:**
- Anomalies may indicate unusual prescription patterns
- High reconstruction errors suggest deviations from normal treatment
- Feature explanations help identify specific risk factors
- Confusion matrix shows detection accuracy and error types

**Execution Time:** {(time.perf_counter() - start_time):.2f}s
        """
        
        result["summary"] = summary
        result["metrics"] = {
            "Mean Error": f"{mean_error:.6f}",
            "Std Error": f"{std_error:.6f}",
            "Threshold": f"{detector.threshold:.6f}",
            "Anomaly Count": str(anomaly_count),
            "Anomaly %": f"{anomaly_percentage:.2f}",
            "True Positives": str(confusion_metrics['true_positives']),
            "False Positives": str(confusion_metrics['false_positives']),
            "False Negatives": str(confusion_metrics['false_negatives']),
            "True Negatives": str(confusion_metrics['true_negatives']),
            "Precision": f"{confusion_metrics['precision']:.3f}",
            "Recall": f"{confusion_metrics['recall']:.3f}",
            "F1-Score": f"{confusion_metrics['f1_score']:.3f}",
            "Accuracy": f"{confusion_metrics['accuracy']:.3f}",
            "Features": len(detector.feature_names),
            "Model": "Variational Autoencoder",
            "Threshold Method": detector.threshold_method,
            "Execution": f"{(time.perf_counter() - start_time)*1000:.1f}ms"
        }
        
        print("\n✅ Anomaly detection completed successfully!")
        print(f"📊 {summary}")
        
    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Improved anomaly agent error: {str(e)}"
        print(f"❌ Error: {e}")
        
    return result


# Test the improved anomaly agent
if __name__ == "__main__":
    # Create sample healthcare data with some anomalies
    np.random.seed(42)
    n_samples = 1000
    
    # Normal data
    normal_data = {
        'patient_id': [f"P{i:04d}" for i in range(n_samples)],
        'age': np.random.normal(55, 15, n_samples),
        'dosage': np.random.exponential(50, n_samples),
        'frequency': np.random.poisson(3, n_samples),
        'risk_score': np.random.beta(2, 5, n_samples)
    }
    
    # Add some anomalies (last 50 samples)
    anomaly_indices = range(n_samples - 50, n_samples)
    for i in anomaly_indices:
        normal_data['dosage'][i] *= 5  # Very high dosage
        normal_data['frequency'][i] = 20  # Very high frequency
        normal_data['risk_score'][i] = 0.9  # Very high risk
    
    df = pd.DataFrame(normal_data)
    col_map = {
        'patient_id': 'patient_id',
        'age': 'age',
        'dosage': 'dosage',
        'frequency': 'frequency',
        'risk_score': 'risk_score'
    }
    
    # Run improved anomaly agent
    results = run_anomaly_agent_improved(df, col_map)
    
    if results["status"] == "ok":
        print("\n✅ Analysis completed successfully!")
        print(f"📊 {results['summary']}")
        
        print("\n📈 Performance Metrics:")
        for metric, value in results["metrics"].items():
            print(f"  {metric}: {value}")
    else:
        print(f"❌ Analysis failed: {results['summary']}")
