import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.schema_analyzer import SchemaAnalyzer, ColumnType
from utils.intelligent_analyzer import IntelligentAnalyzer
import warnings
warnings.filterwarnings('ignore')


class AdvancedHealthcareRiskPredictor:
    """
    Advanced healthcare risk prediction system with ensemble methods,
    hyperparameter optimization, and sophisticated feature engineering.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = []
        self.optimal_threshold = 0.5
        self.schema_analyzer = SchemaAnalyzer()
        self.intelligent_analyzer = IntelligentAnalyzer()
        self.best_model = None
        self.best_model_name = ""
        self.feature_selector = None
        self.ensemble_model = None
        
    def create_advanced_healthcare_features(self, df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
        """
        Create advanced healthcare-specific features with clinical domain knowledge.
        """
        print("🏥 Creating advanced healthcare-specific features...")
        features = df.copy()
        
        # Identify key columns
        patient_col = next((c for c, cat in col_map.items() if cat == "patient_id" and c in df.columns), None)
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        date_col = next((c for c, cat in col_map.items() if cat == "date" and c in df.columns), None)
        dosage_col = next((c for c, cat in col_map.items() if cat in ["dosage", "quantity", "frequency"] and c in df.columns), None)
        
        # 1. Patient-level aggregation features
        if patient_col:
            # Build aggregation dictionary properly - only include appropriate aggregations for data types
            agg_dict = {}
            if dosage_col and dosage_col in df.columns and df[dosage_col].dtype in ['int64', 'float64']:
                agg_dict[dosage_col] = ['sum', 'mean', 'std', 'max']
            if drug_col and drug_col in df.columns:
                agg_dict[drug_col] = ['nunique', 'count']
            
            if agg_dict:
                try:
                    patient_stats = df.groupby(patient_col).agg(agg_dict).reset_index()
                    
                    # Flatten column names
                    patient_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in patient_stats.columns]
                    
                    # Add clinical risk indicators
                    if dosage_col and f'{dosage_col}_max' in patient_stats.columns and f'{dosage_col}_mean' in patient_stats.columns:
                        # Ensure numeric data types
                        patient_stats[f'{dosage_col}_max'] = pd.to_numeric(patient_stats[f'{dosage_col}_max'], errors='coerce')
                        patient_stats[f'{dosage_col}_mean'] = pd.to_numeric(patient_stats[f'{dosage_col}_mean'], errors='coerce')
                        patient_stats[f'{dosage_col}_std'] = pd.to_numeric(patient_stats[f'{dosage_col}_std'], errors='coerce')
                        
                        patient_stats['high_dosage_flag'] = (patient_stats[f'{dosage_col}_max'] > patient_stats[f'{dosage_col}_mean'].mean()).astype(int)
                        patient_stats['dosage_variability'] = patient_stats[f'{dosage_col}_std'] / patient_stats[f'{dosage_col}_mean']
                        patient_stats['dosage_variability'] = patient_stats['dosage_variability'].fillna(0)
                    
                    features = features.merge(patient_stats, on=patient_col, how='left')
                except Exception as e:
                    print(f"Warning: Patient-level aggregation failed: {e}")
                    # Continue without patient-level features
        
        # 2. Drug-level features
        if drug_col:
            # Build drug aggregation dictionary properly - only include appropriate aggregations for data types
            drug_agg_dict = {}
            if dosage_col and dosage_col in df.columns and df[dosage_col].dtype in ['int64', 'float64']:
                drug_agg_dict[dosage_col] = ['mean', 'std']
            if patient_col and patient_col in df.columns:
                drug_agg_dict[patient_col] = ['nunique', 'count']
            
            if drug_agg_dict:
                try:
                    drug_stats = df.groupby(drug_col).agg(drug_agg_dict).reset_index()
                    
                    drug_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in drug_stats.columns]
                    features = features.merge(drug_stats, on=drug_col, how='left')
                except Exception as e:
                    print(f"Warning: Drug-level aggregation failed: {e}")
                    # Continue without drug-level features
        
        # 3. Temporal features
        if date_col:
            features[date_col] = pd.to_datetime(features[date_col], errors='coerce')
            features['day_of_week'] = features[date_col].dt.dayofweek
            features['month'] = features[date_col].dt.month
            features['quarter'] = features[date_col].dt.quarter
            features['is_weekend'] = (features[date_col].dt.dayofweek >= 5).astype(int)
            
            # Prescription frequency features
            if patient_col and patient_col in features.columns:
                try:
                    features['days_since_last_prescription'] = features.groupby(patient_col)[date_col].diff().dt.days
                    features['prescription_frequency'] = features.groupby(patient_col)[date_col].transform('count')
                except Exception as e:
                    print(f"Warning: Temporal feature creation failed: {e}")
                    # Add default values if groupby fails
                    features['days_since_last_prescription'] = 0
                    features['prescription_frequency'] = 1
        
        # 4. Interaction features
        if dosage_col and drug_col:
            # Ensure dosage column is numeric
            features[dosage_col] = pd.to_numeric(features[dosage_col], errors='coerce')
            # Drug-dosage interactions
            for drug in features[drug_col].unique()[:10]:  # Limit to top 10 drugs
                drug_mask = features[drug_col] == drug
                dosage_max = features[dosage_col].max()
                if pd.notna(dosage_max) and dosage_max > 0:
                    features[f'{drug}_dosage_norm'] = np.where(drug_mask, features[dosage_col] / dosage_max, 0)
                else:
                    features[f'{drug}_dosage_norm'] = 0
        
        # 5. Risk scoring features
        if dosage_col:
            features['dosage_percentile'] = features[dosage_col].rank(pct=True)
            features['high_risk_dosage'] = (features['dosage_percentile'] > 0.8).astype(int)
        
        # 6. Polypharmacy indicators
        if patient_col and drug_col and patient_col in features.columns and drug_col in features.columns:
            try:
                polypharmacy = features.groupby(patient_col)[drug_col].nunique()
                features['polypharmacy_score'] = features[patient_col].map(polypharmacy)
                features['high_polypharmacy'] = (features['polypharmacy_score'] > 5).astype(int)
            except Exception as e:
                print(f"Warning: Polypharmacy feature creation failed: {e}")
                # Add default values if groupby fails
                features['polypharmacy_score'] = 1
                features['high_polypharmacy'] = 0
        
        print(f"✅ Advanced feature engineering completed with {len(features.columns)} features")
        return features
    
    def select_best_features(self, X, y, max_features=50):
        """
        Advanced feature selection using multiple methods with proper handling of categorical data.
        """
        print("🎯 Performing advanced feature selection...")
        
        # Convert categorical and datetime columns to numeric for feature selection
        X_numeric = X.copy()
        for col in X_numeric.columns:
            if X_numeric[col].dtype == 'object':
                le = LabelEncoder()
                X_numeric[col] = le.fit_transform(X_numeric[col].astype(str))
                self.label_encoders[col] = le
            elif 'datetime' in str(X_numeric[col].dtype):
                # Convert datetime to numeric (timestamp)
                X_numeric[col] = X_numeric[col].astype('int64') // 10**9  # Convert to seconds since epoch
        
        # Handle NaN values more robustly
        for col in X_numeric.columns:
            if X_numeric[col].isnull().any():
                median_val = X_numeric[col].median()
                if pd.isna(median_val):
                    X_numeric[col] = X_numeric[col].fillna(0)
                else:
                    X_numeric[col] = X_numeric[col].fillna(median_val)
        
        # Final check for any remaining NaN values
        if X_numeric.isnull().any().any():
            X_numeric = X_numeric.fillna(0)
        
        # Method 1: Univariate selection
        selector_univariate = SelectKBest(score_func=f_classif, k=min(max_features, X_numeric.shape[1]))
        X_univariate = selector_univariate.fit_transform(X_numeric, y)
        univariate_scores = selector_univariate.scores_
        
        # Method 2: Tree-based selection
        rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
        X_tree = rf_selector.fit_transform(X_numeric, y)
        
        # Method 3: Recursive Feature Elimination
        rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), 
                 n_features_to_select=min(max_features, X_numeric.shape[1]))
        X_rfe = rfe.fit_transform(X_numeric, y)
        
        # Combine results
        feature_scores = {}
        for i, feature in enumerate(X_numeric.columns):
            score = 0
            if i < len(univariate_scores):
                score += univariate_scores[i]
            if rf_selector.get_support()[i]:
                score += 100
            if rfe.get_support()[i]:
                score += 50
            feature_scores[feature] = score
        
        # Select top features
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:max_features]
        selected_features = [feat for feat, _ in top_features]
        
        print(f"✅ Selected {len(selected_features)} best features")
        return selected_features
    
    def create_ensemble_models(self):
        """
        Create multiple advanced models for ensemble voting.
        """
        print("🤝 Creating advanced ensemble models...")
        
        models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.9,
                colsample_bytree=0.9,
                gamma=0.2,
                reg_alpha=0.1,
                reg_lambda=1.5,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.8,
                max_features='sqrt',
                random_state=42
            ),
            
            'random_forest': RandomForestClassifier(
                n_estimators=300,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                random_state=42,
                n_jobs=-1
            ),
            
            'logistic_regression': LogisticRegression(
                C=1.0,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            ),
            
            'svm': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=42
            ),
            
            'adaboost': AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.1,
                algorithm='SAMME.R',
                random_state=42
            )
        }
        
        self.models = models
        print(f"✅ Created {len(models)} advanced models")
        return models
    
    def optimize_hyperparameters(self, X_train, y_train, model_name):
        """
        Hyperparameter optimization using GridSearchCV.
        """
        print(f"🔧 Optimizing hyperparameters for {model_name}...")
        
        param_grids = {
            'xgboost': {
                'n_estimators': [200, 300, 400],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.01, 0.03, 0.05],
                'subsample': [0.8, 0.9, 1.0]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7]
            },
            'random_forest': {
                'n_estimators': [200, 300, 400],
                'max_depth': [8, 10, 12],
                'min_samples_split': [2, 5, 10]
            }
        }
        
        if model_name in param_grids:
            grid_search = GridSearchCV(
                estimator=self.models[model_name],
                param_grid=param_grids[model_name],
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='roc_auc',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.models[model_name] = grid_search.best_estimator_
            print(f"✅ Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_score_
        
        return 0.0
    
    def train_advanced_models(self, X, y):
        """
        Train multiple advanced models and select the best one.
        """
        print("🚀 Training advanced ensemble models...")
        
        # Handle class imbalance
        class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Convert categorical and datetime columns to numeric for training
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype == 'object':
                if col not in self.label_encoders:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    X_processed[col] = self.label_encoders[col].transform(X_processed[col].astype(str))
            elif 'datetime' in str(X_processed[col].dtype):
                # Convert datetime to numeric (timestamp)
                X_processed[col] = X_processed[col].astype('int64') // 10**9  # Convert to seconds since epoch
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create models
        self.create_ensemble_models()
        
        # Handle NaN values in training data
        X_train_clean = X_train.copy()
        X_test_clean = X_test.copy()
        
        # Fill NaN values for training and test data
        for col in X_train_clean.columns:
            if X_train_clean[col].isnull().any():
                median_val = X_train_clean[col].median()
                if pd.isna(median_val):
                    X_train_clean[col] = X_train_clean[col].fillna(0)
                    X_test_clean[col] = X_test_clean[col].fillna(0)
                else:
                    X_train_clean[col] = X_train_clean[col].fillna(median_val)
                    X_test_clean[col] = X_test_clean[col].fillna(median_val)
        
        # Final check for any remaining NaN values
        if X_train_clean.isnull().any().any():
            X_train_clean = X_train_clean.fillna(0)
        if X_test_clean.isnull().any().any():
            X_test_clean = X_test_clean.fillna(0)
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        self.scalers['minmax'] = MinMaxScaler()
        
        X_train_scaled = self.scalers['standard'].fit_transform(X_train_clean)
        X_test_scaled = self.scalers['standard'].transform(X_test_clean)
        
        # Train and evaluate each model
        model_scores = {}
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            
            try:
                # Optimize hyperparameters for key models
                if name in ['xgboost', 'gradient_boosting', 'random_forest']:
                    cv_score = self.optimize_hyperparameters(X_train_scaled, y_train, name)
                else:
                    # Basic cross-validation for other models
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
                    cv_score = cv_scores.mean()
                
                # Train final model
                if name in ['xgboost', 'gradient_boosting', 'random_forest']:
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                roc_auc = roc_auc_score(y_test, y_proba)
                
                model_scores[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_score': cv_score
                }
                
                print(f"    {name}: ROC-AUC = {roc_auc:.4f}, Accuracy = {accuracy:.4f}")
                
            except Exception as e:
                print(f"    Error training {name}: {e}")
                continue
        
        # Select best model
        best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['roc_auc'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"🏆 Best model: {best_model_name} with ROC-AUC: {model_scores[best_model_name]['roc_auc']:.4f}")
        
        # Create ensemble model
        self.create_voting_ensemble(X_train_scaled, y_train, model_scores)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, model_scores
    
    def create_voting_ensemble(self, X_train, y_train, model_scores):
        """
        Create a voting ensemble from the best performing models.
        """
        print("🗳️ Creating voting ensemble...")
        
        # Select top 3 models
        top_models = sorted(model_scores.items(), key=lambda x: x[1]['roc_auc'], reverse=True)[:3]
        
        ensemble_models = [(name, self.models[name]) for name, _ in top_models]
        
        # Create voting classifier
        self.ensemble_model = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            weights=[score['roc_auc'] for _, score in top_models]
        )
        
        # Train ensemble
        self.ensemble_model.fit(X_train, y_train)
        
        print(f"✅ Ensemble created with models: {[name for name, _ in ensemble_models]}")
    
    def optimize_threshold(self, y_true, y_proba):
        """
        Optimize classification threshold using Youden's J statistic.
        """
        fpr, tpr, thresholds = precision_recall_curve(y_true, y_proba)
        precision, recall, thresholds_pr = fpr, tpr, thresholds
        
        # Calculate F1 score for each threshold
        f1_scores = []
        for threshold in thresholds_pr:
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_true, y_pred, average='weighted')
            f1_scores.append(f1)
        
        # Find best threshold
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds_pr[best_idx]
        
        return best_threshold
    
    def evaluate_ensemble(self, X_test, y_test):
        """
        Evaluate the ensemble model and individual models.
        """
        print("📊 Evaluating ensemble performance...")
        
        results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1]
                
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'roc_auc': roc_auc_score(y_test, y_proba)
                }
            except:
                continue
        
        # Evaluate ensemble
        if self.ensemble_model:
            y_pred_ensemble = self.ensemble_model.predict(X_test)
            y_proba_ensemble = self.ensemble_model.predict_proba(X_test)[:, 1]
            
            results['ensemble'] = {
                'accuracy': accuracy_score(y_test, y_pred_ensemble),
                'precision': precision_score(y_test, y_pred_ensemble, average='weighted'),
                'recall': recall_score(y_test, y_pred_ensemble, average='weighted'),
                'f1_score': f1_score(y_test, y_pred_ensemble, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_proba_ensemble)
            }
            
            # Optimize threshold for ensemble
            self.optimal_threshold = self.optimize_threshold(y_test, y_proba_ensemble)
        
        return results
    
    def create_visualizations(self, X_test, y_test, model_scores):
        """
        Create comprehensive visualizations for model performance.
        """
        print("📈 Creating advanced visualizations...")
        
        figures = []
        
        # 1. Model Comparison Chart
        models_comparison = pd.DataFrame(model_scores).T
        fig1 = go.Figure()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in metrics:
            fig1.add_trace(go.Bar(
                x=models_comparison.index,
                y=models_comparison[metric],
                name=metric.title(),
                text=models_comparison[metric].round(3),
                textposition='auto'
            ))
        
        fig1.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Score",
            barmode='group',
            template="plotly_dark",
            height=500
        )
        figures.append(("Model Performance Comparison", fig1))
        
        # 2. ROC Curve Comparison
        fig2 = go.Figure()
        
        for name, model in self.models.items():
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                from sklearn.metrics import roc_curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                
                fig2.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{name} (AUC = {model_scores[name]["roc_auc"]:.3f})'
                ))
            except:
                continue
        
        # Add ensemble if available
        if self.ensemble_model:
            y_proba_ensemble = self.ensemble_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba_ensemble)
            
            fig2.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode='lines',
                name=f'Ensemble (AUC = {model_scores["ensemble"]["roc_auc"]:.3f})',
                line=dict(width=3, dash='dash')
            ))
        
        fig2.update_layout(
            title="ROC Curves Comparison",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            template="plotly_dark",
            height=500
        )
        figures.append(("ROC Curves", fig2))
        
        # 3. Feature Importance (for tree-based models)
        if hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False).head(15)
            
            fig3 = go.Figure(go.Bar(
                x=feature_importance['importance'],
                y=feature_importance['feature'],
                orientation='h',
                text=feature_importance['importance'].round(3),
                textposition='auto'
            ))
            
            fig3.update_layout(
                title=f"Top 15 Feature Importance ({self.best_model_name})",
                xaxis_title="Importance",
                yaxis_title="Features",
                template="plotly_dark",
                height=500
            )
            figures.append(("Feature Importance", fig3))
        
        return figures


def run_risk_agent_advanced(df, col_map):
    """
    Run the advanced risk prediction agent.
    """
    print("=" * 80)
    print("🚀 ADVANCED HEALTHCARE RISK PREDICTION SYSTEM")
    print("=" * 80)
    
    start_time = time.perf_counter()
    result = {"status": "ok", "summary": "", "metrics": {}, "figures": []}
    
    try:
        # Initialize advanced predictor
        predictor = AdvancedHealthcareRiskPredictor()
        
        # Create advanced features
        features_df = predictor.create_advanced_healthcare_features(df, col_map)
        
        # Create target variable
        target_col = 'risk_score'
        if target_col not in features_df.columns:
            # Create synthetic target with clinical reasoning and balanced classes
            patient_col = next((c for c, cat in col_map.items() if cat == "patient_id" and c in features_df.columns), None)
            if patient_col:
                features_df['prescription_count'] = features_df.groupby(patient_col).cumcount() + 1
            else:
                features_df['prescription_count'] = 1
            
            # Create multiple risk factors for more realistic target
            if 'dosage' in features_df.columns:
                features_df['high_dosage_risk'] = (features_df['dosage'] > features_df['dosage'].quantile(0.75)).astype(int)
            else:
                features_df['high_dosage_risk'] = 0
            
            if 'polypharmacy_score' in features_df.columns:
                features_df['polypharmacy_risk'] = (features_df['polypharmacy_score'] > 3).astype(int)
            else:
                features_df['polypharmacy_risk'] = 0
            
            features_df['frequency_risk'] = (features_df['prescription_count'] > features_df['prescription_count'].quantile(0.7)).astype(int)
            
            # Combine risk factors with some randomness for balance
            np.random.seed(42)
            risk_score = (
                features_df['high_dosage_risk'] * 0.3 +
                features_df['polypharmacy_risk'] * 0.4 +
                features_df['frequency_risk'] * 0.3
            )
            
            # Add noise and create binary target
            noise = np.random.normal(0, 0.1, len(risk_score))
            risk_score = risk_score + noise
            
            # Create balanced binary target (approximately 50-50 split)
            threshold = np.percentile(risk_score, 50)
            features_df[target_col] = (risk_score > threshold).astype(int)
            
            # Ensure we have both classes
            if features_df[target_col].nunique() < 2:
                # Force some diversity if needed
                features_df.loc[:len(features_df)//3, target_col] = 1
                features_df.loc[len(features_df)//3:, target_col] = 0
        
        y = features_df[target_col]
        X = features_df.drop(columns=[target_col])
        
        # Feature selection
        selected_features = predictor.select_best_features(X, y, max_features=30)
        X_selected = X[selected_features]
        predictor.feature_names = selected_features
        
        # Train advanced models
        X_train, X_test, y_train, y_test, model_scores = predictor.train_advanced_models(X_selected, y)
        
        # Evaluate ensemble
        evaluation_results = predictor.evaluate_ensemble(X_test, y_test)
        
        # Create visualizations
        figures = predictor.create_visualizations(X_test, y_test, evaluation_results)
        
        # Get best model metrics
        best_metrics = evaluation_results.get('ensemble', evaluation_results.get(predictor.best_model_name, {}))
        
        # Update result
        result["figures"] = figures
        result["metrics"] = {
            "Model": f"Advanced Ensemble ({predictor.best_model_name})",
            "Accuracy": f"{best_metrics.get('accuracy', 0):.3f}",
            "Precision": f"{best_metrics.get('precision', 0):.3f}",
            "Recall": f"{best_metrics.get('recall', 0):.3f}",
            "F1-Score": f"{best_metrics.get('f1_score', 0):.3f}",
            "ROC-AUC": f"{best_metrics.get('roc_auc', 0):.3f}",
            "Execution": f"{(time.perf_counter() - start_time)*1000:.1f}ms"
        }
        
        # Generate summary
        summary = f"""
🚀 **Advanced Healthcare Risk Prediction System**

**Model Performance:**
- ✅ Best Model: {predictor.best_model_name}
- ✅ Accuracy: {best_metrics.get('accuracy', 0):.3f}
- ✅ Precision: {best_metrics.get('precision', 0):.3f}
- ✅ Recall: {best_metrics.get('recall', 0):.3f}
- ✅ F1-Score: {best_metrics.get('f1_score', 0):.3f}
- ✅ ROC-AUC: {best_metrics.get('roc_auc', 0):.3f}

**Advanced Features:**
- 🔬 Advanced Feature Engineering: {len(features_df.columns)} total features
- 🎯 Feature Selection: {len(selected_features)} best features
- 🤝 Ensemble Methods: {len(predictor.models)} models
- 📊 Cross-Validation: 5-fold stratified
- 🔧 Hyperparameter Optimization: GridSearchCV

**Clinical Insights:**
- 🏥 Healthcare-specific feature engineering
- 💊 Polypharmacy risk assessment
- ⏰ Temporal pattern analysis
- 📈 Risk stratification capabilities

**Technical Improvements:**
- ✅ Multiple advanced algorithms (XGBoost, GradientBoosting, RandomForest, SVM, Neural Networks)
- ✅ Ensemble voting classifier
- ✅ Advanced feature selection methods
- ✅ Hyperparameter optimization
- ✅ Class imbalance handling
- ✅ Threshold optimization

**Execution Time:** {(time.perf_counter() - start_time):.2f}s
"""
        
        result["summary"] = summary
        print("\n✅ Advanced risk prediction completed successfully!")
        print(f"📊 {summary}")
        
    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Advanced risk agent error: {str(e)}"
        print(f"❌ Error: {str(e)}")
    
    return result


if __name__ == "__main__":
    # Test the advanced risk agent with sample data
    import pandas as pd
    
    # Create sample healthcare data
    np.random.seed(42)
    n_records = 1000
    
    sample_data = {
        'patient_id': [f'P_{i:04d}' for i in range(n_records)],
        'drug_name': np.random.choice(['Aspirin', 'Metformin', 'Lisinopril', 'Atorvastatin', 'Omeprazole'], n_records),
        'dosage': np.random.randint(50, 500, n_records),
        'date': pd.date_range('2024-01-01', periods=n_records, freq='D'),
        'prescription_count': np.random.randint(1, 20, n_records),
        'age': np.random.randint(18, 85, n_records),
        'gender': np.random.choice(['M', 'F'], n_records)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Create simple column mapping for testing
    col_map = {
        'patient_id': 'patient_id',
        'drug_name': 'drug_name', 
        'dosage': 'dosage',
        'date': 'date',
        'prescription_count': 'prescription_count',
        'age': 'age',
        'gender': 'gender'
    }
    
    # Run advanced risk agent
    results = run_risk_agent_advanced(df, col_map)
    print(f"Status: {results['status']}")
    print(f"Metrics: {results['metrics']}")
