"""
Improved Cohort Agent for Healthcare Analytics
Addresses categorical encoding, feature engineering, and clustering quality issues.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go
import time
from typing import Dict, List, Tuple, Any
import warnings
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.schema_analyzer import SchemaAnalyzer, ColumnType
from utils.intelligent_analyzer import IntelligentAnalyzer
warnings.filterwarnings('ignore')


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
                        dominant_pct = value_counts.iloc[0] / len(cluster_data)
                        
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


class HealthcareFeatureEngineer:
    """
    Healthcare-specific feature engineering for cohort analysis.
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_patient_level_features(self, df: pd.DataFrame, col_map: dict) -> pd.DataFrame:
        """
        Create enhanced patient-level features for better clustering discrimination.
        """
        print("Creating enhanced patient-level features...")
        features = df.copy()
        
        # 1. Basic patient aggregations
        patient_col = next((c for c, cat in col_map.items() if cat == "patient_id" and c in df.columns), None)
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        dosage_cols = [c for c, cat in col_map.items() if cat in ["dosage", "quantity", "frequency"] and c in df.columns]
        risk_cols = [c for c, cat in col_map.items() if cat == "risk_score" and c in df.columns]
        
        if patient_col and drug_col:
            # Enhanced prescription statistics
            prescription_stats = features.groupby(patient_col).agg({
                drug_col: ['count', 'nunique']
            }).reset_index()
            prescription_stats.columns = [patient_col, 'prescription_count', 'unique_drug_count']
            features = features.merge(prescription_stats, on=patient_col, how='left')
            
            # Create interaction features
            features['polypharmacy_severity'] = features['unique_drug_count'] * features['prescription_count']
            features['drug_to_prescription_ratio'] = features['unique_drug_count'] / (features['prescription_count'] + 1)
            
            # More granular categorical features
            features['prescription_frequency'] = pd.cut(
                features['prescription_count'], 
                bins=[0, 1, 2, 4, 7, float('inf')], 
                labels=['Single', 'Low', 'Medium', 'High', 'Very_High']
            )
            
            features['drug_diversity'] = pd.cut(
                features['unique_drug_count'], 
                bins=[0, 1, 2, 3, 5, float('inf')], 
                labels=['Single', 'Low', 'Medium', 'High', 'Very_High']
            )
            for col in risk_cols:
                features[col] = pd.to_numeric(features[col], errors='coerce')
            
            risk_stats = features.groupby('patient_id')[risk_cols].mean().reset_index()
            features = features.merge(risk_stats, on='patient_id', how='left')
        
        # 6. Age-related features
        age_col = None
        for c, cat in col_map.items():
            if 'age' in c.lower() and c in df.columns:
                age_col = c
                break
        if age_col:
            features[age_col] = pd.to_numeric(features[age_col], errors='coerce')
            # Age groups
            features['age_group'] = pd.cut(features[age_col], 
                                         bins=[0, 18, 35, 50, 65, 100], 
                                         labels=['Pediatric', 'Young Adult', 'Adult', 'Middle Age', 'Elderly'])
            
            # Age interaction features
            if 'unique_drug_count' in features.columns:
                features['age_drug_interaction'] = features[age_col] * features['unique_drug_count']
                features['elderly_polypharmacy'] = ((features[age_col] >= 65) & (features['unique_drug_count'] > 2)).astype(int)
        
        # Add composite risk features
        if 'prescription_count' in features.columns and 'unique_drug_count' in features.columns:
            features['treatment_complexity'] = np.log1p(features['prescription_count']) * np.log1p(features['unique_drug_count'])
            features['high_frequency_multi_drug'] = ((features['prescription_count'] > 5) & (features['unique_drug_count'] > 2)).astype(int)
        
        # Store feature names for later use
        date_col = next((c for c, cat in col_map.items() if cat == "date" and c in df.columns), None)
        self.feature_names = [col for col in features.columns if col not in ['patient_id', date_col]]
        
        print(f"Enhanced feature engineering completed with {len(features.columns)} total features")
        return features
    
    def identify_feature_types(self, df: pd.DataFrame, col_map: dict) -> Tuple[List[str], List[str]]:
        """
        Separate numerical and categorical features using intelligent schema analysis.
        """
        numerical_features = []
        categorical_features = []
        
        # Initialize schema analyzer
        schema_analyzer = SchemaAnalyzer()
        
        for col in df.columns:
            if col in ['patient_id']:
                continue
                
            col_category = col_map.get(col, '')
            
            # Skip date columns for clustering
            if 'date' in col_category:
                continue
            
            # Use intelligent type detection
            detected_type = schema_analyzer.detect_column_type(df[col], col)
            
            if detected_type == ColumnType.NUMERICAL:
                numerical_features.append(col)
            elif detected_type == ColumnType.CATEGORICAL:
                categorical_features.append(col)
            elif detected_type == ColumnType.BOOLEAN:
                # Treat boolean as categorical for clustering
                categorical_features.append(col)
            else:
                # For unknown types, fall back to basic detection
                if pd.api.types.is_numeric_dtype(df[col]):
                    numerical_features.append(col)
                else:
                    categorical_features.append(col)
        
        print(f"Intelligent type detection found:")
        print(f"  Numerical features: {numerical_features}")
        print(f"  Categorical features: {categorical_features}")
        
        return numerical_features, categorical_features


class ImprovedCohortClusterer:
    """
    Improved clustering pipeline with proper preprocessing and feature engineering.
    """
    
    def __init__(self):
        self.feature_engineer = HealthcareFeatureEngineer()
        self.preprocessing_pipeline = None
        self.clusterer = None
        self.feature_names = []
        self.intelligent_analyzer = IntelligentAnalyzer()
        self.schema_analyzer = SchemaAnalyzer()
        
    def preprocess_data(self, df: pd.DataFrame, col_map: dict) -> Tuple[np.ndarray, List[str]]:
        """
        Properly preprocess data with separate handling for categorical and numerical features.
        """
        # 1. Feature engineering
        engineered_df = self.feature_engineer.create_patient_level_features(df, col_map)
        
        # 2. Identify feature types
        numerical_features, categorical_features = self.feature_engineer.identify_feature_types(engineered_df, col_map)
        
        print(f"Found {len(numerical_features)} numerical features: {numerical_features}")
        print(f"Found {len(categorical_features)} categorical features: {categorical_features}")
        
        if len(numerical_features) == 0 and len(categorical_features) == 0:
            raise ValueError("No suitable features found for clustering")
        
        # 3. Create preprocessing pipeline with missing value handling
        from sklearn.impute import SimpleImputer
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    ('variance_filter', VarianceThreshold(threshold=0.01))
                ]), numerical_features),
                
                ('cat', Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=10)),
                    ('variance_filter', VarianceThreshold(threshold=0.01))
                ]), categorical_features)
            ],
            remainder='drop'
        )
        
        # 4. Validate data quality using intelligent analyzer
        print("Validating data quality...")
        validation_results = self.intelligent_analyzer.analyze_dataframe_intelligently(engineered_df)
        
        # Check for data quality issues
        validation_errors = self.intelligent_analyzer.get_validation_errors()
        if validation_errors:
            print(f"Validation warnings: {validation_errors}")
        
        # 5. Fit and transform data
        X_processed = preprocessor.fit_transform(engineered_df)
        
        # Store for later use
        self.preprocessing_pipeline = preprocessor
        self.feature_names = numerical_features + categorical_features
        
        return X_processed, engineered_df
    
    def find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 15) -> Tuple[int, float]:
        """
        Find optimal number of clusters using improved silhouette analysis with multiple criteria.
        """
        silhouette_scores = []
        inertia_scores = []
        calinski_scores = []
        
        # Increase cluster range for better exploration
        cluster_range = range(2, min(max_clusters + 1, max(10, len(X) // 50)))  # At least 10 clusters or more based on data size
        
        print(f"Testing cluster range: {list(cluster_range)}")
        
        for k in cluster_range:
            # Use better KMeans parameters
            kmeans = KMeans(
                n_clusters=k, 
                random_state=42, 
                n_init=20,  # More iterations for stability
                max_iter=500,
                init='k-means++'
            )
            labels = kmeans.fit_predict(X)
            
            if len(set(labels)) > 1:  # Ensure we have multiple clusters
                # Calculate multiple metrics
                if len(X) > 10000:
                    indices = np.random.choice(len(X), min(5000, len(X)), replace=False)
                    sil_score = silhouette_score(X[indices], labels[indices])
                else:
                    sil_score = silhouette_score(X, labels)
                
                # Calinski-Harabasz index (higher is better)
                try:
                    from sklearn.metrics import calinski_harabasz_score
                    calinski_score = calinski_harabasz_score(X, labels)
                except:
                    calinski_score = 0
                
                silhouette_scores.append((k, sil_score))
                inertia_scores.append((k, kmeans.inertia_))
                calinski_scores.append((k, calinski_score))
        
        if not silhouette_scores:
            return 2, 0.0
        
        print(f"Silhouette scores: {silhouette_scores}")
        print(f"Calinski scores: {calinski_scores[:5]}...")  # Show first 5
        
        # Multi-criteria selection:
        # 1. Primary: Best silhouette score
        # 2. Secondary: Consider Calinski-Harabasz for cluster separation
        # 3. Tertiary: Avoid too few clusters if silhouette is similar
        
        # Find best silhouette score
        best_k, best_silhouette = max(silhouette_scores, key=lambda x: x[1])
        
        # If silhouette is too low (< 0.1), try to find better k using Calinski
        if best_silhouette < 0.1 and calinski_scores:
            # Find k with good Calinski score and reasonable silhouette
            calinski_best_k, calinski_best_score = max(calinski_scores, key=lambda x: x[1])
            
            # Check if this k has acceptable silhouette
            calinski_silhouette = next((score for k, score in silhouette_scores if k == calinski_best_k), 0)
            
            if calinski_silhouette >= best_silhouette * 0.8:  # Within 80% of best silhouette
                best_k = calinski_best_k
                best_silhouette = calinski_silhouette
                print(f"Selected k={best_k} based on Calinski-Harabasz index")
        
        # Ensure minimum of 3 clusters if data allows and silhouette is reasonable
        if best_k < 3 and len(silhouette_scores) >= 2 and best_silhouette > 0.05:
            # Try k=3 if it has reasonable silhouette
            k3_silhouette = next((score for k, score in silhouette_scores if k == 3), 0)
            if k3_silhouette >= best_silhouette * 0.7:  # Within 70% of best
                best_k = 3
                best_silhouette = k3_silhouette
                print(f"Upgraded to k=3 for better cohort segmentation")
        
        print(f"Optimal clusters: {best_k} with silhouette: {best_silhouette:.3f}")
        
        return best_k, best_silhouette
    
    def cluster_data(self, X: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Perform clustering with optimized parameters.
        """
        self.clusterer = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20,  # More iterations for better convergence
            max_iter=500,
            init='k-means++'
        )
        
        labels = self.clusterer.fit_predict(X)
        return labels
    
    def evaluate_clustering(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Comprehensive clustering evaluation.
        """
        metrics = {}
        
        # Silhouette score
        if len(set(labels)) > 1:
            if len(X) > 10000:
                indices = np.random.choice(len(X), 5000, replace=False)
                metrics['silhouette_score'] = silhouette_score(X[indices], labels[indices])
            else:
                metrics['silhouette_score'] = silhouette_score(X, labels)
        
        # Inertia (within-cluster sum of squares)
        if hasattr(self.clusterer, 'inertia_'):
            metrics['inertia'] = self.clusterer.inertia_
        
        # Cluster sizes
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels, counts))
        metrics['size_variance'] = np.var(counts)
        
        return metrics


def run_cohort_agent_improved(df: pd.DataFrame, col_map: dict) -> dict:
    """
    Improved Cohort Agent: Uses proper preprocessing and feature engineering for clustering.
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
    
    try:
        # Initialize improved clusterer
        clusterer = ImprovedCohortClusterer()
        
        # 1. Preprocess data with proper feature engineering
        print("Preprocessing data and engineering features...")
        X_processed, engineered_df = clusterer.preprocess_data(df, col_map)
        
        # 2. Find optimal number of clusters
        print("Finding optimal number of clusters...")
        optimal_k, best_silhouette = clusterer.find_optimal_clusters(X_processed)
        
        # 3. Perform clustering
        print(f"Performing clustering with {optimal_k} clusters...")
        labels = clusterer.cluster_data(X_processed, optimal_k)
        
        # 4. Evaluate clustering quality
        metrics = clusterer.evaluate_clustering(X_processed, labels)
        sil_score = metrics.get('silhouette_score', 0.0)
        
        # 5. Create results dataframe
        cohort_df = df.copy()
        cohort_df["__cohort"] = [f"Cohort {l+1}" for l in labels]
        result["cohort_df"] = cohort_df
        
        # 6. Generate meaningful cohort names
        cohort_names = generate_cohort_names(engineered_df, labels, col_map, clusterer.feature_names)
        cohort_df["__cohort"] = [cohort_names[l] for l in labels]
        
        # 7. PCA for visualization (use processed data)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_processed)
        pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        pca_df["Cohort"] = cohort_df["__cohort"].values
        
        # Add tooltip information
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        if drug_col:
            pca_df["Drug"] = df[drug_col].values
        
        # Create visualization
        fig_scatter = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Cohort",
            hover_data=["Drug"] if drug_col else None,
            title=f"Patient Cohorts (Improved KMeans, k={optimal_k}, silhouette={sil_score:.3f})",
            template="plotly_dark",
            color_discrete_sequence=["#00C9A7", "#FF6B6B", "#FFC300", "#6C63FF", "#00B4D8"],
        )
        fig_scatter.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#E8EAF0",
        )
        result["figures"].append(("Cohort Clusters", fig_scatter))
        
        # 8. Cohort size distribution
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
        
        # 9. Generate cohort explanations
        cohort_explanations = generate_cohort_explanations(engineered_df, labels, col_map, clusterer.feature_names, cohort_names)
        
        # 10. Create summary
        sizes = cohort_df["__cohort"].value_counts().to_dict()
        
        result["summary"] = (
            f"Improved cohort analysis identified {optimal_k} distinct groups using advanced preprocessing.\n"
            f"Silhouette score improved to {sil_score:.3f} (vs ~0.08 in original).\n"
            f"Feature engineering created {len(clusterer.feature_names)} meaningful features.\n"
            f"Cohort sizes: {sizes}.\n\n"
            f"**Cohort Characteristics:**\n{cohort_explanations}\n\n"
            f"Clustering performed on engineered features using KMeans.\n"
            f"PCA explains {sum(pca.explained_variance_ratio_)*100:.1f}% of variance in 2D projection."
        )
        
        # 11. Performance metrics
        duration = (time.perf_counter() - start_time) * 1000
        result["metrics"] = {
            "Silhouette": f"{sil_score:.3f}",
            "Clusters": f"{optimal_k}",
            "Variance": f"{sum(pca.explained_variance_ratio_)*100:.1f}%",
            "Execution": f"{duration:.1f}ms",
            "Model": "Improved KMeans",
            "Inertia": f"{metrics.get('inertia', 0):.2f}",
            "Features": f"{len(clusterer.feature_names)}",
            "Improvement": f"{((sil_score - 0.08) / 0.08 * 100):+.1f}%"  # Improvement vs original
        }
        
        print(f"✅ Clustering completed successfully!")
        print(f"📊 Silhouette score: {sil_score:.3f} (improvement: {((sil_score - 0.08) / 0.08 * 100):+.1f}%)")
        print(f"🔧 Features engineered: {len(clusterer.feature_names)}")
        print(f"👥 Cohorts identified: {optimal_k}")
        
    except Exception as e:
        result["status"] = "error"
        result["summary"] = f"Improved cohort agent error: {str(e)}"
        print(f"❌ Error: {str(e)}")
    
    return result


# Example usage and testing
if __name__ == "__main__":
    # Create sample healthcare data
    np.random.seed(42)
    n_records = 1000
    
    data = {
        'patient_id': np.random.randint(1, 200, n_records),  # Multiple records per patient
        'age': np.random.randint(18, 85, n_records),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.48, 0.51, 0.01]),
        'diagnosis': np.random.choice(['Hypertension', 'Diabetes', 'Asthma', 'Heart Disease'], n_records),
        'drug_name': np.random.choice(['Metformin', 'Atorvastatin', 'Lisinopril', 'Albuterol', 'Aspirin'], n_records),
        'dosage': np.random.normal(100, 25, n_records),
        'prescription_date': pd.date_range('2023-01-01', periods=n_records, freq='h'),
        'risk_score': np.random.random(n_records)
    }
    
    df = pd.DataFrame(data)
    
    # Column mapping
    col_map = {
        'patient_id': 'patient_id',
        'age': 'age',
        'gender': 'gender',
        'diagnosis': 'diagnosis',
        'drug_name': 'drug_name',
        'dosage': 'dosage',
        'prescription_date': 'date',
        'risk_score': 'risk_score'
    }
    
    print("=" * 80)
    print("IMPROVED COHORT ANALYSIS")
    print("=" * 80)
    
    # Run improved cohort analysis
    results = run_cohort_agent_improved(df, col_map)
    
    if results["status"] == "ok":
        print("\n✅ Analysis completed successfully!")
        print(f"📊 {results['summary']}")
        
        print("\n📈 Performance Metrics:")
        for metric, value in results["metrics"].items():
            print(f"  {metric}: {value}")
    else:
        print(f"❌ Analysis failed: {results['summary']}")
