import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import ttest_ind, chi2_contingency, f_oneway, mannwhitneyu, kruskal
# Simple implementation of FDR-BH correction without statsmodels
def multipletests(p_values, alpha=0.05, method='fdr_bh'):
    """Simple implementation of Benjamini-Hochberg FDR correction."""
    p_array = np.array(p_values)
    n = len(p_array)
    
    if method == 'fdr_bh':
        # Benjamini-Hochberg procedure
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]
        
        # Find the largest k such that P(k) <= k/n * alpha
        bh_thresholds = [(i + 1) / n * alpha for i in range(n)]
        significant = [p <= threshold for p, threshold in zip(sorted_p, bh_thresholds)]
        
        # Apply correction
        corrected_p = np.minimum.accumulate(np.array(sorted_p) * n / np.arange(1, n + 1))
        corrected_p = np.minimum(corrected_p, 1.0)  # Cap at 1.0
        
        # Restore original order
        original_order = np.argsort(sorted_indices)
        corrected_p = corrected_p[original_order]
        rejected = [sig for sig in significant][original_order]
        
        return rejected, corrected_p, None, None
    
    # Fallback: no correction
    return [p < alpha for p in p_values], p_array, None, None
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class StatisticalValidator:
    """
    Comprehensive statistical validation framework for healthcare analytics.
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.results = []
        self.validation_summary = {}
        
    def calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        s1, s2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / s if s > 0 else 0
    
    def calculate_eta_squared(self, f_stat: float, df_between: int, df_within: int) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        return (df_between * f_stat) / (df_between * f_stat + df_within)
    
    def calculate_cramers_v(self, chi2_stat: float, n: int, min_dim: int) -> float:
        """Calculate Cramér's V effect size for chi-square test."""
        return np.sqrt(chi2_stat / (n * (min_dim - 1)))
    
    def test_normality(self, data: np.ndarray) -> Tuple[bool, float]:
        """Test for normality using Shapiro-Wilk test."""
        if len(data) < 3:
            return False, 1.0
        try:
            stat, p_value = stats.shapiro(data)
            return p_value > self.alpha, p_value
        except:
            return False, 1.0
    
    def test_numerical_groups(self, data: pd.DataFrame, group_col: str, test_col: str) -> Dict[str, Any]:
        """Perform statistical tests on numerical data across groups."""
        # Ensure data is numeric
        if not pd.api.types.is_numeric_dtype(data[test_col]):
            try:
                data[test_col] = pd.to_numeric(data[test_col], errors='coerce')
            except:
                return None
        
        groups = [group[test_col].dropna() for name, group in data.groupby(group_col) if len(group) > 1]
        
        if len(groups) < 2:
            return None
        
        group_names = list(data[group_col].unique())
        
        # Test for normality in each group
        normality_results = [self.test_normality(group) for group in groups]
        all_normal = all(result[0] for result in normality_results)
        
        result = {
            'test_variable': test_col,
            'group_variable': group_col,
            'groups': group_names,
            'sample_sizes': [len(group) for group in groups],
            'means': [np.mean(group) for group in groups],
            'stds': [np.std(group) for group in groups],
            'normality_p_values': [result[1] for result in normality_results],
            'all_normal': all_normal
        }
        
        # Choose appropriate test
        if len(groups) == 2:
            # Two groups
            if all_normal:
                # Parametric t-test
                stat, p_value = ttest_ind(groups[0], groups[1])
                result['test'] = 'Independent t-test'
                result['statistic'] = stat
                result['p_value'] = p_value
                result['effect_size'] = self.calculate_cohens_d(groups[0], groups[1])
                result['effect_type'] = "Cohen's d"
            else:
                # Non-parametric Mann-Whitney U test
                stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                result['test'] = 'Mann-Whitney U test'
                result['statistic'] = stat
                result['p_value'] = p_value
                result['effect_size'] = self.calculate_cohens_d(groups[0], groups[1])  # Still use Cohen's d
                result['effect_type'] = "Cohen's d"
        else:
            # More than two groups
            if all_normal:
                # Parametric ANOVA
                stat, p_value = f_oneway(*groups)
                result['test'] = 'One-way ANOVA'
                result['statistic'] = stat
                result['p_value'] = p_value
                df_between = len(groups) - 1
                df_within = len(data) - len(groups)
                result['effect_size'] = self.calculate_eta_squared(stat, df_between, df_within)
                result['effect_type'] = "η² (eta-squared)"
            else:
                # Non-parametric Kruskal-Wallis test
                stat, p_value = kruskal(*groups)
                result['test'] = 'Kruskal-Wallis test'
                result['statistic'] = stat
                result['p_value'] = p_value
                result['effect_size'] = self.calculate_eta_squared(stat, len(groups) - 1, len(data) - len(groups))
                result['effect_type'] = "η² (eta-squared)"
        
        result['significant'] = result['p_value'] < self.alpha
        return result
    
    def test_categorical_groups(self, data: pd.DataFrame, group_col: str, test_col: str) -> Dict[str, Any]:
        """Perform chi-square test for categorical data."""
        try:
            # Create contingency table
            contingency_table = pd.crosstab(data[group_col], data[test_col])
            
            if contingency_table.size == 0 or contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                return None
            
            # Perform chi-square test
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            
            # Calculate effect size (Cramér's V)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape)
            cramers_v = self.calculate_cramers_v(chi2_stat, n, min_dim)
            
            result = {
                'test_variable': test_col,
                'group_variable': group_col,
                'test': 'Chi-square test of independence',
                'statistic': chi2_stat,
                'p_value': p_value,
                'degrees_of_freedom': dof,
                'effect_size': cramers_v,
                'effect_type': "Cramér's V",
                'contingency_table': contingency_table,
                'expected_frequencies': expected,
                'sample_size': n,
                'significant': p_value < self.alpha,
                'groups': list(data[group_col].unique()),
                'categories': list(data[test_col].unique())
            }
            
            return result
            
        except Exception as e:
            print(f"Error in categorical test for {test_col}: {e}")
            return None
    
    def validate_clustering_results(self, data: pd.DataFrame, cluster_labels: np.ndarray, 
                                  cluster_col: str = 'cluster') -> Dict[str, Any]:
        """Perform statistical validation of clustering results."""
        data_with_clusters = data.copy()
        data_with_clusters[cluster_col] = cluster_labels
        
        # Identify numerical and categorical columns
        numerical_cols = data_with_clusters.select_dtypes(include=[np.number]).columns
        categorical_cols = data_with_clusters.select_dtypes(include=['object', 'category']).columns
        
        # Remove cluster column from lists
        numerical_cols = [col for col in numerical_cols if col != cluster_col]
        categorical_cols = [col for col in categorical_cols if col != cluster_col]
        
        validation_results = {
            'cluster_validation': {},
            'numerical_tests': [],
            'categorical_tests': [],
            'summary': {}
        }
        
        # Test numerical variables
        for col in numerical_cols:
            if col in data_with_clusters.columns and data_with_clusters[col].nunique() > 1:
                result = self.test_numerical_groups(data_with_clusters, cluster_col, col)
                if result:
                    validation_results['numerical_tests'].append(result)
        
        # Test categorical variables
        for col in categorical_cols:
            if col in data_with_clusters.columns and data_with_clusters[col].nunique() > 1:
                result = self.test_categorical_groups(data_with_clusters, cluster_col, col)
                if result:
                    validation_results['categorical_tests'].append(result)
        
        # Multiple testing correction
        all_p_values = []
        for test in validation_results['numerical_tests'] + validation_results['categorical_tests']:
            all_p_values.append(test['p_value'])
        
        if all_p_values:
            rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=self.alpha, method='fdr_bh')
            
            # Update results with corrected p-values
            idx = 0
            for test in validation_results['numerical_tests'] + validation_results['categorical_tests']:
                test['p_value_corrected'] = p_corrected[idx]
                test['significant_corrected'] = rejected[idx]
                idx += 1
        
        # Create summary
        total_tests = len(validation_results['numerical_tests']) + len(validation_results['categorical_tests'])
        significant_tests = sum(1 for test in validation_results['numerical_tests'] + validation_results['categorical_tests'] 
                              if test.get('significant_corrected', test['significant']))
        
        validation_results['summary'] = {
            'total_tests': total_tests,
            'significant_tests': significant_tests,
            'significant_percentage': (significant_tests / total_tests * 100) if total_tests > 0 else 0,
            'numerical_tests': len(validation_results['numerical_tests']),
            'categorical_tests': len(validation_results['categorical_tests']),
            'alpha_level': self.alpha,
            'multiple_testing_correction': 'FDR-BH'
        }
        
        return validation_results
    
    def create_validation_table(self, validation_results: Dict[str, Any]) -> pd.DataFrame:
        """Create a formatted statistical validation results table."""
        table_data = []
        
        # Process numerical tests
        for test in validation_results.get('numerical_tests', []):
            row = {
                'Test Variable': test['test_variable'],
                'Group Variable': test['group_variable'],
                'Test': test['test'],
                'Statistic': round(test['statistic'], 4),
                'P-value': f"{test['p_value']:.4f}",
                'P-value (corrected)': f"{test.get('p_value_corrected', test['p_value']):.4f}",
                'Effect Size': f"{test['effect_size']:.4f}",
                'Effect Type': test['effect_type'],
                'Significant': 'Yes' if test.get('significant_corrected', test['significant']) else 'No',
                'Groups': f"{test['groups']}"
            }
            table_data.append(row)
        
        # Process categorical tests
        for test in validation_results.get('categorical_tests', []):
            row = {
                'Test Variable': test['test_variable'],
                'Group Variable': test['group_variable'],
                'Test': test['test'],
                'Statistic': round(test['statistic'], 4),
                'P-value': f"{test['p_value']:.4f}",
                'P-value (corrected)': f"{test.get('p_value_corrected', test['p_value']):.4f}",
                'Effect Size': f"{test['effect_size']:.4f}",
                'Effect Type': test['effect_type'],
                'Significant': 'Yes' if test.get('significant_corrected', test['significant']) else 'No',
                'Groups': f"{test['groups']}"
            }
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Sort by significance and effect size
        if not df.empty:
            df['Significant_Sort'] = df['Significant'].map({'Yes': 0, 'No': 1})
            df['Effect_Size_Num'] = pd.to_numeric(df['Effect Size'], errors='coerce')
            df = df.sort_values(['Significant_Sort', 'Effect_Size_Num'], ascending=[True, False])
            df = df.drop(['Significant_Sort', 'Effect_Size_Num'], axis=1)
        
        return df
    
    def create_validation_visualization(self, validation_results: Dict[str, Any]) -> go.Figure:
        """Create comprehensive visualization of statistical validation results."""
        # Prepare data for visualization
        all_tests = validation_results.get('numerical_tests', []) + validation_results.get('categorical_tests', [])
        
        if not all_tests:
            # Create empty figure
            fig = go.Figure()
            fig.add_annotation(text="No statistical tests performed", 
                            x=0.5, y=0.5, showarrow=False, 
                            font=dict(size=16))
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('P-value Distribution', 'Effect Size Distribution', 
                          'Significant vs Non-significant Tests', 'Test Types'),
            specs=[[{"type": "histogram"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Extract data
        p_values = [test['p_value'] for test in all_tests]
        p_corrected = [test.get('p_value_corrected', test['p_value']) for test in all_tests]
        effect_sizes = [test['effect_size'] for test in all_tests]
        significant = [test.get('significant_corrected', test['significant']) for test in all_tests]
        test_types = [test['test'] for test in all_tests]
        
        # 1. P-value distribution
        fig.add_trace(go.Histogram(x=p_values, name='Original P-values', 
                                 marker_color='lightblue', opacity=0.7), row=1, col=1)
        fig.add_trace(go.Histogram(x=p_corrected, name='Corrected P-values', 
                                 marker_color='lightcoral', opacity=0.7), row=1, col=1)
        
        # 2. Effect size distribution
        fig.add_trace(go.Histogram(x=effect_sizes, name='Effect Sizes', 
                                 marker_color='lightgreen', opacity=0.7), row=1, col=2)
        
        # 3. Significant vs non-significant
        sig_counts = pd.Series(significant).value_counts()
        fig.add_trace(go.Bar(x=['Non-significant', 'Significant'], 
                            y=[sig_counts.get(False, 0), sig_counts.get(True, 0)],
                            marker_color=['lightcoral', 'lightgreen'], 
                            showlegend=False), row=2, col=1)
        
        # 4. Test types pie chart
        test_type_counts = pd.Series(test_types).value_counts()
        fig.add_trace(go.Pie(labels=test_type_counts.index, values=test_type_counts.values,
                            showlegend=False), row=2, col=2)
        
        # Update layout
        fig.update_layout(
            title="Statistical Validation Results Summary",
            template="plotly_dark",
            height=600,
            showlegend=True
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="P-value", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Effect Size", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        
        return fig
    
    def generate_summary_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate a comprehensive summary report of statistical validation."""
        summary = validation_results.get('summary', {})
        numerical_tests = validation_results.get('numerical_tests', [])
        categorical_tests = validation_results.get('categorical_tests', [])
        
        report = f"""
📊 **Statistical Validation Report**

**Overall Summary:**
- Total Tests Performed: {summary.get('total_tests', 0)}
- Significant Tests: {summary.get('significant_tests', 0)} ({summary.get('significant_percentage', 0):.1f}%)
- Alpha Level: {summary.get('alpha_level', 0.05)}
- Multiple Testing Correction: {summary.get('multiple_testing_correction', 'FDR-BH')}

**Test Distribution:**
- Numerical Tests: {summary.get('numerical_tests', 0)}
- Categorical Tests: {summary.get('categorical_tests', 0)}

**Key Findings:**
"""
        
        # Add most significant results
        all_tests = numerical_tests + categorical_tests
        significant_tests = [test for test in all_tests if test.get('significant_corrected', test['significant'])]
        
        if significant_tests:
            significant_tests.sort(key=lambda x: x['p_value'])
            report += "\n**Most Significant Results:**\n"
            for i, test in enumerate(significant_tests[:5]):  # Top 5
                report += f"• {test['test_variable']}: {test['test']} (p={test['p_value']:.4f}, {test['effect_type']}={test['effect_size']:.3f})\n"
        
        # Add effect size interpretation
        if all_tests:
            effect_sizes = [test['effect_size'] for test in all_tests]
            mean_effect = np.mean(effect_sizes)
            report += f"\n**Effect Size Analysis:**\n"
            report += f"• Mean Effect Size: {mean_effect:.3f}\n"
            report += f"• Large Effects (>0.5): {sum(1 for es in effect_sizes if es > 0.5)}\n"
            report += f"• Medium Effects (0.3-0.5): {sum(1 for es in effect_sizes if 0.3 <= es <= 0.5)}\n"
            report += f"• Small Effects (<0.3): {sum(1 for es in effect_sizes if es < 0.3)}\n"
        
        return report


def perform_statistical_validation(data: pd.DataFrame, cluster_labels: np.ndarray, 
                                 alpha: float = 0.05) -> Dict[str, Any]:
    """
    Main function to perform comprehensive statistical validation.
    """
    validator = StatisticalValidator(alpha=alpha)
    
    # Perform validation
    validation_results = validator.validate_clustering_results(data, cluster_labels)
    
    # Create table
    validation_table = validator.create_validation_table(validation_results)
    
    # Create visualization
    validation_figure = validator.create_validation_visualization(validation_results)
    
    # Generate summary
    summary_report = validator.generate_summary_report(validation_results)
    
    return {
        'validation_results': validation_results,
        'validation_table': validation_table,
        'validation_figure': validation_figure,
        'summary_report': summary_report,
        'validator': validator
    }


# Example usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'age': np.random.normal(50, 15, n_samples),
        'dosage': np.random.exponential(50, n_samples),
        'risk_score': np.random.beta(2, 5, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'diagnosis': np.random.choice(['Diabetes', 'Hypertension', 'Heart Disease'], n_samples)
    })
    
    # Create sample cluster labels
    cluster_labels = np.random.choice([0, 1, 2, 3], n_samples)
    
    # Perform statistical validation
    results = perform_statistical_validation(sample_data, cluster_labels)
    
    print("Statistical Validation Results:")
    print(results['summary_report'])
    print("\nValidation Table:")
    print(results['validation_table'].to_string(index=False))
