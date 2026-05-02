"""
Intelligent Data Analyzer Integration Module
Integrates the SchemaAnalyzer with the existing healthcare analytics pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from utils.schema_analyzer import SchemaAnalyzer, ColumnType
import warnings
import logging

logger = logging.getLogger(__name__)


class IntelligentAnalyzer:
    """
    Integration layer for intelligent data analysis in healthcare systems.
    
    This class provides a clean interface to use the SchemaAnalyzer
    within the existing ML pipeline while preventing common data type errors.
    """
    
    def __init__(self):
        self.schema_analyzer = SchemaAnalyzer()
        self.last_analysis = None
        self.validation_errors = []
    
    def safe_mean(self, series: pd.Series, column_name: str) -> float:
        """
        Safely compute mean only for numerical columns.
        
        Args:
            series: Pandas Series
            column_name: Name of the column
            
        Returns:
            Mean value or raises error for non-numerical columns
        """
        col_type = self.schema_analyzer.detect_column_type(series, column_name)
        
        if col_type == ColumnType.NUMERICAL:
            return pd.to_numeric(series, errors='coerce').mean()
        else:
            error_msg = f"Cannot compute mean on {col_type.value} column '{column_name}'"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            raise ValueError(error_msg)
    
    def safe_std(self, series: pd.Series, column_name: str) -> float:
        """
        Safely compute standard deviation only for numerical columns.
        """
        col_type = self.schema_analyzer.detect_column_type(series, column_name)
        
        if col_type == ColumnType.NUMERICAL:
            return pd.to_numeric(series, errors='coerce').std()
        else:
            error_msg = f"Cannot compute std on {col_type.value} column '{column_name}'"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            raise ValueError(error_msg)
    
    def safe_median(self, series: pd.Series, column_name: str) -> float:
        """
        Safely compute median only for numerical columns.
        """
        col_type = self.schema_analyzer.detect_column_type(series, column_name)
        
        if col_type == ColumnType.NUMERICAL:
            return pd.to_numeric(series, errors='coerce').median()
        else:
            error_msg = f"Cannot compute median on {col_type.value} column '{column_name}'"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            raise ValueError(error_msg)
    
    def safe_mode(self, series: pd.Series, column_name: str) -> Any:
        """
        Safely compute mode for categorical or numerical columns.
        """
        col_type = self.schema_analyzer.detect_column_type(series, column_name)
        
        if col_type in [ColumnType.CATEGORICAL, ColumnType.NUMERICAL, ColumnType.BOOLEAN]:
            return series.mode().iloc[0] if not series.mode().empty else None
        else:
            error_msg = f"Cannot compute mode on {col_type.value} column '{column_name}'"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            raise ValueError(error_msg)
    
    def safe_value_counts(self, series: pd.Series, column_name: str) -> pd.Series:
        """
        Safely compute value counts for categorical columns.
        """
        col_type = self.schema_analyzer.detect_column_type(series, column_name)
        
        if col_type == ColumnType.CATEGORICAL:
            return series.value_counts()
        else:
            error_msg = f"Cannot compute value_counts on {col_type.value} column '{column_name}'"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            raise ValueError(error_msg)
    
    def safe_distribution(self, series: pd.Series, column_name: str) -> Dict[str, float]:
        """
        Safely compute distribution percentages for categorical and boolean columns.
        """
        col_type = self.schema_analyzer.detect_column_type(series, column_name)
        
        if col_type == ColumnType.CATEGORICAL:
            value_counts = series.value_counts()
            total = len(series.dropna())
            return (value_counts / total * 100).round(2).to_dict()
        elif col_type == ColumnType.BOOLEAN:
            # Handle boolean columns specially
            value_counts = series.value_counts()
            total = len(series.dropna())
            return (value_counts / total * 100).round(2).to_dict()
        else:
            error_msg = f"Cannot compute distribution on {col_type.value} column '{column_name}'"
            self.validation_errors.append(error_msg)
            logger.warning(error_msg)
            raise ValueError(error_msg)
    
    def analyze_dataframe_intelligently(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform intelligent analysis on the entire dataframe.
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Comprehensive analysis results with type-safe operations
        """
        self.validation_errors.clear()
        results = self.schema_analyzer.analyze_dataframe(df)
        self.last_analysis = results
        
        # Add safe operation examples
        results['safe_operations_demo'] = self._demonstrate_safe_operations(df)
        
        return results
    
    def _demonstrate_safe_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Demonstrate safe operations on different column types.
        """
        demo_results = {}
        
        for column_name in df.columns:
            column = df[column_name]
            col_type = self.schema_analyzer.detect_column_type(column, column_name)
            
            demo_results[column_name] = {
                'detected_type': col_type.value,
                'safe_operations': []
            }
            
            try:
                if col_type == ColumnType.NUMERICAL:
                    mean_val = self.safe_mean(column, column_name)
                    demo_results[column_name]['safe_operations'].append(f"Mean: {mean_val:.2f}")
                    
                    std_val = self.safe_std(column, column_name)
                    demo_results[column_name]['safe_operations'].append(f"Std: {std_val:.2f}")
                    
                    median_val = self.safe_median(column, column_name)
                    demo_results[column_name]['safe_operations'].append(f"Median: {median_val:.2f}")
                
                elif col_type == ColumnType.CATEGORICAL:
                    if 'gender' in column_name.lower():
                        dist = self.safe_distribution(column, column_name)
                        demo_results[column_name]['safe_operations'].append(f"Gender Distribution: {dist}")
                    else:
                        mode_val = self.safe_mode(column, column_name)
                        demo_results[column_name]['safe_operations'].append(f"Mode: {mode_val}")
                        
                        value_counts = self.safe_value_counts(column, column_name)
                        demo_results[column_name]['safe_operations'].append(f"Value counts: {dict(value_counts.head(3))}")
                
                elif col_type == ColumnType.BOOLEAN:
                    dist = self.safe_distribution(column, column_name)
                    demo_results[column_name]['safe_operations'].append(f"Boolean Distribution: {dist}")
                
                elif col_type == ColumnType.DATETIME:
                    demo_results[column_name]['safe_operations'].append("Date range analysis")
                
            except Exception as e:
                demo_results[column_name]['safe_operations'].append(f"Error: {str(e)}")
        
        return demo_results
    
    def get_validation_errors(self) -> List[str]:
        """Get all validation errors that occurred."""
        return self.validation_errors
    
    def get_column_summary(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a specific column.
        """
        if column_name not in df.columns:
            return {"error": f"Column '{column_name}' not found"}
        
        return self.schema_analyzer.analyze_column(df[column_name], column_name)
    
    def suggest_data_quality_improvements(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest improvements based on data quality analysis.
        """
        suggestions = []
        analysis = self.analyze_dataframe_intelligently(df)
        
        # Check for high missing values
        for col_name, col_analysis in analysis['column_analysis'].items():
            if 'missing' in col_analysis and 'count' in col_analysis:
                missing_pct = (col_analysis['missing'] / col_analysis['count']) * 100
                if missing_pct > 20:
                    suggestions.append(f"Column '{col_name}' has {missing_pct:.1f}% missing values - consider imputation")
        
        # Check for potential type mismatches
        for col_name, col_analysis in analysis['column_analysis'].items():
            if col_analysis.get('type') == 'unknown':
                suggestions.append(f"Column '{col_name}' has unknown type - review data format")
        
        # Gender-specific suggestions
        for col_name, col_analysis in analysis['column_analysis'].items():
            if 'gender' in col_name.lower() and col_analysis.get('type') == 'categorical':
                dist = col_analysis.get('distribution', {})
                if len(dist) > 2:
                    suggestions.append(f"Gender column '{col_name}' has {len(dist)} categories - consider standardizing")
        
        return suggestions


def create_integration_example():
    """Create example showing integration with existing ML pipeline."""
    
    # Sample healthcare data
    np.random.seed(42)
    n_records = 500
    
    data = {
        'patient_id': range(1, n_records + 1),
        'age': np.random.randint(18, 85, n_records),
        'gender': np.random.choice(['Male', 'Female'], n_records),
        'diagnosis': np.random.choice(['Hypertension', 'Diabetes', 'Asthma'], n_records),
        'drug_name': np.random.choice(['Metformin', 'Lisinopril', 'Aspirin'], n_records),
        'dosage': np.random.normal(100, 25, n_records),
        'admission_date': pd.date_range('2023-01-01', periods=n_records, freq='h'),
        'emergency_visit': np.random.choice([True, False], n_records, p=[0.2, 0.8]),
    }
    
    df = pd.DataFrame(data)
    
    # Initialize intelligent analyzer
    analyzer = IntelligentAnalyzer()
    
    # Perform intelligent analysis
    results = analyzer.analyze_dataframe_intelligently(df)
    
    print("=" * 80)
    print("INTELLIGENT HEALTHCARE DATA ANALYSIS")
    print("=" * 80)
    
    print(f"\nSchema Overview: {results['schema_overview']['type_distribution']}")
    
    print("\nSafe Operations Demonstration:")
    for col_name, demo in results['safe_operations_demo'].items():
        print(f"\n{col_name} ({demo['detected_type']}):")
        for op in demo['safe_operations']:
            print(f"  ✓ {op}")
    
    print("\nValidation Errors:", analyzer.get_validation_errors())
    
    print("\nData Quality Suggestions:")
    for suggestion in analyzer.suggest_data_quality_improvements(df):
        print(f"  • {suggestion}")
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = create_integration_example()
