"""
Intelligent Schema Analyzer for Healthcare Analytics
Automatically detects column types and applies appropriate statistical operations.
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import logging
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ColumnType(Enum):
    """Enumeration for column data types."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    UNKNOWN = "unknown"


class SchemaAnalyzer:
    """
    Intelligent schema analyzer for healthcare data.
    
    Automatically detects column types and applies appropriate statistical operations
    based on data characteristics and healthcare domain knowledge.
    """
    
    def __init__(self):
        self.type_cache = {}
        self.operation_log = []
        
        # Healthcare-specific column patterns
        self.healthcare_patterns = {
            'categorical_keywords': {
                'gender', 'sex', 'diagnosis', 'drug', 'medication', 'treatment',
                'condition', 'symptom', 'specialty', 'department', 'hospital',
                'doctor', 'physician', 'patient_type', 'admission_type',
                'discharge_status', 'insurance', 'payment_method', 'ethnicity',
                'race', 'marital_status', 'religion', 'blood_type', 'rh_factor'
            },
            'datetime_keywords': {
                'date', 'time', 'admit', 'discharge', 'birth', 'appointment',
                'visit', 'created', 'updated', 'timestamp', 'dob', 'admission',
                'discharge_date', 'visit_date', 'prescription_date'
            },
            'boolean_keywords': {
                'active', 'inactive', 'alive', 'dead', 'survived', 'expired',
                'readmitted', 'emergency', 'urgent', 'elective', 'smoker',
                'pregnant', 'allergic', 'chronic', 'acute'
            }
        }
    
    def detect_column_type(self, column: pd.Series, column_name: str) -> ColumnType:
        """
        Intelligently detect the data type of a column.
        
        Args:
            column: Pandas Series to analyze
            column_name: Name of the column
            
        Returns:
            ColumnType enum value
        """
        # Check cache first
        cache_key = f"{column_name}_{len(column)}_{column.dtype}"
        if cache_key in self.type_cache:
            return self.type_cache[cache_key]
        
        # Remove missing values for analysis
        clean_series = column.dropna()
        
        if len(clean_series) == 0:
            detected_type = ColumnType.UNKNOWN
        else:
            # Check for boolean first (most specific)
            if self._is_boolean_column(clean_series, column_name):
                detected_type = ColumnType.BOOLEAN
            # Check for datetime next
            elif self._is_datetime_column(clean_series, column_name):
                detected_type = ColumnType.DATETIME
            # Check for numerical
            elif self._is_numerical_column(clean_series, column_name):
                detected_type = ColumnType.NUMERICAL
            # Check for categorical
            elif self._is_categorical_column(clean_series, column_name):
                detected_type = ColumnType.CATEGORICAL
            # Check for text
            elif self._is_text_column(clean_series, column_name):
                detected_type = ColumnType.TEXT
            else:
                detected_type = ColumnType.UNKNOWN
        
        # Cache the result
        self.type_cache[cache_key] = detected_type
        return detected_type
    
    def _is_datetime_column(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains datetime data."""
        # Check column name patterns first
        name_lower = column_name.lower()
        if any(keyword in name_lower for keyword in self.healthcare_patterns['datetime_keywords']):
            return True
        
        # Check pandas dtype first
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        
        # Skip if it's clearly numeric
        if pd.api.types.is_numeric_dtype(series):
            return False
        
        # Try to convert to datetime with better error handling
        try:
            # Check a sample first
            sample = series.head(10).dropna()
            if len(sample) == 0:
                return False
            
            # Only try conversion if it looks like a string date
            sample_str = sample.astype(str)
            # Look for date patterns in the sample
            date_patterns = ['-', '/', ':', ' AM', ' PM', 'T', 'Z']
            has_date_pattern = any(any(pattern in val for pattern in date_patterns) for val in sample_str)
            
            if has_date_pattern:
                converted = pd.to_datetime(sample, errors='coerce')
                if not converted.isna().all():
                    return True
        except:
            pass
        
        return False
    
    def _is_boolean_column(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains boolean data."""
        # Check column name patterns
        name_lower = column_name.lower()
        if any(keyword in name_lower for keyword in self.healthcare_patterns['boolean_keywords']):
            return True
        
        # Check unique values (should be 2 or less)
        unique_values = series.astype(str).str.lower().unique()
        if len(unique_values) <= 2:
            # Check if values are boolean-like
            bool_patterns = {'true', 'false', 'yes', 'no', 'y', 'n', '1', '0', 't', 'f'}
            if all(val in bool_patterns for val in unique_values):
                return True
        
        # Check pandas dtype
        if series.dtype == 'bool':
            return True
        
        return False
    
    def _is_numerical_column(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains numerical data."""
        # Check pandas dtype first
        if pd.api.types.is_numeric_dtype(series):
            # For integer columns, check if they could be categorical codes
            if series.dtype in ['int64', 'int32']:
                # If it's an ID column or has very few unique values, it might be categorical
                if 'id' in column_name.lower():
                    return False
                unique_ratio = series.nunique() / len(series)
                if unique_ratio <= 0.05:  # Less than 5% unique values suggests categorical
                    return False
            return True
        
        # Try to convert to numeric
        try:
            # Check a sample first
            sample = series.head(100).dropna()
            if len(sample) == 0:
                return False
            
            converted = pd.to_numeric(sample, errors='raise')
            # Check if conversion makes sense
            if converted.nunique() > 2:  # More than 2 unique values
                return True
        except:
            pass
        
        return False
    
    def _is_categorical_column(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains categorical data."""
        # Check column name patterns
        name_lower = column_name.lower()
        if any(keyword in name_lower for keyword in self.healthcare_patterns['categorical_keywords']):
            return True
        
        # Check string/object dtype
        if series.dtype == 'object' or series.dtype.name == 'category':
            return True
        
        # Check if numeric but with few unique values
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = series.nunique() / len(series)
            if unique_ratio <= 0.05:  # Less than 5% unique values suggests categorical
                return True
        
        return False
    
    def _is_text_column(self, series: pd.Series, column_name: str) -> bool:
        """Check if column contains free-text data."""
        if series.dtype == 'object':
            # Check average string length
            avg_length = series.astype(str).str.len().mean()
            if avg_length > 50:  # Long text suggests free text
                return True
        
        return False
    
    def analyze_numerical_column(self, column: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Perform numerical analysis on a column.
        
        Args:
            column: Pandas Series with numerical data
            column_name: Name of the column
            
        Returns:
            Dictionary with numerical statistics
        """
        try:
            clean_series = pd.to_numeric(column, errors='coerce').dropna()
            
            if len(clean_series) == 0:
                return {"error": "No valid numerical data"}
            
            stats = {
                'type': 'numerical',
                'count': len(clean_series),
                'missing': len(column) - len(clean_series),
                'mean': clean_series.mean(),
                'median': clean_series.median(),
                'std': clean_series.std(),
                'min': clean_series.min(),
                'max': clean_series.max(),
                'q25': clean_series.quantile(0.25),
                'q75': clean_series.quantile(0.75),
                'skewness': clean_series.skew(),
                'kurtosis': clean_series.kurtosis()
            }
            
            # Healthcare-specific insights
            if 'age' in column_name.lower():
                stats['healthcare_insights'] = {
                    'age_groups': {
                        'pediatric': (clean_series < 18).sum(),
                        'adult': ((clean_series >= 18) & (clean_series < 65)).sum(),
                        'elderly': (clean_series >= 65).sum()
                    }
                }
            
            return stats
            
        except Exception as e:
            self._log_operation_error(f"numerical_analysis", column_name, str(e))
            return {"error": f"Numerical analysis failed: {str(e)}"}
    
    def analyze_categorical_column(self, column: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Perform categorical analysis on a column.
        
        Args:
            column: Pandas Series with categorical data
            column_name: Name of the column
            
        Returns:
            Dictionary with categorical statistics
        """
        try:
            clean_series = column.dropna().astype(str)
            
            if len(clean_series) == 0:
                return {"error": "No valid categorical data"}
            
            value_counts = clean_series.value_counts()
            total_count = len(clean_series)
            
            stats = {
                'type': 'categorical',
                'count': total_count,
                'missing': len(column) - total_count,
                'unique_values': len(value_counts),
                'most_common': value_counts.head(10).to_dict(),
                'distribution': (value_counts / total_count * 100).round(2).to_dict()
            }
            
            # Healthcare-specific insights for gender
            if 'gender' in column_name.lower() or 'sex' in column_name.lower():
                stats['healthcare_insights'] = {
                    'gender_distribution': stats['distribution'],
                    'dominant_gender': value_counts.index[0] if len(value_counts) > 0 else None
                }
            
            # Healthcare-specific insights for drugs/medications
            if 'drug' in column_name.lower() or 'medication' in column_name.lower():
                stats['healthcare_insights'] = {
                    'top_medications': dict(value_counts.head(5)),
                    'medication_diversity': len(value_counts)
                }
            
            return stats
            
        except Exception as e:
            self._log_operation_error(f"categorical_analysis", column_name, str(e))
            return {"error": f"Categorical analysis failed: {str(e)}"}
    
    def analyze_datetime_column(self, column: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Perform datetime analysis on a column.
        
        Args:
            column: Pandas Series with datetime data
            column_name: Name of the column
            
        Returns:
            Dictionary with datetime statistics
        """
        try:
            clean_series = pd.to_datetime(column, errors='coerce').dropna()
            
            if len(clean_series) == 0:
                return {"error": "No valid datetime data"}
            
            stats = {
                'type': 'datetime',
                'count': len(clean_series),
                'missing': len(column) - len(clean_series),
                'min_date': clean_series.min(),
                'max_date': clean_series.max(),
                'date_range_days': (clean_series.max() - clean_series.min()).days
            }
            
            # Time-based analysis
            if len(clean_series) > 1:
                stats['time_insights'] = {
                    'yearly_distribution': clean_series.dt.year.value_counts().head(5).to_dict(),
                    'monthly_distribution': clean_series.dt.month.value_counts().head(12).to_dict(),
                    'weekday_distribution': clean_series.dt.dayofweek.value_counts().to_dict()
                }
            
            # Healthcare-specific insights
            if 'birth' in column_name.lower() or 'dob' in column_name.lower():
                current_year = datetime.now().year
                ages = current_year - clean_series.dt.year
                stats['healthcare_insights'] = {
                    'age_statistics': {
                        'mean_age': ages.mean(),
                        'median_age': ages.median(),
                        'age_distribution': {
                            'pediatric': (ages < 18).sum(),
                            'adult': ((ages >= 18) & (ages < 65)).sum(),
                            'elderly': (ages >= 65).sum()
                        }
                    }
                }
            
            return stats
            
        except Exception as e:
            self._log_operation_error(f"datetime_analysis", column_name, str(e))
            return {"error": f"Datetime analysis failed: {str(e)}"}
    
    def analyze_boolean_column(self, column: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Perform boolean analysis on a column.
        
        Args:
            column: Pandas Series with boolean data
            column_name: Name of the column
            
        Returns:
            Dictionary with boolean statistics
        """
        try:
            # Convert to boolean
            clean_series = column.dropna()
            
            # Standardize boolean values
            bool_mapping = {
                'true': True, 'yes': True, 'y': True, '1': True, 't': True,
                'false': False, 'no': False, 'n': False, '0': False, 'f': False
            }
            
            standardized = clean_series.astype(str).str.lower().map(bool_mapping)
            standardized = standardized.dropna()
            
            if len(standardized) == 0:
                return {"error": "No valid boolean data"}
            
            value_counts = standardized.value_counts()
            total_count = len(standardized)
            
            stats = {
                'type': 'boolean',
                'count': total_count,
                'missing': len(column) - total_count,
                'true_count': value_counts.get(True, 0),
                'false_count': value_counts.get(False, 0),
                'true_percentage': (value_counts.get(True, 0) / total_count * 100).round(2),
                'false_percentage': (value_counts.get(False, 0) / total_count * 100).round(2)
            }
            
            return stats
            
        except Exception as e:
            self._log_operation_error(f"boolean_analysis", column_name, str(e))
            return {"error": f"Boolean analysis failed: {str(e)}"}
    
    def analyze_text_column(self, column: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Perform text analysis on a column.
        
        Args:
            column: Pandas Series with text data
            column_name: Name of the column
            
        Returns:
            Dictionary with text statistics
        """
        try:
            clean_series = column.dropna().astype(str)
            
            if len(clean_series) == 0:
                return {"error": "No valid text data"}
            
            stats = {
                'type': 'text',
                'count': len(clean_series),
                'missing': len(column) - len(clean_series),
                'avg_length': clean_series.str.len().mean(),
                'max_length': clean_series.str.len().max(),
                'min_length': clean_series.str.len().min(),
                'empty_strings': (clean_series == '').sum()
            }
            
            return stats
            
        except Exception as e:
            self._log_operation_error(f"text_analysis", column_name, str(e))
            return {"error": f"Text analysis failed: {str(e)}"}
    
    def analyze_column(self, column: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Main method to analyze a column with appropriate operations.
        
        Args:
            column: Pandas Series to analyze
            column_name: Name of the column
            
        Returns:
            Dictionary with analysis results
        """
        # Detect column type
        column_type = self.detect_column_type(column, column_name)
        
        # Route to appropriate analysis method
        if column_type == ColumnType.NUMERICAL:
            return self.analyze_numerical_column(column, column_name)
        elif column_type == ColumnType.CATEGORICAL:
            return self.analyze_categorical_column(column, column_name)
        elif column_type == ColumnType.DATETIME:
            return self.analyze_datetime_column(column, column_name)
        elif column_type == ColumnType.BOOLEAN:
            return self.analyze_boolean_column(column, column_name)
        elif column_type == ColumnType.TEXT:
            return self.analyze_text_column(column, column_name)
        else:
            return {"error": "Unknown column type", "type": "unknown"}
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze entire dataframe and return structured insights.
        
        Args:
            df: Pandas DataFrame to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        results = {
            'schema_overview': {
                'total_columns': len(df.columns),
                'total_rows': len(df),
                'column_types': {},
                'type_distribution': {}
            },
            'column_analysis': {},
            'summary_insights': []
        }
        
        # Analyze each column
        for column_name in df.columns:
            column_analysis = self.analyze_column(df[column_name], column_name)
            results['column_analysis'][column_name] = column_analysis
            
            # Track type distribution
            if 'type' in column_analysis and 'error' not in column_analysis:
                col_type = column_analysis['type']
                results['schema_overview']['column_types'][column_name] = col_type
                results['schema_overview']['type_distribution'][col_type] = \
                    results['schema_overview']['type_distribution'].get(col_type, 0) + 1
        
        # Generate summary insights
        results['summary_insights'] = self._generate_summary_insights(results)
        
        return results
    
    def _generate_summary_insights(self, analysis_results: Dict[str, Any]) -> List[str]:
        """Generate summary insights from analysis results."""
        insights = []
        
        # Gender distribution insight
        for col_name, analysis in analysis_results['column_analysis'].items():
            if 'gender' in col_name.lower() and 'healthcare_insights' in analysis:
                gender_dist = analysis['healthcare_insights'].get('gender_distribution', {})
                if gender_dist:
                    dominant = max(gender_dist, key=gender_dist.get)
                    insights.append(f"Gender distribution: {dominant} dominant ({gender_dist[dominant]:.1f}%)")
        
        # Age-related insights
        for col_name, analysis in analysis_results['column_analysis'].items():
            if 'age' in col_name.lower() and 'healthcare_insights' in analysis:
                age_groups = analysis['healthcare_insights'].get('age_groups', {})
                if age_groups:
                    insights.append(f"Age groups: Pediatric ({age_groups.get('pediatric', 0)}), "
                                  f"Adult ({age_groups.get('adult', 0)}), Elderly ({age_groups.get('elderly', 0)})")
        
        # Data quality insights
        total_missing = sum(analysis.get('missing', 0) for analysis in analysis_results['column_analysis'].values())
        total_rows = analysis_results['schema_overview']['total_rows']
        if total_missing > 0:
            missing_percentage = (total_missing / (total_rows * len(analysis_results['column_analysis']))) * 100
            insights.append(f"Data quality: {missing_percentage:.1f}% missing values across all columns")
        
        return insights
    
    def _log_operation_error(self, operation: str, column_name: str, error_message: str):
        """Log operation errors for debugging."""
        error_msg = f"Operation '{operation}' failed on column '{column_name}': {error_message}"
        logger.warning(error_msg)
        self.operation_log.append({
            'timestamp': datetime.now(),
            'operation': operation,
            'column': column_name,
            'error': error_message
        })
    
    def get_operation_log(self) -> List[Dict[str, Any]]:
        """Get the operation log for debugging."""
        return self.operation_log


def create_sample_healthcare_data() -> pd.DataFrame:
    """Create sample healthcare dataset for testing."""
    np.random.seed(42)
    n_records = 1000
    
    data = {
        'patient_id': range(1, n_records + 1),
        'age': np.random.randint(18, 85, n_records),
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_records, p=[0.48, 0.51, 0.01]),
        'diagnosis': np.random.choice(['Hypertension', 'Diabetes', 'Asthma', 'Heart Disease', 'None'], n_records),
        'drug_name': np.random.choice(['Metformin', 'Atorvastatin', 'Lisinopril', 'Albuterol', 'Aspirin'], n_records),
        'dosage': np.random.normal(100, 25, n_records),
        'admission_date': pd.date_range('2023-01-01', periods=n_records, freq='h'),
        'discharge_date': pd.date_range('2023-01-02', periods=n_records, freq='h'),
        'emergency_visit': np.random.choice([True, False], n_records, p=[0.2, 0.8]),
        'insurance_type': np.random.choice(['Private', 'Medicare', 'Medicaid', 'Self-pay'], n_records),
        'doctor_notes': ['Patient presents with ' + np.random.choice(['fever', 'cough', 'pain', 'fatigue']) 
                        for _ in range(n_records)]
    }
    
    df = pd.DataFrame(data)
    
    # Ensure proper data types
    df['patient_id'] = df['patient_id'].astype('int64')
    df['age'] = df['age'].astype('int64')
    df['gender'] = df['gender'].astype('category')
    df['diagnosis'] = df['diagnosis'].astype('category')
    df['drug_name'] = df['drug_name'].astype('category')
    df['dosage'] = df['dosage'].astype('float64')
    df['emergency_visit'] = df['emergency_visit'].astype('bool')
    df['insurance_type'] = df['insurance_type'].astype('category')
    df['doctor_notes'] = df['doctor_notes'].astype('string')
    
    return df


# Example usage and testing
if __name__ == "__main__":
    # Create sample data
    sample_df = create_sample_healthcare_data()
    
    # Initialize analyzer
    analyzer = SchemaAnalyzer()
    
    # Analyze the dataframe
    results = analyzer.analyze_dataframe(sample_df)
    
    # Print results
    print("=" * 80)
    print("HEALTHCARE DATA ANALYSIS RESULTS")
    print("=" * 80)
    
    print("\nSCHEMA OVERVIEW:")
    print(f"Total columns: {results['schema_overview']['total_columns']}")
    print(f"Total rows: {results['schema_overview']['total_rows']}")
    print(f"Type distribution: {results['schema_overview']['type_distribution']}")
    
    print("\nCOLUMN ANALYSIS:")
    for col_name, analysis in results['column_analysis'].items():
        print(f"\n{col_name.upper()} ({analysis.get('type', 'unknown')}):")
        if 'error' in analysis:
            print(f"  Error: {analysis['error']}")
        else:
            if analysis['type'] == 'categorical':
                print(f"  Distribution: {analysis.get('distribution', {})}")
            elif analysis['type'] == 'numerical':
                print(f"  Mean: {analysis.get('mean', 'N/A'):.2f}, Std: {analysis.get('std', 'N/A'):.2f}")
            elif analysis['type'] == 'datetime':
                print(f"  Range: {analysis.get('min_date', 'N/A')} to {analysis.get('max_date', 'N/A')}")
            elif analysis['type'] == 'boolean':
                print(f"  True: {analysis.get('true_percentage', 'N/A')}%, False: {analysis.get('false_percentage', 'N/A')}%")
    
    print("\nSUMMARY INSIGHTS:")
    for insight in results['summary_insights']:
        print(f"  • {insight}")
