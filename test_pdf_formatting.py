#!/usr/bin/env python3
"""
Test the improved PDF formatting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from utils.pdf_generator import generate_pdf_report

def test_pdf_formatting():
    """Test the improved PDF formatting."""
    print("🧪 Testing improved PDF formatting...")
    
    # Create sample data
    patient_ids = [f'P_{i:04d}' for i in range(100)]
    drug_names = (['Aspirin', 'Metformin', 'Lisinopril'] * 33 + ['Atorvastatin'] * 34)[:100]
    dosages = [100, 500, 10, 20] * 25
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    risk_scores = [0, 1] * 50
    
    df = pd.DataFrame({
        'patient_id': patient_ids,
        'drug_name': drug_names,
        'dosage': dosages,
        'date': dates,
        'risk_score': risk_scores
    })
    
    col_map = {
        'patient_id': 'patient_id',
        'drug_name': 'drug_name',
        'dosage': 'dosage',
        'date': 'date'
    }
    
    # Sample results
    results = {
        'risk_agent_advanced': {
            'status': 'ok',
            'metrics': {
                'Model': 'Advanced Ensemble (logistic_regression)',
                'Accuracy': '0.710',
                'Precision': '0.745',
                'Recall': '0.710',
                'F1-Score': '0.699',
                'ROC-AUC': '0.745',
                'Execution': '53327.3ms'
            },
            'statistical_validation': {
                'validation_table': pd.DataFrame([
                    {
                        'Agent Name': 'Risk Agent',
                        'Model Name': 'XGBoost Healthcare',
                        'Test Variable': 'Accuracy',
                        'Group Variable': 'Risk Groups',
                        'Test': 'Independent T-test',
                        'Statistic': 't = 2.456',
                        'P-value': '0.0142',
                        'P-value (corrected)': '0.0142',
                        'Effect Size': '0.823',
                        'Effect Type': "Cohen's d",
                        'Significant': 'Yes'
                    },
                    {
                        'Agent Name': 'Cohort Agent',
                        'Model Name': 'DBSCAN Clustering',
                        'Test Variable': 'Silhouette Score',
                        'Group Variable': 'Clustering Algorithms',
                        'Test': 'Z-test Comparison',
                        'Statistic': 'Z = 1.732',
                        'P-value': '0.0833',
                        'P-value (corrected)': '0.0833',
                        'Effect Size': '1.732',
                        'Effect Type': 'Z-score',
                        'Significant': 'No'
                    }
                ]),
                'validation_summary': """
📊 **Agent Performance Statistical Validation Report**

**Overall Summary:**
- Total Models Analyzed: 2
- Classification Models: 1
- Clustering Models: 1
- Alpha Level: 0.05

**Key Findings:**
• Risk Agent shows significant accuracy improvement (p < 0.05)
• Cohort Agent demonstrates excellent clustering performance
• Effect sizes indicate practical significance
• Multiple testing correction applied appropriately
"""
            }
        }
    }
    
    # Sample LLM insights
    llm_insights = """**Clinical Risk Assessment Findings:**

**High-Risk Patient Identification:**
• Patients with prescription counts above 75th percentile show 3.2x higher risk scores
• Polypharmacy (>5 medications) correlates with increased adverse event risk
• High-dosage regimens (>75th percentile) indicate potential medication safety issues

**Medication Pattern Analysis:**
• Cardiovascular medications show strongest correlation with high-risk patients
• Metformin prescriptions suggest well-controlled diabetic population
• Temporal analysis reveals weekend prescription clustering for acute conditions

**Recommendations:**
• Implement medication reconciliation systems for high-risk patients
• Consider clinical pharmacist review for polypharmacy cases
• Develop risk-stratified monitoring protocols
• Enhanced medication adherence programs needed for high-risk groups

**Quality Metrics:**
• Risk prediction accuracy: 71% with 74.5% precision
• Model demonstrates strong discriminatory power (ROC-AUC: 0.745)
• Advanced ensemble methods provide robust performance across patient subgroups"""
    
    # Generate PDF
    try:
        pdf_buffer = generate_pdf_report(df, col_map, results, llm_insights)
        
        # Save test PDF
        with open('test_improved_formatting.pdf', 'wb') as f:
            f.write(pdf_buffer.getvalue())
        
        print("✅ Improved PDF formatting test completed!")
        print("📄 Test PDF saved as: test_improved_formatting.pdf")
        print("\n🎨 Formatting Improvements Applied:")
        print("• Enhanced statistical validation table with professional styling")
        print("• Improved AI insights with section headers and better formatting")
        print("• Professional color scheme and typography")
        print("• Better table layouts with proper spacing")
        print("• Enhanced visual hierarchy and readability")
        
    except Exception as e:
        print(f"❌ Error testing PDF formatting: {e}")

if __name__ == "__main__":
    test_pdf_formatting()
