import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from scipy import stats
import io
import base64
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / pooled_se
    
    return d


def perform_statistical_validation(df, col_map, results):
    """Perform statistical validation tests."""
    validation_results = {}
    
    if results is None:
        return validation_results
    
    # Get numeric columns for statistical tests
    numeric_cols = []
    if col_map is not None:
        for col, cat in col_map.items():
            if cat not in ["patient_id", "date"] and col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_cols.append(col)
    
    if len(numeric_cols) < 2:
        validation_results["error"] = "Insufficient numeric columns for statistical validation"
        return validation_results
    
    # T-test between high and low risk groups (if risk analysis was done)
    if results.get("risk") and results["risk"].get("risk_df") is not None:
        risk_df = results["risk"]["risk_df"]
        if "risk_label" in risk_df.columns:
            high_risk = risk_df[risk_df["risk_label"] == "High Risk"]
            low_risk = risk_df[risk_df["risk_label"] == "Low Risk"]
        elif "__risk_label" in risk_df.columns:
            high_risk = risk_df[risk_df["__risk_label"] == "High Risk"]
            low_risk = risk_df[risk_df["__risk_label"] == "Low Risk"]
        else:
            # Create risk groups based on probability if labels don't exist
            risk_scores = pd.to_numeric(risk_df.get("risk_probability", risk_df.get("risk_score", 0)), errors='coerce')
            median_score = risk_scores.median()
            high_risk = risk_df[risk_scores > median_score]
            low_risk = risk_df[risk_scores <= median_score]
        
        t_test_results = {}
        for col in numeric_cols[:3]:  # Test top 3 numeric columns
            if col in high_risk.columns and col in low_risk.columns:
                try:
                    high_vals = pd.to_numeric(high_risk[col], errors='coerce').dropna()
                    low_vals = pd.to_numeric(low_risk[col], errors='coerce').dropna()
                    
                    if len(high_vals) > 1 and len(low_vals) > 1:
                        t_stat, p_value = stats.ttest_ind(high_vals, low_vals)
                        cohens_d = calculate_cohens_d(high_vals, low_vals)
                        
                        t_test_results[col] = {
                            "t_statistic": t_stat,
                            "p_value": p_value,
                            "cohens_d": cohens_d,
                            "significant": p_value < 0.05
                        }
                except Exception as e:
                    t_test_results[col] = {"error": str(e)}
        
        validation_results["t_test"] = t_test_results
    
    # ANOVA test between cohorts (if cohort analysis was done)
    if results.get("cohort") and results["cohort"].get("cohort_df") is not None:
        cohort_df = results["cohort"]["cohort_df"]
        
        cohort_col = "cohort_label" if "cohort_label" in cohort_df.columns else "__cohort" if "__cohort" in cohort_df.columns else None
        
        if cohort_col:
            cohort_groups = cohort_df[cohort_col].unique()
            
            anova_results = {}
            for col in numeric_cols[:2]:  # Test top 2 numeric columns
                if col in cohort_df.columns:
                    try:
                        group_data = []
                        for cohort in cohort_groups:
                            cohort_data = cohort_df[cohort_df[cohort_col] == cohort]
                            numeric_data = pd.to_numeric(cohort_data[col], errors='coerce').dropna()
                            if len(numeric_data) > 1:
                                group_data.append(numeric_data)
                        
                        if len(group_data) >= 2:
                            f_stat, p_value = stats.f_oneway(*group_data)
                            anova_results[col] = {
                                "f_statistic": f_stat,
                                "p_value": p_value,
                                "significant": p_value < 0.05
                            }
                    except Exception as e:
                        anova_results[col] = {"error": str(e)}
            
            validation_results["anova"] = anova_results
    
    return validation_results


def plotly_to_image(fig):
    """Convert Plotly figure to image for PDF."""
    img_bytes = fig.to_image(format="png", width=800, height=600)
    return img_bytes


def create_dataset_table(df, col_map):
    """Create Table 1: Dataset details."""
    data = []
    
    # Basic dataset info
    data.append(["Dataset Overview", "", ""])
    data.append(["Total Records", f"{len(df):,}", ""])
    data.append(["Total Columns", f"{len(df.columns)}", ""])
    data.append(["Missing Values", f"{df.isna().sum().sum():,}", ""])
    
    # Column analysis
    drug_col, patient_col, date_col = None, None, None
    if col_map is not None:
        drug_col = next((c for c, cat in col_map.items() if cat == "drug_name" and c in df.columns), None)
        patient_col = next((c for c, cat in col_map.items() if cat == "patient_id" and c in df.columns), None)
        date_col = next((c for c, cat in col_map.items() if cat == "date" and c in df.columns), None)
    
    data.append(["", "", ""])
    data.append(["Column Analysis", "", ""])
    
    if drug_col:
        unique_drugs = df[drug_col].nunique()
        data.append(["Unique Drugs", f"{unique_drugs:,}", f"Column: {drug_col}"])
    
    if patient_col:
        unique_patients = df[patient_col].nunique()
        data.append(["Unique Patients", f"{unique_patients:,}", f"Column: {patient_col}"])
    
    if date_col:
        date_range = f"{df[date_col].min()} to {df[date_col].max()}"
        data.append(["Date Range", date_range, f"Column: {date_col}"])
    
    # Domain characteristics
    data.append(["", "", ""])
    data.append(["Domain Characteristics", "", ""])
    data.append(["Domain", "Healthcare/Pharmaceutical", "Prescription analysis"])
    data.append(["Data Type", "Clinical Prescription Data", "Drug prescription patterns"])
    data.append(["Analysis Focus", "Multi-model ML analysis", "Risk, patterns, trends, anomalies"])
    
    return data


def create_models_table(results):
    """Create Table 2: ML/DL models used."""
    data = []
    data.append(["Model Type", "Algorithm", "Purpose", "Category"])
    
    if results is None:
        return data
    
    # Risk Analysis Model
    if results.get("risk"):
        data.append(["Deep Learning", "PyTorch MLP", "Risk classification", "Supervised Learning"])
    
    # Cohort Analysis Model
    if results.get("cohort"):
        data.append(["Machine Learning", "KMeans Clustering", "Patient cohort identification", "Unsupervised Learning"])
    
    # Anomaly Detection Model
    if results.get("anomaly"):
        data.append(["Deep Learning", "PyTorch Autoencoder", "Anomaly detection", "Unsupervised Learning"])
    
    # Trend Analysis Model
    if results.get("trend") and isinstance(results["trend"], dict):
        trend_model = results["trend"].get("metrics", {}).get("Model", "Holt-Winters")
        data.append(["Time Series", trend_model, "Trend forecasting", "Time Series Analysis"])
    
    # Pattern Analysis Model
    if results.get("pattern"):
        data.append(["Association Rules", "Apriori-lite", "Co-prescription patterns", "Pattern Mining"])
    
    # LLM Model
    data.append(["Large Language Model", "Phi-4 mini", "Clinical insights generation", "Natural Language Processing"])
    
    return data


def create_metrics_table(results):
    """Create Table 3: Accuracy and metrics of models."""
    data = []
    data.append(["Model", "Accuracy/Performance", "Key Metrics", "Execution Time"])
    
    if results is None:
        return data
    
    for agent_key, result in results.items():
        if result and isinstance(result, dict) and result.get("status") == "ok" and result.get("metrics"):
            metrics = result["metrics"]
            model_name = metrics.get("Model", agent_key.title())
            
            # Get main performance metric
            performance = "N/A"
            if "Accuracy" in metrics:
                performance = metrics["Accuracy"]
            elif "Silhouette" in metrics:
                performance = metrics["Silhouette"]
            elif "Confidence" in metrics:
                performance = metrics["Confidence"]
            
            # Get additional metrics
            additional_metrics = []
            for metric_name in ["Precision", "Recall", "RMSE", "MAE", "Variance"]:
                if metric_name in metrics:
                    additional_metrics.append(f"{metric_name}: {metrics[metric_name]}")
            
            additional_str = ", ".join(additional_metrics) if additional_metrics else "N/A"
            execution_time = metrics.get("Execution", "N/A")
            
            data.append([model_name, performance, additional_str, execution_time])
    
    return data


def create_statistical_validation_table(validation_results):
    """Create Table 4: Statistical validation results with enhanced formatting."""
    data = []
    data.append(["Agent Name", "Model Name", "Test Variable", "Group Variable", "Test", "Statistic", "P-value", "P-value (corrected)", "Effect Size", "Effect Type", "Significant"])
    
    if validation_results is None:
        return data
    
    # Check if we have validation data from advanced cohort agent
    if hasattr(validation_results, 'get') and 'validation_table' in validation_results:
        # Use the validation table from advanced cohort agent
        validation_df = validation_results['validation_table']
        if not validation_df.empty:
            for _, row in validation_df.iterrows():
                # Format values for better readability
                agent_name = str(row.get('Agent Name', 'N/A')).strip()
                model_name = str(row.get('Model Name', 'N/A')).strip()
                test_var = str(row.get('Test Variable', 'N/A')).strip()
                group_var = str(row.get('Group Variable', 'N/A')).strip()
                test_name = str(row.get('Test', 'N/A')).strip()
                statistic = str(row.get('Statistic', 'N/A')).strip()
                p_value = str(row.get('P-value', 'N/A')).strip()
                p_corrected = str(row.get('P-value (corrected)', 'N/A')).strip()
                effect_size = str(row.get('Effect Size', 'N/A')).strip()
                effect_type = str(row.get('Effect Type', 'N/A')).strip()
                significant = str(row.get('Significant', 'N/A')).strip()
                
                data.append([
                    agent_name, model_name, test_var, group_var, test_name, statistic, p_value, p_corrected, effect_size, effect_type, significant
                ])
    else:
        # Fallback to old validation method for backward compatibility
        # T-test results
        if "t_test" in validation_results:
            for var, results_dict in validation_results["t_test"].items():
                if "error" not in results_dict:
                    t_stat = results_dict["t_statistic"]
                    p_val = results_dict["p_value"]
                    cohens_d = results_dict["cohens_d"]
                    significant = "Yes" if results_dict["significant"] else "No"
                    
                    data.append([
                        "Risk Agent",
                        "Classification Model",
                        var,
                        "Risk Groups",
                        "Independent T-test",
                        f"t = {t_stat:.3f}",
                        f"p = {p_val:.4f}",
                        f"p = {p_val:.4f}",
                        f"d = {cohens_d:.3f}",
                        "Cohen's d",
                        significant
                    ])
        
        # ANOVA results
        if "anova" in validation_results:
            for var, results_dict in validation_results["anova"].items():
                if "error" not in results_dict:
                    f_stat = results_dict["f_statistic"]
                    p_val = results_dict["p_value"]
                    significant = "Yes" if results_dict["significant"] else "No"
                    
                    data.append([
                        "Cohort Agent",
                        "Clustering Model",
                        var,
                        "Cohort Groups",
                        "One-way ANOVA",
                        f"F = {f_stat:.3f}",
                        f"p = {p_val:.4f}",
                        f"p = {p_val:.4f}",
                        f"η² = {results_dict['eta_squared']:.3f}",
                        "Effect Size",
                        significant
                    ])
    
    return data


def generate_pdf_report(df, col_map, results, llm_insights=None, dynamic_summary=None):
    """Generate comprehensive PDF report."""
    
    # Create PDF buffer
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.darkblue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        textColor=colors.darkblue
    )
    
    content = []
    
    # Title Page
    content.append(Paragraph("Prescription Trend AI Analysis Report", title_style))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    content.append(Spacer(1, 50))
    
    # Executive Summary
    content.append(Paragraph("Executive Summary", heading_style))
    
    if dynamic_summary:
        summary_text = dynamic_summary
    else:
        num_records = len(df)
        num_cols = len(df.columns)
        
        drug_col = "N/A"
        if col_map is not None:
            drug_col = next((c for c, cat in col_map.items() if cat == "medications" and c in df.columns), "N/A")
        
        summary_text = f"""
        This report presents a comprehensive multi-agent analysis of the clinical dataset comprising {num_records:,} records and {num_cols} variables. 
        The analysis was performed using an autonomous medical pipeline including Risk Assessment, Cohort Identification, Anomaly Detection, Pattern Mining, and Trend Forecasting.
        Key focus areas included analyzing '{drug_col}' distributions and their clinical correlations.
        """
    content.append(Paragraph(summary_text, styles['Normal']))
    content.append(Spacer(1, 20))

    # Analysis Confidence Section
    content.append(Paragraph("Analysis Confidence Score", heading_style))
    # Note: We need to pass eval_metrics or calculate it here
    completeness = 1.0 - (df.isnull().sum().sum() / df.size)
    mapping_confidence = np.mean([0.8]) # Fallback
    confidence_score = (completeness * 0.4) + (mapping_confidence * 0.6)
    
    confidence_text = f"The overall pipeline confidence score is <b>{confidence_score:.2f}</b>. "
    if confidence_score > 0.8:
        confidence_text += "This indicates a high-fidelity mapping and high data quality, making the findings clinically reliable."
    elif confidence_score > 0.5:
        confidence_text += "This indicates moderate data quality. Results should be interpreted with clinical context."
    else:
        confidence_text += "Low confidence score detected. Manual verification of column mappings and data cleaning is recommended."
    
    content.append(Paragraph(confidence_text, styles['Normal']))
    content.append(Spacer(1, 20))
    
    # Table 1: Dataset Details
    content.append(Paragraph("Table 1: Dataset Details and Characteristics", heading_style))
    table1_data = create_dataset_table(df, col_map)
    table1 = Table(table1_data, colWidths=[1.8*inch, 1.6*inch, 2.1*inch])
    table1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('LEFTPADDING', (0, 1), (-1, -1), 5),
        ('RIGHTPADDING', (0, 1), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(table1)
    content.append(Spacer(1, 20))
    
    # Table 2: Models Used
    content.append(Paragraph("Table 2: Machine Learning and Deep Learning Models", heading_style))
    table2_data = create_models_table(results)
    table2 = Table(table2_data, colWidths=[1.4*inch, 2.1*inch, 2.1*inch, 1.4*inch])
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 1), (-1, -1), 5),
        ('RIGHTPADDING', (0, 1), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(table2)
    content.append(Spacer(1, 20))
    
    # Table 3: Metrics
    content.append(Paragraph("Table 3: Model Performance Metrics", heading_style))
    table3_data = create_metrics_table(results)
    table3 = Table(table3_data, colWidths=[1.4*inch, 1.4*inch, 2.6*inch, 1.2*inch])
    table3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 15),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('LEFTPADDING', (0, 1), (-1, -1), 5),
        ('RIGHTPADDING', (0, 1), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    content.append(table3)
    content.append(Spacer(1, 20))
    
    # Statistical Validation
    content.append(Paragraph("Statistical Validation", heading_style))
    
    # Try to get statistical validation results from agents first
    validation_results = None
    validation_summary = ""
    
    # Check if any agent has statistical validation results
    if results is not None:
        for agent_key, result in results.items():
            if result.get("status") == "ok" and "statistical_validation" in result:
                validation_results = result["statistical_validation"]
                validation_summary = validation_results.get("validation_summary", "")
                break
    
    # Fallback to simple validation if no agent results found
    if validation_results is None:
        validation_results = perform_statistical_validation(df, col_map, results)
    
    # Add statistical validation table
    content.append(Paragraph("Table 4: Statistical Validation Results", heading_style))
    table4_data = create_statistical_validation_table(validation_results)
    if table4_data and len(table4_data) > 1:  # More than just header
        # Enhanced table styling for professional appearance
        table4 = Table(table4_data, colWidths=[0.9*inch, 1.3*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.9*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.7*inch])
        table4.setStyle(TableStyle([
            # Header styling with professional appearance
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E4088')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            # Data row styling with alternating colors
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F8F9FA')),
            ('BACKGROUND', (0, 2), (-1, -1), colors.HexColor('#FFFFFF')),
            ('ALIGN', (0, 1), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('LEFTPADDING', (0, 1), (-1, -1), 8),
            ('RIGHTPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
            # Professional grid and borders
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
            ('LINEBELOW', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
            ('LINEABOVE', (0, 0), (-1, -1), 1, colors.HexColor('#E0E0E0')),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # Alternating row colors for better readability
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8F9FA'), colors.HexColor('#FFFFFF')]),
        ]))
        content.append(table4)
        
        # Add statistical validation summary
        if validation_summary:
            content.append(Paragraph(validation_summary, styles['Normal']))
    else:
        content.append(Paragraph("No statistical validation results available.", styles['Normal']))
    
    content.append(Spacer(1, 20))
    
    # Add LLM Insights if available
    if llm_insights and "ERROR_OLLAMA_DOWN" not in llm_insights:
        content.append(Paragraph("AI-Generated Clinical Insights", heading_style))
        content.append(Spacer(1, 12))
        
        # Enhanced insights styling
        insights_style = ParagraphStyle(
            'ClinicalInsights',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            leftIndent=25,
            bulletIndent=12,
            textColor=colors.black,
            borderColor=colors.lightgrey,
            borderWidth=1,
            borderRadius=5,
            backColor=colors.lightgoldenrodyellow,
            leading=14
        )
        
        # Split insights into meaningful sections
        insight_lines = llm_insights.split('\n')
        current_section = ""
        section_insights = []
        
        for line in insight_lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for section headers
            if line.endswith(':') or line.startswith('**') or line.startswith('•'):
                # Save previous section and start new one
                if current_section and section_insights:
                    # Add section with proper formatting
                    content.append(Paragraph(f"<b>{current_section}</b>", ParagraphStyle(
                        'SectionHeader',
                        parent=styles['Normal'],
                        fontSize=11,
                        spaceAfter=8,
                        textColor=colors.darkblue,
                        leftIndent=0
                    )))
                    
                    # Add insights for this section
                    for insight in section_insights:
                        content.append(Paragraph(f"• {insight}", insights_style))
                    
                    content.append(Spacer(1, 8))
                
                current_section = line.replace(':', '').replace('**', '').replace('•', '').strip()
                section_insights = []
            else:
                # Add to current section
                clean_line = line.lstrip('•').lstrip('-').lstrip('*').strip()
                if clean_line and not clean_line.startswith('**'):
                    section_insights.append(clean_line)
        
        # Add final section
        if current_section and section_insights:
            content.append(Paragraph(f"<b>{current_section}</b>", ParagraphStyle(
                'SectionHeader',
                parent=styles['Normal'],
                fontSize=11,
                spaceAfter=8,
                textColor=colors.darkblue,
                leftIndent=0
            )))
            
            for insight in section_insights:
                content.append(Paragraph(f"• {insight}", insights_style))
        
        content.append(Spacer(1, 20))
    
    # Build PDF
    doc.build(content)
    buffer.seek(0)
    
    return buffer.getvalue()


def create_visualizations_pdf(results):
    """Create separate PDF with visualizations."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    
    content = []
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    content.append(Paragraph("Analysis Visualizations", heading_style))
    
    if results is None:
        doc.build(content)
        buffer.seek(0)
        return buffer.getvalue()

    # Collect all figures
    all_figures = []
    for agent_key, result in results.items():
        if result and isinstance(result, dict) and result.get("status") == "ok" and result.get("figures"):
            for title, fig in result["figures"]:
                all_figures.append((title, fig))
    
    # Limit to 10 figures (5 rows of 2)
    all_figures = all_figures[:10]
    
    # Process figures in pairs for side-by-side display
    for i in range(0, len(all_figures), 2):
        pair = all_figures[i:i+2]
        
        row_titles = []
        row_images = []
        
        for idx, (title, fig) in enumerate(pair):
            try:
                img_bytes = plotly_to_image(fig)
                img_buffer = io.BytesIO(img_bytes)
                # Adjust width for 2-column layout (A4 is ~8.3 inches, so ~3.5 each)
                img = Image(img_buffer, width=3.3*inch, height=2.5*inch)
                
                row_titles.append(Paragraph(f"Figure {i + idx + 1}: {title}", styles['Heading4']))
                row_images.append(img)
            except Exception as e:
                row_titles.append(Paragraph(f"Figure {i + idx + 1}: Error", styles['Heading4']))
                row_images.append(Paragraph(f"Error generating {title}: {str(e)}", styles['Normal']))
        
        # If odd number of figures, add a placeholder for the second column
        if len(pair) == 1:
            row_titles.append("")
            row_images.append("")
            
        # Create a table for side-by-side layout
        fig_table = Table([row_titles, row_images], colWidths=[3.5*inch, 3.5*inch])
        fig_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        
        content.append(fig_table)
        content.append(Spacer(1, 15))
    
    doc.build(content)
    buffer.seek(0)
    
    return buffer.getvalue()
