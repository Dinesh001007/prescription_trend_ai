import os
import json
import uuid
import pandas as pd
from flask import Flask, request, jsonify, render_template
import plotly.utils
from werkzeug.utils import secure_filename

from utils.llm import (
    query_llm,
    get_drug_info,
    identify_columns,
    generate_insights,
    explain_image_report,
    analyze_image_report
)
from utils.data_loader import load_file, get_sample_rows, build_summary
from utils.image_utils import extract_text_from_file, is_pdf_file
from agents.risk_agent import run_risk_agent_v2
from agents.cohort_agent import run_cohort_agent
from agents.anomaly_agent import run_anomaly_agent_v2
from agents.trend_agent import run_trend_agent
from agents.pattern_agent import run_pattern_agent

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ─── MAIN UI ROUTES ───
@app.route('/')
def index():
    return render_template('index.html')


# ─── API ROUTES ───

# 1. Drug Lookup
@app.route('/api/drug_lookup', methods=['POST'])
def api_drug_lookup():
    data = request.json
    query = data.get('query', '')
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    info = get_drug_info(query)
    # Parse the info into sections
    sections = []
    if "ERROR_OLLAMA_DOWN" in info:
        return jsonify({'error': 'Ollama is down. Please run `ollama serve`.'}), 503
        
    for section in info.split("##"):
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        header = lines[0].strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections.append({'header': header, 'body': body})
        
    return jsonify({'sections': sections})

@app.route('/api/drug_chat', methods=['POST'])
def api_drug_chat():
    data = request.json
    prompt = data.get('prompt', '')
    if any(w in prompt.lower() for w in ["drug", "medicine", "medication", "tablet", "pill"]):
        response = get_drug_info(prompt)
    else:
        response = query_llm(prompt, system="You are a clinical pharmacology expert. Answer concisely and accurately using markdown.")
    
    if "ERROR_OLLAMA_DOWN" in response:
        return jsonify({'error': 'Ollama down', 'response': '⚠️ Ollama is not running.'}), 503
    return jsonify({'response': response})

# 2. Image Report
def detect_imaging_modality(filename: str, text: str) -> str | None:
    combined = f"{filename} {text}".lower()
    if any(k in combined for k in ["x-ray", "xray", "chest x-ray", "chest xray"]):
        return "X-ray"
    if any(k in combined for k in ["ct", "computed tomography", "ct scan"]):
        return "CT"
    if any(k in combined for k in ["mri", "magnetic resonance"]):
        return "MRI"
    return None

@app.route('/api/scan_report', methods=['POST'])
def api_scan_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{uuid.uuid4()}_{filename}")
    file.save(filepath)
    
    try:
        extracted_text = extract_text_from_file(filepath)
        modality = detect_imaging_modality(filename, extracted_text or "")
        
        report_analysis = analyze_image_report(filename, extracted_text or "", modality=modality)
        
        if "ERROR_OLLAMA_DOWN" in report_analysis:
             return jsonify({'error': 'Ollama is not running. Please start it with `ollama serve`.'}), 503
             
        os.remove(filepath)
        return jsonify({
            'extracted_text': extracted_text,
            'modality': modality,
            'analysis': report_analysis
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/scan_chat', methods=['POST'])
def api_scan_chat():
    data = request.json
    prompt = data.get('prompt', '')
    analysis = data.get('analysis', '')
    context = f"Scan report analysis:\n{analysis}\n\nFollow-up question: {prompt}"
    response = explain_image_report(context)
    return jsonify({'response': response})


# 3. Dataset Analysis

# In-memory storage for prototype
DATASETS = {}

@app.route('/api/upload_dataset', methods=['POST'])
def api_upload_dataset():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    dataset_id = str(uuid.uuid4())
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{dataset_id}_{filename}")
    file.save(filepath)

    try:
        # Re-using the load_file utility which originally took a streamlit UploadedFile
        # but in Flask we can load directly if we modify load_file to accept path.
        # Actually load_file in utils/data_loader expects file-like object. 
        with open(filepath, 'rb') as f:
            df = load_file(f, filename)
            
        n_rows, n_cols = df.shape
        n_missing = int(df.isna().sum().sum())
        
        sample = get_sample_rows(df, 3)
        col_map = identify_columns(df.columns.tolist(), sample)
        
        DATASETS[dataset_id] = df
        
        os.remove(filepath)
        return jsonify({
            'dataset_id': dataset_id,
            'columns': df.columns.tolist(),
            'rows': n_rows,
            'cols': n_cols,
            'missing': n_missing,
            'col_map': col_map
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze_dataset', methods=['POST'])
def api_analyze_dataset():
    data = request.json
    dataset_id = data.get('dataset_id')
    col_map = data.get('col_map', {})
    selections = data.get('selections', {})
    
    if dataset_id not in DATASETS:
        return jsonify({'error': 'Dataset expired or not found. Please re-upload.'}), 404
        
    df = DATASETS[dataset_id]
    
    results = {}
    summaries = []
    
    try:
        if selections.get('pattern', True):
            res = run_pattern_agent(df, col_map)
            results['pattern'] = handle_figs(res)
            summaries.append(f"[PATTERN AGENT]: {res.get('summary', '')}")
            
        if selections.get('risk', True):
            res = run_risk_agent_v2(df, col_map)
            results['risk'] = handle_figs(res)
            # Remove high-volume data before sending risk_df
            if 'risk_df' in results['risk']:
                 r_df = results['risk']['risk_df']
                 high_risk = len(r_df[r_df["__risk_label"] == "High Risk"])
                 results['risk']['high_risk_count'] = high_risk
                 results['risk']['total_count'] = len(r_df)
                 del results['risk']['risk_df']
            summaries.append(f"[RISK AGENT]: {res.get('summary', '')}")
            
        if selections.get('cohort', True):
            res = run_cohort_agent(df, col_map)
            results['cohort'] = handle_figs(res)
            if 'cohort_df' in results['cohort']:
                 del results['cohort']['cohort_df']
            summaries.append(f"[COHORT AGENT]: {res.get('summary', '')}")
            
        if selections.get('anomaly', True):
            res = run_anomaly_agent_v2(df, col_map)
            results['anomaly'] = handle_figs(res)
            if 'anomaly_df' in results['anomaly']:
                 a_df = results['anomaly']['anomaly_df']
                 anom_count = int((a_df["__anomaly"] == "Anomaly").sum())
                 results['anomaly']['anom_count'] = anom_count
                 del results['anomaly']['anomaly_df']
            summaries.append(f"[ANOMALY AGENT]: {res.get('summary', '')}")
            
        if selections.get('trend', True):
            res = run_trend_agent(df, col_map)
            results['trend'] = handle_figs(res)
            summaries.append(f"[TREND AGENT]: {res.get('summary', '')}")
            
        overall_summary = build_summary(df, col_map) + "\n\n" + "\n".join(summaries)
        llm_insights = generate_insights(overall_summary, col_map)
        
        return jsonify({
            'results': results,
            'llm_insights': llm_insights
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def handle_figs(res):
    """Serialize plotly figures to JSON for the frontend"""
    out = {k: v for k, v in res.items() if k != 'figures'}
    if 'figures' in res and res['figures']:
        json_figs = []
        for title, fig in res['figures']:
            json_figs.append({'title': title, 'plotly_json': fig.to_json()})
        out['figures'] = json_figs
    return out


if __name__ == '__main__':
    app.run(debug=True, port=5000)
