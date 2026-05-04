import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.llm import (
    query_llm,
    extract_intent,
    get_drug_info,
    identify_columns,
    generate_insights,
    explain_analysis,
    analyze_image_report,
    explain_image_report,
)
from utils.data_loader import (
    load_file,
    get_sample_rows,
    build_summary,
)
from agents.risk_agent_improved import run_risk_agent_improved as run_risk_agent
from agents.cohort_agent_advanced import run_cohort_agent_advanced as run_cohort_agent
from agents.anomaly_agent_improved import run_anomaly_agent_improved as run_anomaly_agent
from agents.trend_agent import run_trend_agent
from agents.pattern_agent import run_pattern_agent
from utils.image_utils import (
    is_image_file,
    is_pdf_file,
    load_image_from_bytes,
    extract_text_from_file,
)
from utils.pdf_generator import (
    generate_pdf_report,
    create_visualizations_pdf,
)
from utils.schema_analyzer import SchemaAnalyzer, ColumnType
from utils.intelligent_analyzer import IntelligentAnalyzer

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prescription Trend AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit default header */
#MainMenu, footer, header { visibility: hidden; }

/* App background */
.stApp {
    background: #0D0F14;
}

/* ── Top Logo Bar ── */
.top-bar {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 22px 0 10px 0;
    margin-bottom: 6px;
}
.logo-icon {
    width: 46px; height: 46px;
    background: linear-gradient(135deg, #00C9A7 0%, #007AFF 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 24px rgba(0,201,167,0.35);
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    background: linear-gradient(90deg, #00C9A7, #007AFF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-sub {
    font-size: 12px;
    color: #5A6070;
    font-weight: 400;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #161A22 !important;
    border-right: 1px solid #1E2330;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #5A6070;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 20px;
}

/* ── Metric Cards ── */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.metric-card {
    background: #161A22;
    border: 1px solid #1E2330;
    border-radius: 14px;
    padding: 18px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #00C9A7; }
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00C9A7, #007AFF);
}
.metric-label {
    font-size: 11px;
    color: #5A6070;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 500;
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #E8EAF0;
    line-height: 1.1;
    margin-top: 4px;
}
.metric-delta {
    font-size: 12px;
    color: #00C9A7;
    margin-top: 4px;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 16px;
    font-weight: 700;
    color: #E8EAF0;
    letter-spacing: -0.3px;
    margin: 28px 0 12px 0;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1E2330, transparent);
    margin-left: 8px;
}

/* ── Alert / Risk Boxes ── */
.alert-box {
    background: rgba(255, 107, 107, 0.08);
    border: 1px solid rgba(255, 107, 107, 0.3);
    border-left: 3px solid #FF6B6B;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 10px 0;
}
.alert-box-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 700;
    color: #FF6B6B;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.insight-box {
    background: rgba(0, 201, 167, 0.06);
    border: 1px solid rgba(0, 201, 167, 0.2);
    border-left: 3px solid #00C9A7;
    border-radius: 10px;
    padding: 18px 20px;
    margin: 12px 0;
    line-height: 1.7;
}

/* ── Chat Input ── */
.stChatInput > div {
    background: #161A22 !important;
    border: 1px solid #1E2330 !important;
    border-radius: 14px !important;
}
.stChatInput input {
    color: #E8EAF0 !important;
}

/* ── Chat Messages ── */
.stChatMessage {
    background: #161A22 !important;
    border: 1px solid #1E2330 !important;
    border-radius: 14px !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: #161A22;
    border: 1px dashed #2A3040;
    border-radius: 14px;
    padding: 10px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #161A22;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #5A6070;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #0D0F14 !important;
    color: #00C9A7 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #00C9A7, #007AFF);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 10px 24px;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #161A22 !important;
    border: 1px solid #1E2330 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Pill tag ── */
.pill {
    display: inline-block;
    background: rgba(0, 201, 167, 0.1);
    border: 1px solid rgba(0, 201, 167, 0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    color: #00C9A7;
    margin: 2px;
    font-weight: 500;
}
.pill-red {
    background: rgba(255, 107, 107, 0.1);
    border-color: rgba(255, 107, 107, 0.3);
    color: #FF6B6B;
}
</style>
""", unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="logo-icon">💊</div>
    <div>
        <div class="app-title">Prescription Trend AI</div>
        <div class="app-sub">Clinical Intelligence Platform · Powered by Phi-4 mini</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Input Mode")
    mode = st.radio(
        "",
        ["💊 Drug Lookup", "📂 Dataset Analysis", "🩺 Image Report"],
        label_visibility="collapsed",
    )

    st.markdown("### 📤 Upload File")
    uploaded_file = st.file_uploader(
        "Upload dataset or scan report (supports CSV, JSON, Excel, and image scan reports)",
        type=["csv", "json", "xlsx", "xls", "png", "jpg", "jpeg", "bmp", "tiff", "tif"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.success(f"✓ {uploaded_file.name}")

    st.markdown("### ⚙️ Analysis Options")
    run_risk = st.checkbox("Risk Analysis (XGBoost)", value=True)
    run_cohort = st.checkbox("Cohort Clustering (KMeans)", value=True)
    run_anomaly = st.checkbox("Anomaly Detection (IsoForest)", value=True)
    run_trend = st.checkbox("Trend Forecasting (Holt-Winters)", value=True)
    run_pattern = st.checkbox("Co-Prescription Patterns", value=True)

    st.markdown("### 🤖 Model")
    st.markdown('<span class="pill">Phi-4 mini · Ollama</span>', unsafe_allow_html=True)
    if st.button("Test Ollama Connection"):
        resp = query_llm("Say 'OK' only.")
        if "ERROR_OLLAMA_DOWN" in resp:
            st.error("Ollama not running. Run: `ollama serve`")
        else:
            st.success("Ollama connected ✓")

    st.markdown("---")
    st.markdown('<div style="font-size:11px;color:#3A4050;">Prescription Trend AI v1.0<br>Lab-Scale TRL 4 Prototype</div>', unsafe_allow_html=True)


# ─── Session State ────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "df" not in st.session_state:
    st.session_state.df = None
if "col_map" not in st.session_state:
    st.session_state.col_map = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False


# ─── Drug Lookup Mode ─────────────────────────────────────────────────────────
if mode == "💊 Drug Lookup":
    st.markdown('<div class="section-header">Drug Intelligence Lookup</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        drug_query = st.text_input(
            "",
            placeholder="Enter a drug name  (e.g. Metformin, Atorvastatin, Amoxicillin...)",
            label_visibility="collapsed",
        )
    with col2:
        search_btn = st.button("🔍 Analyze Drug", use_container_width=True)

    if search_btn and drug_query.strip():
        with st.spinner("Consulting clinical knowledge base via Phi-4 mini..."):
            info = get_drug_info(drug_query.strip())

        if "ERROR_OLLAMA_DOWN" in info:
            st.error("⚠️ Cannot reach Ollama. Please run `ollama serve` in your terminal.")
        else:
            st.markdown(f'<div class="section-header">📋 {drug_query.strip().title()} — Clinical Profile</div>', unsafe_allow_html=True)

            # Parse sections and display styled
            sections = info.split("##")
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                lines = section.split("\n", 1)
                header = lines[0].strip()
                body = lines[1].strip() if len(lines) > 1 else ""

                icon_map = {
                    "Overview": "🔬",
                    "Mechanism": "🔬",
                    "Indication": "🏥",
                    "Alternative": "🔄",
                    "Substitute": "🔄",
                    "Interaction": "⚠️",
                    "Do NOT": "🚫",
                    "Warning": "⚠️",
                    "Precaution": "⚠️",
                }
                icon = next((v for k, v in icon_map.items() if k.lower() in header.lower()), "📌")
                is_warning = any(w in header.lower() for w in ["interaction", "do not", "warning", "precaution"])
                box_class = "alert-box" if is_warning else "insight-box"
                title_class = "alert-box-title" if is_warning else ""

                with st.expander(f"{icon} {header}", expanded=not is_warning):
                    st.markdown(body)

    elif search_btn:
        st.warning("Please enter a drug name.")

    # Chat interface for follow-up questions
    st.markdown('<div class="section-header">💬 Ask Follow-up Questions</div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask anything about a drug, interaction, dosage..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_drug_info(prompt) if any(
                    w in prompt.lower() for w in ["drug", "medicine", "medication", "tablet", "pill"]
                ) else query_llm(
                    prompt,
                    system="You are a clinical pharmacology expert. Answer concisely and accurately using markdown.",
                )
                if "ERROR_OLLAMA_DOWN" in response:
                    response = "⚠️ Ollama is not running. Please start it with `ollama serve`."
                st.markdown(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})


# ─── Image Report Mode ───────────────────────────────────────────────────────
elif mode == "🩺 Image Report":
    st.markdown('<div class="section-header">🩺 Scan Report Analysis</div>', unsafe_allow_html=True)

    supported_types = ["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif"]
    report_file = uploaded_file if uploaded_file and uploaded_file.name.lower().endswith(tuple(supported_types)) else None
    if report_file is None:
        st.info("Upload a PDF or image scan report. Use filenames or report text containing xray, ct, or mri for better modality detection.")
        report_file = st.file_uploader(
            "Upload scan report file",
            type=supported_types,
            label_visibility="collapsed",
        )
        if report_file:
            st.success(f"✓ {report_file.name}")

    def detect_imaging_modality(filename: str, text: str) -> str | None:
        combined = f"{filename} {text}".lower()
        if any(k in combined for k in ["x-ray", "xray", "chest x-ray", "chest xray"]):
            return "X-ray"
        if any(k in combined for k in ["ct", "computed tomography", "ct scan"]):
            return "CT"
        if any(k in combined for k in ["mri", "magnetic resonance"]):
            return "MRI"
        return None

    if report_file:
        if st.session_state.get("last_image") != report_file.name:
            st.session_state.report_analysis = None
            st.session_state.image_ocr = None
            st.session_state["last_image"] = report_file.name

        with st.spinner("Extracting text from report..."):
            extracted_text = extract_text_from_file(report_file)
            st.session_state.image_ocr = extracted_text

        if is_pdf_file(report_file.name):
            st.markdown("### Uploaded PDF report")
            if extracted_text:
                st.markdown("Text extracted from PDF.")
                with st.expander("📝 Extracted Text", expanded=False):
                    st.text_area("", extracted_text, height=220)
            else:
                st.warning("No text was extracted from the PDF. The file may be a scanned image PDF.")
        else:
            st.markdown("### Uploaded Image Report")
            if extracted_text:
                with st.expander("📝 OCR Extracted Text", expanded=False):
                    st.text_area("", extracted_text, height=220)
            else:
                st.info("No text was extracted from the image. The current workflow can only interpret text-based reports.")

        modality = detect_imaging_modality(report_file.name, extracted_text or "")
        if modality:
            st.markdown(f"**Detected modality:** {modality}")

        if st.button("🔍 Analyze Scan Report", use_container_width=True):
            with st.spinner("Analyzing report with Phi-4 mini..."):
                report_analysis = analyze_image_report(
                    report_file.name,
                    extracted_text or "",
                    modality=modality,
                )
                st.session_state.report_analysis = report_analysis

        if st.session_state.get("report_analysis"):
            st.markdown('<div class="section-header">📋 Report Findings</div>', unsafe_allow_html=True)
            if "ERROR_OLLAMA_DOWN" in st.session_state.report_analysis:
                st.error("Ollama is not running. Please start it with `ollama serve`.")
            else:
                st.markdown(st.session_state.report_analysis)

            st.markdown('<div class="section-header">💬 Ask Follow-up Questions</div>', unsafe_allow_html=True)
            if prompt := st.chat_input("Ask a question about this scan report..."):
                if not hasattr(st.session_state, "image_chat_history"):
                    st.session_state.image_chat_history = []
                st.session_state.image_chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Formulating answer..."):
                        context = (
                            f"Scan report analysis:\n{st.session_state.report_analysis}\n\n"
                            f"Follow-up question: {prompt}"
                        )
                        response = explain_image_report(context)
                        if "ERROR_OLLAMA_DOWN" in response:
                            response = "⚠️ Ollama is not running. Please start it with `ollama serve`."
                        st.markdown(response)
                        st.session_state.image_chat_history.append({"role": "assistant", "content": response})

            if hasattr(st.session_state, "image_chat_history"):
                for msg in st.session_state.image_chat_history:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

# ─── Dataset Analysis Mode ───────────────────────────────────────────────────
else:
    st.markdown('<div class="section-header">📂 Dataset Analysis</div>', unsafe_allow_html=True)

    # Load file
    if uploaded_file:
        if uploaded_file.name.lower().endswith((".csv", ".json", ".xlsx", ".xls")):
            if st.session_state.df is None or st.session_state.get("last_file") != uploaded_file.name:
                try:
                    df = load_file(uploaded_file)
                    st.session_state.df = df
                    st.session_state.col_map = None
                    st.session_state.analysis_done = False
                    st.session_state["last_file"] = uploaded_file.name
                except Exception as e:
                    st.error(f"Failed to load file: {e}")
        else:
            st.warning("Please upload a dataset file (CSV, JSON, or Excel) for Dataset Analysis.")

    df = st.session_state.df

    if df is None:
        st.markdown("""
        <div style="text-align:center; padding: 60px 20px; background: #161A22;
             border: 1px dashed #2A3040; border-radius: 16px; margin-top: 20px;">
            <div style="font-size:48px; margin-bottom:12px;">📂</div>
            <div style="font-family:'Syne',sans-serif; font-size:18px; font-weight:700; color:#E8EAF0;">
                Upload a Dataset to Begin
            </div>
            <div style="color:#5A6070; font-size:14px; margin-top:8px;">
                Supports CSV, JSON, and Excel files up to 15MB.<br>
                Columns are auto-detected — no schema required.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Dataset Overview metrics ──
        n_rows, n_cols = df.shape
        n_drugs = 0
        drug_col_guess = next((c for c in df.columns if "drug" in c.lower() or "med" in c.lower() or "name" in c.lower()), None)
        if drug_col_guess:
            n_drugs = df[drug_col_guess].nunique()
        n_missing = df.isna().sum().sum()

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Records</div>
                <div class="metric-value">{n_rows:,}</div>
                <div class="metric-delta">Rows loaded</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Columns</div>
                <div class="metric-value">{n_cols}</div>
                <div class="metric-delta">Features detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Unique Drugs</div>
                <div class="metric-value">{n_drugs if n_drugs else "—"}</div>
                <div class="metric-delta">{"From drug column" if n_drugs else "Drug col not found"}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Missing Values</div>
                <div class="metric-value">{n_missing:,}</div>
                <div class="metric-delta">{"⚠️ Needs attention" if n_missing > 0 else "✓ Clean"}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Column Identification ──
        st.markdown('<div class="section-header">🧠 Auto Column Identification (LLM)</div>', unsafe_allow_html=True)

        if st.session_state.col_map is None:
            with st.spinner("Phi-4 mini is identifying your column roles..."):
                sample = get_sample_rows(df, 3)
                col_map = identify_columns(df.columns.tolist(), sample)
                st.session_state.col_map = col_map

        col_map = st.session_state.col_map

        # Intelligent data validation and analysis
        with st.spinner("Performing intelligent data validation..."):
            intelligent_analyzer = IntelligentAnalyzer()
            schema_analyzer = SchemaAnalyzer()
            
            # Analyze data quality and types
            validation_results = intelligent_analyzer.analyze_dataframe_intelligently(df)
            
            # Display data quality insights
            if validation_results['summary_insights']:
                st.markdown('<div class="section-header">🔍 Data Quality Insights</div>', unsafe_allow_html=True)
                for insight in validation_results['summary_insights']:
                    st.info(f"• {insight}")
            
            # Show validation errors if any
            validation_errors = intelligent_analyzer.get_validation_errors()
            if validation_errors:
                st.warning("⚠️ Data Validation Warnings:")
                for error in validation_errors:
                    st.write(f"• {error}")

        # Display column mapping as pills
        cols_display = st.columns(min(4, len(df.columns)))
        color_map = {
            "drug_name": "pill",
            "patient_id": "pill",
            "date": "pill",
            "risk_score": "pill-red",
            "other": "",
        }
        with st.expander("📋 Column Role Mapping", expanded=True):
            html_pills = ""
            for col, cat in col_map.items():
                cls = "pill-red" if cat in ["risk_score"] else "pill"
                html_pills += f'<span class="{cls}">{col} → {cat}</span> '
            st.markdown(f'<div style="line-height:2.2">{html_pills}</div>', unsafe_allow_html=True)
            
            # Show intelligent type detection results
            st.markdown("**🧠 Intelligent Type Detection:**")
            type_counts = validation_results['schema_overview']['type_distribution']
            for col_name, col_type in validation_results['schema_overview']['column_types'].items():
                if col_name in col_map:
                    st.write(f"• {col_name}: `{col_type}` (LLM: `{col_map[col_name]}`)")

        # ── Override column roles (outside expander to avoid nesting) ──
        with st.expander("✏️ Override Column Roles"):
            categories = ["drug_name", "patient_id", "date", "diagnosis", "age", "gender",
                          "dosage", "frequency", "region", "risk_score", "quantity", "prescriber", "other"]
            updated_map = {}
            override_cols = st.columns(3)
            for i, col in enumerate(df.columns):
                with override_cols[i % 3]:
                    col_cat = col_map.get(col, "other")
                    if col_cat not in categories:
                        col_cat = "other"
                    updated_map[col] = st.selectbox(
                        col,
                        categories,
                        index=categories.index(col_cat),
                        key=f"col_{col}",
                    )
            if st.button("Apply Override"):
                st.session_state.col_map = updated_map
                st.session_state.analysis_done = False
                st.rerun()

        # ── Data Preview ──
        with st.expander("🔍 Preview Data (first 50 rows)"):
            st.dataframe(df.head(50), use_container_width=True)

        # ── Run Analysis ──
        st.markdown('<div class="section-header">🚀 Agent Analysis</div>', unsafe_allow_html=True)

        if not st.session_state.analysis_done:
            if st.button("▶ Run Full Analysis", use_container_width=True):
                st.session_state.analysis_results = {}

                progress = st.progress(0, text="Starting agents...")

                agents_to_run = []
                if run_pattern:
                    agents_to_run.append(("pattern", "Co-Prescription Patterns", "agents.pattern_agent"))
                if run_risk:
                    agents_to_run.append(("risk", "Risk Analysis", "agents.risk_agent"))
                if run_cohort:
                    agents_to_run.append(("cohort", "Cohort Clustering", "agents.cohort_agent"))
                if run_anomaly:
                    agents_to_run.append(("anomaly", "Anomaly Detection", "agents.anomaly_agent"))
                if run_trend:
                    agents_to_run.append(("trend", "Trend Forecasting", "agents.trend_agent"))

                for i, (key, label, _) in enumerate(agents_to_run):
                    progress.progress((i) / len(agents_to_run), text=f"Running {label}...")
                    try:
                        if key == "risk":
                            st.session_state.analysis_results["risk"] = run_risk_agent(df, col_map)
                        elif key == "cohort":
                            st.session_state.analysis_results["cohort"] = run_cohort_agent(df, col_map)
                        elif key == "anomaly":
                            st.session_state.analysis_results["anomaly"] = run_anomaly_agent(df, col_map)
                        elif key == "trend":
                            st.session_state.analysis_results["trend"] = run_trend_agent(df, col_map)
                        elif key == "pattern":
                            st.session_state.analysis_results["pattern"] = run_pattern_agent(df, col_map)
                    except Exception as e:
                        st.session_state.analysis_results[key] = {"status": "error", "summary": str(e), "figures": []}

                progress.progress(1.0, text="Generating LLM insights...")

                # Build overall summary for LLM
                summaries = []
                for key, res in st.session_state.analysis_results.items():
                    summaries.append(f"[{key.upper()} AGENT]: {res.get('summary', '')}")
                overall_summary = build_summary(df, col_map) + "\n\n" + "\n".join(summaries)

                with st.spinner("Phi-4 mini generating clinical insights..."):
                    llm_insights = generate_insights(overall_summary, col_map)
                    st.session_state.llm_insights = llm_insights

                st.session_state.analysis_done = True
                progress.empty()
                st.rerun()

        # ── Show Results ──
        if st.session_state.analysis_done and hasattr(st.session_state, "analysis_results"):
            results = st.session_state.analysis_results

            # Model Accuracy Summary
            st.markdown('<div class="section-header">📊 Model Performance Summary</div>', unsafe_allow_html=True)
            
            # Collect all accuracy metrics
            all_accuracy_metrics = []
            for agent_key, res in results.items():
                if res.get("status") == "ok" and res.get("metrics"):
                    metrics = res.get("metrics", {})
                    accuracy_data = {
                        "agent": agent_key.title(),
                        "model": metrics.get("Model", "Unknown"),
                        "accuracy": None,
                        "precision": None,
                        "recall": None,
                        "silhouette": None,
                        "rmse": None,
                        "mae": None,
                        "confidence": None,
                        "execution": metrics.get("Execution", "N/A")
                    }
                    
                    for metric_name in ["accuracy", "precision", "recall", "silhouette", "rmse", "mae", "confidence"]:
                        if metric_name.capitalize() in metrics:
                            accuracy_data[metric_name] = metrics[metric_name.capitalize()]
                    
                    all_accuracy_metrics.append(accuracy_data)
            
            if all_accuracy_metrics:
                # Create accuracy summary table
                accuracy_df = pd.DataFrame(all_accuracy_metrics)
                
                # Display as styled table
                st.markdown("""
                <style>
                .accuracy-table {
                    background: #161A22;
                    border: 1px solid #1E2330;
                    border-radius: 14px;
                    padding: 20px;
                    margin-bottom: 20px;
                }
                .accuracy-table table {
                    width: 100%;
                    border-collapse: collapse;
                    color: #E8EAF0;
                }
                .accuracy-table th {
                    background: #0D0F14;
                    padding: 12px;
                    text-align: left;
                    font-weight: 600;
                    color: #00C9A7;
                    border-bottom: 2px solid #00C9A7;
                }
                .accuracy-table td {
                    padding: 10px 12px;
                    border-bottom: 1px solid #1E2330;
                }
                .accuracy-high { color: #00C9A7; font-weight: 600; }
                .accuracy-medium { color: #FFC300; font-weight: 500; }
                .accuracy-low { color: #FF6B6B; font-weight: 500; }
                </style>
                """, unsafe_allow_html=True)
                
                st.markdown('<div class="accuracy-table">', unsafe_allow_html=True)
                
                # Create HTML table
                table_html = """
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Agent</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>Silhouette</th>
                            <th>RMSE</th>
                            <th>MAE</th>
                            <th>Confidence</th>
                            <th>Execution</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for row in accuracy_df.itertuples():
                    # Determine accuracy class
                    acc_class = "accuracy-high"
                    if row.accuracy:
                        acc_val = float(row.accuracy.rstrip('%')) if '%' in row.accuracy else float(row.accuracy)
                        if acc_val >= 90:
                            acc_class = "accuracy-high"
                        elif acc_val >= 70:
                            acc_class = "accuracy-medium"
                        else:
                            acc_class = "accuracy-low"
                    
                    table_html += f"""
                    <tr>
                        <td><strong>{row.model}</strong></td>
                        <td>{row.agent}</td>
                        <td class="{acc_class}">{row.accuracy or '—'}</td>
                        <td>{row.precision or '—'}</td>
                        <td>{row.recall or '—'}</td>
                        <td>{row.silhouette or '—'}</td>
                        <td>{row.rmse or '—'}</td>
                        <td>{row.mae or '—'}</td>
                        <td>{row.confidence or '—'}</td>
                        <td>{row.execution}</td>
                    </tr>
                    """
                
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # LLM Insights
            st.markdown('<div class="section-header">🤖 Phi-4 mini Clinical Insights</div>', unsafe_allow_html=True)
            if hasattr(st.session_state, "llm_insights"):
                if "ERROR_OLLAMA_DOWN" in st.session_state.llm_insights:
                    st.error("Ollama not running. Start with `ollama serve`.")
                else:
                    st.markdown(
                        f'<div class="insight-box">{st.session_state.llm_insights}</div>',
                        unsafe_allow_html=True,
                    )

            # Risk Alerts
            if "risk" in results and results["risk"].get("risk_df") is not None:
                risk_df = results["risk"]["risk_df"]
                high_risk = risk_df[risk_df["__risk_label"] == "High Risk"]
                if len(high_risk) > 0:
                    st.markdown(f"""
                    <div class="alert-box">
                        <div class="alert-box-title">⚠️ Risk Alert System</div>
                        <div style="color:#E8EAF0; margin-top:6px; font-size:14px;">
                            <strong>{len(high_risk)}</strong> high-risk prescriptions detected
                            ({100*len(high_risk)/len(risk_df):.1f}% of dataset).
                            Review flagged records immediately.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Anomaly Alerts
            if "anomaly" in results and results["anomaly"].get("anomaly_df") is not None:
                anom_df = results["anomaly"]["anomaly_df"]
                n_anom = (anom_df["__anomaly"] == "Anomaly").sum()
                if n_anom > 0:
                    st.markdown(f"""
                    <div class="alert-box">
                        <div class="alert-box-title">🔴 Anomaly Alert</div>
                        <div style="color:#E8EAF0; margin-top:6px; font-size:14px;">
                            <strong>{n_anom}</strong> anomalous prescriptions identified by Isolation Forest.
                            These may indicate unusual dosing, rare combinations, or data errors.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Visualization Tabs ──
            tab_names = []
            if "pattern" in results:
                tab_names.append("📊 Patterns")
            if "risk" in results:
                tab_names.append("🔴 Risk")
            if "cohort" in results:
                tab_names.append("👥 Cohorts")
            if "anomaly" in results:
                tab_names.append("🚨 Anomalies")
            if "trend" in results:
                tab_names.append("📈 Trends")
            tab_names.append("🗂 Data Table")

            if tab_names:
                tabs = st.tabs(tab_names)
                tab_idx = 0

                def show_agent_tab(tab, agent_key, label):
                    with tab:
                        if agent_key in results:
                            res = results[agent_key]
                            if res["status"] == "error":
                                st.error(f"Agent error: {res['summary']}")
                            elif res["status"] in ["insufficient_columns", "no_drug_col", "no_date"]:
                                st.info(res["summary"])
                            else:
                                # Metrics Row with Accuracy Highlight
                                metrics = res.get("metrics", {})
                                if metrics:
                                    # Separate accuracy-related metrics from others
                                    accuracy_metrics = {}
                                    other_metrics = {}
                                    
                                    for m_label, m_value in metrics.items():
                                        if m_label.lower() in ["accuracy", "precision", "recall", "silhouette", "rmse", "mae", "confidence"]:
                                            accuracy_metrics[m_label] = m_value
                                        else:
                                            other_metrics[m_label] = m_value
                                    
                                    # Display accuracy metrics prominently first
                                    if accuracy_metrics:
                                        st.markdown('<div style="margin-bottom: 16px;"><h4 style="color: #00C9A7; font-size: 14px; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 1px;">🎯 Model Performance Metrics</h4></div>', unsafe_allow_html=True)
                                        accuracy_html = '<div class="metric-row" style="grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-bottom: 12px;">'
                                        for m_label, m_value in accuracy_metrics.items():
                                            # Add special styling for accuracy
                                            if m_label.lower() == "accuracy":
                                                accuracy_html += f'<div class="metric-card" style="border: 2px solid #00C9A7; box-shadow: 0 0 16px rgba(0,201,167,0.3);"><div class="metric-label" style="color: #00C9A7;">⭐ {m_label}</div><div class="metric-value" style="font-size: 20px; color: #00C9A7; font-weight: 800;">{m_value}</div></div>'
                                            elif m_label.lower() in ["precision", "recall"]:
                                                accuracy_html += f'<div class="metric-card" style="border: 1px solid #FFC300;"><div class="metric-label" style="color: #FFC300;">📊 {m_label}</div><div class="metric-value" style="font-size: 18px; color: #FFC300;">{m_value}</div></div>'
                                            elif m_label.lower() == "silhouette":
                                                accuracy_html += f'<div class="metric-card" style="border: 1px solid #007AFF;"><div class="metric-label" style="color: #007AFF;">🔷 {m_label}</div><div class="metric-value" style="font-size: 18px; color: #007AFF;">{m_value}</div></div>'
                                            elif m_label.lower() in ["rmse", "mae"]:
                                                accuracy_html += f'<div class="metric-card" style="border: 1px solid #FF6B6B;"><div class="metric-label" style="color: #FF6B6B;">📉 {m_label}</div><div class="metric-value" style="font-size: 18px; color: #FF6B6B;">{m_value}</div></div>'
                                            else:
                                                accuracy_html += f'<div class="metric-card"><div class="metric-label">{m_label}</div><div class="metric-value" style="font-size: 18px;">{m_value}</div></div>'
                                        accuracy_html += "</div>"
                                        st.markdown(accuracy_html, unsafe_allow_html=True)
                                    
                                    # Display other metrics
                                    if other_metrics:
                                        st.markdown('<div style="margin-bottom: 16px;"><h4 style="color: #5A6070; font-size: 14px; margin: 0 0 8px 0; text-transform: uppercase; letter-spacing: 1px;">⚙️ Model Details</h4></div>', unsafe_allow_html=True)
                                        other_html = '<div class="metric-row" style="grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));">'
                                        for m_label, m_value in other_metrics.items():
                                            other_html += f'<div class="metric-card"><div class="metric-label">{m_label}</div><div class="metric-value" style="font-size: 16px;">{m_value}</div></div>'
                                        other_html += "</div>"
                                        st.markdown(other_html, unsafe_allow_html=True)


                                # Summary
                                if res.get("summary"):
                                    st.markdown(f'<div class="insight-box" style="font-size:13px">{res["summary"]}</div>', unsafe_allow_html=True)

                                # Charts
                                figs = res.get("figures", [])
                                if figs:
                                    for i in range(0, len(figs), 2):
                                        chart_cols = st.columns(2 if i + 1 < len(figs) else 1)
                                        for j, (title, fig) in enumerate(figs[i:i+2]):
                                            with chart_cols[j]:
                                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


                for tname in tab_names:
                    if tname == "📊 Patterns":
                        show_agent_tab(tabs[tab_idx], "pattern", "Patterns")
                    elif tname == "🔴 Risk":
                        show_agent_tab(tabs[tab_idx], "risk", "Risk")
                    elif tname == "👥 Cohorts":
                        show_agent_tab(tabs[tab_idx], "cohort", "Cohorts")
                    elif tname == "🚨 Anomalies":
                        show_agent_tab(tabs[tab_idx], "anomaly", "Anomalies")
                    elif tname == "📈 Trends":
                        show_agent_tab(tabs[tab_idx], "trend", "Trends")
                    elif tname == "🗂 Data Table":
                        with tabs[tab_idx]:
                            st.markdown("### Full Dataset")
                            # Add agent columns to display
                            display_df = df.copy()
                            if "risk" in results and results["risk"].get("risk_df") is not None:
                                display_df["Risk Score"] = results["risk"]["risk_df"]["__risk_score"].round(3)
                                display_df["Risk Label"] = results["risk"]["risk_df"]["__risk_label"]
                            if "cohort" in results and results["cohort"].get("cohort_df") is not None:
                                display_df["Cohort"] = results["cohort"]["cohort_df"]["__cohort"]
                            if "anomaly" in results and results["anomaly"].get("anomaly_df") is not None:
                                display_df["Anomaly"] = results["anomaly"]["anomaly_df"]["__anomaly"]
                            st.dataframe(display_df, use_container_width=True)
                            csv_export = display_df.to_csv(index=False).encode()
                            st.download_button(
                                "⬇ Download Analyzed Dataset",
                                csv_export,
                                "prescription_analysis.csv",
                                "text/csv",
                            )
                    tab_idx += 1

            # PDF Download Section
            st.markdown("---")
            st.markdown('<div class="section-header">📄 Download Analysis Report</div>', unsafe_allow_html=True)
            
            # Initialize session state for download buttons
            if 'generate_full_report' not in st.session_state:
                st.session_state.generate_full_report = False
            if 'generate_visualizations' not in st.session_state:
                st.session_state.generate_visualizations = False
            if 'generate_complete_package' not in st.session_state:
                st.session_state.generate_complete_package = False
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Download Full Report", use_container_width=True):
                    st.session_state.generate_full_report = True
                    st.session_state.generate_visualizations = False
                    st.session_state.generate_complete_package = False
                    st.rerun()
                
                if st.session_state.generate_full_report:
                    with st.spinner("Generating comprehensive PDF report..."):
                        try:
                            llm_insights = getattr(st.session_state, 'llm_insights', None)
                            pdf_bytes = generate_pdf_report(df, col_map, results, llm_insights)
                            st.download_button(
                                label="⬇ Full Analysis Report (PDF)",
                                data=pdf_bytes,
                                file_name=f"prescription_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.session_state.generate_full_report = False
                        except Exception as e:
                            st.error(f"Error generating PDF report: {str(e)}")
                            st.session_state.generate_full_report = False
            
            with col2:
                if st.button("📈 Download Visualizations", use_container_width=True):
                    st.session_state.generate_full_report = False
                    st.session_state.generate_visualizations = True
                    st.session_state.generate_complete_package = False
                    st.rerun()
                
                if st.session_state.generate_visualizations:
                    with st.spinner("Generating visualizations PDF..."):
                        try:
                            viz_pdf_bytes = create_visualizations_pdf(results)
                            st.download_button(
                                label="⬇ Visualizations Only (PDF)",
                                data=viz_pdf_bytes,
                                file_name=f"prescription_visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                            st.session_state.generate_visualizations = False
                        except Exception as e:
                            st.error(f"Error generating visualizations PDF: {str(e)}")
                            st.session_state.generate_visualizations = False
            
            with col3:
                # Generate combined report (both text and visualizations)
                if st.button("📋 Download Complete Package", use_container_width=True):
                    st.session_state.generate_full_report = False
                    st.session_state.generate_visualizations = False
                    st.session_state.generate_complete_package = True
                    st.rerun()
                
                if st.session_state.generate_complete_package:
                    with st.spinner("Generating complete analysis package..."):
                        try:
                            from reportlab.lib.utils import ImageReader
                            import zipfile
                            import io
                            
                            # Create zip file
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                # Add main report
                                llm_insights = getattr(st.session_state, 'llm_insights', None)
                                main_pdf = generate_pdf_report(df, col_map, results, llm_insights)
                                zip_file.writestr(f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", main_pdf)
                                
                                # Add visualizations
                                viz_pdf = create_visualizations_pdf(results)
                                zip_file.writestr(f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", viz_pdf)
                                
                                # Add CSV data
                                csv_data = df.to_csv(index=False)
                                zip_file.writestr(f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", csv_data)
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                label="⬇ Complete Package (ZIP)",
                                data=zip_buffer.getvalue(),
                                file_name=f"prescription_complete_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                            st.session_state.generate_complete_package = False
                        except Exception as e:
                            st.error(f"Error generating complete package: {str(e)}")
                            st.session_state.generate_complete_package = False

            # Reset button
            st.markdown("---")
            if st.button("🔄 Reset Analysis"):
                st.session_state.analysis_done = False
                st.session_state.col_map = None
                if hasattr(st.session_state, "analysis_results"):
                    del st.session_state.analysis_results
                if hasattr(st.session_state, "llm_insights"):
                    del st.session_state.llm_insights
                st.rerun()

        # ── Chat for dataset queries ──
        st.markdown('<div class="section-header">💬 Ask About Your Dataset</div>', unsafe_allow_html=True)

        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask a question about your data or analysis..."):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    context = (
                        f"Dataset summary:\n{build_summary(df, col_map)}\n\n"
                        f"User question: {prompt}"
                    )
                    response = explain_analysis(context)
                    if "ERROR_OLLAMA_DOWN" in response:
                        response = "⚠️ Ollama is not running. Please start it with `ollama serve`."
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})