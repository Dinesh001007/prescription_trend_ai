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
    generate_pdf_executive_summary,
)

def convert_df_to_csv(df):
    """Convert dataframe to CSV for download."""
    if df is None:
        return None
    return df.to_csv(index=False).encode('utf-8')

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
    page_title="Prescription Trend AI - Clinical Intelligence Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Dinesh001007/prescription_trend_ai',
        'Report a bug': "https://github.com/Dinesh001007/prescription_trend_ai/issues",
        'About': "# Prescription Trend AI\nAn advanced multi-agent clinical intelligence platform for autonomous medical data discovery."
    }
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Hide Streamlit default header */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { 
    background: transparent !important; 
}
header[data-testid="stHeader"] > div:not(:first-child) {
    display: none;
}

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
        "Select Mode",
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
            "Drug Query",
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
            if extracted_text and not extracted_text.startswith("["):
                st.success("✓ Text extracted from PDF.")
                with st.expander("📝 Extracted Text", expanded=False):
                    st.text_area("Extracted Text", extracted_text, height=220, label_visibility="collapsed")
            elif extracted_text and extracted_text.startswith("[Error]"):
                st.error(extracted_text)
            elif extracted_text and extracted_text.startswith("[Warning]"):
                st.warning(extracted_text)
                st.info("💡 If this is a scanned PDF, please ensure Tesseract-OCR is installed on your system.")
            else:
                st.warning("No text was extracted from the PDF. The file may be a scanned image PDF.")
        else:
            st.markdown("### Uploaded Image Report")
            if extracted_text and not extracted_text.startswith("["):
                st.success("✓ Text extracted from image.")
                with st.expander("📝 OCR Extracted Text", expanded=False):
                    st.text_area("OCR Text", extracted_text, height=220, label_visibility="collapsed")
            elif extracted_text and extracted_text.startswith("[Error]"):
                st.error(extracted_text)
            else:
                st.info("No text was extracted from the image. Please ensure the image contains clear text and Tesseract-OCR is installed.")

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
                Upload Medical Dataset to Begin
            </div>
            <div style="color:#5A6070; font-size:14px; margin-top:8px;">
                Supports any unknown medical structured data (CSV, JSON, Excel).<br>
                The system will dynamically infer schema and medical meaning.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Step 1: Advanced Column Understanding ──
        st.markdown('<div class="section-header">🧠 STEP 1: Advanced Column Understanding</div>', unsafe_allow_html=True)
        
        from utils.medical_pipeline import MedicalDataPipeline
        
        if "pipeline" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
            with st.spinner("🧠 Intelligently mapping medical schema and determining field semantics..."):
                st.session_state.pipeline = MedicalDataPipeline(df)
                st.session_state.mapping_table = st.session_state.pipeline.step1_column_understanding()
                st.session_state.last_file = uploaded_file.name
                st.session_state.analysis_done = False

        st.dataframe(st.session_state.mapping_table, use_container_width=True)

        # ── Step 2: Data Preprocessing ──
        with st.expander("⚙️ STEP 2: Data Preprocessing (Automatic)", expanded=False):
            if "preprocessing_log" not in st.session_state:
                with st.spinner("⚙️ Executing automated data quality checks and feature engineering..."):
                    res = st.session_state.pipeline.step2_preprocessing()
                    st.session_state.preprocessing_log = res["preprocessing_log"]
            
            for log in st.session_state.preprocessing_log:
                st.write(f"• {log}")

        # ── Step 3: Agent Execution ──
        st.markdown('<div class="section-header">🚀 STEP 3: Agent Execution</div>', unsafe_allow_html=True)

        if not st.session_state.analysis_done:
            if st.button("▶ Run Full Multi-Agent Analysis", use_container_width=True):
                progress = st.progress(0, text="Initializing Medical Agents...")
                
                # Configure agents
                agents_config = {
                    "risk": run_risk,
                    "cohort": run_cohort,
                    "anomaly": run_anomaly,
                    "trend": run_trend,
                    "pattern": run_pattern
                }
                
                # Execute Pipeline
                with st.spinner("Executing pipeline agents..."):
                    results = st.session_state.pipeline.step3_agent_execution(agents_config)
                    st.session_state.analysis_results = results
                
                # STEP 4: Evaluating results
                with st.spinner("Step 4: Evaluating results..."):
                    st.session_state.eval_metrics = st.session_state.pipeline.step4_evaluation(st.session_state.analysis_results)
                
                # Step 5: Final Insights (LLM)
                with st.spinner("Phi-4 mini generating clinical report..."):
                    summaries = []
                    for key, res in results.items():
                        summaries.append(f"[{key.upper()} AGENT]: {res.get('summary', '')}")
                    
                    overall_summary = f"MEDICAL DATASET ANALYSIS\n{df.shape}\n" + "\n\n" + "\n".join(summaries)
                    llm_insights = generate_insights(overall_summary, {})
                    st.session_state.llm_insights = llm_insights
                    st.session_state.pdf_summary = generate_pdf_executive_summary(overall_summary)

                st.session_state.analysis_done = True
                st.rerun()

        # ── STEP 5: FINAL OUTPUT FORMAT ──
        if st.session_state.analysis_done:
            col_title, col_download = st.columns([3, 1])
            with col_title:
                st.markdown('<div class="section-header">📋 STEP 5: FINAL MEDICAL ANALYSIS REPORT</div>', unsafe_allow_html=True)
            with col_download:
                try:
                    pdf_data = generate_pdf_report(
                        st.session_state.df, 
                        st.session_state.col_map, 
                        st.session_state.analysis_results,
                        st.session_state.llm_insights,
                        dynamic_summary=st.session_state.get("pdf_summary")
                    )
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_data,
                        file_name=f"medical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as e:
                    st.error(f"PDF generation failed: {e}")

            tabs = st.tabs([
                "📊 Dataset Summary", 
                "🧬 Mapping & Quality", 
                "⚠️ Risk Analysis", 
                "👥 Cohort Analysis", 
                "🔍 Anomaly Detection", 
                "🧩 Pattern Analysis",
                "📈 Trend Analysis",
                "⭐ Final Insights"
            ])

            with tabs[0]:
                n_rows, n_cols = df.shape
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-label">Records</div>
                        <div class="metric-value">{n_rows:,}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Features</div>
                        <div class="metric-value">{n_cols}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Quality Score</div>
                        <div class="metric-value">{st.session_state.eval_metrics.get('Analysis Confidence Score', 0)*100:.0f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.dataframe(df.head(10), use_container_width=True)

            with tabs[1]:
                st.markdown("### Column Mapping Table")
                st.dataframe(st.session_state.mapping_table, use_container_width=True)
                st.markdown("### Evaluation Metrics")
                for k, v in st.session_state.eval_metrics.items():
                    if k != "statistical_validation":
                        st.write(f"**{k}:** {v}")

                # --- Statistical Validation ---
                st.markdown("---")
                st.markdown("### 🔬 Multi-Agent Statistical Validation")
                val_data = st.session_state.eval_metrics.get("statistical_validation")
                if val_data and val_data.get("validation_figure"):
                    st.plotly_chart(val_data["validation_figure"], use_container_width=True)
                    
                    with st.expander("📄 View Statistical Validation Report"):
                        summary = val_data["validation_results"].get("summary_report", "No summary available")
                        st.markdown(f"```\n{summary}\n```")
                        
                        table = val_data["validation_results"].get("validation_table")
                        if table is not None and not table.empty:
                            st.dataframe(table, use_container_width=True)
                else:
                    st.info("Insufficient data for multi-agent statistical comparison.")

            with tabs[2]:
                if "risk" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["risk"]
                    st.markdown(res.get("summary", "No summary"))
                    
                    if "risk_df" in res:
                        with st.expander("📄 View Processed Risk Data Preview"):
                            st.dataframe(res["risk_df"].head(100), use_container_width=True)
                            col_risk1, col_risk2 = st.columns(2)
                            with col_risk1:
                                st.download_button(
                                    label="📥 Download Full Risk Analysis CSV",
                                    data=convert_df_to_csv(res["risk_df"]),
                                    file_name="medical_risk_analysis_full.csv",
                                    mime="text/csv",
                                    key="download_risk_full"
                                )
                            with col_risk2:
                                high_risk_df = res["risk_df"][res["risk_df"]["risk_label"] == "High Risk"]
                                st.download_button(
                                    label="⚠️ Download High Risk Patients Only",
                                    data=convert_df_to_csv(high_risk_df),
                                    file_name="high_risk_patients.csv",
                                    mime="text/csv",
                                    key="download_risk_high"
                                )
                    
                    figs = res.get("figures", [])
                    for i in range(0, len(figs), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(figs):
                                title, fig = figs[i+j]
                                with cols[j]:
                                    st.plotly_chart(fig, use_container_width=True)

                    # --- Agent-Specific Statistical Validation ---
                    if "statistical_validation" in res:
                        with st.expander("🔬 View Agent-Specific Statistical Validation"):
                            val = res["statistical_validation"]
                            st.markdown(val.get("validation_summary", ""))
                            table = val.get("validation_table")
                            if table is not None and not table.empty:
                                st.dataframe(table, use_container_width=True)

            with tabs[3]:
                if "cohort" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["cohort"]
                    st.markdown(res.get("summary", "No summary"))
                    
                    if "cohort_df" in res:
                        with st.expander("📄 View Patient Cohort Data Preview"):
                            st.dataframe(res["cohort_df"].head(100), use_container_width=True)
                            st.download_button(
                                label="📥 Download Full Patient Cohort Data (CSV)",
                                data=convert_df_to_csv(res["cohort_df"]),
                                file_name="patient_cohort_analysis.csv",
                                mime="text/csv",
                                key="download_cohort"
                            )
                        
                    figs = res.get("figures", [])
                    for i in range(0, len(figs), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(figs):
                                title, fig = figs[i+j]
                                with cols[j]:
                                    st.plotly_chart(fig, use_container_width=True)

                    # --- Agent-Specific Statistical Validation ---
                    if "statistical_validation" in res:
                        with st.expander("🔬 View Agent-Specific Statistical Validation"):
                            val = res["statistical_validation"]
                            st.markdown(val.get("validation_summary", ""))
                            table = val.get("validation_table")
                            if table is not None and not table.empty:
                                st.dataframe(table, use_container_width=True)

            with tabs[4]:
                if "anomaly" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["anomaly"]
                    st.markdown(res.get("summary", "No summary"))
                    
                    if "anomaly_df" in res:
                        with st.expander("📄 View Anomaly Detection Data Preview"):
                            st.dataframe(res["anomaly_df"].head(100), use_container_width=True)
                            col_anom1, col_anom2 = st.columns(2)
                            with col_anom1:
                                st.download_button(
                                    label="📥 Download Full Anomaly Data CSV",
                                    data=convert_df_to_csv(res["anomaly_df"]),
                                    file_name="prescription_anomalies_full.csv",
                                    mime="text/csv",
                                    key="download_anomaly_full"
                                )
                            with col_anom2:
                                anomalies_only_df = res["anomaly_df"][res["anomaly_df"]["anomaly_label"] == "Anomaly"]
                                st.download_button(
                                    label="🔍 Download Detected Anomalies Only",
                                    data=convert_df_to_csv(anomalies_only_df),
                                    file_name="detected_anomalies.csv",
                                    mime="text/csv",
                                    key="download_anomaly_only"
                                )
                        
                    figs = res.get("figures", [])
                    for i in range(0, len(figs), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(figs):
                                title, fig = figs[i+j]
                                with cols[j]:
                                    st.plotly_chart(fig, use_container_width=True)

                    # --- Agent-Specific Statistical Validation ---
                    if "statistical_validation" in res:
                        with st.expander("🔬 View Agent-Specific Statistical Validation"):
                            val = res["statistical_validation"]
                            st.markdown(val.get("validation_summary", ""))
                            table = val.get("validation_table")
                            if table is not None and not table.empty:
                                st.dataframe(table, use_container_width=True)

            with tabs[5]:
                if "pattern" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["pattern"]
                    st.markdown(res.get("summary", "No summary"))
                    
                    if "pattern_df" in res:
                        with st.expander("📄 View Pattern Mining Data Preview"):
                            st.dataframe(res["pattern_df"].head(100), use_container_width=True)
                            st.download_button(
                                label="📥 Download Full Pattern Mining Data (CSV)",
                                data=convert_df_to_csv(res["pattern_df"]),
                                file_name="prescription_patterns.csv",
                                mime="text/csv",
                                key="download_pattern"
                            )
                        
                    figs = res.get("figures", [])
                    for i in range(0, len(figs), 2):
                        cols = st.columns(2)
                        for j in range(2):
                            if i + j < len(figs):
                                title, fig = figs[i+j]
                                with cols[j]:
                                    st.plotly_chart(fig, use_container_width=True)

                    # --- Agent-Specific Statistical Validation ---
                    if "statistical_validation" in res:
                        with st.expander("🔬 View Agent-Specific Statistical Validation"):
                            val = res["statistical_validation"]
                            st.markdown(val.get("validation_summary", ""))
                            table = val.get("validation_table")
                            if table is not None and not table.empty:
                                st.dataframe(table, use_container_width=True)

            with tabs[6]:
                if "trend" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["trend"]
                    if res.get("status") == "ok":
                        st.markdown(res.get("summary", "No summary"))
                        
                        if "trend_df" in res:
                            with st.expander("📄 View Trend & Forecast Data Preview"):
                                st.dataframe(res["trend_df"].head(100), use_container_width=True)
                                st.download_button(
                                    label="📥 Download Full Trend & Forecast Data (CSV)",
                                    data=convert_df_to_csv(res["trend_df"]),
                                    file_name="prescription_trends.csv",
                                    mime="text/csv",
                                    key="download_trend"
                                )
                            
                        figs = res.get("figures", [])
                        for i in range(0, len(figs), 2):
                            cols = st.columns(2)
                            for j in range(2):
                                if i + j < len(figs):
                                    title, fig = figs[i+j]
                                    with cols[j]:
                                        st.plotly_chart(fig, use_container_width=True)

                        # --- Agent-Specific Statistical Validation ---
                        if "statistical_validation" in res:
                            with st.expander("🔬 View Agent-Specific Statistical Validation"):
                                val = res["statistical_validation"]
                                st.markdown(val.get("validation_summary", ""))
                                table = val.get("validation_table")
                                if table is not None and not table.empty:
                                    st.dataframe(table, use_container_width=True)
                    else:
                        st.info("Temporal data not detected or trend analysis skipped.")

            with tabs[7]:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown(st.session_state.llm_insights)
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown(f"### Analysis Confidence Score: **{st.session_state.eval_metrics.get('Analysis Confidence Score', 0)*100:.0f}%**")

            # ── Reset Analysis ──
            if st.button("🔄 Reset Analysis"):
                st.session_state.analysis_done = False
                st.session_state.analysis_results = {}
                st.rerun()

            # ── Chat for dataset queries ──
            st.markdown('<div class="section-header">💬 Ask About Your Dataset</div>', unsafe_allow_html=True)

            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if prompt := st.chat_input("Ask a question about your data...", key="dataset_chat"):
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing..."):
                        context = f"Dataset context: {st.session_state.mapping_table.to_string()}\nInsights: {st.session_state.llm_insights}\nQuestion: {prompt}"
                        response = explain_analysis(context)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "content": response})