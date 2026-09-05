import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils.db import (
    init_db,
    register_user,
    authenticate_user,
    get_user_by_id,
    create_analysis_session,
    get_user_analysis_sessions,
    get_analysis_session,
    update_analysis_session_data,
    delete_analysis_session,
    add_analysis_message,
    get_analysis_messages,
)
from utils.llm_core import (
    query_llm,
    get_drug_info,
    identify_columns,
    generate_insights,
    explain_analysis,
    analyze_image_report,
    detect_scan_modality,
    explain_image_report,
    generate_pdf_executive_summary,
    is_ollama_running,
    ensure_ollama_running,
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
from utils.media_utils import (
    is_image_file,
    is_pdf_file,
    load_image_from_bytes,
    extract_text_from_file,
)
from utils.media_utils import (
    generate_pdf_report,
)
from utils.data_profiling import SchemaAnalyzer, ColumnType
from utils.core_pipeline import IntelligentAnalyzer
from utils.core_pipeline import MedicalDataPipeline
from utils.data_profiling import DatasetProfiler
from utils.data_profiling import SemanticMapper
from utils.core_pipeline import CapabilityMatrix
from utils.core_pipeline import AgentOrchestrator
from utils.llm_core import AIReasoner
from tools.tool_registry import ToolRegistry

# Initialize database tables on startup
init_db()

# Auto-start Ollama server in background if not already running
@st.cache_resource(show_spinner=False)
def init_ollama_service():
    """Ensure Ollama local server is running in the background."""
    ensure_ollama_running(timeout=4.0)
    return True

init_ollama_service()

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prescription Trend AI - Clinical Intelligence Platform",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Dinesh001007/prescription_trend_ai',
        'Report a bug': "https://github.com/Dinesh001007/prescription_trend_ai/issues",
        'About': "# Prescription Trend AI\nAn advanced multi-agent clinical intelligence platform for autonomous medical data discovery & conversational reasoning."
    }
)

# ─── Custom Premium CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', 'DM Sans', sans-serif;
}

/* Hide Streamlit default header & footer */
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent !important; }
header[data-testid="stHeader"] > div:not(:first-child) { display: none; }

/* Base App Background */
.stApp {
    background: #090C10;
    color: #F0F3F8;
}

/* ── Top Header Navigation Bar ── */
.top-nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 18px;
    background: linear-gradient(180deg, #131822 0%, #0D111A 100%);
    border: 1px solid #1C2333;
    border-radius: 16px;
    margin-bottom: 20px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}
.brand-box {
    display: flex;
    align-items: center;
    gap: 14px;
}
.brand-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, #00E5BE 0%, #0A84FF 100%);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
    box-shadow: 0 0 20px rgba(0, 229, 190, 0.35);
}
.brand-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    background: linear-gradient(90deg, #FFFFFF 0%, #CBD5E1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.brand-subtitle {
    font-size: 11px;
    color: #00E5BE;
    letter-spacing: 1px;
    text-transform: uppercase;
    font-weight: 600;
}

/* ── User Profile Badge ── */
.user-profile-badge {
    display: flex;
    align-items: center;
    gap: 10px;
    background: #151B26;
    border: 1px solid #222C3E;
    border-radius: 30px;
    padding: 5px 14px;
}
.user-avatar {
    width: 30px; height: 30px;
    background: linear-gradient(135deg, #0A84FF, #00E5BE);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700;
    font-size: 11px;
    color: white;
}
.user-info-name {
    font-size: 12px;
    font-weight: 700;
    color: #F0F3F8;
}
.user-info-role {
    font-size: 10px;
    color: #00E5BE;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Mode Selection Navigation Tabs ── */
.nav-card-container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.nav-tab-card {
    background: #111622;
    border: 1px solid #1C2436;
    border-radius: 14px;
    padding: 14px 18px;
    cursor: pointer;
    transition: all 0.25s ease;
    display: flex;
    align-items: center;
    gap: 12px;
}
.nav-tab-card.active {
    background: linear-gradient(145deg, rgba(0, 229, 190, 0.12) 0%, rgba(10, 132, 255, 0.08) 100%);
    border: 1.5px solid #00E5BE;
    box-shadow: 0 0 16px rgba(0, 229, 190, 0.2);
}

/* ── Content Action Cards ── */
.action-card {
    background: #101520;
    border: 1px solid #1A2234;
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
    position: relative;
    overflow: hidden;
}
.action-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    border-bottom: 1px solid #1B2336;
    padding-bottom: 10px;
}
.action-card-title {
    font-size: 16px;
    font-weight: 700;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Stepper Guide ── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #00E5BE;
    color: #090C10;
    font-weight: 800;
    font-size: 11px;
    border-radius: 50%;
    width: 22px; height: 22px;
}
.step-pill-done {
    background: rgba(0, 229, 190, 0.15);
    border: 1px solid rgba(0, 229, 190, 0.4);
    color: #00E5BE;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
}

/* ── Metrics Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 16px 0;
}
.metric-box {
    background: #141A26;
    border: 1px solid #1F283C;
    border-radius: 12px;
    padding: 14px 16px;
    position: relative;
    overflow: hidden;
}
.metric-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #00E5BE, #0A84FF);
}
.metric-label {
    font-size: 11px;
    color: #7A8699;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 700;
}
.metric-val {
    font-size: 24px;
    font-weight: 800;
    color: #FFFFFF;
    margin-top: 4px;
}

/* ── Insight Box ── */
.insight-container {
    background: #121824;
    border: 1px solid #1C263A;
    border-left: 4px solid #00E5BE;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 14px 0;
    color: #E2E8F0;
    line-height: 1.6;
}

/* ── Chips / Pills ── */
.chip {
    display: inline-block;
    background: rgba(0, 229, 190, 0.08);
    border: 1px solid rgba(0, 229, 190, 0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #00E5BE;
    font-weight: 600;
}
.chip-blue {
    background: rgba(10, 132, 255, 0.08);
    border: 1px solid rgba(10, 132, 255, 0.25);
    color: #0A84FF;
}

/* ── Sidebar Session Card ── */
.session-card {
    background: #121722;
    border: 1px solid #1C2333;
    border-radius: 10px;
    padding: 10px 12px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}
.session-card.active {
    border: 1.5px solid #00E5BE;
    background: rgba(0, 229, 190, 0.06);
}

/* ── Auth Cards ── */
.auth-wrapper {
    max-width: 480px;
    margin: 30px auto;
    background: #111622;
    border: 1px solid #1C2538;
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ────────────────────────────────────────────
if "user" not in st.session_state:
    st.session_state.user = None
if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = "📂 Dataset Analysis"
if "active_dataset_session_id" not in st.session_state:
    st.session_state.active_dataset_session_id = None
if "active_image_session_id" not in st.session_state:
    st.session_state.active_image_session_id = None
if "df" not in st.session_state:
    st.session_state.df = None
if "col_map" not in st.session_state:
    st.session_state.col_map = None
if "mapping_table" not in st.session_state:
    st.session_state.mapping_table = None
if "preprocessing_log" not in st.session_state:
    st.session_state.preprocessing_log = []
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = {}
if "eval_metrics" not in st.session_state:
    st.session_state.eval_metrics = {}
if "llm_insights" not in st.session_state:
    st.session_state.llm_insights = ""
if "pdf_summary" not in st.session_state:
    st.session_state.pdf_summary = ""
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "scan_extracted_text" not in st.session_state:
    st.session_state.scan_extracted_text = ""
if "scan_report_analysis" not in st.session_state:
    st.session_state.scan_report_analysis = ""
if "scan_filename" not in st.session_state:
    st.session_state.scan_filename = ""


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 1. AUTHENTICATION & SIGNUP PAGE ──────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.user is None:
    st.markdown("""
    <div style="text-align: center; margin: 40px auto 10px auto;">
        <div style="width: 58px; height: 58px; background: linear-gradient(135deg, #00E5BE, #0A84FF);
                    border-radius: 16px; display: inline-flex; align-items: center; justify-content: center;
                    font-size: 30px; box-shadow: 0 0 28px rgba(0, 229, 190, 0.4); margin-bottom: 12px;">💊</div>
        <div style="font-size: 26px; font-weight: 800; color: #FFFFFF; letter-spacing: -0.5px;">Prescription Trend AI</div>
        <div style="color: #718096; font-size: 13px; margin-top: 4px;">Clinical Intelligence & Multi-Agent Analytics Platform</div>
    </div>
    """, unsafe_allow_html=True)

    col_a1, col_a2, col_a3 = st.columns([1, 2, 1])
    with col_a2:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Sign In", "✨ Create Account"])

        # ── SIGN IN FORM ──
        with auth_tab1:
            st.markdown("##### Secure Practitioner Sign In")
            login_username = st.text_input("Username or Email", key="login_user", placeholder="Enter username or email")
            login_password = st.text_input("Password", type="password", key="login_pwd", placeholder="••••••••")

            if st.button("Sign In to Workspace →", use_container_width=True, key="btn_signin"):
                if not login_username or not login_password:
                    st.error("Please enter both username/email and password.")
                else:
                    success, msg, user_data = authenticate_user(login_username, login_password)
                    if success:
                        st.session_state.user = user_data
                        st.success("Signed in successfully!")
                        st.rerun()
                    else:
                        st.error(msg)

        # ── SIGN UP FORM ──
        with auth_tab2:
            st.markdown("##### Register Practitioner Account")
            new_fullname = st.text_input("Full Name", placeholder="e.g. Dr. Sarah Jenkins, MD", key="reg_fullname")
            new_username = st.text_input("Username", placeholder="e.g. sjenkins", key="reg_user")
            new_email = st.text_input("Email Address", placeholder="e.g. sarah.jenkins@hospital.org", key="reg_email")
            new_role = st.selectbox(
                "Clinical Role", 
                ["Clinician / Physician", "Medical Data Scientist", "Radiologist", "Clinical Researcher", "Healthcare Specialist"],
                key="reg_role"
            )
            reg_pwd1 = st.text_input("Password (min 6 characters)", type="password", key="reg_pwd1", placeholder="••••••••")
            reg_pwd2 = st.text_input("Confirm Password", type="password", key="reg_pwd2", placeholder="••••••••")

            if st.button("Create Account & Access Platform →", use_container_width=True, key="btn_signup"):
                if not new_username or not new_email or not reg_pwd1:
                    st.error("Please fill in all required fields.")
                elif reg_pwd1 != reg_pwd2:
                    st.error("Passwords do not match. Please re-enter.")
                elif len(reg_pwd1) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    success, msg, user_data = register_user(
                        username=new_username,
                        email=new_email,
                        password=reg_pwd1,
                        full_name=new_fullname,
                        role=new_role
                    )
                    if success:
                        st.session_state.user = user_data
                        st.success(f"Account created successfully! Welcome, {user_data['full_name']}.")
                        st.rerun()
                    else:
                        st.error(msg)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 2. AUTHENTICATED TOP HEADER ──────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
current_user = st.session_state.user

col_top_left, col_top_right = st.columns([3, 1])
with col_top_left:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-icon">💊</div>
        <div>
            <div class="brand-title">Prescription Trend AI</div>
            <div class="brand-subtitle">Clinical Intelligence & Multi-Agent Analytics Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_top_right:
    c1, c2 = st.columns([2, 1])
    with c1:
        initials = "".join([part[0] for part in current_user.get("full_name", "DR").split()[:2]]).upper() or "MD"
        st.markdown(f"""
        <div class="user-profile-badge">
            <div class="user-avatar">{initials}</div>
            <div>
                <div class="user-info-name">{current_user.get('full_name', current_user['username'])}</div>
                <div class="user-info-role">{current_user.get('role', 'Clinician')}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        if st.button("🚪 Logout", key="btn_logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.active_dataset_session_id = None
            st.session_state.active_image_session_id = None
            st.session_state.df = None
            st.rerun()


# ─── Helper: Session State Loaders ───────────────────────────────────────────
def render_agent_figures(agent_name: str, title: str = None):
    """Render all Plotly figures produced by a specific agent, if available."""
    result = st.session_state.analysis_results.get(agent_name, {})
    figures = result.get("figures") or []
    if not figures:
        return

    st.markdown(f"##### {title or agent_name.title()} Visuals")
    for i in range(0, len(figures), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(figures):
                fig_title, fig = figures[i + j]
                with cols[j]:
                    st.markdown(f"**{fig_title}**")
                    st.plotly_chart(fig, use_container_width=True)


def load_dataset_session_state(session_id: str, user_id: int):
    session = get_analysis_session(session_id, user_id)
    if not session:
        return
    st.session_state.active_dataset_session_id = session_id
    try:
        data = json.loads(session.get("data_json") or "{}")
    except Exception:
        data = {}
    
    st.session_state.last_file = session.get("filename", "")
    st.session_state.analysis_done = data.get("analysis_done", False)
    st.session_state.llm_insights = data.get("llm_insights", "")
    st.session_state.pdf_summary = data.get("pdf_summary", "")
    st.session_state.eval_metrics = data.get("eval_metrics", {})
    st.session_state.preprocessing_log = data.get("preprocessing_log", [])
    
    if "mapping_table" in data and data["mapping_table"] is not None:
        st.session_state.mapping_table = pd.DataFrame(data["mapping_table"])
    else:
        st.session_state.mapping_table = None
        
    if "dataset_records" in data and data["dataset_records"] is not None:
        st.session_state.df = pd.DataFrame(data["dataset_records"])
    else:
        st.session_state.df = None
        
    st.session_state.analysis_results = data.get("analysis_results", {})


def load_image_session_state(session_id: str, user_id: int):
    session = get_analysis_session(session_id, user_id)
    if not session:
        return
    st.session_state.active_image_session_id = session_id
    try:
        data = json.loads(session.get("data_json") or "{}")
    except Exception:
        data = {}
    st.session_state.scan_filename = session.get("filename", "")
    st.session_state.scan_extracted_text = data.get("extracted_text", "")
    st.session_state.scan_report_analysis = data.get("analysis", "")


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 3. PROMINENT TOP WORKSPACE NAVIGATOR ─────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("##### 🧭 Select Workspace Module")
nav_col1, nav_col2, nav_col3 = st.columns(3)

with nav_col1:
    is_ds_active = st.session_state.selected_mode == "📂 Dataset Analysis"
    btn_label = "📊 Dataset Intelligence (Active ✓)" if is_ds_active else "📊 Dataset Intelligence"
    if st.button(btn_label, use_container_width=True, key="nav_btn_ds", type="primary" if is_ds_active else "secondary"):
        st.session_state.selected_mode = "📂 Dataset Analysis"
        st.rerun()

with nav_col2:
    is_img_active = st.session_state.selected_mode == "🩺 Image & Scan Report"
    btn_label = "🩻 Radiology Reports (Active ✓)" if is_img_active else "🩻 Radiology Reports"
    if st.button(btn_label, use_container_width=True, key="nav_btn_img", type="primary" if is_img_active else "secondary"):
        st.session_state.selected_mode = "🩺 Image & Scan Report"
        st.rerun()

with nav_col3:
    is_drug_active = st.session_state.selected_mode == "💊 Drug Intelligence"
    btn_label = "💊 Pharmacology Database (Active ✓)" if is_drug_active else "💊 Pharmacology Database"
    if st.button(btn_label, use_container_width=True, key="nav_btn_drug", type="primary" if is_drug_active else "secondary"):
        st.session_state.selected_mode = "💊 Drug Intelligence"
        st.rerun()

app_mode = st.session_state.selected_mode
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 4. SIDEBAR: CLEAN SESSIONS ARCHIVE & LLM HEALTH ──────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🗂️ Analysis Archive")
    
    if app_mode == "📂 Dataset Analysis":
        if st.button("➕  New Dataset Analysis", use_container_width=True, key="btn_new_ds_session"):
            new_s = create_analysis_session(current_user["id"], session_type="dataset_analysis", title="New Dataset Analysis")
            st.session_state.active_dataset_session_id = new_s["id"]
            st.session_state.df = None
            st.session_state.mapping_table = None
            st.session_state.preprocessing_log = []
            st.session_state.analysis_results = {}
            st.session_state.eval_metrics = {}
            st.session_state.llm_insights = ""
            st.session_state.pdf_summary = ""
            st.session_state.analysis_done = False
            st.session_state.last_file = ""
            st.rerun()

        st.markdown("##### 📂 Saved Dataset Analyses")
        ds_sessions = get_user_analysis_sessions(current_user["id"], "dataset_analysis")

        if not ds_sessions:
            new_s = create_analysis_session(current_user["id"], session_type="dataset_analysis", title="Prescription Analysis Session")
            st.session_state.active_dataset_session_id = new_s["id"]
            ds_sessions = [new_s]

        if not st.session_state.active_dataset_session_id and ds_sessions:
            st.session_state.active_dataset_session_id = ds_sessions[0]["id"]
            load_dataset_session_state(ds_sessions[0]["id"], current_user["id"])

        for s in ds_sessions:
            is_active = s["id"] == st.session_state.active_dataset_session_id
            col_s1, col_s2 = st.columns([5, 1])
            with col_s1:
                label = f"📁 {s['title']}" if not is_active else f"👉 **{s['title']}**"
                if st.button(label, key=f"sel_ds_{s['id']}", use_container_width=True):
                    load_dataset_session_state(s["id"], current_user["id"])
                    st.rerun()
            with col_s2:
                if st.button("🗑️", key=f"del_ds_{s['id']}", help="Delete analysis data safely for privacy"):
                    delete_analysis_session(s["id"], current_user["id"])
                    remaining = get_user_analysis_sessions(current_user["id"], "dataset_analysis")
                    if remaining:
                        load_dataset_session_state(remaining[0]["id"], current_user["id"])
                    else:
                        st.session_state.active_dataset_session_id = None
                        st.session_state.df = None
                        st.session_state.analysis_done = False
                    st.rerun()

    elif app_mode == "🩺 Image & Scan Report":
        if st.button("➕  New Scan Report", use_container_width=True, key="btn_new_img_session"):
            new_s = create_analysis_session(current_user["id"], session_type="image_report", title="New Scan Report")
            st.session_state.active_image_session_id = new_s["id"]
            st.session_state.scan_extracted_text = ""
            st.session_state.scan_report_analysis = ""
            st.session_state.scan_filename = ""
            st.rerun()

        st.markdown("##### 🩺 Saved Scan Reports")
        img_sessions = get_user_analysis_sessions(current_user["id"], "image_report")

        if not img_sessions:
            new_s = create_analysis_session(current_user["id"], session_type="image_report", title="Radiology Scan Report")
            st.session_state.active_image_session_id = new_s["id"]
            img_sessions = [new_s]

        if not st.session_state.active_image_session_id and img_sessions:
            st.session_state.active_image_session_id = img_sessions[0]["id"]
            load_image_session_state(img_sessions[0]["id"], current_user["id"])

        for s in img_sessions:
            is_active = s["id"] == st.session_state.active_image_session_id
            col_s1, col_s2 = st.columns([5, 1])
            with col_s1:
                label = f"🩻 {s['title']}" if not is_active else f"👉 **{s['title']}**"
                if st.button(label, key=f"sel_img_{s['id']}", use_container_width=True):
                    load_image_session_state(s["id"], current_user["id"])
                    st.rerun()
            with col_s2:
                if st.button("🗑️", key=f"del_img_{s['id']}", help="Delete scan data safely for privacy"):
                    delete_analysis_session(s["id"], current_user["id"])
                    remaining = get_user_analysis_sessions(current_user["id"], "image_report")
                    if remaining:
                        load_image_session_state(remaining[0]["id"], current_user["id"])
                    else:
                        st.session_state.active_image_session_id = None
                        st.session_state.scan_extracted_text = ""
                        st.session_state.scan_report_analysis = ""
                    st.rerun()

    else:
        st.info("Search and review pharmacology mechanism of action, interactions, and therapeutic alternatives.")

    # ── Model Status ──
    st.markdown("---")
    st.markdown("### 🤖 Clinical LLM")
    st.markdown('<span class="chip">Phi-4 mini · Reasoning Engine</span>', unsafe_allow_html=True)
    
    ollama_online = is_ollama_running()
    if ollama_online:
        st.markdown(
            '<div style="display:flex; align-items:center; gap:8px; margin:8px 0;">'
            '<span style="height:10px; width:10px; background-color:#00E5BE; border-radius:50%; display:inline-block;"></span>'
            '<span style="font-size:12px; color:#00E5BE; font-weight:700;">Ollama Live & Connected ✓</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="display:flex; align-items:center; gap:8px; margin:8px 0;">'
            '<span style="height:10px; width:10px; background-color:#F59E0B; border-radius:50%; display:inline-block;"></span>'
            '<span style="font-size:12px; color:#F59E0B; font-weight:700;">Ollama Server Offline</span>'
            '</div>',
            unsafe_allow_html=True
        )
        if st.button("⚡ Start Ollama Server", key="btn_start_ollama"):
            with st.spinner("Starting Ollama background server..."):
                ok, msg = ensure_ollama_running(timeout=5.0)
                if ok:
                    st.success("Ollama started successfully!")
                    st.rerun()
                else:
                    st.error(f"Could not start Ollama: {msg}")

    if st.button("Test LLM Inference", key="btn_test_ollama"):
        with st.spinner("Testing model inference..."):
            resp = query_llm("Respond with 'OK - Clinical LLM ready' only.")
            if "ERROR_OLLAMA_DOWN" in resp:
                st.warning("Ollama offline (`ollama serve`). Intelligent clinical fallbacks are active.")
            elif "ERROR:" in resp:
                st.error(f"Inference error: {resp}")
            else:
                st.success(f"Response: {resp}")

    st.markdown("---")
    st.markdown(f'<div style="font-size:11px; color:#64748B;">Prescription Trend AI v2.0<br>Logged in as {current_user["username"]}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 5. DATASET INTELLIGENCE WORKSPACE ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
if app_mode == "📂 Dataset Analysis":
    if not st.session_state.active_dataset_session_id:
        new_s = create_analysis_session(current_user["id"], "dataset_analysis", "Dataset Analysis Session")
        st.session_state.active_dataset_session_id = new_s["id"]

    active_session = get_analysis_session(st.session_state.active_dataset_session_id, current_user["id"])
    if not active_session:
        ds_list = get_user_analysis_sessions(current_user["id"], "dataset_analysis")
        if ds_list:
            st.session_state.active_dataset_session_id = ds_list[0]["id"]
            active_session = ds_list[0]
            load_dataset_session_state(active_session["id"], current_user["id"])

    # ── Session Action Toolbar ──
    col_bar1, col_bar2, col_bar3 = st.columns([5, 2, 1])
    with col_bar1:
        st.markdown(f"### 📂 {active_session['title']}")
    with col_bar2:
        with st.expander("✏️ Rename Analysis", expanded=False):
            new_title_val = st.text_input("Title", value=active_session["title"], key="rename_ds_input")
            if st.button("Save Title", key="btn_save_ds_title"):
                if new_title_val.strip():
                    update_analysis_session_data(active_session["id"], title=new_title_val.strip())
                    st.rerun()
    with col_bar3:
        if st.button("🗑️ Delete", key="btn_del_curr_ds", help="Permanently delete this dataset analysis for data privacy"):
            delete_analysis_session(active_session["id"], current_user["id"])
            remaining = get_user_analysis_sessions(current_user["id"], "dataset_analysis")
            if remaining:
                load_dataset_session_state(remaining[0]["id"], current_user["id"])
            else:
                st.session_state.active_dataset_session_id = None
                st.session_state.df = None
                st.session_state.analysis_done = False
            st.rerun()

    # ── STEP 1: Upload & Data Ingestion Card ──
    st.markdown("""
    <div class="action-card">
        <div class="action-card-header">
            <div class="action-card-title">
                <span class="step-badge">1</span>
                Dataset Ingestion & File Upload
            </div>
            <span class="chip">CSV · JSON · Excel XLSX</span>
        </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose a clinical prescription dataset file",
        type=["csv", "json", "xlsx", "xls"],
        key="main_ds_file_uploader",
        help="Upload CSV, JSON, or Excel dataset. The platform automatically maps medical schema and features."
    )

    if uploaded_file:
        if st.session_state.df is None or st.session_state.get("last_file") != uploaded_file.name:
            try:
                df = load_file(uploaded_file)
                st.session_state.df = df
                st.session_state.col_map = None
                st.session_state.mapping_table = None
                st.session_state.preprocessing_log = []
                st.session_state.analysis_results = {}
                st.session_state.eval_metrics = {}
                st.session_state.analysis_done = False
                st.session_state["last_file"] = uploaded_file.name
                
                # Profile dataset dynamically
                profiler = DatasetProfiler()
                prof = profiler.profile_dataframe(df, filename=uploaded_file.name)
                st.session_state.dataset_profile = prof
                
                # Layered semantic mapping
                mapper = SemanticMapper()
                mapping_res = mapper.map_columns(df, prof.get("columns", {}), use_llm=False)
                st.session_state.canonical_map = mapping_res["canonical_mapping"]
                st.session_state.mapping_details = mapping_res["mapping_details"]
                
                # Capability Matrix
                cap_eval = CapabilityMatrix()
                st.session_state.capabilities = cap_eval.evaluate_capabilities(st.session_state.canonical_map, prof)

                update_analysis_session_data(active_session["id"], title=f"Dataset: {uploaded_file.name}", filename=uploaded_file.name)
                st.success(f"✓ Ingested {uploaded_file.name} ({df.shape[0]:,} rows, {df.shape[1]} columns) — Data Quality Score: {prof['data_quality_score']}/100")
            except Exception as e:
                st.error(f"Failed to parse file: {e}")

    df = st.session_state.df
    
    # Render Profiling & Capability Banner if dataset is present
    if df is not None and "dataset_profile" in st.session_state:
        prof = st.session_state.dataset_profile
        caps = st.session_state.get("capabilities", {})
        
        st.markdown(f"""
        <div style="background:#0D111A; border:1px solid #1C263A; border-radius:12px; padding:14px; margin-top:10px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-weight:700; font-size:14px; color:#F0F3F8;">📋 Automated Dataset Health & Capability Matrix</span>
                <span class="chip-green">Quality Score: {prof.get('data_quality_score', 100)}/100</span>
            </div>
            <div style="font-size:12px; color:#94A3B8; margin-bottom:10px;">
                <strong>Dimensions:</strong> {prof.get('row_count', len(df)):,} rows · {prof.get('column_count', len(df.columns))} columns · Missing Cells: {prof.get('missing_rate_pct', 0.0)}% · Duplicate Rows: {prof.get('duplicate_row_count', 0)}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Capability badges
        cap_cols = st.columns(5)
        cap_dict = caps.get("capabilities", {})
        tool_names = ["trend", "cohort", "risk", "anomaly", "pattern"]
        tool_icons = {"trend": "📈 Trend", "cohort": "👥 Cohort", "risk": "⚠️ Risk", "anomaly": "🔍 Anomaly", "pattern": "🧩 Pattern"}
        
        for idx, t_name in enumerate(tool_names):
            with cap_cols[idx]:
                c_info = cap_dict.get(t_name, {})
                if c_info.get("feasible", False):
                    st.success(f"{tool_icons[t_name]}\n\n✓ Ready")
                else:
                    st.warning(f"{tool_icons[t_name]}\n\nUnavailable")

    st.markdown("</div>", unsafe_allow_html=True)

    if df is None:
        st.info("💡 **Ready to begin**: Upload a medical prescription dataset above to launch the autonomous multi-agent analysis pipeline.")
    else:
        # ── STEP 2: Agent Configuration & Pipeline Execution Card ──
        st.markdown("""
        <div class="action-card">
            <div class="action-card-header">
                <div class="action-card-title">
                    <span class="step-badge">2</span>
                    Dynamic AI-Agent Planner & Multi-Model Execution
                </div>
                <span class="chip-blue">Dynamic Tool Orchestrator</span>
            </div>
        """, unsafe_allow_html=True)

        user_analysis_query = st.text_input(
            "💬 Clinical Query / Intent Directive (Optional)",
            value="Perform comprehensive multi-model prescription trend, risk, and anomaly analysis.",
            help="The AI Supervisor dynamically evaluates capabilities, plans DAG stages, and runs tool competitions."
        )

        col_ag1, col_ag2, col_ag3 = st.columns(3)
        with col_ag1:
            run_risk = st.checkbox("⚠️ **Risk Tool** (XGBoost / Composite)", value=True)
            run_cohort = st.checkbox("👥 **Cohort Tool** (KMeans / DBSCAN / Agglom)", value=True)
        with col_ag2:
            run_anomaly = st.checkbox("🔍 **Anomaly Tool** (Isolation Forest / LOF / OCSVM)", value=True)
            run_trend = st.checkbox("📈 **Trend Tool** (Prophet / ETS / ARIMA)", value=True)
        with col_ag3:
            run_pattern = st.checkbox("🧩 **Pattern Tool** (Association & Polypharmacy)", value=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 Launch Dynamic AI Multi-Agent Pipeline", use_container_width=True, key="btn_run_pipeline", type="primary"):
            # Ensure profiling & capabilities are current
            if "dataset_profile" not in st.session_state:
                profiler = DatasetProfiler()
                st.session_state.dataset_profile = profiler.profile_dataframe(df)
            if "canonical_map" not in st.session_state:
                mapper = SemanticMapper()
                mapping_res = mapper.map_columns(df, st.session_state.dataset_profile.get("columns", {}), use_llm=False)
                st.session_state.canonical_map = mapping_res["canonical_mapping"]
                st.session_state.mapping_details = mapping_res["mapping_details"]
            if "capabilities" not in st.session_state:
                cap_eval = CapabilityMatrix()
                st.session_state.capabilities = cap_eval.evaluate_capabilities(st.session_state.canonical_map, st.session_state.dataset_profile)

            orchestrator = AgentOrchestrator()
            reasoner = AIReasoner()

            with st.spinner("🤖 AI Supervisor: Formulating DAG execution plan..."):
                plan = orchestrator.plan_execution(user_analysis_query, st.session_state.capabilities)
                st.session_state.execution_plan = plan

            with st.spinner("⚡ Running candidate model competitions concurrently (Holdout validation & metric rankings)..."):
                exec_result = orchestrator.execute_plan(df, st.session_state.canonical_map, plan)
                st.session_state.analysis_results = exec_result["tool_results"]
                st.session_state.total_duration_ms = exec_result["total_duration_ms"]

            with st.spinner("🔬 Evidence-Grounded AI Reasoning Agent: Synthesizing clinical intelligence report..."):
                synthesis = reasoner.synthesize_findings(
                    query=user_analysis_query,
                    tool_results=st.session_state.analysis_results,
                    canonical_map=st.session_state.canonical_map,
                    dataset_profile=st.session_state.dataset_profile
                )
                st.session_state.llm_insights = synthesis
                st.session_state.pdf_summary = synthesis

            st.session_state.analysis_done = True
            
            # Create clean serializable version for SQLite persistence
            serializable_results = {}
            for k, v in st.session_state.analysis_results.items():
                if isinstance(v, dict):
                    serializable_results[k] = {
                        "tool": v.get("tool", k),
                        "model": v.get("model", ""),
                        "status": v.get("status", ""),
                        "summary": v.get("summary", ""),
                        "metrics": v.get("metrics", {}),
                        "findings": v.get("findings", []),
                        "evidence": v.get("evidence", []),
                        "leaderboard": v.get("leaderboard", [])
                    }
                else:
                    serializable_results[k] = str(v)

            update_analysis_session_data(
                active_session["id"],
                data_dict={
                    "analysis_done": True,
                    "row_count": df.shape[0],
                    "col_count": df.shape[1],
                    "canonical_map": st.session_state.canonical_map,
                    "analysis_results": serializable_results,
                    "llm_insights": st.session_state.llm_insights
                }
            )
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        # ── STEP 3: Results & Comprehensive Reports Hub ──
        if st.session_state.analysis_done or (st.session_state.analysis_results and len(st.session_state.analysis_results) > 0):
            st.markdown("""
            <div class="action-card">
                <div class="action-card-header">
                    <div class="action-card-title">
                        <span class="step-badge">3</span>
                        Clinical Intelligence Findings & Dynamic Model Selection Leaderboard
                    </div>
                    <span class="step-pill-done">Analysis Verified ✓</span>
                </div>
            """, unsafe_allow_html=True)

            # High-level Metrics Strip
            n_rows, n_cols = df.shape
            conf_val = st.session_state.eval_metrics.get('Analysis Confidence Score', 0.95)
            conf_str = f"{conf_val*100:.0f}%" if isinstance(conf_val, (int, float)) else str(conf_val)

            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-box">
                    <div class="metric-label">Total Records</div>
                    <div class="metric-val">{n_rows:,}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Feature Columns</div>
                    <div class="metric-val">{n_cols}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Agents Executed</div>
                    <div class="metric-val">{len(st.session_state.analysis_results)} / 5</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Confidence Score</div>
                    <div class="metric-val">{conf_str}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # PDF Download Button
            try:
                pdf_data = generate_pdf_report(
                    st.session_state.df,
                    st.session_state.col_map,
                    st.session_state.analysis_results,
                    st.session_state.llm_insights,
                    dynamic_summary=st.session_state.get("pdf_summary"),
                    mapping_details=st.session_state.get("mapping_details")
                )
                st.download_button(
                    label="📥 Export Full Clinical PDF Report",
                    data=pdf_data,
                    file_name=f"clinical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="btn_dl_pdf_top"
                )
            except Exception as e:
                st.caption(f"PDF generator notice: {e}")

            # ── Detailed Report Tabs ──
            tabs = st.tabs([
                "📊 Summary & Data", 
                "🧬 Canonical Schema & Evidence",
                "🏆 Dynamic Model Selection",
                "⚠️ Risk Stratification", 
                "👥 Cohort Phenotypes", 
                "🔍 Anomaly Detection", 
                "🧩 Co-Prescriptions",
                "📈 Temporal Trends",
                "⭐ AI Executive Synthesis"
            ])

            with tabs[0]:
                st.markdown("##### 📋 Raw Data Preview (First 100 Records)")
                st.dataframe(df.head(100), use_container_width=True)

            with tabs[1]:
                st.markdown("##### 🧬 Canonical Semantic Schema & Detection Evidence")
                if "mapping_details" in st.session_state and st.session_state.mapping_details:
                    map_rows = []
                    for col_name, d in st.session_state.mapping_details.items():
                        conf_pct = int(d.get("confidence", 0.0) * 100)
                        map_rows.append({
                            "Dataset Column": col_name,
                            "Canonical Concept": d.get("canonical", "OTHER"),
                            "Mapping Confidence": f"{conf_pct}%",
                            "Inference Layer": d.get("layer", ""),
                            "Detection Evidence": d.get("evidence", "")
                        })
                    st.dataframe(pd.DataFrame(map_rows), use_container_width=True)
                elif st.session_state.mapping_table is not None:
                    st.dataframe(st.session_state.mapping_table, use_container_width=True)

            def create_model_performance_chart(tool_name: str, leaderboard: list, winner_name: str = ""):
                """Generates an interactive Plotly benchmark bar chart comparing candidate model performance."""
                if not leaderboard or not isinstance(leaderboard, list):
                    return None
                
                valid_records = [c for c in leaderboard if isinstance(c, dict) and c.get("valid", False)]
                if not valid_records:
                    valid_records = [c for c in leaderboard if isinstance(c, dict)]
                
                if not valid_records:
                    return None
                
                df_board = pd.DataFrame(valid_records)
                tool_key = tool_name.lower()
                
                try:
                    # 1. Trend Tool
                    if "trend" in tool_key and "rmse" in df_board.columns:
                        plot_df = df_board[df_board["rmse"].fillna(999999) < 90000].copy()
                        if plot_df.empty:
                            plot_df = df_board.copy()
                        
                        fig = px.bar(
                            plot_df, x="model", y="rmse",
                            color="is_winner" if "is_winner" in plot_df.columns else None,
                            title=f"🏆 {tool_name.upper()} Model Holdout Validation Error (RMSE — Lower is Better)",
                            template="plotly_dark",
                            color_discrete_map={True: "#00E5BE", False: "#3B82F6"} if "is_winner" in plot_df.columns else None,
                            text=plot_df["rmse"].round(2)
                        )
                        fig.update_layout(
                            paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                            xaxis_title="Forecasting Algorithm", yaxis_title="Holdout RMSE",
                            font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                            showlegend=False,
                            height=320
                        )
                        return fig

                    # 2. Cohort Tool
                    elif "cohort" in tool_key and "silhouette_score" in df_board.columns:
                        plot_df = df_board[df_board["silhouette_score"].notna()].copy()
                        if not plot_df.empty:
                            fig = px.bar(
                                plot_df, x="model", y="silhouette_score",
                                color="is_winner" if "is_winner" in plot_df.columns else None,
                                title=f"🏆 {tool_name.upper()} Model Clustering Quality (Silhouette Score — Higher is Better)",
                                template="plotly_dark",
                                color_discrete_map={True: "#00E5BE", False: "#3B82F6"} if "is_winner" in plot_df.columns else None,
                                text=plot_df["silhouette_score"].round(3)
                            )
                            fig.update_layout(
                                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                                xaxis_title="Clustering Algorithm", yaxis_title="Silhouette Score",
                                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                                showlegend=False,
                                height=320
                            )
                            return fig

                    # 3. Anomaly Tool
                    elif "anomaly" in tool_key and "separation_score" in df_board.columns:
                        plot_df = df_board[df_board["separation_score"].notna()].copy()
                        if not plot_df.empty:
                            fig = px.bar(
                                plot_df, x="model", y="separation_score",
                                color="is_winner" if "is_winner" in plot_df.columns else None,
                                title=f"🏆 {tool_name.upper()} Anomaly Separation Contrast (Z-Score Contrast — Higher is Better)",
                                template="plotly_dark",
                                color_discrete_map={True: "#00E5BE", False: "#3B82F6"} if "is_winner" in plot_df.columns else None,
                                text=plot_df["separation_score"].round(2)
                            )
                            fig.update_layout(
                                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                                xaxis_title="Anomaly Algorithm", yaxis_title="Separation Contrast (SD)",
                                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                                showlegend=False,
                                height=320
                            )
                            return fig

                    # 4. Risk Tool
                    elif "risk" in tool_key:
                        if "roc_auc" in df_board.columns and df_board["roc_auc"].notna().any():
                            metrics_to_plot = [m for m in ["roc_auc", "f1_score", "accuracy"] if m in df_board.columns and df_board[m].notna().any()]
                            if len(metrics_to_plot) > 1:
                                melted = pd.melt(df_board, id_vars=["model"], value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
                                fig = px.bar(
                                    melted, x="model", y="Score", color="Metric",
                                    barmode="group",
                                    title=f"🏆 {tool_name.upper()} Supervised Classifier Holdout Benchmark (Multi-Metric)",
                                    template="plotly_dark",
                                    color_discrete_sequence=["#00E5BE", "#0A84FF", "#F59E0B"]
                                )
                            else:
                                fig = px.bar(
                                    df_board, x="model", y="roc_auc",
                                    color="is_winner" if "is_winner" in df_board.columns else None,
                                    title=f"🏆 {tool_name.upper()} Classifier Benchmark (ROC-AUC — Higher is Better)",
                                    template="plotly_dark",
                                    color_discrete_map={True: "#00E5BE", False: "#3B82F6"} if "is_winner" in df_board.columns else None,
                                    text=df_board["roc_auc"].round(3)
                                )
                            fig.update_layout(
                                paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                                xaxis_title="Classifier Algorithm", yaxis_title="Test Score (0 - 1.0)",
                                font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                                height=320
                            )
                            return fig

                    # Generic numeric metric fallback
                    num_cols = df_board.select_dtypes(include=[np.number]).columns.tolist()
                    num_cols = [c for c in num_cols if c not in ["valid", "is_winner"]]
                    if num_cols and len(df_board) > 0:
                        primary_col = num_cols[0]
                        fig = px.bar(
                            df_board, x="model", y=primary_col,
                            color="is_winner" if "is_winner" in df_board.columns else None,
                            title=f"🏆 {tool_name.upper()} Evaluation Benchmark ({primary_col})",
                            template="plotly_dark",
                            color_discrete_map={True: "#00E5BE", False: "#3B82F6"} if "is_winner" in df_board.columns else None
                        )
                        fig.update_layout(
                            paper_bgcolor="#0D111A", plot_bgcolor="#090C10",
                            font=dict(family="Plus Jakarta Sans, DM Sans, sans-serif"),
                            showlegend=False,
                            height=320
                        )
                        return fig
                except Exception:
                    return None
                return None

            with tabs[2]:
                st.markdown("##### 🏆 Dynamic Model Selection Leaderboard & Validation Evidence")
                st.markdown("<p style='font-size:13px; color:#94A3B8;'>Objective mathematical evaluation scores determine the best performing algorithm for each tool — no guesswork.</p>", unsafe_allow_html=True)
                
                for t_name, res in st.session_state.analysis_results.items():
                    if isinstance(res, dict) and "leaderboard" in res and res["leaderboard"]:
                        st.markdown(f"**Tool: `{t_name.upper()}`** — Winning Model: `{res.get('model', 'Winner')}` (Execution: {res.get('execution_time_ms', 0.0)}ms)")
                        st.dataframe(pd.DataFrame(res["leaderboard"]), use_container_width=True)
                        if res.get("evidence"):
                            st.info("📌 **Selection Evidence:** " + " · ".join(res["evidence"]))
                        
                        # Render visual model performance comparison chart
                        perf_fig = create_model_performance_chart(t_name, res["leaderboard"], res.get("model", ""))
                        if perf_fig is not None:
                            st.plotly_chart(perf_fig, use_container_width=True)
                        
                        st.markdown("---")

            def render_tool_figures(figs):
                if not figs:
                    return
                for i in range(0, len(figs), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        if i + j < len(figs):
                            item = figs[i+j]
                            if isinstance(item, (list, tuple)) and len(item) == 2:
                                title, fig = item
                            else:
                                title, fig = "Visualization", item
                            with cols[j]:
                                st.plotly_chart(fig, use_container_width=True)

            with tabs[3]:
                if "risk" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["risk"]
                    st.markdown(res.get("summary", ""))
                    render_tool_figures(res.get("figures", []))

            with tabs[4]:
                if "cohort" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["cohort"]
                    st.markdown(res.get("summary", ""))
                    
                    # Display structured phenotype characteristics table if available
                    c_profiles = res.get("data", {}).get("cohort_profiles", [])
                    if c_profiles:
                        st.markdown("##### 🧬 Discovered Patient Phenotype Characteristics")
                        table_rows = []
                        for cp in c_profiles:
                            row = {
                                "Phenotype Cohort": cp.get("cohort", ""),
                                "Patient Count": f"{cp.get('patient_count', 0):,}",
                                "Cohort Share": cp.get("percentage", "")
                            }
                            traits = cp.get("traits", {})
                            if "mean_age" in traits: row["Mean Age"] = f"{traits['mean_age']} yrs"
                            if "primary_drug" in traits: row["Dominant Medication"] = traits["primary_drug"]
                            if "mean_quantity" in traits: row["Avg Units"] = str(traits["mean_quantity"])
                            if "common_dosage" in traits: row["Typical Dose"] = str(traits["common_dosage"])
                            if "risk_index" in traits: row["Risk Index"] = str(traits["risk_index"])
                            if "primary_diagnosis" in traits: row["Primary Diagnosis"] = str(traits["primary_diagnosis"])
                            if "dominant_gender" in traits: row["Gender Breakdown"] = str(traits["dominant_gender"])
                            table_rows.append(row)
                        if table_rows:
                            st.dataframe(pd.DataFrame(table_rows), use_container_width=True)

                    render_tool_figures(res.get("figures", []))

            with tabs[5]:
                if "anomaly" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["anomaly"]
                    st.markdown(res.get("summary", ""))
                    render_tool_figures(res.get("figures", []))

            with tabs[6]:
                if "pattern" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["pattern"]
                    st.markdown(res.get("summary", ""))
                    render_tool_figures(res.get("figures", []))

            with tabs[7]:
                if "trend" in st.session_state.analysis_results:
                    res = st.session_state.analysis_results["trend"]
                    st.markdown(res.get("summary", "Temporal trend analysis complete."))
                    render_tool_figures(res.get("figures", []))

            with tabs[8]:
                st.markdown('<div class="insight-container">', unsafe_allow_html=True)
                st.markdown(st.session_state.llm_insights)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # ── STEP 4: Interactive Dataset AI Copilot (Q&A) ──
            st.markdown("""
            <div class="action-card">
                <div class="action-card-header">
                    <div class="action-card-title">
                        <span class="step-badge">4</span>
                        Interactive Dataset AI Copilot (Session Q&A)
                    </div>
                    <span class="chip">Phi-4 mini Context-Aware</span>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<p style='font-size:13px; color:#94A3B8;'>Ask questions about this specific dataset analysis. Responses are grounded in your ML agent findings.</p>", unsafe_allow_html=True)

            # Quick Prompt Starter Chips
            col_q1, col_q2, col_q3, col_q4 = st.columns(4)
            quick_prompt = None
            with col_q1:
                if st.button("💡 Risk Drivers", key="qp_1", use_container_width=True):
                    quick_prompt = "What are the primary clinical risk drivers and patient features identified in this dataset?"
            with col_q2:
                if st.button("💡 Cohort Breakdown", key="qp_2", use_container_width=True):
                    quick_prompt = "Explain the clinical characteristics of the patient cohorts discovered by KMeans."
            with col_q3:
                if st.button("💡 Polypharmacy Risks", key="qp_3", use_container_width=True):
                    quick_prompt = "What high-risk co-prescription patterns or drug combinations were mined by the Pattern Agent?"
            with col_q4:
                if st.button("💡 Recommendations", key="qp_4", use_container_width=True):
                    quick_prompt = "Provide 3 actionable clinical recommendations based on the overall analysis findings."

            ds_messages = get_analysis_messages(active_session["id"])
            for msg in ds_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_query = st.chat_input("Ask about this dataset (e.g., 'Summarize high risk patient findings')...", key="chat_ds_input")
            query_to_run = quick_prompt or user_query

            if query_to_run:
                add_analysis_message(active_session["id"], "user", query_to_run)
                with st.chat_message("user"):
                    st.markdown(query_to_run)

                with st.chat_message("assistant"):
                    with st.spinner("Analyzing dataset findings with clinical intelligence..."):
                        findings_context = []
                        for tool_k, tool_v in st.session_state.analysis_results.items():
                            if isinstance(tool_v, dict):
                                tool_findings = tool_v.get("findings", [])
                                if tool_findings:
                                    findings_context.append(f"[{tool_k.upper()} FINDINGS]:\n" + "\n".join([f"• {f}" for f in tool_findings]))

                        detailed_evidence = "\n\n".join(findings_context)

                        context_prompt = (
                            f"Dataset: {active_session.get('filename', 'Medical Dataset')}\n"
                            f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
                            f"Executive Insights:\n{st.session_state.llm_insights}\n\n"
                            f"Detailed Multi-Agent Tool Findings & Discovered Cluster Profiles:\n{detailed_evidence}\n\n"
                            f"User Query: {query_to_run}\n"
                            "Instructions: Provide a comprehensive, direct, evidence-based clinical answer using the specific findings, numbers, and discovered phenotype traits above."
                        )
                        reply = explain_analysis(context_prompt)
                        st.markdown(reply)
                
                add_analysis_message(active_session["id"], "assistant", reply)
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 6. IMAGE & SCAN REPORT WORKSPACE ─────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "🩺 Image & Scan Report":
    if not st.session_state.active_image_session_id:
        new_s = create_analysis_session(current_user["id"], "image_report", "Radiology Scan Report")
        st.session_state.active_image_session_id = new_s["id"]

    active_session = get_analysis_session(st.session_state.active_image_session_id, current_user["id"])
    if not active_session:
        img_list = get_user_analysis_sessions(current_user["id"], "image_report")
        if img_list:
            st.session_state.active_image_session_id = img_list[0]["id"]
            active_session = img_list[0]
            load_image_session_state(active_session["id"], current_user["id"])

    # Action Toolbar
    col_bar1, col_bar2, col_bar3 = st.columns([5, 2, 1])
    with col_bar1:
        st.markdown(f"### 🩻 {active_session['title']}")
    with col_bar2:
        with st.expander("✏️ Rename Report", expanded=False):
            new_title_val = st.text_input("Title", value=active_session["title"], key="rename_img_input")
            if st.button("Save Title", key="btn_save_img_title"):
                if new_title_val.strip():
                    update_analysis_session_data(active_session["id"], title=new_title_val.strip())
                    st.rerun()
    with col_bar3:
        if st.button("🗑️ Delete", key="btn_del_curr_img", help="Permanently delete this scan report for patient privacy"):
            delete_analysis_session(active_session["id"], current_user["id"])
            remaining = get_user_analysis_sessions(current_user["id"], "image_report")
            if remaining:
                load_image_session_state(remaining[0]["id"], current_user["id"])
            else:
                st.session_state.active_image_session_id = None
                st.session_state.scan_extracted_text = ""
                st.session_state.scan_report_analysis = ""
            st.rerun()

    # Upload Card
    st.markdown("""
    <div class="action-card">
        <div class="action-card-header">
            <div class="action-card-title">
                <span class="step-badge">1</span>
                Scan Report Ingestion & OCR Extraction
            </div>
            <span class="chip">PDF · PNG · JPG · TIFF (X-Ray, CT, MRI, Ultrasound)</span>
        </div>
    """, unsafe_allow_html=True)

    supported_types = ["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "tif", "webp"]
    report_file = st.file_uploader(
        "Upload Radiology Scan, Medication Image, or Clinical Report",
        type=supported_types,
        key="main_img_file_uploader",
        help="Upload radiology scan, medication box, blister pack, prescription slip, or PDF report."
    )

    if report_file:
        if st.session_state.get("scan_filename") != report_file.name:
            with st.spinner("🔍 Extracting clinical text & analyzing medical scan..."):
                extracted_text = extract_text_from_file(report_file)
                st.session_state.scan_extracted_text = extracted_text
                st.session_state.scan_filename = report_file.name
                
                # Automatically perform deep clinical / diagnostic analysis
                if extracted_text and not extracted_text.startswith("[Error]"):
                    analysis = analyze_image_report(report_file.name, extracted_text)
                    st.session_state.scan_report_analysis = analysis
                else:
                    st.session_state.scan_report_analysis = ""

                update_analysis_session_data(
                    active_session["id"],
                    title=f"Scan: {report_file.name}",
                    filename=report_file.name,
                    data_dict={
                        "extracted_text": extracted_text,
                        "analysis": st.session_state.scan_report_analysis
                    }
                )

    extracted_text = st.session_state.scan_extracted_text
    current_fname = st.session_state.get("scan_filename", "")
    mod_info = detect_scan_modality(current_fname, extracted_text or "")

    if extracted_text:
        if extracted_text.startswith("[Error]"):
            st.error(f"⚠️ {extracted_text}")
        elif extracted_text.startswith("[Warning]"):
            st.warning(f"ℹ️ {extracted_text}")
        else:
            st.success(f"✓ Processed {current_fname} — {mod_info['label']}")
            
            # Display split visual scan preview and extracted text
            col_img_prev, col_ocr_prev = st.columns([1, 1])
            with col_img_prev:
                if report_file and is_image_file(current_fname):
                    st.image(report_file, caption=f"📸 Scan Image: {current_fname}", use_container_width=True)
                else:
                    st.info(f"📄 Document File: `{current_fname}`")
            
            with col_ocr_prev:
                st.markdown(f"**Detected Category**: `{mod_info['label']}`")
                st.text_area("📝 Extracted Report / Clinical Text", extracted_text, height=260)
    st.markdown("</div>", unsafe_allow_html=True)

    if not extracted_text and not st.session_state.scan_report_analysis:
        st.info("💡 **Ready to begin**: Upload a radiology report (PDF, Image, CT, X-ray) or medication scan above to evaluate findings and clinical impressions.")
    else:
        # Step 2: Clinical / Radiology AI Analysis
        badge_name = mod_info.get("badge", "Clinical Reasoning Model")
        card_title = "Clinical Pharmacology & Safety Analysis" if mod_info["category"] == "medication" else "Radiology Impression & AI Diagnostic Summary"
        
        st.markdown(f"""
        <div class="action-card">
            <div class="action-card-header">
                <div class="action-card-title">
                    <span class="step-badge">2</span>
                    {card_title}
                </div>
                <span class="chip-blue">{badge_name}</span>
            </div>
        """, unsafe_allow_html=True)

        if not st.session_state.scan_report_analysis and extracted_text and not extracted_text.startswith("[Error]"):
            if st.button("🔍 Generate Comprehensive Clinical Analysis", use_container_width=True, key="btn_analyze_scan", type="primary"):
                with st.spinner("Generating clinical and pharmacological diagnostic analysis..."):
                    analysis = analyze_image_report(current_fname, extracted_text or "")
                    st.session_state.scan_report_analysis = analysis
                    update_analysis_session_data(
                        active_session["id"],
                        data_dict={"analysis": analysis, "extracted_text": extracted_text}
                    )
                    st.rerun()

        if st.session_state.scan_report_analysis:
            st.markdown('<div class="insight-container">', unsafe_allow_html=True)
            st.markdown(st.session_state.scan_report_analysis)
            st.markdown('</div>', unsafe_allow_html=True)

            col_btn1, _ = st.columns([1, 3])
            with col_btn1:
                if st.button("🔄 Re-Analyze Scan", key="btn_reanalyze_scan", use_container_width=True):
                    with st.spinner("Re-evaluating clinical findings..."):
                        analysis = analyze_image_report(current_fname, extracted_text or "", temperature=0.3)
                        st.session_state.scan_report_analysis = analysis
                        update_analysis_session_data(
                            active_session["id"],
                            data_dict={"analysis": analysis, "extracted_text": extracted_text}
                        )
                        st.rerun()

            # Step 3: Interactive Clinical AI Assistant
            st.markdown("<hr style='border-color:#1C263A;'>", unsafe_allow_html=True)
            st.markdown("##### 💬 Interactive Clinical & Pharmacology Assistant")
            st.markdown("<p style='font-size:13px; color:#94A3B8;'>Ask follow-up questions about this specific medication, dosing rules, drug interactions, or radiological follow-up protocols.</p>", unsafe_allow_html=True)

            img_messages = get_analysis_messages(active_session["id"])
            for msg in img_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            user_query = st.chat_input("Ask about this scan/medication (e.g. 'What are the main drug interactions for Risperdal?')...", key="chat_img_input")
            if user_query:
                add_analysis_message(active_session["id"], "user", user_query)
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Evaluating clinical inquiry..."):
                        context_prompt = (
                            f"Document / Scan: {current_fname}\n"
                            f"Category: {mod_info['label']}\n"
                            f"Extracted Text:\n{extracted_text}\n\n"
                            f"Clinical Analysis:\n{st.session_state.scan_report_analysis}\n\n"
                            f"User Question: {user_query}"
                        )
                        reply = explain_image_report(context_prompt)
                        st.markdown(reply)
                
                add_analysis_message(active_session["id"], "assistant", reply)
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ─── 7. DRUG INTELLIGENCE WORKSPACE ───────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════
elif app_mode == "💊 Drug Intelligence":
    st.markdown("""
    <div class="action-card">
        <div class="action-card-header">
            <div class="action-card-title">
                💊 Comprehensive Pharmacology Knowledge Base
            </div>
            <span class="chip">Mechanism · Indications · Alternatives · Warnings</span>
        </div>
    """, unsafe_allow_html=True)

    # Search Bar
    col_search1, col_search2 = st.columns([4, 1])
    with col_search1:
        drug_query = st.text_input(
            "Drug Search",
            placeholder="Search pharmaceutical agent (e.g., Metformin, Atorvastatin, Warfarin, Lisinopril...)",
            label_visibility="collapsed",
            key="drug_search_input"
        )
    with col_search2:
        search_btn = st.button("🔍 Search Profile", use_container_width=True, key="btn_analyze_drug", type="primary")

    # Quick Suggestion Chips
    st.markdown("<p style='font-size:12px; color:#64748B; margin-top:8px;'>Quick Search Suggestions:</p>", unsafe_allow_html=True)
    c_s1, c_s2, c_s3, c_s4, c_s5, c_s6 = st.columns(6)
    suggested_drug = None
    with c_s1:
        if st.button("Metformin", key="sug_1", use_container_width=True): suggested_drug = "Metformin"
    with c_s2:
        if st.button("Atorvastatin", key="sug_2", use_container_width=True): suggested_drug = "Atorvastatin"
    with c_s3:
        if st.button("Warfarin", key="sug_3", use_container_width=True): suggested_drug = "Warfarin"
    with c_s4:
        if st.button("Lisinopril", key="sug_4", use_container_width=True): suggested_drug = "Lisinopril"
    with c_s5:
        if st.button("Amoxicillin", key="sug_5", use_container_width=True): suggested_drug = "Amoxicillin"
    with c_s6:
        if st.button("Semaglutide", key="sug_6", use_container_width=True): suggested_drug = "Semaglutide"

    target_drug = suggested_drug or (drug_query.strip() if search_btn and drug_query.strip() else None)

    if target_drug:
        with st.spinner(f"Consulting clinical pharmacology knowledge base for {target_drug}..."):
            info = get_drug_info(target_drug)

        st.markdown(f"#### 📋 {target_drug.title()} — Comprehensive Clinical Profile")

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

            with st.expander(f"{icon} {header}", expanded=True):
                st.markdown(body)

    st.markdown("</div>", unsafe_allow_html=True)