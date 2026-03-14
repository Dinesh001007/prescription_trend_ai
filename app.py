import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils.llm import (
    query_llm,
    extract_intent,
    get_drug_info,
    identify_columns,
    generate_insights,
    explain_analysis,
)
from utils.data_loader import (
    load_file,
    get_sample_rows,
    build_summary,
)
from agents.risk_agent import run_risk_agent
from agents.cohort_agent import run_cohort_agent
from agents.anomaly_agent import run_anomaly_agent
from agents.trend_agent import run_trend_agent
from agents.pattern_agent import run_pattern_agent

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
        <div class="app-sub">Clinical Intelligence Platform · Powered by Llama 3</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Input Mode")
    mode = st.radio(
        "",
        ["💊 Drug Lookup", "📂 Dataset Analysis"],
        label_visibility="collapsed",
    )

    st.markdown("### 📤 Upload Data")
    uploaded_file = st.file_uploader(
        "CSV / JSON / Excel",
        type=["csv", "json", "xlsx", "xls"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        st.success(f"✓ {uploaded_file.name}")

    st.markdown("### ⚙️ Analysis Options")
    run_risk = st.checkbox("Risk Analysis (XGBoost)", value=True)
    run_cohort = st.checkbox("Cohort Clustering (KMeans)", value=True)
    run_anomaly = st.checkbox("Anomaly Detection (IsoForest)", value=True)
    run_trend = st.checkbox("Trend Forecasting (Prophet)", value=True)
    run_pattern = st.checkbox("Co-Prescription Patterns", value=True)

    st.markdown("### 🤖 Model")
    st.markdown('<span class="pill">Llama 3 · Ollama</span>', unsafe_allow_html=True)
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
        with st.spinner("Consulting clinical knowledge base via Llama 3..."):
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


# ─── Dataset Analysis Mode ────────────────────────────────────────────────────
else:
    st.markdown('<div class="section-header">📂 Dataset Analysis</div>', unsafe_allow_html=True)

    # Load file
    if uploaded_file:
        if st.session_state.df is None or st.session_state.get("last_file") != uploaded_file.name:
            try:
                df = load_file(uploaded_file)
                st.session_state.df = df
                st.session_state.col_map = None
                st.session_state.analysis_done = False
                st.session_state["last_file"] = uploaded_file.name
            except Exception as e:
                st.error(f"Failed to load file: {e}")

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
            with st.spinner("Llama 3 is identifying your column roles..."):
                sample = get_sample_rows(df, 3)
                col_map = identify_columns(df.columns.tolist(), sample)
                st.session_state.col_map = col_map

        col_map = st.session_state.col_map

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

        # ── Override column roles (outside expander to avoid nesting) ──
        with st.expander("✏️ Override Column Roles"):
            categories = ["drug_name", "patient_id", "date", "diagnosis", "age", "gender",
                          "dosage", "frequency", "region", "risk_score", "quantity", "prescriber", "other"]
            updated_map = {}
            override_cols = st.columns(3)
            for i, col in enumerate(df.columns):
                with override_cols[i % 3]:
                    updated_map[col] = st.selectbox(
                        col,
                        categories,
                        index=categories.index(col_map.get(col, "other")),
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

                with st.spinner("Llama 3 generating clinical insights..."):
                    llm_insights = generate_insights(overall_summary, col_map)
                    st.session_state.llm_insights = llm_insights

                st.session_state.analysis_done = True
                progress.empty()
                st.rerun()

        # ── Show Results ──
        if st.session_state.analysis_done and hasattr(st.session_state, "analysis_results"):
            results = st.session_state.analysis_results

            # LLM Insights
            st.markdown('<div class="section-header">🤖 Llama 3 Clinical Insights</div>', unsafe_allow_html=True)
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
                                # Summary
                                if res.get("summary"):
                                    cols = st.columns([2, 1])
                                    with cols[0]:
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