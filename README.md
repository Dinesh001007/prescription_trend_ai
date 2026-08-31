# Prescription Trend AI — Clinical Intelligence Platform

An advanced multi-agent clinical intelligence platform combining multi-agent machine learning (XGBoost, KMeans, Isolation Forest, Holt-Winters, Apriori) with conversational LLM reasoning (Phi-4 mini / SLM), persistent SQLite storage, secure authentication, session history with safe data deletion, and radiological scan interpretation.

---

## 🌟 Key Features

### 🔐 1. User Authentication & Persistent Database
- **Secure Sign Up & Sign In**: Dedicated registration and sign-in page with PBKDF2 + SHA-256 salted password hashing.
- **Selectable Clinical Roles**: Clinician / Physician, Medical Data Scientist, Radiologist, Clinical Researcher, Healthcare Specialist.
- **SQLite Relational Database (`medical_platform.db`)**:
  - `users`: User profiles, credentials, clinical roles.
  - `analysis_sessions`: Stored dataset analysis and scan report sessions.
  - `analysis_messages`: Contextual interactive Q&A history linked to specific datasets and scan reports.

### 📂 2. Multi-Agent Dataset Analysis Pipeline with Session History & Safe Deletion
- **`+ New Dataset Analysis`**: Start a clean dataset analysis session anytime from the sidebar.
- **Saved Analyses List**: Switch between previous dataset analyses, review full agent outputs, and download reports.
- **Safe Data Deletion (🗑️)**: Permanently and securely delete any previous dataset analysis and its associated chat history for patient privacy and regulatory compliance.
- **Autonomous Analysis Pipeline**:
  - **Step 1**: Autonomous medical schema mapping and semantic classification.
  - **Step 2**: Automated preprocessing & feature engineering.
  - **Step 3**: Execution of 5 specialized medical agents:
    - ⚠️ Risk Agent (XGBoost)
    - 👥 Cohort Agent (KMeans Clustering)
    - 🔍 Anomaly Detection (Isolation Forest)
    - 📈 Trend Forecasting (Holt-Winters)
    - 🧩 Pattern Mining (Apriori Co-Prescriptions)
  - **Step 4 & 5**: Automated statistical validation, executive insights, and downloadable PDF clinical report.
- **Interactive Dataset AI Assistant**: In-session clinical Q&A specifically tailored to the active dataset's findings and risk factors.

### 🩺 3. Scan & Image Report Analysis with Session History
- **`+ New Scan Report`**: Initialize fresh imaging analysis sessions.
- **Saved Scan Reports List**: Browse, rename, or safely delete previous scan analyses.
- **OCR-Assisted Radiological Interpretation**: Text extraction and clinical impression summaries for X-ray, CT, MRI, and ultrasound reports.
- **Interactive Radiology AI Assistant**: Contextual Q&A to ask follow-up questions about specific scan findings and recommendations.

### 💊 4. Drug Intelligence & Pharmacology Profiles
- Search comprehensive drug mechanism of action, clinical indications, therapeutic alternatives, high-risk drug-drug interactions, and precautions.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

### 3. Sign Up & Sign In
- Register your user account on the **Create Account** tab with your clinical role, then sign in to access the workspace.
