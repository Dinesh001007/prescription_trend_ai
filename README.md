# Prescription Trend AI — Dynamic AI-Agent Architecture & Clinical Intelligence Platform

An enterprise-grade, schema-independent healthcare AI platform powered by a dynamic multi-agent architecture. The system profiles arbitrary datasets without assuming fixed column names, automatically discovers capabilities, competes candidate machine learning models with objective metrics, and synthesizes evidence-grounded clinical intelligence reports.

---

## 🌟 Core Architecture Principles

1. **No Fixed Dataset Schema**: The system never assumes rigid column names (`drug_name`, `patient_id`, `date`, `age`). Any tabular format (CSV, Excel, JSON) is dynamically parsed and mapped.
2. **3-Layer Semantic Schema Mapping**:
   - **Layer 1**: Rule & fuzzy name similarity matching against clinical ontological keywords.
   - **Layer 2**: Content & distribution heuristics (date parsing, regex dosage patterns, biological age bounds, ICD diagnostic terms, gender domains).
   - **Layer 3**: LLM semantic inference for ambiguous dimensions with confidence and evidence tracking.
   - **Canonical Concepts**: `DRUG`, `DATE`, `PATIENT_ID`, `DIAGNOSIS`, `AGE`, `GENDER`, `DOSAGE`, `REGION`, `QUANTITY`, `PRESCRIBER`, `RISK_SCORE`.
3. **Automated Capability Matrix & Graceful Degradation**:
   - Analyzes available vs missing canonical fields and dataset statistics to determine feasible tools.
   - Strict graceful degradation: explicitly reports data limitations without fabricating missing variables.
4. **Dynamic ML Tool Competition & Mathematical Scoring**:
   - ML algorithms are specialized tools evaluated objectively:
     - **Cohort Tool**: Competes `KMeans`, `DBSCAN`, `AgglomerativeClustering`, and `GaussianMixture` evaluated by `Silhouette Score` (higher is better), `Davies-Bouldin Index` (lower is better), `Calinski-Harabasz`, and cluster validity.
     - **Trend Tool**: Competes `Prophet`, `ExponentialSmoothing (ETS)`, `ARIMA`, and `LinearTrend` on holdout validation data evaluated by `RMSE`, `MAE`, and `MAPE`.
     - **Anomaly Tool**: Competes `IsolationForest`, `LocalOutlierFactor (LOF)`, `OneClassSVM`, and `Statistical IQR-Envelope` evaluated by separation contrast.
     - **Risk Tool**: Competes `XGBoost`, `RandomForest`, `GradientBoosting`, and `LogisticRegression` (`ROC-AUC`, `F1-Score`, `Accuracy`) when a target exists; routes gracefully to multi-factor composite risk scoring when no target is present.
     - **Pattern Tool**: Discovers frequent co-prescribing itemsets, association rules (Support, Confidence, Lift), and polypharmacy interactions.
5. **Supervisor Agent & Concurrent Execution**:
   - Formulates DAG execution stages based on user intent and capability reports.
   - Executes independent tools in parallel (`concurrent.futures.ThreadPoolExecutor` / `asyncio`) and cascades dependent tools.
6. **Evidence-Grounded AI Reasoning Agent**:
   - Normalizes all tool outputs into standard JSON contracts.
   - Synthesizes findings strictly from measurable mathematical evidence, highlighting uncertainty and actionable clinical guidance.
7. **FastAPI Backend + Streamlit Frontend + SSE Streaming**:
   - Asynchronous REST API and Server-Sent Events streaming progress and real-time synthesis.
   - Visual model competition leaderboards, interactive schema confidence inspection, and downloadable clinical PDF reports.

---

## 🏗️ System Architecture Flow

```
USER / FRONTEND
      │
      ▼
FastAPI API / Streamlit Gateway (Port 8000 / 8501)
      │
      ├──► Dataset Ingestion & Profiler (Dtype, Cardinality, Missing Rate, Health Score)
      │         │
      │         ▼
      ├──► 3-Layer Semantic Schema Mapper (Canonical: DRUG, DATE, AGE, DOSAGE, etc.)
      │         │
      │         ▼
      ├──► Capability Matrix & Data Quality Report (Feasibility Verification)
      │         │
      │         ▼
      ├──► AI Supervisor / Planner (Query Intent -> DAG Execution Plan)
      │         │
      │         ├─────────────── Concurrent Multi-Model Execution ───────────────┐
      │         │                     │                      │                   │
      │         ▼                     ▼                      ▼                   ▼
      │    [Risk Tool]          [Trend Tool]          [Cohort Tool]       [Anomaly Tool]
      │   XGBoost / RF          Prophet / ETS         KMeans / DBSCAN      IsoForest / LOF
      │  Holdout ROC-AUC        Holdout RMSE          Silhouette / DB     Separation Score
      │         │                     │                      │                   │
      │         └─────────────────────┴──────────────────────┴───────────────────┘
      │                                       │
      │                                       ▼
      ├──► Pattern Tool (Apriori Co-Prescriptions & Polypharmacy Lift Matrix)
      │         │
      │         ▼
      ├──► Normalized Results Aggregator
      │         │
      │         ▼
      ├──► Evidence-Grounded AI Reasoning Agent
      │         │
      │         ▼
      └──► Streaming Response & Visual Multi-Model Leaderboards
```

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive OpenAPI documentation is available at `http://127.0.0.1:8000/docs`.

### 3. Launch Streamlit Clinical Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 4. Run Automated Test Suite
```bash
python -m unittest tests/test_dynamic_pipeline.py
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/upload` | Ingest tabular dataset, profile statistics, and generate initial mapping |
| `GET` | `/api/profile/{dataset_id}` | Retrieve comprehensive column distributions and data quality metrics |
| `POST` | `/api/semantic_map/{dataset_id}` | Update canonical semantic schema mapping |
| `POST` | `/api/plan/{dataset_id}` | Formulate DAG execution plan based on query intent |
| `POST` | `/api/execute/{dataset_id}` | Execute concurrent multi-model competition and AI reasoning synthesis |
| `GET` | `/api/stream_events/{dataset_id}` | Server-Sent Events (SSE) stream for real-time progress and token synthesis |
| `POST` | `/api/scan_report` | Multi-modality clinical scan, packaging, and OCR diagnostic analysis |
| `POST` | `/api/drug_lookup` | Complete pharmacology profile search |
| `GET` | `/health` | System health check and LLM connectivity status |
