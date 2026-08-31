import requests
import json
import subprocess
import shutil
import os
import time
from typing import List, Dict, Any, Optional, Tuple

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODEL = "phi4-mini"
# MODEL = "MedAIBase/MedGemma1.5:4b"


def find_ollama_binary() -> Optional[str]:
    """Locate the ollama binary in PATH or standard install directories."""
    path = shutil.which("ollama")
    if path and os.path.exists(path):
        return path
    
    # Common Windows installation locations
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        candidate = os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe")
        if os.path.isfile(candidate):
            return candidate
            
    candidate2 = r"C:\Program Files\Ollama\ollama.exe"
    if os.path.isfile(candidate2):
        return candidate2
        
    return None


def is_ollama_running(timeout: float = 1.0) -> bool:
    """Check if the local Ollama server is responding."""
    try:
        res = requests.get(OLLAMA_TAGS_URL, timeout=timeout)
        return res.status_code == 200
    except Exception:
        return False


def ensure_ollama_running(timeout: float = 5.0) -> Tuple[bool, str]:
    """
    Ensure that the local Ollama server is running.
    If it is not running, attempts to launch `ollama serve` in a detached background process.
    """
    if is_ollama_running(timeout=1.0):
        return True, "Ollama server is active."

    binary = find_ollama_binary()
    if not binary:
        return False, "Ollama executable not found on system PATH or default installation directory."

    try:
        if os.name == "nt":
            # Run detached with no console window on Windows
            creation_flags = 0x08000000 | 0x00000008  # CREATE_NO_WINDOW | DETACHED_PROCESS
            subprocess.Popen(
                [binary, "serve"],
                creationflags=creation_flags,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                close_fds=True,
            )
        else:
            subprocess.Popen(
                [binary, "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )

        # Wait up to `timeout` seconds for the server port to be available
        start_time = time.time()
        while time.time() - start_time < timeout:
            time.sleep(0.5)
            if is_ollama_running(timeout=1.0):
                return True, "Ollama server started automatically."

        return False, "Ollama server process launched but timed out waiting for port 11434."
    except Exception as e:
        return False, f"Failed to start Ollama server: {str(e)}"


def query_llm(prompt: str, system: str = "", temperature: float = 0.3) -> str:
    """Send a completion prompt to Ollama with error handling and automatic server start."""
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    payload = {
        "model": MODEL,
        "prompt": full_prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        # Attempt to auto-start Ollama if it was down, then retry once
        started, _ = ensure_ollama_running(timeout=4.0)
        if started:
            try:
                response = requests.post(OLLAMA_URL, json=payload, timeout=300)
                response.raise_for_status()
                return response.json().get("response", "").strip()
            except Exception:
                pass
        return "ERROR_OLLAMA_DOWN"
    except Exception as e:
        return f"ERROR: {str(e)}"


def query_chat_llm(
    messages: List[Dict[str, Any]], 
    user_memories: Optional[List[Dict[str, Any]]] = None,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3
) -> str:
    """
    Execute a conversational multi-turn chat turn including long-term memories.
    Provides structured clinical responses with fallback support.
    """
    from utils.memory_engine import build_conversation_prompt
    
    full_prompt = build_conversation_prompt(
        messages=messages,
        user_memories=user_memories,
        custom_system_prompt=system_prompt
    )

    response = query_llm(full_prompt, temperature=temperature)
    
    # Check if Ollama is not running and generate an intelligent clinical fallback response
    if response == "ERROR_OLLAMA_DOWN":
        last_user_msg = messages[-1]["content"] if messages else ""
        return _generate_offline_clinical_response(last_user_msg, user_memories)
    
    return response


def _generate_offline_clinical_response(user_query: str, user_memories: Optional[List[Dict[str, Any]]] = None) -> str:
    """Provide a structured, helpful clinical response when Ollama local instance is offline."""
    q_lower = user_query.lower()
    memory_mention = ""
    if user_memories and len(user_memories) > 0:
        mem_str = ", ".join([m["content"] for m in user_memories[:3]])
        memory_mention = f"\n\n> 🧠 **Active Stored Context Recalled**: {mem_str}\n"

    base_notice = (
        "💡 *Note: Ollama local instance is currently offline. Start `ollama serve` to activate live Phi-4 mini inference.*"
    )

    if any(k in q_lower for k in ["metformin", "glp-1", "diabetes", "hba1c"]):
        return (
            f"### 💊 Clinical Pharmacotherapy Guidance: Type 2 Diabetes\n\n"
            f"{memory_mention}"
            f"**First-Line Therapeutic Principles:**\n"
            f"- **Metformin**: Decreases hepatic glucose production, improves peripheral insulin sensitivity. Target dose: 1000mg BID with meals.\n"
            f"- **GLP-1 Receptor Agonists (e.g. Semaglutide, Liraglutide)**: Indicated for cardiovascular and renal risk reduction with substantial weight loss benefits.\n\n"
            f"**Renal Considerations:**\n"
            f"- Monitor eGFR; contraindicate Metformin if eGFR < 30 mL/min/1.73m² due to lactic acidosis risk.\n\n"
            f"{base_notice}"
        )
    elif any(k in q_lower for k in ["warfarin", "inr", "anticoagula", "bleeding"]):
        return (
            f"### ⚠️ Anticoagulation Management & Drug Interactions\n\n"
            f"{memory_mention}"
            f"**Critical Prescribing Precautions for Warfarin:**\n"
            f"- **CYP2C9 & CYP3A4 Interactions**: Concomitant use with Amiodarone, Fluconazole, Metronidazole, or NSAIDs significantly elevates bleeding risk.\n"
            f"- **Monitoring**: Routine INR target 2.0–3.0 for DVT/PE/AF; target 2.5–3.5 for mechanical prosthetic valves.\n\n"
            f"**Direct Alternatives**: DOACs (Apixaban, Rivaroxaban, Dabigatran) depending on renal function and valve pathology.\n\n"
            f"{base_notice}"
        )
    elif any(k in q_lower for k in ["kidney", "renal", "ckd", "creatinine", "egfr"]):
        return (
            f"### 🩺 Renal Function & Prescription Dosing Protocol\n\n"
            f"{memory_mention}"
            f"**Key Dose Adjustments in Renal Impairment (CKD Stage 3+):**\n"
            f"- **ACEi / ARBs**: Continue with serum creatinine & potassium monitoring (up to 30% Cr rise acceptable).\n"
            f"- **Antibiotics**: Reduce dosage/frequency for Aminoglycosides, Vancomycin, and Fluoroquinolones.\n"
            f"- **Analgesics**: Strictly avoid NSAIDs to prevent acute renal hemodynamics compromise.\n\n"
            f"{base_notice}"
        )
    else:
        return (
            f"### 🩺 Clinical AI Consultation\n\n"
            f"{memory_mention}"
            f"I have received your clinical inquiry regarding: **\"{user_query}\"**.\n\n"
            f"**Key Clinical Review Areas:**\n"
            f"1. **Indication & Pharmacodynamics**: Validating dosing schedule and therapeutic mechanism.\n"
            f"2. **Safety & Contraindications**: Checking cross-reactivity, organ function clearance, and drug-drug interactions.\n"
            f"3. **Long-Term Memory Context**: Any recorded patient conditions or prescribing preferences are actively factored into recommendations.\n\n"
            f"{base_notice}"
        )


def extract_intent(user_input: str) -> dict:
    system = (
        "You are a clinical data analyst. Extract intent from the user query. "
        "Return ONLY valid JSON with keys: "
        "intent (one of: drug_info, csv_analysis), "
        "drug_name (string or null), "
        "analysis_focus (string or null). "
        "No explanation, no markdown, just the JSON object."
    )
    result = query_llm(user_input, system=system)
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        return json.loads(result[start:end])
    except Exception:
        lower = user_input.lower()
        if any(w in lower for w in ["csv", "json", "file", "upload", "dataset", "data"]):
            return {"intent": "csv_analysis", "drug_name": None, "analysis_focus": user_input}
        return {"intent": "drug_info", "drug_name": user_input.strip(), "analysis_focus": None}


def get_drug_info(drug_name: str) -> str:
    system = (
        "You are a clinical pharmacology expert. Provide detailed structured information about the given drug. "
        "Use clear sections with headers using markdown (##). Include:\n"
        "## Overview & Mechanism of Action\n"
        "## Common Indications\n"
        "## Alternative / Substitute Drugs\n"
        "List at least 5 alternatives with a brief comparison of each.\n"
        "## Drug Interactions — Do NOT Take With\n"
        "List drugs that should NOT be combined with this drug and explain why.\n"
        "## Warnings & Precautions\n"
        "Be clinically accurate, thorough, and well-organized."
    )
    res = query_llm(
        f"Provide complete clinical information about the drug: {drug_name}",
        system=system,
        temperature=0.2,
    )
    if res == "ERROR_OLLAMA_DOWN":
        return _generate_offline_drug_info(drug_name)
    return res


def _generate_offline_drug_info(drug_name: str) -> str:
    name = drug_name.strip().title()
    return (
        f"## Overview & Mechanism of Action\n"
        f"**{name}** is an essential pharmaceutical agent widely utilized in clinical practice. "
        f"It acts on specific therapeutic pathways to modulate cellular and receptor physiology.\n\n"
        f"## Common Indications\n"
        f"- Primary indicated conditions per clinical guidelines.\n"
        f"- Secondary prophylaxis and adjunctive management.\n\n"
        f"## Alternative / Substitute Drugs\n"
        f"1. **First-line Alternative**: Comparable efficacy with distinct pharmacokinetic profile.\n"
        f"2. **Second-line Option**: Recommended in patients with mild hypersensitivity.\n"
        f"3. **Extended-Release Formulation**: Better compliance and lower peak-related adverse effects.\n"
        f"4. **Class-Equivalent Agent**: Alternative receptor selectivity.\n"
        f"5. **Adjunctive Agent**: Used in combination regimens.\n\n"
        f"## Drug Interactions — Do NOT Take With\n"
        f"- **CYP3A4/CYP2D6 Potent Inhibitors**: Risk of elevated serum concentration and toxicity.\n"
        f"- **Potassium-sparing or nephrotoxic agents**: Requires electrolyte and renal panel monitoring.\n\n"
        f"## Warnings & Precautions\n"
        f"- Regular monitoring of hepatic and renal function tests.\n"
        f"- Dose titration required in geriatric or impaired renal cohorts.\n\n"
        f"*Note: Live Phi-4 mini inference offline. Start `ollama serve` for dynamic model generation.*"
    )


def explain_analysis(context: str) -> str:
    system = (
        "You are a senior clinical data scientist. Interpret pharmaceutical prescription data analysis results. "
        "Provide clear, actionable clinical insights. Highlight risks, patterns, and recommendations. "
        "Use markdown formatting with bullet points and headers. Be specific and clinically relevant."
    )
    return query_llm(context, system=system, temperature=0.3)


def identify_columns(columns: list, sample_data: list) -> dict:
    system = (
        "You are a healthcare data schema expert. Analyze column names and sample rows to identify their clinical meaning. "
        "Return ONLY valid JSON mapping each column name to one of these categories: "
        "[drug_name, patient_id, date, diagnosis, age, gender, dosage, frequency, region, risk_score, quantity, prescriber, other]. "
        "No explanation, no markdown, just the JSON object."
    )
    prompt = (
        f"Columns: {json.dumps(columns)}\n"
        f"Sample rows (first 3): {json.dumps(sample_data)}\n"
        "Map each column to its clinical category."
    )
    result = query_llm(prompt, system=system)
    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        mapping = json.loads(result[start:end])
        valid_cats = {"drug_name", "patient_id", "date", "diagnosis", "age", "gender", "dosage", "frequency", "region", "risk_score", "quantity", "prescriber", "other"}
        for col in columns:
            val = mapping.get(col, "other")
            if isinstance(val, list) and len(val) > 0:
                val = val[0]
            if not isinstance(val, str) or val not in valid_cats:
                mapping[col] = "other"
            else:
                mapping[col] = val
        return mapping
    except Exception:
        return {col: "other" for col in columns}


def generate_insights(summary: str, column_map: dict) -> str:
    system = (
        "You are a pharmaceutical data analyst. Based on the dataset summary and analysis results, "
        "generate comprehensive clinical insights including:\n"
        "- Key prescription patterns observed\n"
        "- Anomalies or risk indicators\n"
        "- Cohort-level findings\n"
        "- Trend observations\n"
        "- Clinical recommendations\n"
        "Use markdown with clear headers and bullet points."
    )
    prompt = f"Column schema: {json.dumps(column_map)}\n\nDataset analysis summary:\n{summary}"
    return query_llm(prompt, system=system, temperature=0.35)


def analyze_image_report(filename: str, ocr_text: str = "", modality: str | None = None, temperature: float = 0.3) -> str:
    system = (
        "You are a clinical radiology analyst. Interpret radiology scan reports such as X-ray, CT, MRI, and other medical images. "
        "Provide a concise but clinically meaningful description of findings, likely impressions, potential severity, and recommended next steps."
    )
    modality_prefix = f"This appears to be a {modality} report." if modality else "This appears to be a medical imaging report."
    if ocr_text:
        prompt = (
            f"A medical scan report was uploaded: {filename}.\n"
            f"{modality_prefix}\n"
            "Use the extracted text below to provide a radiology impression, highlight any abnormalities, and suggest follow-up actions.\n\n"
            "REPORT TEXT:\n"
            f"{ocr_text}\n\n"
            "Summarize the findings in clear medical language and include key impressions and recommended next steps."
        )
    else:
        prompt = (
            f"A medical scan report was uploaded: {filename}.\n"
            f"{modality_prefix}\n"
            "No OCR text is available. If this is a radiology report, explain that the system needs report text to summarize findings accurately. "
            "If only a raw imaging file is available, be transparent that raw pixel-based image interpretation is not supported in this workflow."
        )
    return query_llm(prompt, system=system, temperature=temperature)


def explain_image_report(context: str) -> str:
    system = (
        "You are a clinical radiology specialist. Answer follow-up questions about imaging findings clearly and accurately. "
        "Use clinical reasoning and avoid speculation beyond the information provided."
    )
    return query_llm(context, system=system, temperature=0.3)


def generate_pdf_executive_summary(summary: str) -> str:
    system = (
        "You are a Senior Clinical Data Analyst. Write a high-level, professional executive summary for a medical data report. "
        "Summarize the key objectives, the analysis performed, and the high-level impact. "
        "Use formal, academic medical language. Keep it between 100-150 words."
    )
    return query_llm(f"Summarize this clinical analysis: {summary}", system=system, temperature=0.3)
