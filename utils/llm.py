import requests
import json

OLLAMA_URL = "http://localhost:11434/api/generate"
# MODEL = "phi4-mini"
MODEL = "MedAIBase/MedGemma1.5:4b"


def query_llm(prompt: str, system: str = "", temperature: float = 0.3) -> str:
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
        return "ERROR_OLLAMA_DOWN"
    except Exception as e:
        return f"ERROR: {str(e)}"


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
    return query_llm(
        f"Provide complete clinical information about the drug: {drug_name}",
        system=system,
        temperature=0.2,
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
