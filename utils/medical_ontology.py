from enum import Enum
from typing import List, Dict, Any

class SemanticCategory(Enum):
    IDENTIFIERS = "A. IDENTIFIERS"
    DEMOGRAPHICS = "B. DEMOGRAPHICS"
    VITAL_SIGNS = "C. VITAL_SIGNS"
    LAB_RESULTS = "D. LAB_RESULTS"
    DIAGNOSIS = "E. DIAGNOSIS / CONDITIONS"
    MEDICATIONS = "F. MEDICATIONS / TREATMENTS"
    TEMPORAL = "G. TEMPORAL DATA"
    IMAGING = "H. IMAGING / REPORT REFERENCES"
    CLINICAL_NOTES = "I. CLINICAL NOTES (TEXT)"
    UTILIZATION = "J. UTILIZATION / ADMINISTRATIVE"
    OUTCOMES = "K. OUTCOME-RELATED"
    DEVICE_DATA = "L. DEVICE / MONITORING DATA"
    GENOMIC = "M. GENOMIC / SPECIALIZED DATA"
    GENERAL = "N. GENERAL FEATURES"
    UNKNOWN = "O. UNKNOWN_MEDICAL_FEATURE"

MEDICAL_ONTOLOGY = {
    SemanticCategory.IDENTIFIERS: {
        "keywords": ["id", "mrn", "encounter", "visit_id", "record_id", "patient_id", "admission_id", "uid"],
        "description": "Unique identifiers for patients, visits, or records."
    },
    SemanticCategory.DEMOGRAPHICS: {
        "keywords": ["age", "gender", "sex", "ethnicity", "race", "weight", "height", "bmi", "dob", "birth"],
        "description": "Patient basic information."
    },
    SemanticCategory.VITAL_SIGNS: {
        "keywords": ["heart_rate", "pulse", "bp", "systolic", "diastolic", "temp", "spo2", "respiratory", "oxygen"],
        "description": "Basic physiological measurements."
    },
    SemanticCategory.LAB_RESULTS: {
        "keywords": ["glucose", "hba1c", "creatinine", "hemoglobin", "wbc", "platelets", "sodium", "potassium", "level", "result", "lab"],
        "description": "Laboratory test results."
    },
    SemanticCategory.DIAGNOSIS: {
        "keywords": ["icd", "diagnosis", "disease", "comorbidity", "condition", "status", "stage"],
        "description": "Medical conditions and diagnosis codes."
    },
    SemanticCategory.MEDICATIONS: {
        "keywords": ["drug", "medication", "dose", "frequency", "route", "prescription", "rx", "therapy"],
        "description": "Drugs, treatments, and dosages."
    },
    SemanticCategory.TEMPORAL: {
        "keywords": ["date", "time", "admission", "discharge", "timestamp", "start", "end", "duration"],
        "description": "Time-related data."
    },
    SemanticCategory.IMAGING: {
        "keywords": ["scan", "mri", "ct", "xray", "radiology", "imaging", "report", "dicom"],
        "description": "References to imaging or radiology reports."
    },
    SemanticCategory.CLINICAL_NOTES: {
        "keywords": ["note", "summary", "text", "description", "comment", "clinical", "history"],
        "description": "Free-text clinical observations."
    },
    SemanticCategory.UTILIZATION: {
        "keywords": ["stay", "cost", "billing", "insurance", "type", "department", "ward", "provider"],
        "description": "Administrative and hospital utilization data."
    },
    SemanticCategory.OUTCOMES: {
        "keywords": ["mortality", "death", "readmission", "outcome", "complication", "survival", "score"],
        "description": "Patient outcomes and complications."
    },
    SemanticCategory.DEVICE_DATA: {
        "keywords": ["ecg", "eeg", "monitor", "wearable", "signal", "icv", "stream"],
        "description": "Data from medical devices or wearables."
    },
    SemanticCategory.GENOMIC: {
        "keywords": ["gene", "biomarker", "dna", "rna", "marker", "mutation", "expression"],
        "description": "Genomic or specialized biomarker data."
    }
}
