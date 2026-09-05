"""
FastAPI Routes for Prescription Trend AI - Dynamic AI-Agent Architecture
"""

import os
import uuid
import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from utils.data_profiling import DatasetProfiler
from utils.data_profiling import SemanticMapper
from utils.core_pipeline import CapabilityMatrix
from utils.core_pipeline import AgentOrchestrator
from utils.llm_core import AIReasoner
from utils.media_utils import extract_text_from_file
from utils.llm_core import analyze_image_report, explain_image_report, get_drug_info, query_llm

router = APIRouter()

# Shared In-Memory State & Singletons
profiler = DatasetProfiler()
mapper = SemanticMapper()
capability_evaluator = CapabilityMatrix()
orchestrator = AgentOrchestrator()
reasoner = AIReasoner()

DATASET_METADATA: Dict[str, Dict[str, Any]] = {}


# --- Request/Response Models ---
class MapUpdateRequest(BaseModel):
    custom_mapping: Dict[str, str]

class PlanRequest(BaseModel):
    query: Optional[str] = "Perform comprehensive multi-tool analysis."
    custom_mapping: Optional[Dict[str, str]] = None

class DrugLookupRequest(BaseModel):
    query: str

class ScanChatRequest(BaseModel):
    prompt: str
    scan_filename: Optional[str] = "scan_report"
    extracted_text: Optional[str] = ""
    analysis: Optional[str] = ""


# --- Endpoints ---

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Accepts CSV or Excel datasets, profiles them, and returns a unique dataset_id."""
    filename = file.filename or "uploaded_data.csv"
    content = await file.read()

    try:
        import io
        dataset_id, df, profile = profiler.ingest_dataset(io.BytesIO(content), filename=filename)
        
        # Immediate layered semantic mapping
        mapping_result = mapper.map_columns(df, profile.get("columns", {}), use_llm=True)
        canonical_map = mapping_result["canonical_mapping"]
        
        # Capability matrix evaluation
        cap_report = capability_evaluator.evaluate_capabilities(canonical_map, profile)

        DATASET_METADATA[dataset_id] = {
            "filename": filename,
            "profile": profile,
            "canonical_mapping": canonical_map,
            "mapping_details": mapping_result["mapping_details"],
            "capabilities": cap_report
        }

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "filename": filename,
            "row_count": profile["row_count"],
            "column_count": profile["column_count"],
            "data_quality_score": profile["data_quality_score"],
            "canonical_mapping": canonical_map,
            "capabilities": cap_report["capabilities"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to ingest dataset: {str(e)}")


@router.get("/profile/{dataset_id}")
async def get_profile(dataset_id: str):
    """Retrieves full column profile and data distributions."""
    df = profiler.get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")
    meta = DATASET_METADATA.get(dataset_id, {})
    return {
        "dataset_id": dataset_id,
        "profile": meta.get("profile", profiler.profile_dataframe(df)),
        "canonical_mapping": meta.get("canonical_mapping", {}),
        "mapping_details": meta.get("mapping_details", {}),
        "capabilities": meta.get("capabilities", {})
    }


@router.post("/semantic_map/{dataset_id}")
async def update_semantic_mapping(dataset_id: str, payload: MapUpdateRequest):
    """Overrides or updates canonical semantic mapping and recomputes capability matrix."""
    df = profiler.get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")
    
    meta = DATASET_METADATA.setdefault(dataset_id, {})
    canonical_map = payload.custom_mapping
    meta["canonical_mapping"] = canonical_map

    # Persist explicit corrections so future uploads can reuse them.
    for column_name, canonical in canonical_map.items():
        mapper.meaning_agent.teach(column_name, canonical, meta.get("profile", {}).get("columns", {}).get(column_name))

    # Re-evaluate capabilities
    profile = meta.get("profile", profiler.profile_dataframe(df))
    cap_report = capability_evaluator.evaluate_capabilities(canonical_map, profile)
    meta["capabilities"] = cap_report

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "updated_canonical_mapping": canonical_map,
        "capabilities": cap_report
    }


@router.post("/plan/{dataset_id}")
async def plan_execution(dataset_id: str, payload: PlanRequest):
    """Generates execution DAG plan based on user intent and dataset capabilities."""
    df = profiler.get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")

    meta = DATASET_METADATA.get(dataset_id, {})
    cap_report = meta.get("capabilities", capability_evaluator.evaluate_capabilities(meta.get("canonical_mapping", {}), meta.get("profile", {})))

    plan = orchestrator.plan_execution(payload.query or "", cap_report)
    return {
        "dataset_id": dataset_id,
        "plan": plan
    }


@router.post("/execute/{dataset_id}")
async def execute_analysis(dataset_id: str, payload: PlanRequest):
    """Executes dynamic multi-model tool competition and synthesizes AI explanation."""
    df = profiler.get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")

    meta = DATASET_METADATA.get(dataset_id, {})
    canonical_map = payload.custom_mapping or meta.get("canonical_mapping", {})
    profile = meta.get("profile", profiler.profile_dataframe(df))
    cap_report = meta.get("capabilities", capability_evaluator.evaluate_capabilities(canonical_map, profile))

    # 1. Plan
    plan = orchestrator.plan_execution(payload.query or "", cap_report)

    # 2. Execute Concurrently
    exec_result = orchestrator.execute_plan(df, canonical_map, plan)
    tool_results = exec_result["tool_results"]

    # 3. AI Reasoning Synthesis
    synthesis = reasoner.synthesize_findings(
        query=payload.query or "Comprehensive Clinical Analysis",
        tool_results=tool_results,
        canonical_map=canonical_map,
        dataset_profile=profile
    )

    return {
        "status": "success",
        "dataset_id": dataset_id,
        "execution_summary": {
            "duration_ms": exec_result["total_duration_ms"],
            "tool_count": exec_result["executed_tool_count"]
        },
        "tool_results": tool_results,
        "ai_synthesis": synthesis
    }


@router.get("/stream_events/{dataset_id}")
async def stream_analysis_events(dataset_id: str, query: str = "Analyze dataset"):
    """Server-Sent Events (SSE) streaming real-time execution progress and synthesis."""
    df = profiler.get_dataset(dataset_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset ID not found.")

    async def event_generator():
        meta = DATASET_METADATA.get(dataset_id, {})
        canonical_map = meta.get("canonical_mapping", {})
        profile = meta.get("profile", profiler.profile_dataframe(df))
        cap_report = meta.get("capabilities", capability_evaluator.evaluate_capabilities(canonical_map, profile))

        yield f"data: {json.dumps({'event': 'profiling_complete', 'health_score': profile.get('data_quality_score', 100)})}\n\n"
        await asyncio.sleep(0.1)

        plan = orchestrator.plan_execution(query, cap_report)
        yield f"data: {json.dumps({'event': 'plan_generated', 'selected_tools': plan['selected_tools']})}\n\n"
        await asyncio.sleep(0.1)

        # Run tools
        exec_result = orchestrator.execute_plan(df, canonical_map, plan)
        tool_results = exec_result["tool_results"]

        for t_name, res in tool_results.items():
            yield f"data: {json.dumps({'event': 'tool_completed', 'tool': t_name, 'winner': res.get('model', ''), 'status': res.get('status', 'success')})}\n\n"
            await asyncio.sleep(0.05)

        yield f"data: {json.dumps({'event': 'synthesis_started'})}\n\n"
        await asyncio.sleep(0.1)

        synthesis = reasoner.synthesize_findings(query, tool_results, canonical_map, profile)
        yield f"data: {json.dumps({'event': 'final_synthesis', 'content': synthesis})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/scan_report")
async def api_scan_report(file: UploadFile = File(...)):
    """Processes medical/radiology image or prescription and generates AI diagnosis."""
    content = await file.read()
    filename = file.filename or "scan_report.jpg"

    try:
        import io
        extracted_text = extract_text_from_file(io.BytesIO(content))
        analysis = analyze_image_report(filename, extracted_text or "")

        return {
            "status": "success",
            "filename": filename,
            "extracted_text": extracted_text,
            "clinical_analysis": analysis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR or Analysis failed: {str(e)}")


@router.post("/drug_lookup")
async def api_drug_lookup(payload: DrugLookupRequest):
    """Clinical pharmacology lookup for drugs, mechanisms, and interactions."""
    info = get_drug_info(payload.query)
    return {
        "status": "success",
        "drug": payload.query,
        "clinical_profile": info
    }


@router.post("/scan_chat")
async def api_scan_chat(payload: ScanChatRequest):
    """Interactive clinical chat on analyzed scans."""
    context = (
        f"Scan Report: {payload.scan_filename}\n"
        f"Extracted OCR Text:\n{payload.extracted_text}\n\n"
        f"Diagnostic Analysis:\n{payload.analysis}\n\n"
        f"Question: {payload.prompt}"
    )
    reply = explain_image_report(context)
    return {
        "status": "success",
        "response": reply
    }
