"""
FastAPI Application Entry Point
Prescription Trend AI - Dynamic AI-Agent Architecture
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router

app = FastAPI(
    title="Prescription Trend AI — Dynamic AI-Agent Architecture",
    description="Schema-independent clinical analytics API with dynamic multi-model tool competition and evidence-grounded AI reasoning.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for all frontends (Streamlit, React, Vue, Next.js)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
def root():
    return {
        "message": "Prescription Trend AI — Dynamic AI-Agent Architecture API is active.",
        "version": "2.0.0",
        "documentation": "/docs"
    }


@app.get("/health")
def health_check():
    from utils.llm_core import is_ollama_running
    return {
        "status": "healthy",
        "api_engine": "FastAPI",
        "llm_service": "active" if is_ollama_running() else "offline_fallback_mode"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
