"""
BMD-HS Heart Disease Detection Agentic AI - FastAPI Backend Server.

Supports:
1. Four-valve input patient sessions (Aortic, Mitral, Pulmonary, Tricuspid).
2. Specialized Multi-Agent Orchestration:
   - Agent 1: Signal & Data Quality Agent
   - Agent 2: RosaNet Prediction & XAI Agent
   - Agent 3: ACC/AHA Clinical RAG Evidence Agent
3. Comprehensive Clinical Report Synthesis.
"""

import os
import io
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.orchestrator import DiagnosticPipelineOrchestrator
from src.agents.schemas import FinalClinicalReport

app = FastAPI(
    title="BMD-HS Multi-Valvular Cardiovascular Diagnostic Assistant API",
    description="Agentic AI system for multi-valve phonocardiogram classification and ACC/AHA guideline retrieval.",
    version="2.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ORCHESTRATOR: Optional[DiagnosticPipelineOrchestrator] = None


@app.on_event("startup")
def initialize_system():
    global ORCHESTRATOR
    print("[INFO] Initializing Multi-Agent Diagnostic Pipeline Orchestrator...")
    try:
        model_path = ROOT_DIR / "dataset" / "models" / "best_rosanet_model.keras"
        chroma_dir = ROOT_DIR / "dataset" / "rag" / "chroma_db"
        ORCHESTRATOR = DiagnosticPipelineOrchestrator(
            model_checkpoint_path=str(model_path) if model_path.exists() else None,
            chroma_db_dir=str(chroma_dir) if chroma_dir.exists() else None
        )
        print("[SUCCESS] Multi-Agent Pipeline Orchestrator initialized.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[ERROR] Failed to initialize pipeline orchestrator: {e}")


@app.get("/api/status")
def get_system_status():
    return {
        "status": "online",
        "engine": "RosaNet 4-Valve Phonocardiogram Multi-Agent Engine",
        "target_sr": 2000,
        "input_valves_required": ["Aor", "Mit", "Pul", "Tri"],
        "orchestrator_ready": ORCHESTRATOR is not None
    }


@app.get("/api/sample_patients")
def get_sample_patients():
    """
    Returns curated authentic clinical 4-valve recording sets from the dataset.
    """
    train_dir = ROOT_DIR / "train"
    if not train_dir.exists():
        return {"patients": []}

    # Curate authentic patients across all disease categories
    curated_pids = [
        ("patient_089", "Normal Healthy Subject", "Normal", "N_089"),
        ("patient_005", "Aortic Stenosis (AS)", "Aortic Stenosis", "AS_005"),
        ("patient_016", "Aortic Regurgitation (AR)", "Aortic Regurgitation", "AR_016"),
        ("patient_002", "Mitral Regurgitation (MR)", "Mitral Regurgitation", "MR_002"),
        ("patient_006", "Mitral Stenosis (MS)", "Mitral Stenosis", "MS_006"),
        ("patient_003", "Combined Multi-Valvular Disease (AS+AR)", "Multi-Disease", "MD_003")
    ]

    patients = []
    for pid, label, category, prefix in curated_pids:
        # Match sitting files for the 4 valves
        aor = f"{prefix}_sit_Aor.wav"
        mit = f"{prefix}_sit_Mit.wav"
        pul = f"{prefix}_sit_Pul.wav"
        tri = f"{prefix}_sit_Tri.wav"

        if all((train_dir / f).exists() for f in [aor, mit, pul, tri]):
            patients.append({
                "patient_id": pid,
                "label": label,
                "category": category,
                "valves": {
                    "Aor": aor,
                    "Mit": mit,
                    "Pul": pul,
                    "Tri": tri
                }
            })

    return {"patients": patients}


@app.get("/api/sample_audio/{filename}")
def stream_sample_audio(filename: str):
    file_path = ROOT_DIR / "train" / filename
    if not file_path.exists():
        # Check augmented
        file_path = ROOT_DIR / "dataset" / "augmented" / "train" / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(str(file_path), media_type="audio/wav")


@app.post("/api/predict_patient")
async def analyze_patient_session(
    file_aor: Optional[UploadFile] = File(None),
    file_mit: Optional[UploadFile] = File(None),
    file_pul: Optional[UploadFile] = File(None),
    file_tri: Optional[UploadFile] = File(None),
    sample_patient_id: Optional[str] = Form(None)
):
    """
    Main Multi-Valvular Diagnostic Endpoint:
    Accepts 4 distinct recordings for Aortic, Mitral, Pulmonary, and Tricuspid sites.
    Executes Agent 1 (Quality) -> Agent 2 (Prediction & XAI) -> Agent 3 (RAG) -> Clinical Report.
    """
    if ORCHESTRATOR is None:
        raise HTTPException(status_code=500, detail="Diagnostic Orchestrator is not initialized.")

    valve_inputs = {}
    train_dir = ROOT_DIR / "train"

    if sample_patient_id:
        # Load authentic sample set
        samples = get_sample_patients()["patients"]
        matched = next((p for p in samples if p["patient_id"] == sample_patient_id), None)
        if not matched:
            raise HTTPException(status_code=404, detail=f"Sample patient {sample_patient_id} not found.")

        for v_name in ["Aor", "Mit", "Pul", "Tri"]:
            f_path = train_dir / matched["valves"][v_name]
            with open(f_path, "rb") as f:
                valve_inputs[v_name] = f.read()

        session_id = f"SESSION-{sample_patient_id.upper()}"
        metadata = {"patient_id": sample_patient_id}

    else:
        # User uploaded 4 files
        uploaded = {
            "Aor": file_aor,
            "Mit": file_mit,
            "Pul": file_pul,
            "Tri": file_tri
        }
        missing = [v for v, f in uploaded.items() if f is None]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Analysis requires recordings for all 4 valves. Missing: {', '.join(missing)}"
            )

        for v_name, up_file in uploaded.items():
            content = await up_file.read()
            valve_inputs[v_name] = content

        session_id = f"SESSION-CUSTOM-{os.urandom(3).hex().upper()}"
        metadata = None

    try:
        report: FinalClinicalReport = ORCHESTRATOR.process_session(
            valve_files=valve_inputs,
            session_id=session_id,
            patient_metadata=metadata
        )
        return report.dict()
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# Serve Web Frontend
web_dir = ROOT_DIR / "web"
if web_dir.exists():
    app.mount("/", StaticFiles(directory=str(web_dir), html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
