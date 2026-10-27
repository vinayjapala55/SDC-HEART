"""
Pydantic Schemas for Multi-Agent Clinical Heart Sound Analysis.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ValveSignalInfo(BaseModel):
    valve_location: str  # 'Aor', 'Mit', 'Pul', 'Tri'
    filename: str
    sample_rate: int
    duration_seconds: float
    is_valid: bool = True
    clipping_detected: bool = False
    snr_db: Optional[float] = None
    warning: Optional[str] = None


class QualityAgentOutput(BaseModel):
    agent_name: str = "Agent 1: Signal and Data Quality Agent"
    session_id: str
    timestamp: str
    status: str  # 'PASS', 'WARN', 'FAIL'
    required_valves_present: bool
    valves_inspected: List[ValveSignalInfo]
    spectrograms_generated: bool
    summary: str
    error_message: Optional[str] = None


class ValveAttribution(BaseModel):
    valve_name: str
    branch_score: float
    mean_saliency: float
    peak_saliency: float


class DetectedCondition(BaseModel):
    disease_code: str  # 'AS', 'AR', 'MR', 'MS', 'N'
    disease_name: str
    probability: float
    threshold: float
    is_positive: bool
    primary_contributing_valve: str


class PredictionAgentOutput(BaseModel):
    agent_name: str = "Agent 2: RosaNet Prediction and XAI Agent"
    session_id: str
    model_version: str = "RosaNet-v1.0"
    raw_probabilities: Dict[str, float]
    decision_thresholds: Dict[str, float]
    detected_conditions: List[DetectedCondition]
    is_normal: bool
    overall_status: str
    risk_level: str  # 'Low Risk', 'Moderate Risk', 'High Risk'
    valve_attributions: List[Dict[str, Any]]
    model_limitations: str


class GuidelineEvidenceChunk(BaseModel):
    chunk_id: str
    guideline_title: str
    organization: str = "ACC/AHA"
    year: int = 2020
    page_number: int
    section_name: str
    target_disease: str
    clinical_topic: str
    class_of_rec: Optional[str] = None
    level_of_evidence: Optional[str] = None
    content_text: str
    relevance_score: float


class EvidenceAgentOutput(BaseModel):
    agent_name: str = "Agent 3: Medical Evidence and Risk Context Agent"
    session_id: str
    evidence_found: bool
    retrieved_guidelines: List[GuidelineEvidenceChunk]
    disease_context_summaries: Dict[str, str]
    citation_references: List[str]
    missing_evidence_warning: Optional[str] = None


class FinalClinicalReport(BaseModel):
    report_title: str = "BMD-HS Multi-Valvular PCG Diagnostic & Risk Stratification Report"
    generation_timestamp: str
    patient_session_id: str
    model_version: str
    quality_summary: QualityAgentOutput
    prediction_summary: PredictionAgentOutput
    evidence_summary: EvidenceAgentOutput
    structured_markdown_report: str
    clinical_disclaimer: str
