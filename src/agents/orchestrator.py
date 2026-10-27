"""
Diagnostic Orchestrator for Multi-Agent RosaNet AI Pipeline.

Manages the deterministic execution lifecycle:
Audio Inputs (4 Valves)
       │
       ▼
[Agent 1: SignalQualityAgent] ───► Validate Quality, Filter, Standardize Spectrograms
       │
       ▼
[Agent 2: PredictionXAIAgent] ───► RosaNet 4-Valve Inference, Multilabel Probs, XAI Heatmaps
       │
       ▼
[Agent 3: MedicalEvidenceAgent] ─► ACC/AHA RAG Knowledge Retrieval, Verifiable Citations
       │
       ▼
[ClinicalReportGenerator] ───────► Comprehensive Clinical Synthesis Report
"""

import uuid
from typing import Dict, Any, Optional, Tuple

from src.agents.schemas import FinalClinicalReport
from src.agents.quality_agent import SignalQualityAgent
from src.agents.prediction_agent import PredictionXAIAgent
from src.agents.evidence_agent import MedicalEvidenceAgent
from src.agents.report_generator import ClinicalReportGenerator


class DiagnosticPipelineOrchestrator:
    """
    Coordinates execution across the 3 specialized agents and report generator.
    """

    def __init__(self, model_checkpoint_path: Optional[str] = None, chroma_db_dir: Optional[str] = None):
        self.quality_agent = SignalQualityAgent()
        self.prediction_agent = PredictionXAIAgent(model_checkpoint_path=model_checkpoint_path)
        self.evidence_agent = MedicalEvidenceAgent(chroma_db_dir=chroma_db_dir)
        self.report_generator = ClinicalReportGenerator()

    def process_session(
        self,
        valve_files: Dict[str, Any],  # maps 'Aor', 'Pul', 'Tri', 'Mit' to audio bytes or paths
        session_id: Optional[str] = None,
        patient_metadata: Optional[Dict[str, Any]] = None,
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> FinalClinicalReport:
        """
        Executes complete end-to-end multi-agent clinical workflow.
        """
        if not session_id:
            session_id = f"SES-{uuid.uuid4().hex[:8].upper()}"

        # 1. Execute Agent 1: Signal and Data Quality Inspection
        quality_summary, spectrogram_dict = self.quality_agent.inspect_and_preprocess(
            valve_files=valve_files,
            session_id=session_id
        )

        if quality_summary.status == "FAIL":
            # Early abort on critical quality failure
            return self.report_generator.synthesize_report(
                session_id=session_id,
                quality=quality_summary,
                prediction=self.prediction_agent.predict_and_explain(
                    # Fallback zeros
                    {v: np.zeros((64, 626, 1), dtype=np.float32) for v in ['Aor', 'Mit', 'Pul', 'Tri']},
                    session_id=session_id
                ),
                evidence=self.evidence_agent.retrieve_clinical_evidence({}, [], session_id=session_id),
                patient_metadata=patient_metadata
            )

        # 2. Execute Agent 2: RosaNet Prediction & XAI
        prediction_summary = self.prediction_agent.predict_and_explain(
            spectrogram_dict=spectrogram_dict,
            session_id=session_id,
            custom_thresholds=custom_thresholds
        )

        # 3. Execute Agent 3: Medical Evidence & Guideline Context Retrieval
        evidence_summary = self.evidence_agent.retrieve_clinical_evidence(
            predicted_probabilities=prediction_summary.raw_probabilities,
            detected_conditions=prediction_summary.detected_conditions,
            session_id=session_id
        )

        # 4. Synthesize Final Clinical Report
        final_report = self.report_generator.synthesize_report(
            session_id=session_id,
            quality=quality_summary,
            prediction=prediction_summary,
            evidence=evidence_summary,
            patient_metadata=patient_metadata
        )

        return final_report
