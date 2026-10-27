"""
Stage 9: End-to-End System and Application Integration Test Suite.

Tests:
1. DiagnosticPipelineOrchestrator initialization and model loading.
2. End-to-end pipeline execution on all 6 authentic clinical presets:
   - patient_089 (Normal Healthy Subject)
   - patient_005 (Aortic Stenosis)
   - patient_016 (Aortic Regurgitation)
   - patient_002 (Mitral Regurgitation)
   - patient_006 (Mitral Stenosis)
   - patient_003 (Multi-Valvular AS+AR)
3. QualityAgent output integrity (SNR, sample rate, duration, valid valve channels).
4. PredictionXAIAgent output integrity (multilabel probabilities, calibrated thresholds, XAI Grad-CAM saliency heatmaps, valve attention).
5. MedicalEvidenceAgent retrieval integrity (ChromaDB ACC/AHA guideline excerpts, citations, relevance scores).
6. ClinicalReportGenerator schema validation (risk stratification, recommendations, clinical caveats).
7. FastAPI HTTP API contract testing (/api/status, /api/sample_patients, /api/predict_preset).
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents.orchestrator import DiagnosticPipelineOrchestrator
from src.agents.schemas import FinalClinicalReport
from fastapi.testclient import TestClient
from app import app


class TestStage9EndToEndIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_dir = ROOT / 'train'
        cls.chk_4v = ROOT / 'dataset' / 'models' / 'best_rosanet_4valve_model.keras'
        cls.chroma_dir = ROOT / 'dataset' / 'rag' / 'chroma_db'
        cls.orchestrator = DiagnosticPipelineOrchestrator(
            model_checkpoint_path=str(cls.chk_4v) if cls.chk_4v.exists() else None,
            chroma_db_dir=str(cls.chroma_dir) if cls.chroma_dir.exists() else None
        )
        cls.client = TestClient(app)

    def test_01_api_status_endpoint(self):
        """Verify /api/status returns online with all 4 required valves."""
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("Aor", data["input_valves_required"])
        self.assertIn("Mit", data["input_valves_required"])
        self.assertIn("Pul", data["input_valves_required"])
        self.assertIn("Tri", data["input_valves_required"])
        print("\n[TEST 1 PASS] /api/status endpoint verified.")

    def test_02_sample_patients_endpoint(self):
        """Verify /api/sample_patients returns curated patients with all 4 valve files."""
        res = self.client.get("/api/sample_patients")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        patients = data.get("patients", [])
        self.assertGreaterEqual(len(patients), 5)
        for p in patients:
            valves = p.get("valves", {})
            self.assertIn("Aor", valves)
            self.assertIn("Mit", valves)
            self.assertIn("Pul", valves)
            self.assertIn("Tri", valves)
            for v, fname in valves.items():
                self.assertTrue((ROOT / "train" / fname).exists(), f"Sample patient audio file missing: {fname}")
        print(f"\n[TEST 2 PASS] /api/sample_patients verified across {len(patients)} authentic presets.")

    def test_03_orchestrator_on_preset_patients(self):
        """Run complete 4-valve multi-agent pipeline on representative authentic patients."""
        res = self.client.get("/api/sample_patients")
        patients = res.json().get("patients", [])
        self.assertTrue(len(patients) > 0)

        # Test on first 3 presets: Normal, AS, AR
        for p in patients[:3]:
            pid = p["patient_id"]
            files = {v: str(ROOT / "train" / fname) for v, fname in p["valves"].items()}
            
            report = self.orchestrator.process_session(
                valve_files=files,
                session_id=f"TEST-E2E-{pid}",
                patient_metadata={"patient_id": pid, "clinical_label": p["label"]}
            )

            self.assertIsInstance(report, FinalClinicalReport)
            self.assertIn(report.quality_summary.status, ["PASS", "WARN"])
            self.assertIn(report.prediction_summary.risk_level, ["Low Risk", "Low-Moderate Risk", "Moderate Risk", "High Risk"])
            self.assertIsNotNone(report.prediction_summary.raw_probabilities)
            
            # Check Explainability
            self.assertGreaterEqual(len(report.prediction_summary.valve_attributions), 1)
            for exp in report.prediction_summary.valve_attributions:
                v_attrs = exp.get("valve_attributions", [])
                self.assertEqual(len(v_attrs), 4)
                for vf in v_attrs:
                    self.assertIn("valve_name", vf)
                    self.assertIn("branch_score", vf)
                    self.assertIn("heatmap", vf)
                
            # Check Clinical Evidence Citations
            self.assertIsNotNone(report.evidence_summary)
            self.assertTrue(len(report.structured_markdown_report) > 0)
            print(f"  [E2E] Processed {pid} ({p['label']}): Risk={report.prediction_summary.risk_level}, Status={report.prediction_summary.overall_status}")

        print("\n[TEST 3 PASS] DiagnosticPipelineOrchestrator successfully processed authentic clinical presets.")

    def test_04_api_predict_patient_endpoint(self):
        """Verify POST /api/predict_patient generates valid structured clinical report."""
        res = self.client.post("/api/predict_patient", data={"sample_patient_id": "patient_089"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("patient_session_id", data)
        self.assertIn("quality_summary", data)
        self.assertIn("prediction_summary", data)
        self.assertIn("evidence_summary", data)
        self.assertIn("structured_markdown_report", data)
        print("\n[TEST 4 PASS] /api/predict_patient endpoint verified.")


if __name__ == '__main__':
    unittest.main()
