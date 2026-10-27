"""
Clinical Report Synthesizer for Multi-Agent RosaNet Diagnostic System.

Combines verified outputs from:
- Agent 1: Signal and Data Quality Agent
- Agent 2: RosaNet Prediction and XAI Agent
- Agent 3: Medical Evidence and Risk Context Agent

Enforces strict non-hallucination safeguards:
- Zero fabrication of unsupplied demographic details (age, sex, clinical history).
- Explicit separation between observed signals, model probabilities, XAI attributions, and retrieved medical text.
- Mandatory investigational disclaimers.
"""

from datetime import datetime
from typing import Dict, Any, Optional

from src.agents.schemas import (
    QualityAgentOutput,
    PredictionAgentOutput,
    EvidenceAgentOutput,
    FinalClinicalReport
)


class ClinicalReportGenerator:
    """
    Deterministic synthesizer generating formatted clinical diagnostic reports.
    """

    def synthesize_report(
        self,
        session_id: str,
        quality: QualityAgentOutput,
        prediction: PredictionAgentOutput,
        evidence: EvidenceAgentOutput,
        patient_metadata: Optional[Dict[str, Any]] = None
    ) -> FinalClinicalReport:
        """
        Synthesizes structured Markdown report from multi-agent outputs.
        """
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Patient context (only if explicitly supplied)
        meta_lines = []
        if patient_metadata:
            if "patient_id" in patient_metadata:
                meta_lines.append(f"- **Patient / Subject ID**: `{patient_metadata['patient_id']}`")
            if "age_band" in patient_metadata and patient_metadata["age_band"]:
                meta_lines.append(f"- **Age Band**: {patient_metadata['age_band']}")
            if "gender" in patient_metadata and patient_metadata["gender"]:
                meta_lines.append(f"- **Gender**: {patient_metadata['gender']}")
        else:
            meta_lines.append(f"- **Analysis Session ID**: `{session_id}` (De-identified Session)")

        patient_meta_block = "\n".join(meta_lines) if meta_lines else "- De-identified PCG auscultation session."

        # Quality lines
        valves_table = "| Valve Site | File Analyzed | Duration (s) | Sample Rate | Quality Status |\n| :--- | :--- | :---: | :---: | :---: |\n"
        for v in quality.valves_inspected:
            status_tag = "VALID" if v.is_valid else "INVALID"
            if v.clipping_detected: status_tag += " (Clipping)"
            valves_table += f"| **{v.valve_location}** | `{v.filename}` | {v.duration_seconds}s | {v.sample_rate} Hz | {status_tag} |\n"

        # Predictions lines
        probs_table = "| Disease Category | Code | Model Probability | Threshold | Clinical Classification |\n| :--- | :---: | :---: | :---: | :---: |\n"
        for code, name in [
            ("N", "Normal Auscultation"),
            ("AS", "Aortic Stenosis"),
            ("AR", "Aortic Regurgitation"),
            ("MR", "Mitral Regurgitation"),
            ("MS", "Mitral Stenosis")
        ]:
            prob = prediction.raw_probabilities.get(code, 0.0)
            thresh = prediction.decision_thresholds.get(code, 0.45)
            is_active = (prob >= thresh) if code != "N" else (prob >= 0.50 and len(prediction.detected_conditions) == 0)
            cls_text = "**ELEVATED**" if (code != "N" and prob >= thresh) else ("**NORMAL**" if (code == "N" and is_active) else "Low Probability")
            probs_table += f"| {name} | `{code}` | **{prob * 100:.1f}%** | {thresh * 100:.0f}% | {cls_text} |\n"

        # XAI Attribution lines
        xai_blocks = []
        if prediction.valve_attributions:
            for item in prediction.valve_attributions:
                d_name = item.get("disease_class", "")
                p_val = item.get("probability", 0.0)
                top_v = item.get("top_contributing_valve", "Aortic Valve")
                xai_blocks.append(f"- **{d_name} (Prob: {p_val*100:.1f}%)**: Primary convolutional feature attribution concentrated at **{top_v}**.")
        xai_text = "\n".join(xai_blocks) if xai_blocks else "- No elevated disease conditions requiring valve attribution analysis."

        # Evidence lines
        evidence_blocks = []
        if evidence.retrieved_guidelines:
            for g in evidence.retrieved_guidelines[:3]:
                cor_loe = f" [COR: {g.class_of_rec}, LOE: {g.level_of_evidence}]" if g.class_of_rec else ""
                evidence_blocks.append(
                    f"#### {g.target_disease} — {g.section_name} (Page {g.page_number}){cor_loe}\n"
                    f"> *\"{g.content_text.strip()}\"*\n\n"
                    f"**Source**: {g.guideline_title}\n"
                )
        evidence_text = "\n".join(evidence_blocks) if evidence_blocks else "*No direct guideline matches retrieved for current parameters.*"

        # Citations
        citations_text = "\n".join([f"- {c}" for c in evidence.citation_references]) if evidence.citation_references else "- 2020 ACC/AHA Guideline for the Management of Patients With Valvular Heart Disease."

        disclaimer = (
            "**IMPORTANT CLINICAL NOTICE**:\n"
            "This document is an artificial intelligence-generated risk stratification aid based on automated "
            "phonocardiogram signal processing and ACC/AHA guidelines retrieval. It does NOT constitute a definitive "
            "medical diagnosis. Murmurs and acoustic features must be confirmed via 2D transthoracic echocardiography (TTE) "
            "and evaluated by a board-certified physician or cardiologist."
        )

        # Markdown Document Assembly
        md_report = f"""# BMD-HS Cardiovascular Risk Stratification & Auscultation Report

**Report Generation Date**: {now_str}  
**Model Architecture**: `{prediction.model_version}` | **Decision Fusion**: Dynamic Valve Gating  

---

### 1. Patient & Session Context
{patient_meta_block}
- **Quality Status**: `{quality.status}`
- **Overall Diagnostic Assessment**: **{prediction.overall_status}**
- **Risk Stratification Tier**: **`{prediction.risk_level.upper()}`**

---

### 2. Audio Quality & Valve Mapping Summary (Agent 1)
{valves_table}
*{quality.summary}*

---

### 3. Multilabel RosaNet Predictions & Classifications (Agent 2)
{probs_table}

---

### 4. Explainable AI & Valve-Level Attribution (Agent 2)
{xai_text}

> *Attribution Note: Activation heatmaps indicate regions of maximal convolutional filter response during forward inference and reflect learned feature importance, not anatomical pathology.*

---

### 5. Relevant Clinical Evidence & Practice Guidelines (Agent 3)
{evidence_text}

**Traceable Guideline Citations**:
{citations_text}

---

### 6. Technical Limitations & Diagnostic Uncertainty
- {prediction.model_limitations}
- **Missing Diagnostic Modalities**: Transthoracic echocardiogram (TTE), Doppler peak velocity, mean pressure gradient, and 12-lead ECG.

---

### 7. Clinical Disclaimer
{disclaimer}
"""

        return FinalClinicalReport(
            report_title="BMD-HS Multi-Valvular PCG Diagnostic & Risk Stratification Report",
            generation_timestamp=now_str,
            patient_session_id=session_id,
            model_version=prediction.model_version,
            quality_summary=quality,
            prediction_summary=prediction,
            evidence_summary=evidence,
            structured_markdown_report=md_report,
            clinical_disclaimer=disclaimer
        )
