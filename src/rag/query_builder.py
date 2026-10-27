"""
Clinical RAG Query Builder for BMD-HS Project.

Transforms RosaNet multi-label predicted probabilities into focused,
topic-aware clinical guideline retrieval queries.
"""

from typing import Dict, List, Any, Optional


class ClinicalQueryBuilder:
    """
    Constructs deterministic, guideline-aligned retrieval queries from RosaNet predictions.
    """

    # Disease Topic Query Templates aligned with 2020 ACC/AHA VHD Guideline
    QUERY_TEMPLATES = {
        "AS": [
            {
                "topic": "Auscultation",
                "query": "Aortic stenosis auscultation findings harsh systolic crescendo-decrescendo murmur right upper sternal border radiation"
            },
            {
                "topic": "Staging",
                "query": "Aortic stenosis echocardiographic severity criteria jet velocity mean gradient valve area stages A B C D"
            },
            {
                "topic": "Intervention",
                "query": "Indications and timing of intervention for severe aortic stenosis TAVR versus SAVR"
            }
        ],
        "AR": [
            {
                "topic": "Auscultation",
                "query": "Aortic regurgitation early diastolic decrescendo murmur left sternal border peripheral pulses"
            },
            {
                "topic": "Staging",
                "query": "Chronic aortic regurgitation severity staging criteria regurgitant volume effective regurgitant orifice"
            },
            {
                "topic": "Surgery",
                "query": "Indications for surgical valve replacement or repair in chronic severe aortic regurgitation"
            }
        ],
        "MR": [
            {
                "topic": "Auscultation",
                "query": "Mitral regurgitation auscultation holosystolic murmur apex axilla radiation S3"
            },
            {
                "topic": "Staging",
                "query": "Primary and secondary mitral regurgitation severity staging criteria EROA regurgitant fraction"
            },
            {
                "topic": "Intervention",
                "query": "Transcatheter edge-to-edge repair TEER and surgery indications in severe mitral regurgitation"
            }
        ],
        "MS": [
            {
                "topic": "Auscultation",
                "query": "Mitral stenosis auscultation opening snap low-pitched diastolic rumble apex presystolic accentuation"
            },
            {
                "topic": "Staging",
                "query": "Rheumatic mitral stenosis severity staging criteria mitral valve area mean gradient pressure half-time"
            },
            {
                "topic": "Intervention",
                "query": "Percutaneous mitral balloon commissurotomy PMBC and surgical intervention indications in mitral stenosis"
            }
        ],
        "N": [
            {
                "topic": "Physical Examination",
                "query": "Normal heart sounds S1 S2 auscultation physiological splitting and innocent flow murmurs evaluation"
            },
            {
                "topic": "General",
                "query": "Initial clinical evaluation and physical examination of patients with suspected valvular heart disease"
            }
        ],
        "MULTI": [
            {
                "topic": "Staging",
                "query": "Evaluation and staging of mixed and multivalvular heart disease combined aortic and mitral lesions"
            },
            {
                "topic": "Intervention",
                "query": "Timing and indications for intervention in patients with mixed and multiple valve disease"
            }
        ]
    }

    def __init__(self, confidence_threshold: float = 0.50):
        """
        Initialize query builder with a configurable retrieval confidence threshold.
        Note: Threshold is purely for retrieval candidate selection, not a clinical diagnostic threshold.
        """
        self.confidence_threshold = confidence_threshold

    def build_queries(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """
        Convert patient-level predicted probabilities into retrieval queries.
        """
        ordered_classes = ["N", "AS", "AR", "MR", "MS"]
        preds_clean = {k: float(predictions.get(k, 0.0)) for k in ordered_classes}

        # 1. Identify retrieval candidate diseases based on confidence threshold
        candidates = [c for c in ordered_classes if preds_clean[c] >= self.confidence_threshold]

        # If no class reaches threshold, fallback to top predicted signal (dominant probability)
        if not candidates:
            top_class = max(preds_clean, key=preds_clean.get)
            candidates = [top_class]

        # If 'N' (Normal) is dominant or only candidate
        if "N" in candidates and len(candidates) == 1:
            selected_diseases = ["N"]
            is_multi = False
        else:
            # Filter out 'N' if pathological diseases are also detected
            selected_diseases = [c for c in candidates if c != "N"]
            if not selected_diseases:
                selected_diseases = ["N"]
            is_multi = len(selected_diseases) > 1

        # 2. Assemble queries
        queries = []
        for dis in selected_diseases:
            if dis in self.QUERY_TEMPLATES:
                for item in self.QUERY_TEMPLATES[dis]:
                    queries.append({
                        "disease": dis,
                        "topic": item["topic"],
                        "query": item["query"]
                    })

        # Add MULTI query if multiple valvular diseases have elevated probability
        if is_multi:
            for item in self.QUERY_TEMPLATES["MULTI"]:
                queries.append({
                    "disease": "MULTI",
                    "topic": item["topic"],
                    "query": item["query"]
                })

        # 3. Create descriptive clinical summary
        elevated_str = ", ".join([f"{d} ({preds_clean[d]:.3f})" for d in selected_diseases])
        clinical_summary = f"RosaNet prediction indicates elevated probability of: {elevated_str}."

        return {
            "predictions": preds_clean,
            "confidence_threshold": self.confidence_threshold,
            "candidate_diseases": selected_diseases,
            "is_multi_disease": is_multi,
            "clinical_intent_summary": clinical_summary,
            "queries": queries
        }
