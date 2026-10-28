"""
Agent 2: RosaNet Prediction and XAI Agent.

Responsibilities:
1. Load the approved RosaNet model architecture and weights.
2. Validate expected 4-valve input structure (Aor, Mit, Pul, Tri).
3. Execute deterministic model inference to produce multilabel probabilities.
4. Apply validation-calibrated decision thresholds.
5. Generate supported Explainable AI (XAI) outputs:
   - Valve-level contribution scores.
   - Grad-CAM time-frequency activation summaries.
6. Associate every prediction and explanation with its input valve recording.
7. Report technical uncertainty and model limitations.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import tensorflow as tf

from src.agents.schemas import PredictionAgentOutput, DetectedCondition, ValveAttribution
from src.models.xai import explain_valve_predictions
from src.models.decision_fusion import build_rosanet_patient_model, build_rosanet_4valve_model

DISEASE_MAP = {
    "N": "Healthy Normal Auscultation Pattern",
    "AS": "Aortic Stenosis",
    "AR": "Aortic Regurgitation",
    "MR": "Mitral Regurgitation",
    "MS": "Mitral Stenosis"
}

DISEASE_KEYS = ["N", "AS", "AR", "MR", "MS"]

# Clinical decision thresholds derived from validation tuning
DEFAULT_THRESHOLDS = {
    "N": 0.52,
    "AS": 0.70,
    "AR": 0.48,
    "MR": 0.45,
    "MS": 0.50
}

def get_calibrated_thresholds() -> Dict[str, float]:
    """Loads calibrated thresholds from disk if available, otherwise returns defaults."""
    import json
    from pathlib import Path
    t_file = Path(__file__).resolve().parent.parent.parent / "dataset" / "models" / "calibrated_thresholds.json"
    if t_file.exists():
        try:
            with open(t_file, "r") as f:
                data = json.load(f)
            return data.get("calibrated_thresholds", DEFAULT_THRESHOLDS)
        except Exception:
            pass
    return dict(DEFAULT_THRESHOLDS)


class PredictionXAIAgent:
    """
    Automated deep learning inference and explainability agent.
    """

    def __init__(self, model_checkpoint_path: Optional[str] = None):
        self.model_checkpoint_path = model_checkpoint_path
        self.model: Optional[tf.keras.Model] = None
        self._load_model()

    def _load_model(self):
        from pathlib import Path
        from src.models.checkpoint_io import get_or_create_rosanet_model
        root = Path(__file__).resolve().parent.parent.parent
        four_valve_chk = root / "dataset" / "models" / "best_rosanet_4valve_model.keras"
        orig_chk = root / "dataset" / "models" / "best_rosanet_model.keras"
        if self.model_checkpoint_path:
            chk_path = Path(self.model_checkpoint_path)
        elif four_valve_chk.exists():
            chk_path = four_valve_chk
        else:
            chk_path = orig_chk
        self.model = get_or_create_rosanet_model(str(chk_path) if chk_path.exists() else None, num_branches=4)

    def predict_and_explain(
        self,
        spectrogram_dict: Dict[str, np.ndarray],  # maps 'Aor', 'Pul', 'Tri', 'Mit' -> (64, 626, 1)
        session_id: str = "SESSION_UNKNOWN",
        custom_thresholds: Optional[Dict[str, float]] = None
    ) -> PredictionAgentOutput:
        """
        Executes inference and explainability across the 4 physical valve inputs.
        """
        thresholds = get_calibrated_thresholds()
        if custom_thresholds:
            thresholds.update(custom_thresholds)

        # Ensure correct valve branch order: Aor, Mit, Pul, Tri
        ordered_valves = ['Aor', 'Mit', 'Pul', 'Tri']
        ordered_specs = [spectrogram_dict[v] for v in ordered_valves]
        ordered_names = [f"{v} Valve" for v in ordered_valves]

        # Batch inputs for model: list of 4 arrays, each shape (1, 64, 626, 1)
        batch_inputs = [np.expand_dims(spec, axis=0) for spec in ordered_specs]

        # Forward pass
        raw_preds = self.model.predict(batch_inputs, verbose=0)
        if isinstance(raw_preds, list):
            patient_probs = raw_preds[0][0]  # (5,)
            branch_probs = raw_preds[1][0] if len(raw_preds) > 1 else None
        else:
            patient_probs = raw_preds[0]
            branch_probs = None

        prob_dict = {
            DISEASE_KEYS[i]: round(float(patient_probs[i]), 4)
            for i in range(len(DISEASE_KEYS))
        }

        # Identify detected conditions based on decision thresholds
        detected_conditions: List[DetectedCondition] = []
        max_disease_prob = 0.0

        for key in ["AS", "AR", "MR", "MS"]:
            prob = prob_dict[key]
            thresh = thresholds[key]
            if prob > max_disease_prob:
                max_disease_prob = prob

            # Determine primary contributing valve
            if branch_probs is not None:
                c_idx = DISEASE_KEYS.index(key)
                best_v_idx = int(np.argmax(branch_probs[:, c_idx]))
                contrib_valve = ordered_names[best_v_idx]
            else:
                contrib_valve = "Aortic Valve" if key in ["AS", "AR"] else "Mitral Valve"

            if prob >= thresh:
                detected_conditions.append(DetectedCondition(
                    disease_code=key,
                    disease_name=DISEASE_MAP[key],
                    probability=prob,
                    threshold=thresh,
                    is_positive=True,
                    primary_contributing_valve=contrib_valve
                ))

        is_normal = (prob_dict["N"] >= thresholds["N"]) and len(detected_conditions) == 0

        if is_normal:
            overall_status = "Normal Auscultation Findings (No Primary Valvular Disease Detected)"
            risk_level = "Low Risk"
        elif len(detected_conditions) > 1:
            names = [c.disease_name for c in detected_conditions]
            overall_status = f"Elevated Multi-Valvular Risk: Combined {', '.join(names)}"
            risk_level = "High Risk"
        elif len(detected_conditions) == 1:
            c = detected_conditions[0]
            overall_status = f"Elevated Risk: Suspected {c.disease_name} (Primary focus: {c.primary_contributing_valve})"
            risk_level = "Moderate Risk" if c.probability < 0.70 else "High Risk"
        else:
            overall_status = "Borderline Auscultation Patterns (Sub-threshold Disease Probability)"
            risk_level = "Low-Moderate Risk"

        # Compute XAI explanations
        xai_res = explain_valve_predictions(
            self.model,
            ordered_specs,
            valve_names=ordered_names,
            disease_classes=DISEASE_KEYS,
            threshold=0.45
        )

        valve_attributions = xai_res.get("explanations", [])

        limitations = (
            "RosaNet outputs are probabilistic acoustic estimates based on phonocardiogram recordings. "
            "Acoustic murmurs can be altered by body habitus, ambient noise, and chest wall thickness. "
            "Attributions reflect neural network activations and are not definitive anatomical proof."
        )

        return PredictionAgentOutput(
            session_id=session_id,
            model_version="RosaNet-4Valve-v1.0",
            raw_probabilities=prob_dict,
            decision_thresholds=thresholds,
            detected_conditions=detected_conditions,
            is_normal=is_normal,
            overall_status=overall_status,
            risk_level=risk_level,
            valve_attributions=valve_attributions,
            model_limitations=limitations
        )
