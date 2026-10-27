"""
Explainable AI (XAI) Module for RosaNet Heart Sound Auscultation.

Implements:
1. Grad-CAM for Log-Mel Spectrograms:
   Computes class-specific convolutional feature attribution maps highlighting
   which time-frequency segments (e.g. systolic ejection clicks, diastolic rumbles)
   drove the model's prediction.
2. Valve-Level Diagnostic Contribution Analysis:
   Extracts branch confidence weights from the Dynamic Decision Fusion layer
   to quantify the relative diagnostic importance of each auscultation site.
3. Clinical Attribution Safeguards:
   Clearly distinguishes model feature attribution from confirmed clinical causation.
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import tensorflow as tf
from scipy.ndimage import zoom


def compute_gradcam_spectrogram(
    recording_model: tf.keras.Model,
    spectrogram_tensor: np.ndarray,
    target_class_idx: int,
    conv_layer_name: str = "block4_conv1"
) -> np.ndarray:
    """
    Computes 2D Grad-CAM saliency map for a single log-mel spectrogram input.
    
    Args:
        recording_model: Keras model with shared_rosanet_branch architecture
        spectrogram_tensor: Shape (64, 626, 1) or (1, 64, 626, 1)
        target_class_idx: Integer class index in [0..4] (0: N, 1: AS, 2: AR, 3: MR, 4: MS)
        conv_layer_name: Target deep convolutional layer name
        
    Returns:
        Saliency heatmap array of shape (64, 626) normalized to [0.0, 1.0].
    """
    if spectrogram_tensor.ndim == 3:
        input_batch = np.expand_dims(spectrogram_tensor, axis=0)
    else:
        input_batch = spectrogram_tensor

    input_batch = tf.cast(input_batch, tf.float32)

    # Find the target conv layer
    try:
        conv_layer = recording_model.get_layer(conv_layer_name)
    except ValueError:
        # Fallback to any suitable Conv2D layer in the model
        conv_layers = [l for l in recording_model.layers if isinstance(l, tf.keras.layers.Conv2D)]
        if conv_layers:
            conv_layer = conv_layers[-1]
            conv_layer_name = conv_layer.name
        else:
            return np.zeros((64, 626), dtype=np.float32)

    # Construct sub-model mapping input -> [conv_features, predictions]
    grad_model = tf.keras.models.Model(
        inputs=recording_model.inputs,
        outputs=[conv_layer.output, recording_model.outputs[0]]
    )

    with tf.GradientTape() as tape:
        tape.watch(input_batch)
        conv_outputs, predictions = grad_model(input_batch, training=False)
        loss = predictions[:, target_class_idx]

    # Gradients of target class score with respect to conv feature map
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return np.zeros((64, 626), dtype=np.float32)

    # Global Average Pooling of gradients across spatial/temporal dimensions
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (filters,)

    # Weight feature channels by pooled gradients
    conv_outputs_val = conv_outputs[0]  # shape: (H, W, filters)
    heatmap = tf.reduce_sum(conv_outputs_val * pooled_grads, axis=-1)  # shape: (H, W)

    # Apply ReLU: keep only positive influence
    heatmap = tf.maximum(heatmap, 0.0).numpy()

    # Normalize heatmap
    max_val = np.max(heatmap)
    if max_val > 1e-7:
        heatmap = heatmap / max_val
    else:
        heatmap = np.zeros_like(heatmap)

    # Upsample heatmap to target spectrogram dimensions (64, 626)
    target_h, target_w = 64, 626
    curr_h, curr_w = heatmap.shape
    if curr_h > 0 and curr_w > 0 and (curr_h != target_h or curr_w != target_w):
        zoom_factors = (target_h / curr_h, target_w / curr_w)
        heatmap = zoom(heatmap, zoom_factors, order=1)
        heatmap = np.clip(heatmap, 0.0, 1.0)

    return heatmap.astype(np.float32)


def explain_valve_predictions(
    model: tf.keras.Model,
    valve_spectrograms: List[np.ndarray],
    valve_names: List[str] = None,
    disease_classes: List[str] = None,
    threshold: float = 0.45
) -> Dict[str, Any]:
    """
    Generates explainability report for a patient's valve auscultation set.
    
    Args:
        model: Trained RosaNet model (4-valve or 8-branch)
        valve_spectrograms: List of 4 or 8 spectrogram arrays, each (64, 626, 1)
        valve_names: Names corresponding to branches (e.g. ['Aortic', 'Mitral', 'Pulmonary', 'Tricuspid'])
        disease_classes: Class names ['N', 'AS', 'AR', 'MR', 'MS']
        threshold: Decision threshold for reporting
        
    Returns:
        Structured dictionary with predictions, valve contributions, and Grad-CAM heatmaps.
    """
    if disease_classes is None:
        disease_classes = ['N', 'AS', 'AR', 'MR', 'MS']
    if valve_names is None:
        if len(valve_spectrograms) == 4:
            valve_names = ['Aortic (Aor)', 'Mitral (Mit)', 'Pulmonary (Pul)', 'Tricuspid (Tri)']
        else:
            valve_names = [f'Branch {i+1}' for i in range(len(valve_spectrograms))]

    # Prepare inputs as batch of size 1
    branch_batches = [np.expand_dims(spec, axis=0) for spec in valve_spectrograms]

    preds = model.predict(branch_batches, verbose=0)
    if isinstance(preds, list):
        patient_probs = preds[0][0]  # (5,)
        branch_preds = preds[1][0] if len(preds) > 1 else None  # (K, 5)
    else:
        patient_probs = preds[0]
        branch_preds = None

    # Identify shared recording classifier backbone for Grad-CAM
    recording_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "shared_rosanet" in layer.name:
            recording_model = layer
            break

    explanations = []
    prob_dict = {disease_classes[i]: float(patient_probs[i]) for i in range(len(disease_classes))}

    # Explain each detected condition or top condition
    for c_idx, c_name in enumerate(disease_classes):
        prob = float(patient_probs[c_idx])
        if prob >= threshold or (c_name == 'N' and prob >= 0.50):
            valve_attributions = []
            
            for b_idx, (spec, v_name) in enumerate(zip(valve_spectrograms, valve_names)):
                # 1. Grad-CAM saliency
                if recording_model is not None:
                    cam = compute_gradcam_spectrogram(
                        recording_model, spec, target_class_idx=c_idx, conv_layer_name="block4_conv1"
                    )
                else:
                    cam = np.zeros((64, 626), dtype=np.float32)

                # 2. Branch score if available
                br_score = float(branch_preds[b_idx, c_idx]) if branch_preds is not None else prob

                valve_attributions.append({
                    "valve_name": v_name,
                    "branch_score": round(br_score, 4),
                    "peak_saliency": round(float(np.max(cam)), 4),
                    "mean_saliency": round(float(np.mean(cam)), 4),
                    "heatmap": cam.tolist()  # (64, 626) array
                })

            # Sort valves by diagnostic relevance
            valve_attributions.sort(key=lambda v: (v["branch_score"], v["mean_saliency"]), reverse=True)

            explanations.append({
                "disease_class": c_name,
                "probability": round(prob, 4),
                "is_active": True,
                "top_contributing_valve": valve_attributions[0]["valve_name"] if valve_attributions else "Unknown",
                "valve_attributions": valve_attributions
            })

    return {
        "probabilities": prob_dict,
        "explanations": explanations,
        "disclaimer": (
            "Model saliency heatmaps indicate regions of convolutional activation during inference, "
            "not clinical causation. They should be reviewed by a qualified specialist alongside echocardiography."
        )
    }
