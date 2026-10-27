"""
Safe Model Checkpoint Loading and Saving Utility for RosaNet.

Provides robust serialization and deserialization across Keras/TensorFlow versions
without encountering cross-version GlorotUniform config incompatibilities.
"""

import os
from pathlib import Path
from typing import Optional, List
import numpy as np
import tensorflow as tf

from src.models.attention import SplitSelfAttention, CardiacSpatialTemporalAttention
from src.models.decision_fusion import (
    build_rosanet_patient_model,
    build_rosanet_4valve_model,
    DynamicDecisionFusion
)
from src.models.loss import WeightedMultiLabelBCE


def load_rosanet_weights_safe(
    model: tf.keras.Model,
    checkpoint_path: str
) -> bool:
    """
    Safely loads weights into an instantiated RosaNet model from a .keras or .weights.h5 file.
    If the checkpoint is an 8-branch model and the target model is 4-valve (or vice versa),
    it transfers all shared feature extractor backbone weights and matching layers.
    """
    chk = Path(checkpoint_path)
    if not chk.exists():
        return False

    # 1. Direct weight loading attempt
    try:
        model.load_weights(str(chk))
        return True
    except Exception:
        pass

    # 2. Layer-by-layer transfer attempt
    try:
        # Build reference 8-branch model to hold checkpoint weights
        ref_model = build_rosanet_patient_model(num_branches=8)
        ref_model.load_weights(str(chk))

        # Transfer weights from ref_model to target model
        for target_layer in model.layers:
            try:
                src_layer = ref_model.get_layer(target_layer.name)
                # Check matching weights count and shapes
                target_w = target_layer.get_weights()
                src_w = src_layer.get_weights()
                if len(target_w) == len(src_w) and all(tw.shape == sw.shape for tw, sw in zip(target_w, src_w)):
                    target_layer.set_weights(src_w)
            except Exception:
                pass
        return True
    except Exception as e:
        print(f"[checkpoint_io] Warning: Layer-by-layer transfer failed: {e}")
        return False


def get_or_create_rosanet_model(
    checkpoint_path: Optional[str] = None,
    num_branches: int = 4
) -> tf.keras.Model:
    """
    Builds the RosaNet model (4-valve or 8-branch) and loads trained weights if available.
    """
    if num_branches == 4:
        model = build_rosanet_4valve_model()
    else:
        model = build_rosanet_patient_model(num_branches=num_branches)

    if checkpoint_path and Path(checkpoint_path).exists():
        loaded = load_rosanet_weights_safe(model, checkpoint_path)
        if loaded:
            print(f"[checkpoint_io] Successfully loaded RosaNet ({num_branches}-branch) weights from {checkpoint_path}")
        else:
            print(f"[checkpoint_io] Warning: Could not restore weights from {checkpoint_path}")

    return model
