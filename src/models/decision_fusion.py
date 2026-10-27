"""
Decision Fusion Module and Patient-Level RosaNet Architecture.

Conceptual Flow:
========================================================================================
Branch 1 (Aor) ───► RosaNet ──► Attention ──► Branch Decision 1 (5-D) ──┐
Branch 2 (Mit) ───► RosaNet ──► Attention ──► Branch Decision 2 (5-D) ──┤
Branch 3 (Pul) ───► RosaNet ──► Attention ──► Branch Decision 3 (5-D) ──┼─► DECISION FUSION
Branch 4 (Tri) ───► RosaNet ──► Attention ──► Branch Decision 4 (5-D) ──┘        │
                                                                                 ▼
                                                                      PATIENT-LEVEL MULTI-LABEL
                                                                [ P(N), P(AS), P(AR), P(MR), P(MS) ]
========================================================================================

Architecture Details:
1. Multi-Branch Input: 4 physical valve auscultation sites (Aor, Mit, Pul, Tri)
   OR 8 physical sites (4 valves x 2 postures: sitting + supine).
2. Shared RosaNet + Attention Feature Extractor per branch.
3. Individual Branch Decision Heads: Produce 5-D sigmoid probabilities [P(N), P(AS), P(AR), P(MR), P(MS)].
4. Learnable Decision Fusion:
   - Dynamic branch confidence weighting computes relative diagnostic importance of each valve site.
5. Final Patient Decision: Aggregated patient-level multi-label probabilities bounded in [0, 1]^5.
"""

from typing import List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models

from src.models.rosanet import build_rosanet_backbone, build_rosanet_recording_classifier


@tf.keras.utils.register_keras_serializable(package="RosaNet")
class DynamicDecisionFusion(layers.Layer):
    """
    Learnable Dynamic Decision Fusion Layer.
    
    Takes:
    - branch_decisions: (B, num_branches, num_classes=5) -> individual recording disease probabilities
    - branch_features: (B, num_branches, embedding_dim=128) -> recording-level acoustic feature representations
    
    Computes:
    - branch_confidence_weights: Softmax across the branches for each disease class
    - patient_decision: Sum_k (branch_confidence_weights_k * branch_decisions_k)
    """
    def __init__(self, num_branches: int = 8, num_classes: int = 5, **kwargs):
        super(DynamicDecisionFusion, self).__init__(**kwargs)
        self.num_branches = num_branches
        self.num_classes = num_classes

        # Gating network to compute valve diagnostic confidence weights
        self.confidence_dense1 = layers.Dense(32, activation='relu', name='fusion_conf_dense1')
        self.confidence_dense2 = layers.Dense(num_classes, activation='linear', name='fusion_conf_dense2')
        self.softmax = layers.Softmax(axis=1, name='fusion_branch_softmax')
        
        # Patient refinement head
        self.refine_dense = layers.Dense(32, activation='relu', name='fusion_refine_dense')
        self.final_sigmoid = layers.Dense(num_classes, activation='sigmoid', name='patient_final_prob')

    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor], training=None) -> tf.Tensor:
        branch_decisions, branch_features = inputs

        # Concatenate decisions and features for each branch -> (B, num_branches, 128 + 5 = 133)
        combined_branch_info = layers.concatenate([branch_features, branch_decisions], axis=-1)

        # Compute branch-level confidence scores -> (B, num_branches, 5)
        conf_hidden = self.confidence_dense1(combined_branch_info)
        conf_logits = self.confidence_dense2(conf_hidden)
        
        # Softmax over branches (axis=1) -> weights sum to 1.0 across valve positions for each disease
        branch_weights = self.softmax(conf_logits)

        # Weighted Decision Aggregation -> (B, 5)
        weighted_decisions = tf.reduce_sum(branch_decisions * branch_weights, axis=1)

        # Non-linear clinical interaction refinement head
        refine_in = layers.concatenate([weighted_decisions, tf.reduce_max(branch_decisions, axis=1)], axis=-1)
        refine_act = self.refine_dense(refine_in)
        final_patient_prediction = self.final_sigmoid(refine_act)

        return final_patient_prediction

    def get_config(self):
        config = super(DynamicDecisionFusion, self).get_config()
        config.update({
            'num_branches': self.num_branches,
            'num_classes': self.num_classes
        })
        return config


def build_rosanet_patient_model(
    input_shape: tuple = (64, 626, 1),
    num_branches: int = 8,
    num_classes: int = 5,
    embedding_dim: int = 128,
    dropout_rate: float = 0.3,
    model_name: str = "rosanet_patient_decision_fusion"
) -> tf.keras.Model:
    """
    Builds the complete end-to-end Patient-Level RosaNet model with Attention and Decision Fusion.
    Supports either:
    - num_branches = 4: 4 anatomical valve locations [Aor, Mit, Pul, Tri]
    - num_branches = 8: 8 valve/posture combinations (4 valves x 2 postures)
    """
    if num_branches == 4:
        branch_names = [
            "Aor_spectrogram",
            "Mit_spectrogram",
            "Pul_spectrogram",
            "Tri_spectrogram",
        ]
    elif num_branches == 8:
        branch_names = [
            "sit_Aor_spectrogram",
            "sit_Mit_spectrogram",
            "sit_Pul_spectrogram",
            "sit_Tri_spectrogram",
            "sup_Aor_spectrogram",
            "sup_Mit_spectrogram",
            "sup_Pul_spectrogram",
            "sup_Tri_spectrogram",
        ]
    else:
        branch_names = [f"branch_{i}_spectrogram" for i in range(num_branches)]
    
    # 1. Create distinct physical recording inputs
    branch_inputs = [
        layers.Input(shape=input_shape, name=name)
        for name in branch_names
    ]

    # 2. Instantiate shared RosaNet + Attention recording classifier
    recording_model = build_rosanet_recording_classifier(
        input_shape=input_shape,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        model_name="shared_rosanet_branch"
    )

    # 3. Process each branch through RosaNet + Attention
    branch_decisions_list = []
    branch_features_list = []
    
    for idx, inp in enumerate(branch_inputs):
        pred_k, feat_k = recording_model(inp)
        pred_k_exp = layers.Reshape((1, num_classes), name=f"expand_pred_{idx}")(pred_k)
        feat_k_exp = layers.Reshape((1, embedding_dim), name=f"expand_feat_{idx}")(feat_k)
        branch_decisions_list.append(pred_k_exp)
        branch_features_list.append(feat_k_exp)

    # 4. Stack branch outputs -> (B, num_branches, 5) and (B, num_branches, 128)
    branch_decisions_stacked = layers.concatenate(branch_decisions_list, axis=1, name="stacked_branch_decisions")
    branch_features_stacked = layers.concatenate(branch_features_list, axis=1, name="stacked_branch_features")

    # 5. Apply Dynamic Decision Fusion
    fusion_layer = DynamicDecisionFusion(
        num_branches=num_branches,
        num_classes=num_classes,
        name="dynamic_decision_fusion"
    )
    patient_prediction = fusion_layer([branch_decisions_stacked, branch_features_stacked])

    # 6. Assemble Patient-Level Model
    patient_model = models.Model(
        inputs=branch_inputs,
        outputs=[patient_prediction, branch_decisions_stacked],
        name=model_name
    )

    return patient_model


def build_rosanet_4valve_model(
    input_shape: tuple = (64, 626, 1),
    num_classes: int = 5,
    embedding_dim: int = 128,
    dropout_rate: float = 0.3,
    model_name: str = "rosanet_4valve_decision_fusion"
) -> tf.keras.Model:
    """
    Convenience constructor for the dedicated 4-Valve RosaNet Model [Aor, Mit, Pul, Tri].
    """
    return build_rosanet_patient_model(
        input_shape=input_shape,
        num_branches=4,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout_rate=dropout_rate,
        model_name=model_name
    )
