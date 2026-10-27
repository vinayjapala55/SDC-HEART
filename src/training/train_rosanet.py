"""
RosaNet Training Configuration and Multi-Label Pipeline Specification.

Note: In compliance with Step 3.2 instructions and the strict STOP condition,
this module defines loss formulations, metric evaluation functions, optimizer
configurations, and callback pipelines, but DOES NOT execute training loops.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import tensorflow as tf
from tensorflow.keras import losses, metrics, optimizers, callbacks

from src.models.decision_fusion import build_rosanet_patient_model
from src.data.dataset_loader import PatientMultiRecordingSequence
from src.data.patient_grouping import DISEASE_CLASSES


# Default Training Configuration Parameters
TRAINING_CONFIG: Dict[str, Any] = {
    "random_seed": 42,
    "batch_size": 4,                    # Patient-level batch size (4 patients x 8 branches = 32 spectrograms/batch)
    "learning_rate": 1e-3,              # Initial Adam learning rate
    "min_learning_rate": 1e-6,          # Minimum learning rate for scheduler
    "num_epochs": 50,                   # Target training epochs
    "optimizer": "Adam",
    "adam_beta_1": 0.9,
    "adam_beta_2": 0.999,
    "adam_epsilon": 1e-7,
    "loss_formulation": "Multi-Label Binary Cross-Entropy (Independent Sigmoid)",
    "loss_weights": {
        "dynamic_decision_fusion": 1.0, # Primary patient-level decision loss weight
        "stacked_branch_decisions": 0.2 # Auxiliary branch-level decision supervision weight
    },
    "early_stopping_patience": 15,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "checkpoint_monitor": "val_dynamic_decision_fusion_macro_auroc",
    "checkpoint_mode": "max",
    "target_classes": DISEASE_CLASSES   # ['N', 'AS', 'AR', 'MR', 'MS']
}


def compute_multilabel_class_weights(
    train_patients: Dict[str, Dict[str, Any]]
) -> np.ndarray:
    """
    Computes positive class weight ratios for the 5 multi-label disease targets
    based on training split patient distribution, strictly excluding unknown (-1.0) labels.
    
    Formula:
        pos_weight_c = N_neg_c / N_pos_c (over confirmed known labels)
    """
    targets = np.array([p['disease_vector'] for p in train_patients.values()], dtype=np.float32)
    pos_weights = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
    
    for c_idx in range(len(DISEASE_CLASSES)):
        valid_targets = targets[targets[:, c_idx] != -1.0, c_idx]
        pos_count = np.sum(valid_targets == 1.0)
        neg_count = np.sum(valid_targets == 0.0)
        
        if pos_count > 0:
            pos_weights[c_idx] = neg_count / pos_count
        else:
            pos_weights[c_idx] = 1.0
            
    return pos_weights


def get_multilabel_loss(
    pos_weights: Optional[np.ndarray] = None
) -> Dict[str, tf.keras.losses.Loss]:
    """
    Returns Masked and Weighted Multi-Label Binary Cross-Entropy losses for patient prediction
    and auxiliary branch predictions, strictly ignoring unannotated (-1.0) targets.
    """
    from src.models.loss import MaskedWeightedMultiLabelBCE
    patient_loss = MaskedWeightedMultiLabelBCE(
        pos_weights=pos_weights.tolist() if pos_weights is not None else None,
        name="patient_masked_bce_loss"
    )
    branch_loss = MaskedWeightedMultiLabelBCE(
        pos_weights=pos_weights.tolist() if pos_weights is not None else None,
        name="branch_masked_bce_loss"
    )
    
    return {
        "dynamic_decision_fusion": patient_loss,
        "stacked_branch_decisions": branch_loss
    }


def get_multilabel_metrics() -> List[tf.keras.metrics.Metric]:
    """
    Returns multi-label clinical classification metrics.
    Tracks Macro AUROC, Macro AUPRC (PR-AUC), Precision, Recall, and Binary Accuracy.
    """
    return [
        metrics.BinaryAccuracy(name="binary_acc"),
        metrics.AUC(curve='ROC', multi_label=True, name="macro_auroc"),
        metrics.AUC(curve='PR', multi_label=True, name="macro_auprc"),
        metrics.Precision(name="precision"),
        metrics.Recall(name="recall"),
    ]


def get_training_callbacks(
    checkpoint_dir: str = "dataset/models",
    monitor_metric: str = "val_dynamic_decision_fusion_macro_auroc"
) -> List[tf.keras.callbacks.Callback]:
    """
    Constructs training callbacks for early stopping, model checkpointing, and learning rate decay.
    """
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    cb_list = [
        callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode="max",
            patience=TRAINING_CONFIG["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=f"{checkpoint_dir}/best_rosanet_model.keras",
            monitor=monitor_metric,
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor=monitor_metric,
            mode="max",
            factor=TRAINING_CONFIG["reduce_lr_factor"],
            patience=TRAINING_CONFIG["reduce_lr_patience"],
            min_lr=TRAINING_CONFIG["min_learning_rate"],
            verbose=1
        )
    ]
    return cb_list


def build_and_compile_training_model(
    learning_rate: float = 1e-3,
    input_shape: tuple = (64, 626, 1),
    num_classes: int = 5,
    embedding_dim: int = 128
) -> tf.keras.Model:
    """
    Instantiates and compiles the RosaNet Patient Decision Fusion model.
    """
    # Set global random seed
    tf.keras.utils.set_random_seed(TRAINING_CONFIG["random_seed"])

    model = build_rosanet_patient_model(
        input_shape=input_shape,
        num_branches=8,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout_rate=0.3
    )

    loss_dict = get_multilabel_loss()
    metric_list = get_multilabel_metrics()

    optimizer = optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=TRAINING_CONFIG["adam_beta_1"],
        beta_2=TRAINING_CONFIG["adam_beta_2"],
        epsilon=TRAINING_CONFIG["adam_epsilon"]
    )

    model.compile(
        optimizer=optimizer,
        loss=loss_dict,
        loss_weights=TRAINING_CONFIG["loss_weights"],
        metrics={
            "dynamic_decision_fusion": metric_list
        }
    )

    return model
