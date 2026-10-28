"""
High-Accuracy RosaNet 4-Valve Model Training and Optimization.

Optimizes RosaNet 4-valve model accuracy to reach >= 90% via:
1. Graph-compiled @tf.function training step for maximum CPU throughput.
2. On-the-fly Mel SpecAugment (Frequency & Temporal acoustic masking).
3. Masked Focal Multi-Label Cross-Entropy Loss (gamma=1.5, label smoothing=0.02).
4. Cosine Decay Learning Rate Scheduler.
5. In-memory caching of spectrogram representations.
6. Direct per-class accuracy threshold calibration targeting >= 90% binary accuracy.
7. Evaluation on Train, Validation, and Test cohorts.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data.patient_grouping import (
    load_patient_records,
    get_split_patients,
    DISEASE_CLASSES,
    VALVE_LOCATIONS
)
from src.data.dataset_loader import (
    DEFAULT_FREQ_BINS,
    DEFAULT_TIME_STEPS,
    load_and_standardize_spectrogram
)
from src.models.decision_fusion import build_rosanet_4valve_model
from src.models.checkpoint_io import load_rosanet_weights_safe

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

VALVES_4 = ['Aor', 'Mit', 'Pul', 'Tri']


def apply_spec_augment(
    spec: tf.Tensor,
    freq_mask_max: int = 6,
    time_mask_max: int = 24
) -> tf.Tensor:
    """Applies on-the-fly random frequency and time masking to Mel spectrograms."""
    # spec shape: (B, 64, 626, 1)
    B = tf.shape(spec)[0]
    H = 64
    W = 626
    
    # Frequency mask: zero out horizontal stripe
    f_len = tf.random.uniform([], minval=1, maxval=freq_mask_max, dtype=tf.int32)
    f_start = tf.random.uniform([], minval=0, maxval=H - f_len, dtype=tf.int32)
    f_mask = tf.concat([
        tf.ones([B, f_start, W, 1]),
        tf.zeros([B, f_len, W, 1]),
        tf.ones([B, H - (f_start + f_len), W, 1])
    ], axis=1)
    
    # Time mask: zero out vertical stripe
    t_len = tf.random.uniform([], minval=1, maxval=time_mask_max, dtype=tf.int32)
    t_start = tf.random.uniform([], minval=0, maxval=W - t_len, dtype=tf.int32)
    t_mask = tf.concat([
        tf.ones([B, H, t_start, 1]),
        tf.zeros([B, H, t_len, 1]),
        tf.ones([B, H, W - (t_start + t_len), 1])
    ], axis=2)
    
    return spec * f_mask * t_mask


class MaskedFocalMultiLabelLoss(tf.keras.losses.Loss):
    """
    Masked Multi-Label Focal Loss with Label Smoothing.
    Focuses on hard examples and masks unannotated (-1.0) entries.
    """
    def __init__(self, gamma: float = 1.5, alpha: float = 0.65, label_smoothing: float = 0.02, **kwargs):
        super(MaskedFocalMultiLabelLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        y_clean = tf.maximum(y_true, 0.0)
        
        # Label smoothing
        if self.label_smoothing > 0.0:
            y_clean = y_clean * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Focal weights
        p_t = y_clean * y_pred + (1.0 - y_clean) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        
        alpha_t = y_clean * self.alpha + (1.0 - y_clean) * (1.0 - self.alpha)
        bce = -(y_clean * tf.math.log(y_pred) + (1.0 - y_clean) * tf.math.log(1.0 - y_pred))
        
        loss = alpha_t * focal_weight * bce * mask
        normalizer = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(loss) / normalizer


def build_4valve_instances(
    patient: Dict[str, Any],
    include_augmented: bool = True
) -> List[Dict[str, str]]:
    recs = patient['recordings']
    orig_sit, orig_sup, aug_sit, aug_sup = {}, {}, {}, {}
    
    for r in recs:
        v = r['valve_location']
        post = r['posture']
        is_aug = (r['original_or_augmented'] == 'augmented')
        path = r['numerical_path']
        if not is_aug:
            if post in ['sit', 'sitting']:
                orig_sit[v] = path
            elif post in ['sup', 'supine']:
                orig_sup[v] = path
        else:
            if post in ['sit', 'sitting']:
                aug_sit[v] = path
            elif post in ['sup', 'supine']:
                aug_sup[v] = path
                
    instances = []
    if all(v in orig_sit for v in VALVES_4):
        instances.append(dict(orig_sit))
    elif len(orig_sit) > 0:
        instances.append({v: orig_sit.get(v, orig_sup.get(v, '')) for v in VALVES_4})
        
    if all(v in orig_sup for v in VALVES_4):
        instances.append(dict(orig_sup))
    elif len(orig_sup) > 0:
        instances.append({v: orig_sup.get(v, orig_sit.get(v, '')) for v in VALVES_4})
        
    if include_augmented and len(aug_sit) > 0:
        instances.append({v: aug_sit.get(v, orig_sit.get(v, '')) for v in VALVES_4})
    if include_augmented and len(aug_sup) > 0:
        instances.append({v: aug_sup.get(v, orig_sup.get(v, '')) for v in VALVES_4})
        
    return instances


def prepare_tensors(
    patients_dict: Dict[str, Any],
    spec_cache: Dict[str, np.ndarray],
    include_augmented: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    samples = []
    for pid, pdata in patients_dict.items():
        insts = build_4valve_instances(pdata, include_augmented=include_augmented)
        for inst in insts:
            samples.append({'patient_id': pid, 'disease_vector': pdata['disease_vector'], 'valves': inst})
            
    n_samples = len(samples)
    branch_arrays = [
        np.zeros((n_samples, DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1), dtype=np.float32)
        for _ in range(4)
    ]
    patient_targets = np.zeros((n_samples, len(DISEASE_CLASSES)), dtype=np.float32)
    branch_targets = np.zeros((n_samples, 4, len(DISEASE_CLASSES)), dtype=np.float32)
    sample_pids = []
    
    for idx, s in enumerate(samples):
        sample_pids.append(s['patient_id'])
        patient_targets[idx] = s['disease_vector']
        for b in range(4):
            branch_targets[idx, b] = s['disease_vector']
        v_paths = s['valves']
        for v_idx, v_name in enumerate(VALVES_4):
            npy_path = v_paths.get(v_name, '')
            if npy_path and npy_path in spec_cache:
                branch_arrays[v_idx][idx] = spec_cache[npy_path]
            elif npy_path:
                arr = load_and_standardize_spectrogram(npy_path, root_dir=str(ROOT))
                spec_cache[npy_path] = arr
                branch_arrays[v_idx][idx] = arr
            else:
                branch_arrays[v_idx][idx] = np.zeros((DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1), dtype=np.float32)
                
    return branch_arrays, patient_targets, branch_targets, sample_pids


def optimize_thresholds_for_target_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_accuracy: float = 0.90
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Finds decision thresholds that directly maximize overall classification accuracy,
    aiming to reach >= target_accuracy across classes.
    """
    threshold_grid = np.linspace(0.20, 0.85, 66)
    best_thresholds = {}
    per_class_acc = {}
    
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        valid = (y_true[:, c_idx] != -1.0)
        c_true = y_true[valid, c_idx]
        c_prob = y_prob[valid, c_idx]
        
        best_acc = 0.0
        best_t = 0.50
        
        for t in threshold_grid:
            c_pred = (c_prob >= t).astype(np.float32)
            acc = float(accuracy_score(c_true, c_pred))
            f1 = float(f1_score(c_true, c_pred, zero_division=0))
            
            # Primary criterion: binary accuracy, secondary: F1 tie-breaker
            score = acc + 0.01 * f1
            if score > best_acc:
                best_acc = score
                best_t = float(round(t, 2))
                
        # Compute true accuracy at best_t
        c_pred_final = (c_prob >= best_t).astype(np.float32)
        final_acc = float(accuracy_score(c_true, c_pred_final))
        best_thresholds[c_name] = best_t
        per_class_acc[c_name] = final_acc
        
    overall_acc = float(np.mean(list(per_class_acc.values())))
    return best_thresholds, overall_acc, per_class_acc


def main():
    print("=" * 75)
    print("HIGH-ACCURACY ROSANET 4-VALVE MODEL TRAINING & OPTIMIZATION")
    print("TARGET: ACCURACY >= 90%")
    print("=" * 75)
    
    manifest_csv = ROOT / 'dataset' / 'metadata' / 'spectrogram_manifest.csv'
    all_patients = load_patient_records(manifest_path='dataset/metadata/spectrogram_manifest.csv', root_dir=str(ROOT))
    train_patients, val_patients, test_patients = get_split_patients(all_patients)
    
    print(f"Dataset Splits: {len(train_patients)} Train, {len(val_patients)} Val, {len(test_patients)} Test patients.")
    
    spec_cache: Dict[str, np.ndarray] = {}
    X_train, y_train_pat, y_train_br, train_pids = prepare_tensors(train_patients, spec_cache, include_augmented=True)
    X_val, y_val_pat, y_val_br, val_pids = prepare_tensors(val_patients, spec_cache, include_augmented=False)
    X_test, y_test_pat, y_test_br, test_pids = prepare_tensors(test_patients, spec_cache, include_augmented=False)
    
    print(f"Total Cached Spectrograms in RAM: {len(spec_cache)}")
    print(f"Training Instances: {len(y_train_pat)}, Validation Instances: {len(y_val_pat)}, Test Instances: {len(y_test_pat)}")
    
    # Build Model
    model = build_rosanet_4valve_model()
    
    # Load pretrained weights from existing checkpoint if available
    chk_existing = ROOT / 'dataset' / 'models' / 'best_rosanet_4valve_model.keras'
    if chk_existing.exists():
        print(f"Initializing from trained 4-valve weights: {chk_existing}...")
        load_rosanet_weights_safe(model, str(chk_existing))
    else:
        orig = ROOT / 'dataset' / 'models' / 'best_rosanet_model.keras'
        if orig.exists():
            print(f"Transferring weights from backbone: {orig}...")
            load_rosanet_weights_safe(model, str(orig))
            
    loss_fn = MaskedFocalMultiLabelLoss(gamma=1.5, alpha=0.60, label_smoothing=0.02)
    
    # Cosine decay schedule for smooth convergence
    num_train = len(y_train_pat)
    batch_size = 4
    epochs = 40
    steps_per_epoch = int(np.ceil(num_train / batch_size))
    total_steps = steps_per_epoch * epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0004,
        decay_steps=total_steps,
        alpha=0.05
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    # Compiled @tf.function train step for rapid training
    @tf.function
    def train_step(b_X, b_y_pat, b_y_br):
        # Apply SpecAugment during training
        augmented_inputs = [apply_spec_augment(inp) for inp in b_X]
        with tf.GradientTape() as tape:
            p_pat, p_br = model(augmented_inputs, training=True)
            loss_pat = loss_fn(b_y_pat, p_pat)
            loss_br = loss_fn(b_y_br, p_br)
            total_loss = loss_pat + 0.2 * loss_br
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss
        
    print("\nStarting Accelerated Training Loop (40 Epochs with SpecAugment & Focal Loss)...")
    indices = np.arange(num_train)
    best_val_score = 0.0
    best_overall_acc = 0.0
    best_weights = None
    
    checkpoint_out = ROOT / 'dataset' / 'models' / 'best_rosanet_4valve_model.keras'
    
    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)
        epoch_losses = []
        
        for s_idx in range(0, num_train, batch_size):
            b_idx = indices[s_idx:s_idx + batch_size]
            b_X = [tf.constant(arr[b_idx]) for arr in X_train]
            b_y_pat = tf.constant(y_train_pat[b_idx])
            b_y_br = tf.constant(y_train_br[b_idx])
            loss_val = train_step(b_X, b_y_pat, b_y_br)
            epoch_losses.append(float(loss_val.numpy()))
            
        mean_train_loss = float(np.mean(epoch_losses))
        
        # Validation Evaluation
        val_X_tf = [tf.constant(arr) for arr in X_val]
        val_preds_pat, _ = model(val_X_tf, training=False)
        val_probs = val_preds_pat.numpy()
        
        # Patient-level aggregation on validation
        unique_val_pids = sorted(list(set(val_pids)))
        val_pat_y_true = []
        val_pat_y_prob = []
        for pid in unique_val_pids:
            p_indices = [i for i, p in enumerate(val_pids) if p == pid]
            val_pat_y_true.append(y_val_pat[p_indices[0]])
            val_pat_y_prob.append(np.mean(val_probs[p_indices], axis=0))
        val_pat_y_true = np.array(val_pat_y_true)
        val_pat_y_prob = np.array(val_pat_y_prob)
        
        # Optimize thresholds and evaluate accuracy
        cal_t, val_acc, per_class_acc = optimize_thresholds_for_target_accuracy(val_pat_y_true, val_pat_y_prob)
        
        # Also compute Macro AUROC
        roc_list = []
        for c in range(5):
            v_idx = (val_pat_y_true[:, c] != -1.0)
            if np.sum(val_pat_y_true[v_idx, c] == 1) > 0 and np.sum(val_pat_y_true[v_idx, c] == 0) > 0:
                roc_list.append(roc_auc_score(val_pat_y_true[v_idx, c], val_pat_y_prob[v_idx, c]))
            else:
                roc_list.append(0.5)
        val_auroc = float(np.mean(roc_list))
        
        # Track best model based on joint accuracy and AUROC
        score = val_acc + 0.5 * val_auroc
        if score > best_val_score:
            best_val_score = score
            best_overall_acc = val_acc
            best_thresholds = cal_t
            best_per_class = per_class_acc
            model.save(str(checkpoint_out))
            save_star = " *** NEW BEST ACCURACY ***"
        else:
            save_star = ""
            
        if ep % 5 == 0 or ep == 1 or save_star:
            print(f"Epoch {ep:02d}/{epochs} | Train Loss: {mean_train_loss:.4f} | Val Accuracy: {val_acc*100:.2f}% | Val AUROC: {val_auroc:.4f}{save_star}")
            
    print(f"\nTraining Complete. Peak Validation Accuracy: {best_overall_acc * 100:.2f}%")
    print(f"Selected Best Thresholds: {best_thresholds}")
    for c, acc in best_per_class.items():
        print(f"  Class {c:2s} -> Threshold: {best_thresholds[c]:.2f} | Accuracy: {acc * 100:.2f}%")
        
    # Reload best checkpoint
    model.load_weights(str(checkpoint_out))
    
    # --------------------------------------------------------------------------
    # FULL EVALUATION ON UNTOUCHED TEST SET
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("FINAL TEST SET ACCURACY & PERFORMANCE EVALUATION (16 TEST PATIENTS)")
    print("=" * 75)
    
    test_X_tf = [tf.constant(arr) for arr in X_test]
    test_preds_pat, _ = model(test_X_tf, training=False)
    test_probs_inst = test_preds_pat.numpy()
    
    unique_test_pids = sorted(list(set(test_pids)))
    test_pat_y_true = []
    test_pat_y_prob = []
    for pid in unique_test_pids:
        p_indices = [i for i, p in enumerate(test_pids) if p == pid]
        test_pat_y_true.append(y_test_pat[p_indices[0]])
        test_pat_y_prob.append(np.mean(test_probs_inst[p_indices], axis=0))
    test_pat_y_true = np.array(test_pat_y_true)
    test_pat_y_prob = np.array(test_pat_y_prob)
    
    # Evaluate with calibrated thresholds
    test_pred_bin = np.zeros_like(test_pat_y_prob)
    per_class_test_acc = {}
    per_class_test_f1 = {}
    per_class_test_sens = {}
    per_class_test_spec = {}
    
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        t = best_thresholds.get(c_name, 0.50)
        c_true = test_pat_y_true[:, c_idx]
        c_pred = (test_pat_y_prob[:, c_idx] >= t).astype(np.float32)
        test_pred_bin[:, c_idx] = c_pred
        
        acc_c = float(accuracy_score(c_true, c_pred))
        f1_c = float(f1_score(c_true, c_pred, zero_division=0))
        rec_c = float(recall_score(c_true, c_pred, zero_division=0))
        
        # Specificity
        neg_idx = (c_true == 0)
        spec_c = float(np.mean(c_pred[neg_idx] == 0)) if np.sum(neg_idx) > 0 else 1.0
        
        per_class_test_acc[c_name] = acc_c
        per_class_test_f1[c_name] = f1_c
        per_class_test_sens[c_name] = rec_c
        per_class_test_spec[c_name] = spec_c
        
    overall_test_acc = float(np.mean(list(per_class_test_acc.values())))
    overall_test_f1 = float(np.mean(list(per_class_test_f1.values())))
    overall_test_sens = float(np.mean(list(per_class_test_sens.values())))
    overall_test_spec = float(np.mean(list(per_class_test_spec.values())))
    
    # Patient-level Disease vs Normal Screening Accuracy
    true_has_disease = (test_pat_y_true[:, 0] == 0)  # Class 0 is Normal
    pred_has_disease = (test_pred_bin[:, 0] == 0)
    screening_acc = float(np.mean(true_has_disease == pred_has_disease))
    
    print(f"\n>>> OVERALL TEST ACCURACY (MEAN BINARY ACCURACY): {overall_test_acc * 100:.2f}% <<<")
    print(f">>> PATIENT SCREENING ACCURACY (NORMAL VS DISEASE): {screening_acc * 100:.2f}% <<<")
    print(f"Macro F1 Score:       {overall_test_f1:.4f}")
    print(f"Macro Sensitivity:    {overall_test_sens * 100:.2f}%")
    print(f"Macro Specificity:    {overall_test_spec * 100:.2f}%")
    print("\nPer-Class Breakdown on Unseen Test Cohort:")
    for c in DISEASE_CLASSES:
        print(f"  Class {c:2s} -> Accuracy: {per_class_test_acc[c]*100:5.2f}% | F1: {per_class_test_f1[c]:.3f} | Sensitivity: {per_class_test_sens[c]*100:5.2f}% | Specificity: {per_class_test_spec[c]*100:5.2f}% | Threshold: {best_thresholds[c]:.2f}")
        
    # Save updated thresholds and metrics
    thresh_path = ROOT / 'dataset' / 'models' / 'calibrated_thresholds.json'
    with open(thresh_path, 'w') as f:
        json.dump({
            'calibrated_thresholds': best_thresholds,
            'peak_validation_accuracy': best_overall_acc,
            'test_overall_accuracy': overall_test_acc,
            'test_screening_accuracy': screening_acc,
            'test_per_class_accuracy': per_class_test_acc,
            'test_macro_f1': overall_test_f1,
            'test_macro_sensitivity': overall_test_sens,
            'test_macro_specificity': overall_test_spec
        }, f, indent=2)
    print(f"\nSaved optimized thresholds to: {thresh_path}")
    print("\n[SUCCESS] Model training and accuracy optimization completed.")


if __name__ == '__main__':
    main()
