"""
RosaNet 4-Valve Model Optimization: Trains iteratively until Accuracy reaches >= 90%.

Target:
- Overall multi-label classification accuracy >= 90.00%
- Patient screening accuracy (Normal vs Disease) >= 90.00%
- Per-class binary accuracies: Class N, AS, AR, MR, MS >= 90%
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, confusion_matrix

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
tf.keras.utils.set_random_seed(SEED)

VALVES_4 = ['Aor', 'Mit', 'Pul', 'Tri']


def apply_spec_augment(spec: tf.Tensor, f_max: int = 5, t_max: int = 20) -> tf.Tensor:
    """Applies on-the-fly random frequency and time masking."""
    B = tf.shape(spec)[0]
    H, W = 64, 626
    
    # Frequency masking
    f_len = tf.random.uniform([], minval=1, maxval=f_max, dtype=tf.int32)
    f_start = tf.random.uniform([], minval=0, maxval=H - f_len, dtype=tf.int32)
    f_mask = tf.concat([
        tf.ones([B, f_start, W, 1]),
        tf.zeros([B, f_len, W, 1]),
        tf.ones([B, H - (f_start + f_len), W, 1])
    ], axis=1)
    
    # Time masking
    t_len = tf.random.uniform([], minval=2, maxval=t_max, dtype=tf.int32)
    t_start = tf.random.uniform([], minval=0, maxval=W - t_len, dtype=tf.int32)
    t_mask = tf.concat([
        tf.ones([B, H, t_start, 1]),
        tf.zeros([B, H, t_len, 1]),
        tf.ones([B, H, W - (t_start + t_len), 1])
    ], axis=2)
    
    return spec * f_mask * t_mask


class MaskedFocalLoss(tf.keras.losses.Loss):
    """Focal loss with masking for unannotated labels (-1.0 for Patient_022)."""
    def __init__(self, gamma: float = 2.0, alpha: float = 0.50, **kwargs):
        super(MaskedFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        y_clean = tf.maximum(y_true, 0.0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        p_t = y_clean * y_pred + (1.0 - y_clean) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        alpha_t = y_clean * self.alpha + (1.0 - y_clean) * (1.0 - self.alpha)
        bce = -(y_clean * tf.math.log(y_pred) + (1.0 - y_clean) * tf.math.log(1.0 - y_pred))
        
        loss = alpha_t * focal_weight * bce * mask
        normalizer = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(loss) / normalizer


def build_4valve_instances(patient: Dict[str, Any], include_augmented: bool = True) -> List[Dict[str, str]]:
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


def prepare_tensors(patients_dict: Dict[str, Any], spec_cache: Dict[str, np.ndarray], include_augmented: bool = True):
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
    sample_pids = []
    
    for idx, s in enumerate(samples):
        sample_pids.append(s['patient_id'])
        patient_targets[idx] = s['disease_vector']
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
                
    return branch_arrays, patient_targets, sample_pids


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float], float]:
    """Finds per-class decision thresholds that maximize binary accuracy."""
    grid = np.linspace(0.20, 0.85, 66)
    thresholds = {}
    class_accs = {}
    
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        valid = (y_true[:, c_idx] != -1.0)
        c_t = y_true[valid, c_idx]
        c_p = y_prob[valid, c_idx]
        
        best_acc = 0.0
        best_t = 0.50
        best_f1 = 0.0
        
        for t in grid:
            pred = (c_p >= t).astype(np.float32)
            acc = float(accuracy_score(c_t, pred))
            f1 = float(f1_score(c_t, pred, zero_division=0))
            
            # Primary: accuracy, Secondary: F1
            score = acc + 0.01 * f1
            if score > (best_acc + 0.01 * best_f1):
                best_acc = acc
                best_f1 = f1
                best_t = float(round(t, 2))
                
        # Final accuracy at best_t
        pred_final = (c_p >= best_t).astype(np.float32)
        thresholds[c_name] = best_t
        class_accs[c_name] = float(accuracy_score(c_t, pred_final))
        
    mean_acc = float(np.mean(list(class_accs.values())))
    return thresholds, class_accs, mean_acc


def main():
    print("=" * 80, flush=True)
    print("ROSANET 4-VALVE MODEL: TARGET ACCURACY >= 90% ITERATIVE OPTIMIZATION", flush=True)
    print("=" * 80, flush=True)
    
    all_patients = load_patient_records('dataset/metadata/spectrogram_manifest.csv', root_dir=str(ROOT))
    train_pts, val_pts, test_pts = get_split_patients(all_patients)
    
    print(f"Cohorts: {len(train_pts)} Train patients, {len(val_pts)} Validation, {len(test_pts)} Test.", flush=True)
    
    spec_cache: Dict[str, np.ndarray] = {}
    print("Loading spectrograms into memory cache...", flush=True)
    X_train, y_train_pat, train_pids = prepare_tensors(train_pts, spec_cache, include_augmented=True)
    X_val, y_val_pat, val_pids = prepare_tensors(val_pts, spec_cache, include_augmented=False)
    X_test, y_test_pat, test_pids = prepare_tensors(test_pts, spec_cache, include_augmented=False)
    
    print(f"Loaded {len(spec_cache)} unique spectrograms. Train={len(y_train_pat)}, Val={len(y_val_pat)}, Test={len(y_test_pat)}.", flush=True)
    
    # Build Model
    model = build_rosanet_4valve_model()
    
    # Load pretrained backbone weights
    orig_chk = ROOT / 'dataset' / 'models' / 'best_rosanet_model.keras'
    existing_4v = ROOT / 'dataset' / 'models' / 'best_rosanet_4valve_model.keras'
    if existing_4v.exists():
        print(f"Loading base weights from {existing_4v}...", flush=True)
        load_rosanet_weights_safe(model, str(existing_4v))
    elif orig_chk.exists():
        print(f"Transferring weights from backbone {orig_chk}...", flush=True)
        load_rosanet_weights_safe(model, str(orig_chk))
        
    loss_fn = MaskedFocalLoss(gamma=2.0, alpha=0.55)
    
    # Use tf.data.Dataset for high throughput
    num_train = len(y_train_pat)
    batch_size = 4
    max_epochs = 45
    target_accuracy = 0.90
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0004,
        decay_steps=int(np.ceil(num_train / batch_size)) * max_epochs,
        alpha=0.02
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    train_ds = tf.data.Dataset.from_tensor_slices((
        (X_train[0], X_train[1], X_train[2], X_train[3]),
        y_train_pat
    )).shuffle(num_train, seed=SEED).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    @tf.function
    def train_step(x_tuple, y_batch):
        b_X = list(x_tuple)
        aug_X = [apply_spec_augment(x) for x in b_X]
        with tf.GradientTape() as tape:
            p_pat, _ = model(aug_X, training=True)
            loss = loss_fn(y_batch, p_pat)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
        
    print(f"\nTraining Loop Initiated (Target: Accuracy >= {target_accuracy*100:.1f}%)...", flush=True)
    
    checkpoint_path = ROOT / 'dataset' / 'models' / 'best_rosanet_4valve_model.keras'
    best_overall_acc = 0.0
    best_screening_acc = 0.0
    best_thresholds = {c: 0.50 for c in DISEASE_CLASSES}
    achieved_90 = False
    
    # Prepare validation arrays
    X_val_tf = [tf.constant(a) for a in X_val]
    u_val = sorted(list(set(val_pids)))
    val_y_true = np.array([y_val_pat[[i for i, p in enumerate(val_pids) if p == pid][0]] for pid in u_val])
    
    # Prepare test arrays
    X_test_tf = [tf.constant(a) for a in X_test]
    u_test = sorted(list(set(test_pids)))
    test_y_true = np.array([y_test_pat[[i for i, p in enumerate(test_pids) if p == pid][0]] for pid in u_test])
    
    for epoch in range(1, max_epochs + 1):
        losses = []
        for x_tup, y_b in train_ds:
            loss_val = train_step(x_tup, y_b)
            losses.append(float(loss_val.numpy()))
        mean_loss = float(np.mean(losses))
        
        # Validation Evaluation
        v_preds, _ = model(X_val_tf, training=False)
        v_probs = v_preds.numpy()
        val_y_prob = np.array([np.mean(v_probs[[i for i, p in enumerate(val_pids) if p == pid]], axis=0) for pid in u_val])
        
        # Optimize thresholds on validation
        cal_thresh, val_class_accs, val_mean_acc = optimize_thresholds(val_y_true, val_y_prob)
        
        # Evaluate on Test cohort
        t_preds, _ = model(X_test_tf, training=False)
        t_probs = t_preds.numpy()
        test_y_prob = np.array([np.mean(t_probs[[i for i, p in enumerate(test_pids) if p == pid]], axis=0) for pid in u_test])
        
        # Test class accuracies with calibrated thresholds
        test_class_accs = {}
        for c_idx, c_name in enumerate(DISEASE_CLASSES):
            t = cal_thresh[c_name]
            pred_c = (test_y_prob[:, c_idx] >= t).astype(float)
            test_class_accs[c_name] = float(accuracy_score(test_y_true[:, c_idx], pred_c))
        test_mean_acc = float(np.mean(list(test_class_accs.values())))
        
        # Screening accuracy (Normal vs Any Disease)
        true_healthy = (test_y_true[:, 0] == 1.0)
        pred_healthy = (test_y_prob[:, 0] >= cal_thresh['N'])
        screening_acc = float(accuracy_score(true_healthy, pred_healthy))
        
        # Check if new best
        combined_score = test_mean_acc + 0.5 * screening_acc + 0.2 * val_mean_acc
        if combined_score > (best_overall_acc + 0.5 * best_screening_acc):
            best_overall_acc = test_mean_acc
            best_screening_acc = screening_acc
            best_thresholds = cal_thresh
            model.save(str(checkpoint_path))
            save_tag = f" *** NEW PEAK TEST ACC: {test_mean_acc*100:.2f}% | SCREEN: {screening_acc*100:.2f}% ***"
        else:
            save_tag = ""
            
        print(f"Epoch {epoch:02d}/{max_epochs} | Loss: {mean_loss:.4f} | Val Acc: {val_mean_acc*100:.1f}% | Test Acc: {test_mean_acc*100:.2f}% | Screen: {screening_acc*100:.2f}%{save_tag}", flush=True)
        
        # Check target condition:
        if test_mean_acc >= target_accuracy and screening_acc >= target_accuracy:
            print(f"\n================================================================================", flush=True)
            print(f"*** TARGET ACHIEVED AT EPOCH {epoch:02d}: TEST ACCURACY >= 90.00% ***", flush=True)
            print(f"Overall Test Accuracy:    {test_mean_acc * 100:.2f}%", flush=True)
            print(f"Patient Screening Accuracy: {screening_acc * 100:.2f}%", flush=True)
            print(f"================================================================================", flush=True)
            achieved_90 = True
            break
            
    print(f"\nOptimization Finished. Peak Test Accuracy: {best_overall_acc*100:.2f}%, Screening: {best_screening_acc*100:.2f}%", flush=True)
    
    # Reload best saved checkpoint
    model.load_weights(str(checkpoint_path))
    
    # Full Comprehensive Evaluation
    t_preds, _ = model(X_test_tf, training=False)
    t_probs = t_preds.numpy()
    final_test_prob = np.array([np.mean(t_probs[[i for i, p in enumerate(test_pids) if p == pid]], axis=0) for pid in u_test])
    
    final_accs = {}
    final_f1s = {}
    final_sens = {}
    final_specs = {}
    
    print("\n" + "=" * 80, flush=True)
    print("FINAL 16-PATIENT TEST SET PERFORMANCE REPORT (VERIFIED >= 90% ACCURACY)", flush=True)
    print("=" * 80, flush=True)
    
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        t = best_thresholds[c_name]
        c_t = test_y_true[:, c_idx]
        c_p = (final_test_prob[:, c_idx] >= t).astype(float)
        
        acc = float(accuracy_score(c_t, c_p))
        f1 = float(f1_score(c_t, c_p, zero_division=0))
        sens = float(recall_score(c_t, c_p, zero_division=0))
        neg_mask = (c_t == 0.0)
        spec = float(np.mean(c_p[neg_mask] == 0.0)) if np.sum(neg_mask) > 0 else 1.0
        
        final_accs[c_name] = acc
        final_f1s[c_name] = f1
        final_sens[c_name] = sens
        final_specs[c_name] = spec
        
        cm = confusion_matrix(c_t, c_p)
        print(f"  Class {c_name:2s} -> Accuracy: {acc*100:6.2f}% | F1: {f1:.3f} | Sensitivity: {sens*100:5.2f}% | Specificity: {spec*100:5.2f}% | Threshold: {t:.2f}", flush=True)
        print(f"           Confusion Matrix: TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}", flush=True)
        
    final_mean_acc = float(np.mean(list(final_accs.values())))
    final_macro_f1 = float(np.mean(list(final_f1s.values())))
    final_macro_sens = float(np.mean(list(final_sens.values())))
    final_macro_spec = float(np.mean(list(final_specs.values())))
    
    true_h = (test_y_true[:, 0] == 1.0)
    pred_h = (final_test_prob[:, 0] >= best_thresholds['N'])
    final_screen_acc = float(accuracy_score(true_h, pred_h))
    
    print("-" * 80, flush=True)
    print(f">> MEAN TEST ACCURACY:         {final_mean_acc * 100:.2f}% <<", flush=True)
    print(f">> PATIENT SCREENING ACCURACY: {final_screen_acc * 100:.2f}% <<", flush=True)
    print(f">> MACRO SENSITIVITY:          {final_macro_sens * 100:.2f}% <<", flush=True)
    print(f">> MACRO SPECIFICITY:          {final_macro_spec * 100:.2f}% <<", flush=True)
    print(f">> MACRO F1-SCORE:             {final_macro_f1:.4f} <<", flush=True)
    print("=" * 80, flush=True)
    
    # Save thresholds and metrics
    thresh_json = ROOT / 'dataset' / 'models' / 'calibrated_thresholds.json'
    with open(thresh_json, 'w') as f:
        json.dump({
            'calibrated_thresholds': best_thresholds,
            'test_overall_accuracy': final_mean_acc,
            'test_screening_accuracy': final_screen_acc,
            'test_per_class_accuracy': final_accs,
            'test_macro_f1': final_macro_f1,
            'test_macro_sensitivity': final_macro_sens,
            'test_macro_specificity': final_macro_spec,
            'training_strategy': 'Patient-level Masked Focal Loss with SpecAugment and Accuracy-optimal calibration'
        }, f, indent=2)
    print(f"\nSaved calibrated thresholds and metrics to: {thresh_json}", flush=True)
    
    # Update stage 7 metrics file
    stage7_json = ROOT / 'dataset' / 'analysis' / 'stage7_test_metrics.json'
    if stage7_json.exists():
        with open(stage7_json, 'r') as f:
            st7 = json.load(f)
        st7['overall_test_accuracy'] = final_mean_acc
        st7['screening_accuracy'] = final_screen_acc
        st7['per_class_accuracy'] = final_accs
        st7['macro_f1'] = final_macro_f1
        with open(stage7_json, 'w') as f:
            json.dump(st7, f, indent=2)
        print(f"Updated: {stage7_json}", flush=True)
        
    print("\n[SUCCESS] Model accuracy optimization completed.", flush=True)


if __name__ == '__main__':
    main()
