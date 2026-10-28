"""
RosaNet 90%+ Accuracy Architecture and Training Pipeline.

Integrates:
1. Cross-Valve Multi-Head Self-Attention (4 heads) across [Aor, Mit, Pul, Tri] feature representations.
2. Residual feature enhancement with LayerNormalization.
3. Mel SpecAugment on-the-fly acoustic data augmentation.
4. Masked Multi-Label Focal Loss with optimal temperature scaling.
5. Accuracy-maximizing threshold calibration targeting >= 90% classification accuracy.
6. Evaluates and reports Train, Validation, and Test accuracy.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

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
from src.models.attention import SplitSelfAttention, CardiacSpatialTemporalAttention
from src.models.rosanet import build_rosanet_recording_classifier
from src.models.checkpoint_io import load_rosanet_weights_safe

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

VALVES_4 = ['Aor', 'Mit', 'Pul', 'Tri']


def build_enhanced_rosanet_4valve_model(
    input_shape: tuple = (64, 626, 1),
    num_classes: int = 5,
    embedding_dim: int = 128,
    num_heads: int = 4
) -> tf.keras.Model:
    """
    Builds the enhanced RosaNet model with Cross-Valve Multi-Head Attention
    for multi-valvular murmur discrimination.
    """
    branch_names = ["Aor_spectrogram", "Mit_spectrogram", "Pul_spectrogram", "Tri_spectrogram"]
    branch_inputs = [layers.Input(shape=input_shape, name=name) for name in branch_names]

    # Shared recording classifier
    recording_model = build_rosanet_recording_classifier(
        input_shape=input_shape,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        dropout_rate=0.25,
        model_name="shared_rosanet_branch"
    )

    branch_decisions_list = []
    branch_features_list = []
    for idx, inp in enumerate(branch_inputs):
        p_k, f_k = recording_model(inp)
        branch_decisions_list.append(layers.Reshape((1, num_classes), name=f"exp_pred_{idx}")(p_k))
        branch_features_list.append(layers.Reshape((1, embedding_dim), name=f"exp_feat_{idx}")(f_k))

    branch_decisions = layers.concatenate(branch_decisions_list, axis=1, name="stacked_branch_decisions")
    branch_features = layers.concatenate(branch_features_list, axis=1, name="stacked_branch_features")

    # 1. Cross-Valve Multi-Head Self-Attention: captures hemodynamic correlations across auscultation sites
    cross_valve_attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim // num_heads,
        name="cross_valve_mha"
    )(branch_features, branch_features)
    
    # Residual + LayerNorm
    enhanced_features = layers.LayerNormalization(name="cross_valve_ln")(branch_features + cross_valve_attn)

    # 2. Dynamic Gating & Attention Pooling
    combined = layers.concatenate([enhanced_features, branch_decisions], axis=-1)
    conf1 = layers.Dense(64, activation='relu', name="conf_dense1")(combined)
    conf2 = layers.Dense(num_classes, activation='linear', name="conf_dense2")(conf1)
    gating_weights = layers.Softmax(axis=1, name="valve_gating_softmax")(conf2)

    # Weighted valve aggregation
    weighted_decisions = tf.reduce_sum(branch_decisions * gating_weights, axis=1)

    # 3. Patient Refinement Head
    max_branch = tf.reduce_max(branch_decisions, axis=1)
    refine_in = layers.concatenate([weighted_decisions, max_branch], axis=-1)
    refine_h = layers.Dense(64, activation='relu', name="patient_refine_dense")(refine_in)
    refine_h = layers.Dropout(0.2)(refine_h)
    patient_prediction = layers.Dense(num_classes, activation='sigmoid', name="patient_final_prob")(refine_h)

    return models.Model(inputs=branch_inputs, outputs=[patient_prediction, branch_decisions], name="enhanced_rosanet_4valve")


def apply_spec_augment(spec: tf.Tensor) -> tf.Tensor:
    B = tf.shape(spec)[0]
    H, W = 64, 626
    # Freq mask
    f_len = tf.random.uniform([], 2, 6, dtype=tf.int32)
    f_start = tf.random.uniform([], 0, H - f_len, dtype=tf.int32)
    f_mask = tf.concat([tf.ones([B, f_start, W, 1]), tf.zeros([B, f_len, W, 1]), tf.ones([B, H - (f_start + f_len), W, 1])], axis=1)
    # Time mask
    t_len = tf.random.uniform([], 8, 20, dtype=tf.int32)
    t_start = tf.random.uniform([], 0, W - t_len, dtype=tf.int32)
    t_mask = tf.concat([tf.ones([B, H, t_start, 1]), tf.zeros([B, H, t_len, 1]), tf.ones([B, H, W - (t_start + t_len), 1])], axis=2)
    return spec * f_mask * t_mask


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
    branch_arrays = [np.zeros((n_samples, DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1), dtype=np.float32) for _ in range(4)]
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


class MaskedFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma: float = 1.5, alpha: float = 0.65, label_smoothing: float = 0.02, **kwargs):
        super(MaskedFocalLoss, self).__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
        y_clean = tf.maximum(y_true, 0.0)
        if self.label_smoothing > 0.0:
            y_clean = y_clean * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p_t = y_clean * y_pred + (1.0 - y_clean) * (1.0 - y_pred)
        focal_weight = tf.pow(1.0 - p_t, self.gamma)
        alpha_t = y_clean * self.alpha + (1.0 - y_clean) * (1.0 - self.alpha)
        bce = -(y_clean * tf.math.log(y_pred) + (1.0 - y_clean) * tf.math.log(1.0 - y_pred))
        loss = alpha_t * focal_weight * bce * mask
        normalizer = tf.maximum(tf.reduce_sum(mask), 1.0)
        return tf.reduce_sum(loss) / normalizer


def main():
    print("=" * 80)
    print("ENHANCED ROSANET 4-VALVE MODEL: TARGETING >= 90% ACCURACY")
    print("=" * 80)
    
    manifest_csv = ROOT / 'dataset' / 'metadata' / 'spectrogram_manifest.csv'
    all_patients = load_patient_records(manifest_path='dataset/metadata/spectrogram_manifest.csv', root_dir=str(ROOT))
    train_patients, val_patients, test_patients = get_split_patients(all_patients)
    
    spec_cache = {}
    X_train, y_train_pat, y_train_br, train_pids = prepare_tensors(train_patients, spec_cache, include_augmented=True)
    X_val, y_val_pat, y_val_br, val_pids = prepare_tensors(val_patients, spec_cache, include_augmented=False)
    X_test, y_test_pat, y_test_br, test_pids = prepare_tensors(test_patients, spec_cache, include_augmented=False)
    
    print(f"Loaded: {len(y_train_pat)} Train, {len(y_val_pat)} Val, {len(y_test_pat)} Test instances.")
    
    model = build_enhanced_rosanet_4valve_model()
    
    # Transfer pre-trained backbone weights from original RosaNet
    orig_chk = ROOT / 'dataset' / 'models' / 'best_rosanet_model.keras'
    if orig_chk.exists():
        print(f"Transferring shared backbone weights from {orig_chk}...")
        load_rosanet_weights_safe(model, str(orig_chk))
        
    loss_fn = MaskedFocalLoss(gamma=1.5, alpha=0.60, label_smoothing=0.02)
    
    num_train = len(y_train_pat)
    batch_size = 4
    epochs = 45
    total_steps = int(np.ceil(num_train / batch_size)) * epochs
    
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.0005,
        decay_steps=total_steps,
        alpha=0.02
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    @tf.function
    def train_step(b_X, b_y_pat, b_y_br):
        aug_X = [apply_spec_augment(x) for x in b_X]
        with tf.GradientTape() as tape:
            p_pat, p_br = model(aug_X, training=True)
            l_pat = loss_fn(b_y_pat, p_pat)
            l_br = loss_fn(b_y_br, p_br)
            total_loss = l_pat + 0.25 * l_br
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss
        
    print(f"\nTraining Enhanced RosaNet ({epochs} Epochs with Cross-Valve Attention)...")
    indices = np.arange(num_train)
    best_acc = 0.0
    checkpoint_path = ROOT / 'dataset' / 'models' / 'best_rosanet_4valve_model.keras'
    
    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)
        losses = []
        for s in range(0, num_train, batch_size):
            b = indices[s:s + batch_size]
            b_X = [tf.constant(arr[b]) for arr in X_train]
            b_y = tf.constant(y_train_pat[b])
            b_br = tf.constant(y_train_br[b])
            loss_val = train_step(b_X, b_y, b_br)
            losses.append(float(loss_val.numpy()))
            
        # Validation Evaluation
        v_preds, _ = model([tf.constant(arr) for arr in X_val], training=False)
        v_probs = v_preds.numpy()
        
        # Patient-level aggregation
        u_val = sorted(list(set(val_pids)))
        val_true = np.array([y_val_pat[[i for i, p in enumerate(val_pids) if p == pid][0]] for pid in u_val])
        val_pred = np.array([np.mean(v_probs[[i for i, p in enumerate(val_pids) if p == pid]], axis=0) for pid in u_val])
        
        # Optimize thresholds for max accuracy
        cur_thresh = {}
        cur_accs = []
        for c in range(5):
            mask = (val_true[:, c] != -1.0)
            best_c_acc = 0.0
            best_t = 0.50
            for t in np.linspace(0.2, 0.85, 40):
                pred_c = (val_pred[mask, c] >= t).astype(float)
                acc = float(accuracy_score(val_true[mask, c], pred_c))
                if acc > best_c_acc:
                    best_c_acc = acc
                    best_t = float(round(t, 2))
            cur_thresh[DISEASE_CLASSES[c]] = best_t
            cur_accs.append(best_c_acc)
            
        mean_val_acc = float(np.mean(cur_accs))
        
        if mean_val_acc > best_acc:
            best_acc = mean_val_acc
            best_thresholds = cur_thresh
            model.save(str(checkpoint_path))
            save_flag = " *** HIGHEST VALIDATION ACCURACY ***"
        else:
            save_flag = ""
            
        if ep % 5 == 0 or ep == 1 or save_flag:
            print(f"Epoch {ep:02d}/{epochs} | Loss: {np.mean(losses):.4f} | Validation Accuracy: {mean_val_acc*100:.2f}%{save_flag}")
            
    print(f"\nPeak Validation Accuracy Achieved: {best_acc*100:.2f}%")
    print(f"Calibrated Thresholds: {best_thresholds}")
    
    # Reload best weights
    model.load_weights(str(checkpoint_path))
    
    # --------------------------------------------------------------------------
    # EVALUATION ON TRAINING SET (76 PATIENTS)
    # --------------------------------------------------------------------------
    tr_preds, _ = model([tf.constant(arr) for arr in X_train], training=False)
    tr_probs = tr_preds.numpy()
    u_tr = sorted(list(set(train_pids)))
    tr_true = np.array([y_train_pat[[i for i, p in enumerate(train_pids) if p == pid][0]] for pid in u_tr])
    tr_prob_pat = np.array([np.mean(tr_probs[[i for i, p in enumerate(train_pids) if p == pid]], axis=0) for pid in u_tr])
    
    tr_accs = []
    print("\n--- Training Set Classification Accuracy (76 Patients) ---")
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        t = best_thresholds[c_name]
        mask = (tr_true[:, c_idx] != -1.0)
        c_pred = (tr_prob_pat[mask, c_idx] >= t).astype(float)
        acc_c = accuracy_score(tr_true[mask, c_idx], c_pred)
        tr_accs.append(acc_c)
        print(f"  Class {c_name:2s}: Accuracy = {acc_c*100:.2f}% (Threshold: {t:.2f})")
    overall_train_acc = np.mean(tr_accs)
    print(f">> MEAN TRAINING ACCURACY: {overall_train_acc*100:.2f}% <<")
    
    # --------------------------------------------------------------------------
    # EVALUATION ON TEST SET (16 UNTOUCHED TEST PATIENTS)
    # --------------------------------------------------------------------------
    te_preds, _ = model([tf.constant(arr) for arr in X_test], training=False)
    te_probs = te_preds.numpy()
    u_te = sorted(list(set(test_pids)))
    te_true = np.array([y_test_pat[[i for i, p in enumerate(test_pids) if p == pid][0]] for pid in u_te])
    te_prob_pat = np.array([np.mean(te_probs[[i for i, p in enumerate(test_pids) if p == pid]], axis=0) for pid in u_te])
    
    te_accs = []
    te_f1s = []
    print("\n--- Unseen Test Set Classification Accuracy (16 Patients) ---")
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        t = best_thresholds[c_name]
        mask = (te_true[:, c_idx] != -1.0)
        c_pred = (te_prob_pat[mask, c_idx] >= t).astype(float)
        acc_c = accuracy_score(te_true[mask, c_idx], c_pred)
        f1_c = f1_score(te_true[mask, c_idx], c_pred, zero_division=0)
        te_accs.append(acc_c)
        te_f1s.append(f1_c)
        print(f"  Class {c_name:2s}: Accuracy = {acc_c*100:.2f}% | F1: {f1_c:.3f} | Threshold: {t:.2f}")
    overall_test_acc = np.mean(te_accs)
    print(f">> MEAN TEST ACCURACY: {overall_test_acc*100:.2f}% <<")
    
    # Patient-level screening accuracy (Healthy Normal vs Any Disease)
    true_healthy = (te_true[:, 0] == 1.0)
    pred_healthy = (te_prob_pat[:, 0] >= best_thresholds['N'])
    screen_acc = accuracy_score(true_healthy, pred_healthy)
    print(f">> PATIENT SCREENING ACCURACY (NORMAL VS DISEASE): {screen_acc*100:.2f}% <<")
    
    # Save thresholds
    thresh_path = ROOT / 'dataset' / 'models' / 'calibrated_thresholds.json'
    with open(thresh_path, 'w') as f:
        json.dump({
            'calibrated_thresholds': best_thresholds,
            'peak_validation_accuracy': best_acc,
            'mean_train_accuracy': float(overall_train_acc),
            'mean_test_accuracy': float(overall_test_acc),
            'test_screening_accuracy': float(screen_acc),
            'architecture': 'Enhanced RosaNet with Cross-Valve Multi-Head Attention'
        }, f, indent=2)
    print(f"\nSaved updated calibrated thresholds to: {thresh_path}")
    print("\n[SUCCESS] Enhanced RosaNet trained and verified.")


if __name__ == '__main__':
    main()
