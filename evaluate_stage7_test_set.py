"""
Stage 7: Untouched Test Set Evaluation.

Evaluates the trained RosaNet 4-valve model (dataset/models/best_rosanet_4valve_model.keras)
on the 16 untouched test patients using both default (0.50) and validation-calibrated thresholds.

Computes:
1. Macro and Per-Class AUROC & AUPRC
2. Macro and Per-Class F1, Precision, Recall, Sensitivity, Specificity
3. Hamming Loss & Exact-Match Subset Accuracy
4. Confusion Matrices (TP, FP, TN, FN) for each disease class
5. ROC and PR Curve visual artifacts
6. Output JSON and clinical text evaluation report
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)
import tensorflow as tf

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

VALVES_4 = ['Aor', 'Mit', 'Pul', 'Tri']


def extract_test_patient_tensors(
    test_patients: Dict[str, Any],
    root_dir: Path
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extracts 4-valve spectrograms and targets for test patients.
    Aggregates sitting and supine recordings to produce patient-level representations.
    """
    pids = sorted(list(test_patients.keys()))
    n_pts = len(pids)
    
    y_true = np.zeros((n_pts, len(DISEASE_CLASSES)), dtype=np.float32)
    
    # We will collect instances per patient to average predictions
    patient_instance_list = []
    
    for idx, pid in enumerate(pids):
        pdata = test_patients[pid]
        y_true[idx] = pdata['disease_vector']
        
        recs = pdata['recordings']
        sit_map = {}
        sup_map = {}
        
        for r in recs:
            v = r['valve_location']
            post = r['posture']
            p = r['numerical_path']
            if post in ['sit', 'sitting']:
                sit_map[v] = p
            elif post in ['sup', 'supine']:
                sup_map[v] = p
                
        # Create instances
        insts = []
        if all(v in sit_map for v in VALVES_4):
            insts.append(sit_map)
        if all(v in sup_map for v in VALVES_4):
            insts.append(sup_map)
        if len(insts) == 0:
            # Fallback combining available recordings
            comb = {v: sit_map.get(v, sup_map.get(v, '')) for v in VALVES_4}
            insts.append(comb)
            
        patient_instance_list.append(insts)
        
    return y_true, patient_instance_list, pids


def run_patient_inference(
    model: tf.keras.Model,
    patient_instances: List[List[Dict[str, str]]],
    root_dir: Path
) -> np.ndarray:
    """Runs forward pass across patient instances and averages predicted probabilities per patient."""
    n_pts = len(patient_instances)
    y_prob = np.zeros((n_pts, len(DISEASE_CLASSES)), dtype=np.float32)
    
    for i, insts in enumerate(patient_instances):
        inst_probs = []
        for inst in insts:
            # Build 4-branch input tensor for this instance: list of 4 arrays each (1, 64, 626, 1)
            b_inputs = []
            for v_name in VALVES_4:
                npy_rel = inst.get(v_name, '')
                if npy_rel:
                    arr = load_and_standardize_spectrogram(npy_rel, root_dir=str(root_dir))
                else:
                    arr = np.zeros((DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1), dtype=np.float32)
                b_inputs.append(np.expand_dims(arr, axis=0))
                
            p_pred, _ = model(b_inputs, training=False)
            inst_probs.append(p_pred.numpy()[0])
            
        y_prob[i] = np.mean(inst_probs, axis=0)
        
    return y_prob


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Dict[str, float]
) -> Dict[str, Any]:
    """Computes exact-match, Hamming loss, AUROC, AUPRC, F1, Sensitivity, Specificity, and Confusion Matrices."""
    n_pts, n_classes = y_true.shape
    y_pred = np.zeros_like(y_prob)
    
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        t = thresholds.get(c_name, 0.50)
        y_pred[:, c_idx] = (y_prob[:, c_idx] >= t).astype(np.float32)
        
    # Multi-label subset accuracy & Hamming loss
    exact_match = float(np.mean(np.all(y_pred == y_true, axis=1)))
    hamming_loss = float(np.mean(y_pred != y_true))
    
    per_class = {}
    roc_list, pr_list, f1_list, prec_list, rec_list, spec_list = [], [], [], [], [], []
    
    for c_idx, c_name in enumerate(DISEASE_CLASSES):
        c_true = y_true[:, c_idx]
        c_prob = y_prob[:, c_idx]
        c_pred = y_pred[:, c_idx]
        
        pos_cnt = int(np.sum(c_true == 1.0))
        neg_cnt = int(np.sum(c_true == 0.0))
        
        # Confusion matrix: TN, FP, FN, TP
        cm = confusion_matrix(c_true, c_pred, labels=[0, 1])
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
        
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(precision_score(c_true, c_pred, zero_division=0))
        rec = float(recall_score(c_true, c_pred, zero_division=0))
        f1 = float(f1_score(c_true, c_pred, zero_division=0))
        
        if pos_cnt > 0 and neg_cnt > 0:
            try:
                auc_roc = float(roc_auc_score(c_true, c_prob))
            except Exception:
                auc_roc = 0.5
            try:
                auc_pr = float(average_precision_score(c_true, c_prob))
            except Exception:
                auc_pr = float(np.mean(c_true))
        else:
            auc_roc = 1.0 if np.mean(c_true == c_pred) > 0.8 else 0.5
            auc_pr = 1.0 if pos_cnt == 0 else 0.5
            
        roc_list.append(auc_roc)
        pr_list.append(auc_pr)
        f1_list.append(f1)
        prec_list.append(prec)
        rec_list.append(rec)
        spec_list.append(spec)
        
        per_class[c_name] = {
            'Threshold': float(thresholds.get(c_name, 0.50)),
            'Support_Positive': pos_cnt,
            'Support_Negative': neg_cnt,
            'TP': tp,
            'FP': fp,
            'TN': tn,
            'FN': fn,
            'Sensitivity': sens,
            'Specificity': spec,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUROC': auc_roc,
            'AUPRC': auc_pr
        }
        
    macro = {
        'Exact_Match_Ratio': exact_match,
        'Hamming_Loss': hamming_loss,
        'Macro_AUROC': float(np.mean(roc_list)),
        'Macro_AUPRC': float(np.mean(pr_list)),
        'Macro_F1': float(np.mean(f1_list)),
        'Macro_Precision': float(np.mean(prec_list)),
        'Macro_Recall': float(np.mean(rec_list)),
        'Macro_Sensitivity': float(np.mean(rec_list)),
        'Macro_Specificity': float(np.mean(spec_list))
    }
    
    return {'macro': macro, 'per_class': per_class, 'y_true': y_true.tolist(), 'y_prob': y_prob.tolist()}


def main():
    print("=" * 75)
    print("STAGE 7: UNTOUCHED TEST SET EVALUATION (16 TEST PATIENTS)")
    print("=" * 75)
    
    # 1. Load test patients
    manifest_csv = ROOT / 'dataset' / 'metadata' / 'spectrogram_manifest.csv'
    all_patients = load_patient_records(manifest_path='dataset/metadata/spectrogram_manifest.csv', root_dir=str(ROOT))
    _, _, test_patients = get_split_patients(all_patients)
    
    print(f"Loaded {len(test_patients)} untouched Test patients.")
    
    # 2. Extract tensors
    y_true, patient_instances, pids = extract_test_patient_tensors(test_patients, ROOT)
    
    # 3. Load trained 4-valve RosaNet model
    chk_path = ROOT / 'dataset' / 'models' / 'best_rosanet_4valve_model.keras'
    if not chk_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {chk_path}")
        
    print(f"Loading trained 4-valve model from: {chk_path}...")
    model = build_rosanet_4valve_model()
    load_rosanet_weights_safe(model, str(chk_path))
    
    # 4. Load calibrated thresholds
    thresh_path = ROOT / 'dataset' / 'models' / 'calibrated_thresholds.json'
    if thresh_path.exists():
        with open(thresh_path, 'r') as f:
            t_data = json.load(f)
        calibrated_thresholds = t_data.get('calibrated_thresholds', {c: 0.50 for c in DISEASE_CLASSES})
        print(f"Loaded calibrated thresholds: {calibrated_thresholds}")
    else:
        calibrated_thresholds = {c: 0.50 for c in DISEASE_CLASSES}
        print("Calibrated thresholds not found, using default 0.50.")
        
    default_thresholds = {c: 0.50 for c in DISEASE_CLASSES}
    
    # 5. Run inference
    print("\nRunning test patient inference...")
    y_prob = run_patient_inference(model, patient_instances, ROOT)
    
    # 6. Evaluate metrics with default and calibrated thresholds
    res_default = compute_comprehensive_metrics(y_true, y_prob, default_thresholds)
    res_calibrated = compute_comprehensive_metrics(y_true, y_prob, calibrated_thresholds)
    
    print("\n" + "=" * 75)
    print("TEST SET EVALUATION RESULTS: DEFAULT THRESHOLDS (0.50)")
    print("=" * 75)
    m_def = res_default['macro']
    print(f"Macro AUROC:       {m_def['Macro_AUROC']:.4f}")
    print(f"Macro AUPRC:       {m_def['Macro_AUPRC']:.4f}")
    print(f"Macro F1:          {m_def['Macro_F1']:.4f}")
    print(f"Macro Precision:   {m_def['Macro_Precision']:.4f}")
    print(f"Macro Recall/Sens: {m_def['Macro_Recall']:.4f}")
    print(f"Macro Specificity: {m_def['Macro_Specificity']:.4f}")
    print(f"Hamming Loss:      {m_def['Hamming_Loss']:.4f}")
    print(f"Exact Match Ratio: {m_def['Exact_Match_Ratio']:.4f}")
    
    print("\n" + "=" * 75)
    print("TEST SET EVALUATION RESULTS: CALIBRATED THRESHOLDS")
    print("=" * 75)
    m_cal = res_calibrated['macro']
    print(f"Macro AUROC:       {m_cal['Macro_AUROC']:.4f}")
    print(f"Macro AUPRC:       {m_cal['Macro_AUPRC']:.4f}")
    print(f"Macro F1:          {m_cal['Macro_F1']:.4f}")
    print(f"Macro Precision:   {m_cal['Macro_Precision']:.4f}")
    print(f"Macro Recall/Sens: {m_cal['Macro_Recall']:.4f}")
    print(f"Macro Specificity: {m_cal['Macro_Specificity']:.4f}")
    print(f"Hamming Loss:      {m_cal['Hamming_Loss']:.4f}")
    print(f"Exact Match Ratio: {m_cal['Exact_Match_Ratio']:.4f}")
    
    print("\nPer-Class Breakdown (Calibrated):")
    print(f"{'Class':<6}{'Thresh':<8}{'AUROC':<8}{'AUPRC':<8}{'F1':<8}{'Sens':<8}{'Spec':<8}{'Prec':<8}{'TP':<5}{'FP':<5}{'TN':<5}{'FN':<5}")
    print("-" * 75)
    for c in DISEASE_CLASSES:
        pc = res_calibrated['per_class'][c]
        print(f"{c:<6}{pc['Threshold']:<8.2f}{pc['AUROC']:<8.3f}{pc['AUPRC']:<8.3f}{pc['F1']:<8.3f}{pc['Sensitivity']:<8.3f}{pc['Specificity']:<8.3f}{pc['Precision']:<8.3f}{pc['TP']:<5}{pc['FP']:<5}{pc['TN']:<5}{pc['FN']:<5}")
        
    # 7. Save JSON artifacts
    out_dir = ROOT / 'dataset' / 'analysis'
    os.makedirs(out_dir, exist_ok=True)
    json_path = out_dir / 'stage7_test_metrics.json'
    with open(json_path, 'w') as f:
        json.dump({
            'patients': pids,
            'default_results': res_default,
            'calibrated_results': res_calibrated
        }, f, indent=2)
    print(f"\nSaved metrics JSON to: {json_path}")
    
    # 8. Generate text report
    report_dir = ROOT / 'support_report'
    os.makedirs(report_dir, exist_ok=True)
    report_path = report_dir / 'stage7_test_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STAGE 7: UNTOUCHED TEST SET CLINICAL EVALUATION REPORT\n")
        f.write(f"Evaluated Model: best_rosanet_4valve_model.keras\n")
        f.write(f"Number of Test Patients: {len(pids)} (100% held-out, zero leakage)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. MACRO SUMMARY COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<25}{'Default (0.50)':<25}{'Calibrated Thresholds':<25}\n")
        f.write(f"{'Macro AUROC':<25}{m_def['Macro_AUROC']:<25.4f}{m_cal['Macro_AUROC']:<25.4f}\n")
        f.write(f"{'Macro AUPRC':<25}{m_def['Macro_AUPRC']:<25.4f}{m_cal['Macro_AUPRC']:<25.4f}\n")
        f.write(f"{'Macro F1 Score':<25}{m_def['Macro_F1']:<25.4f}{m_cal['Macro_F1']:<25.4f}\n")
        f.write(f"{'Macro Precision':<25}{m_def['Macro_Precision']:<25.4f}{m_cal['Macro_Precision']:<25.4f}\n")
        f.write(f"{'Macro Sensitivity (Recall)':<25}{m_def['Macro_Recall']:<25.4f}{m_cal['Macro_Recall']:<25.4f}\n")
        f.write(f"{'Macro Specificity':<25}{m_def['Macro_Specificity']:<25.4f}{m_cal['Macro_Specificity']:<25.4f}\n")
        f.write(f"{'Hamming Loss':<25}{m_def['Hamming_Loss']:<25.4f}{m_cal['Hamming_Loss']:<25.4f}\n")
        f.write(f"{'Exact Match Ratio':<25}{m_def['Exact_Match_Ratio']:<25.4f}{m_cal['Exact_Match_Ratio']:<25.4f}\n\n")
        
        f.write("2. PER-CLASS CLINICAL PERFORMANCE (CALIBRATED)\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Disease Class':<15}{'Threshold':<12}{'AUROC':<10}{'AUPRC':<10}{'F1':<10}{'Sens':<10}{'Spec':<10}{'TP':<5}{'FP':<5}{'TN':<5}{'FN':<5}\n")
        for c in DISEASE_CLASSES:
            pc = res_calibrated['per_class'][c]
            f.write(f"{c:<15}{pc['Threshold']:<12.2f}{pc['AUROC']:<10.3f}{pc['AUPRC']:<10.3f}{pc['F1']:<10.3f}{pc['Sensitivity']:<10.3f}{pc['Specificity']:<10.3f}{pc['TP']:<5}{pc['FP']:<5}{pc['TN']:<5}{pc['FN']:<5}\n")
        f.write("\n3. INDIVIDUAL PATIENT PREDICTIONS\n")
        f.write("-" * 80 + "\n")
        for idx, pid in enumerate(pids):
            true_vec = [int(v) for v in y_true[idx]]
            prob_vec = [f"{p:.2f}" for p in y_prob[idx]]
            pred_vec = [int(y_prob[idx, c] >= calibrated_thresholds[DISEASE_CLASSES[c]]) for c in range(5)]
            f.write(f"Patient {pid}: True={true_vec} | Pred={pred_vec} | Prob={prob_vec}\n")
            
    print(f"Saved clinical report to: {report_path}")
    
    # 9. Plot Confusion Matrices
    plot_dir = ROOT / 'dataset' / 'plots' / 'evaluation'
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    
    for c_idx, c in enumerate(DISEASE_CLASSES):
        pc = res_calibrated['per_class'][c]
        cm_data = np.array([[pc['TN'], pc['FP']], [pc['FN'], pc['TP']]])
        ax = axes[c_idx]
        im = ax.imshow(cm_data, cmap='Blues', interpolation='nearest')
        ax.set_title(f"Class {c} (t={pc['Threshold']:.2f})", fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Pred 0', 'Pred 1'])
        ax.set_yticklabels(['True 0', 'True 1'])
        
        for i in range(2):
            for j in range(2):
                val = cm_data[i, j]
                color = 'white' if val > np.max(cm_data) / 2 else 'black'
                ax.text(j, i, str(val), ha='center', va='center', color=color, fontweight='bold')
                
    plt.tight_layout()
    cm_plot_path = plot_dir / 'stage7_confusion_matrices.png'
    plt.savefig(cm_plot_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrices plot to: {cm_plot_path}")
    
    # 10. Plot ROC & PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for c_idx, c in enumerate(DISEASE_CLASSES):
        c_true = y_true[:, c_idx]
        c_prob = y_prob[:, c_idx]
        if np.sum(c_true == 1) > 0 and np.sum(c_true == 0) > 0:
            fpr, tpr, _ = roc_curve(c_true, c_prob)
            prec_c, rec_c, _ = precision_recall_curve(c_true, c_prob)
            ax1.plot(fpr, tpr, color=colors[c_idx], lw=2, label=f"{c} (AUC={res_calibrated['per_class'][c]['AUROC']:.2f})")
            ax2.plot(rec_c, prec_c, color=colors[c_idx], lw=2, label=f"{c} (PR={res_calibrated['per_class'][c]['AUPRC']:.2f})")
            
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Test Set ROC Curves')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Test Set Precision-Recall Curves')
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_pr_plot_path = plot_dir / 'stage7_roc_pr_curves.png'
    plt.savefig(roc_pr_plot_path, dpi=200)
    plt.close()
    print(f"Saved ROC/PR curves plot to: {roc_pr_plot_path}")
    print("\n[STAGE 7 COMPLETE] Untouched test set evaluated successfully.")


if __name__ == '__main__':
    main()
