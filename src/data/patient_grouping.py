"""
Patient grouping and metadata structuring module for the BMD-HS dataset.

Preserves the hierarchical clinical relationship:
Patient -> Multiple Recordings (8 auscultation sites: 4 valves x 2 postures) -> Spectrograms
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd

DISEASE_CLASSES = ['N', 'AS', 'AR', 'MR', 'MS']
VALVE_LOCATIONS = ['Aor', 'Mit', 'Pul', 'Tri']
POSTURES = ['sit', 'sup']


def parse_disease_labels(label_str: Any) -> np.ndarray:
    """
    Parses a disease label string (e.g. 'AS+AR', 'N', 'MR') into a 5-element binary vector.
    Class order: [N, AS, AR, MR, MS]
    
    Special case:
    If label_str is NaN / None / 'N/A' / 'MVP' (e.g. patient_022 diagnosed with MVP
    whose five disease targets are unannotated / unavailable):
    returns [-1.0, -1.0, -1.0, -1.0, -1.0] representing unknown targets that are masked out
    during loss computation and metric evaluation, rather than treating them as confirmed negatives.
    """
    if pd.isna(label_str) or label_str is None or str(label_str).strip() in ['', 'nan', 'NaN', 'N/A', 'MVP', '-1', 'None', 'ALL_ZERO_MVP', 'UNKNOWN_MVP']:
        return np.full(len(DISEASE_CLASSES), -1.0, dtype=np.float32)
    
    target = np.zeros(len(DISEASE_CLASSES), dtype=np.float32)
    labels = [l.strip() for l in str(label_str).split('+')]
    matched = False
    for l in labels:
        if l in DISEASE_CLASSES:
            idx = DISEASE_CLASSES.index(l)
            target[idx] = 1.0
            matched = True
            
    if not matched:
        return np.full(len(DISEASE_CLASSES), -1.0, dtype=np.float32)
        
    return target


def load_patient_records(
    manifest_path: str = 'dataset/metadata/spectrogram_manifest.csv',
    root_dir: str = '.'
) -> Dict[str, Dict[str, Any]]:
    """
    Loads spectrogram manifest and groups recordings by patient_id.
    
    Returns a dictionary keyed by patient_id:
    {
        'patient_001': {
            'patient_id': 'patient_001',
            'split': 'validation',
            'disease_labels_str': 'AS+AR',
            'disease_vector': array([0., 1., 1., 0., 0.]),
            'is_all_zero_special': False,
            'recordings': [...]
        },
        ...
    }
    """
    root = Path(root_dir)
    manifest = pd.read_csv(root / manifest_path)
    
    patients: Dict[str, Dict[str, Any]] = {}
    
    for _, row in manifest.iterrows():
        pid = str(row['patient_id'])
        split = str(row['split'])
        label_str = row['disease_labels']
        vec = parse_disease_labels(label_str)
        is_unknown = bool(np.all(vec == -1.0))
        
        if pid not in patients:
            disp_label = str(label_str) if (not pd.isna(label_str) and str(label_str).strip() not in ['', 'nan', 'NaN', 'N/A', '-1', 'None']) else 'UNKNOWN_MVP'
            patients[pid] = {
                'patient_id': pid,
                'split': split,
                'disease_labels_str': disp_label,
                'disease_vector': vec,
                'is_all_zero_special': is_unknown,
                'recordings': []
            }
        
        fname = str(row['processed_filename'])
        posture = 'sit' if '_sit_' in fname else ('sup' if '_sup_' in fname else 'unknown')
        
        raw_shape = row['spectrogram_shape']
        shape_val = eval(str(raw_shape)) if isinstance(raw_shape, str) else raw_shape
        
        patients[pid]['recordings'].append({
            'processed_filename': fname,
            'spectrogram_filename': str(row['spectrogram_filename']),
            'valve_location': str(row['valve_location']),
            'posture': posture,
            'original_or_augmented': str(row['original_or_augmented']),
            'spectrogram_shape': shape_val,
            'numerical_path': str(row['numerical_path']),
            'visualization_path': str(row['visualization_path'])
        })
        
    return patients


def get_split_patients(
    patients: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Separates patients dictionary into train, validation, and test splits.
    """
    train_pts = {k: v for k, v in patients.items() if v['split'] == 'train'}
    val_pts = {k: v for k, v in patients.items() if v['split'] == 'validation'}
    test_pts = {k: v for k, v in patients.items() if v['split'] == 'test'}
    return train_pts, val_pts, test_pts
