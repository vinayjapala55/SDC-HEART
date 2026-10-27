"""
Dataset loading and batch collation module for RosaNet.

Provides:
1. PatientMultiRecordingSequence:
   - Groups recordings by patient.
   - Preserves the 8 physical auscultation locations (4 valves x 2 postures).
   - Augmented recordings in the training set are treated as additional training instances
     (never as an extraneous 9th branch), where augmented recordings populate their matching
     physical auscultation branch slots.
2. Standardized Spectrogram Loading:
   - Input shape: (64 Mel bins, 626 Time steps, 1 channel).
   - Time dimension standardization: Constant padding with minimum dB floor for shorter signals,
     symmetric central windowing for longer signals.
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import tensorflow as tf

from src.data.patient_grouping import DISEASE_CLASSES, load_patient_records

STANDARD_POSITIONS = [
    ('sit', 'Aor'),
    ('sit', 'Mit'),
    ('sit', 'Pul'),
    ('sit', 'Tri'),
    ('sup', 'Aor'),
    ('sup', 'Mit'),
    ('sup', 'Pul'),
    ('sup', 'Tri'),
]

DEFAULT_FREQ_BINS = 64
DEFAULT_TIME_STEPS = 626


def load_and_standardize_spectrogram(
    npy_path: str,
    target_time_steps: int = DEFAULT_TIME_STEPS,
    root_dir: str = '.'
) -> np.ndarray:
    """
    Loads a .npy Mel spectrogram from disk and standardizes it to (64, target_time_steps, 1).
    
    Standardization Strategy:
    - If spectrogram time steps T < 626:
      Pads the temporal dimension (axis=1) with the minimum dB value (silence floor)
      to avoid introducing artificial energy steps.
    - If spectrogram time steps T > 626:
      Extracts a central temporal window of length 626.
    - If spectrogram time steps T == 626:
      Preserved as is.
    - Adds channel dimension -> (64, 626, 1).
    """
    full_path = Path(root_dir) / npy_path
    spec = np.load(full_path, allow_pickle=False).astype(np.float32)
    
    n_mels, t_steps = spec.shape
    
    if t_steps < target_time_steps:
        pad_width = target_time_steps - t_steps
        pad_val = float(np.min(spec)) if spec.size > 0 else 0.0
        spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant', constant_values=pad_val)
    elif t_steps > target_time_steps:
        start = (t_steps - target_time_steps) // 2
        spec = spec[:, start:start + target_time_steps]
        
    return np.expand_dims(spec, axis=-1)


def build_patient_8branch_sample_instances(
    patient: Dict[str, Any],
    include_augmented_instances: bool = True
) -> List[Dict[Tuple[str, str], str]]:
    """
    Constructs 8-branch mapping configurations for a patient.
    
    Each configuration maps the 8 standard physical auscultation positions:
      (posture, valve_location) -> numerical_path
    
    How Augmented Recordings are Handled:
    - Instance 1 (Base): Formed by the 8 original recordings.
    - Instance 2 (Augmented, for train patients with augmentation):
      Populates the 8 branch slots with the patient's augmented recordings where available,
      and uses the original recordings for any unaugmented positions.
    - An augmented recording is NEVER treated as a 9th branch. It is an augmented sample
      for that specific physical auscultation site within an additional patient instance.
    """
    recs = patient['recordings']
    
    orig_map = {}
    aug_map = {}
    for r in recs:
        key = (r['posture'], r['valve_location'])
        if r['original_or_augmented'] == 'original':
            orig_map[key] = r['numerical_path']
        else:
            aug_map[key] = r['numerical_path']
            
    instances = [orig_map]
    
    if include_augmented_instances and len(aug_map) > 0:
        # Create augmented configuration combining augmented recordings with base originals
        aug_instance = dict(orig_map)
        for k, v in aug_map.items():
            aug_instance[k] = v
        instances.append(aug_instance)
        
    return instances


class PatientMultiRecordingSequence(tf.keras.utils.Sequence):
    """
    Keras Sequence for Patient-Level Training and Evaluation.
    
    Output X: List of 8 numpy arrays (one per physical auscultation branch),
              each array of shape (batch_size, 64, 626, 1).
    Output Y:
        - patient_target: (batch_size, 5) multi-label binary vector [N, AS, AR, MR, MS]
        - branch_targets: (batch_size, 8, 5) replicated patient target for branch supervision
    """
    def __init__(
        self,
        patients_dict: Dict[str, Dict[str, Any]],
        batch_size: int = 4,
        time_steps: int = DEFAULT_TIME_STEPS,
        shuffle: bool = True,
        include_augmented_instances: bool = True,
        root_dir: str = '.'
    ):
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.shuffle = shuffle
        self.root_dir = root_dir
        
        # Flatten into patient sample instances
        self.samples = []
        for pid, pdata in patients_dict.items():
            inst_list = build_patient_8branch_sample_instances(
                pdata,
                include_augmented_instances=include_augmented_instances
            )
            for inst in inst_list:
                self.samples.append({
                    'patient_id': pid,
                    'split': pdata['split'],
                    'disease_vector': pdata['disease_vector'],
                    'branch_paths': inst
                })
                
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.samples) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[List[np.ndarray], Dict[str, np.ndarray]]:
        batch_indices = self.indices[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_samples = [self.samples[i] for i in batch_indices]
        b_size = len(batch_samples)
        
        # 8 physical auscultation branches
        branch_arrays = [
            np.zeros((b_size, DEFAULT_FREQ_BINS, self.time_steps, 1), dtype=np.float32)
            for _ in range(8)
        ]
        patient_targets = np.zeros((b_size, len(DISEASE_CLASSES)), dtype=np.float32)
        branch_targets = np.zeros((b_size, 8, len(DISEASE_CLASSES)), dtype=np.float32)

        for b_idx, s in enumerate(batch_samples):
            patient_targets[b_idx] = s['disease_vector']
            for branch_idx in range(8):
                branch_targets[b_idx, branch_idx] = s['disease_vector']
            
            b_paths = s['branch_paths']
            for pos_idx, pos_key in enumerate(STANDARD_POSITIONS):
                if pos_key in b_paths:
                    npy_rel = b_paths[pos_key]
                    spec = load_and_standardize_spectrogram(
                        npy_rel, target_time_steps=self.time_steps, root_dir=self.root_dir
                    )
                    branch_arrays[pos_idx][b_idx] = spec
                else:
                    # Silence pad if location missing
                    branch_arrays[pos_idx][b_idx] = np.zeros(
                        (DEFAULT_FREQ_BINS, self.time_steps, 1), dtype=np.float32
                    )

        targets_dict = {
            "dynamic_decision_fusion": patient_targets,
            "stacked_branch_decisions": branch_targets
        }
        return branch_arrays, targets_dict
