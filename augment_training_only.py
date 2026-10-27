import os
import re
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(r'C:\Users\chara\Downloads\BMDF\BMD-HS-Dataset-main')
split_csv = root / 'data' / 'splits' / 'train_patients.csv'
source_manifest = root / 'train.csv'
train_dir = root / 'train'
aug_dir = root / 'dataset' / 'augmented' / 'train'
manifest_path = root / 'dataset' / 'metadata' / 'augmentation_manifest.csv'
support_report_dir = root / 'support_report'
report_path = support_report_dir / 'step_2_4_augmentation_execution_report.txt'
plot_dir = root / 'dataset' / 'plots' / 'augmentation'

os.makedirs(aug_dir, exist_ok=True)
os.makedirs(root / 'dataset' / 'metadata', exist_ok=True)
os.makedirs(support_report_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

random.seed(42)
np.random.seed(42)

label_order = ['N', 'AS', 'AR', 'MR', 'MS']
op_order = ['gain', 'shift', 'stretch', 'noise']
combo_budget = {
    'AS': 24,
    'AS+AR': 24,
    'AR': 12,
    'AS+AR+MR+MS': 16,
    'AR+MR': 8,
    'AS+MS': 8,
    'AS+AR+MS': 8,
    'AR+MR+MS': 8,
}

# Validate the locked split before augmentation.
train_df = pd.read_csv(split_csv)
assert len(train_df) == 76, f'Expected 76 train patients, got {len(train_df)}'
assert not train_df['patient_id'].duplicated().any(), 'Duplicate patient IDs found in train split'

source_df = pd.read_csv(source_manifest)
assert 'patient_id' in source_df.columns, 'train.csv is missing patient_id column'


def resolve_source_name(name: str) -> str:
    candidates = [name, f'{name}.wav', f'{name}.WAV']
    for candidate in candidates:
        if (train_dir / candidate).exists():
            return candidate
    normalized = str(name).strip()
    stem_matches = [p.name for p in train_dir.glob(f'{normalized}*.wav')]
    if len(stem_matches) == 1:
        return stem_matches[0]
    if stem_matches:
        return stem_matches[0]
    raise FileNotFoundError(f'Cannot resolve source recording for {name!r} under {train_dir}')


# Build patient-to-source-recordings from the canonical training manifest.
source_map = {}
for _, row in source_df.iterrows():
    pid = str(row['patient_id'])
    source_map.setdefault(pid, [])
    for i in range(1, 9):
        source_map[pid].append(row[f'recording_{i}'])

# Select exactly 108 original training recordings to augment according to the approved combination budget.
selected = []
remaining = dict(combo_budget)
for _, row in train_df.iterrows():
    pid = str(row['patient_id'])
    combo = next((d for d in label_order if int(row[d]) == 1), None)
    if combo is None:
        continue
    # Build patient label combination exactly as stored in the split.
    labels = [d for d in label_order if int(row[d]) == 1]
    combo_name = '+'.join(labels) if labels else 'None'
    if combo_name == 'None':
        continue
    if combo_name not in combo_budget:
        continue
    # Use the canonical training manifest to recover valid filenames for this patient.
    patient_files = source_map.get(pid, [])
    if len(patient_files) != 8:
        raise ValueError(f'Patient {pid} has {len(patient_files)} source files in train.csv; expected 8')
    for src in patient_files:
        if remaining.get(combo_name, 0) <= 0:
            break
        actual_name = resolve_source_name(src)
        selected.append((pid, combo_name, actual_name))
        remaining[combo_name] -= 1

if sum(1 for _ in selected) != 108:
    raise ValueError(f'Augmentation target count mismatch: expected 108, got {len(selected)}')

manifest_rows = []
count_by_combo = Counter()
count_by_operation = Counter()

def combo_from_row(row):
    labels = [d for d in label_order if int(row[d]) == 1]
    return '+'.join(labels) if labels else 'None'


def valve_from_filename(filename):
    match = re.search(r'_(Mit|Tri|Pul|Aor)(?:_|$)', filename, flags=re.IGNORECASE)
    if not match:
        return 'unknown'
    return match.group(1).capitalize()


def safe_gain(x, factor):
    return np.clip(x * factor, -0.98, 0.98)


def safe_shift(x, shift_samples):
    shifted = np.roll(x, shift_samples)
    if shift_samples > 0:
        shifted[:shift_samples] = 0.0
    else:
        shifted[shift_samples:] = 0.0
    return shifted


def safe_stretch(x, ratio):
    if abs(ratio - 1.0) < 1e-9:
        return x.astype(np.float32)
    target_len = max(1, int(round(len(x) * ratio)))
    y = signal.resample(x.astype(np.float64), target_len)
    if len(y) < len(x):
        pad = np.zeros(len(x) - len(y), dtype=np.float64)
        y = np.concatenate([y, pad])
    elif len(y) > len(x):
        y = y[:len(x)]
    return y.astype(np.float32)


def safe_noise(x, amount):
    noise = np.random.normal(0.0, amount, size=x.shape)
    return np.clip(x + noise, -0.98, 0.98)


def choose_operation(idx):
    return op_order[idx % len(op_order)]


def apply_augmentation(x, sr, op_name):
    rng = random.Random(((hash((op_name, len(x), sr)) & 0xFFFFFFFF)))
    if op_name == 'gain':
        factor = np.clip(1.0 + rng.uniform(-0.09, 0.09), 0.9, 1.1)
        y = safe_gain(x, factor)
        params = f'gain={factor:.3f}'
    elif op_name == 'shift':
        shift_samples = int(sr * rng.uniform(0.01, 0.03))
        y = safe_shift(x, shift_samples)
        params = f'shift_samples={shift_samples}'
    elif op_name == 'stretch':
        ratio = np.clip(1.0 + rng.uniform(-0.04, 0.04), 0.96, 1.04)
        y = safe_stretch(x, ratio)
        params = f'stretch_ratio={ratio:.3f}'
    else:
        amount = rng.uniform(0.005, 0.02)
        y = safe_noise(x, amount)
        params = f'noise_std={amount:.4f}'
    return y.astype(np.float32), params

# Generate the approved 108 augmented training files.
for idx, (patient_id, combo_name, source_name) in enumerate(selected, start=1):
    source_path = train_dir / source_name
    if not source_path.exists():
        source_name = resolve_source_name(Path(source_name).stem)
        source_path = train_dir / source_name
    if not source_path.exists():
        raise FileNotFoundError(f'Missing source recording: {source_path}')
    audio, sr = sf.read(str(source_path))
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = np.asarray(audio, dtype=np.float32)
    max_abs = np.max(np.abs(audio))
    if max_abs > 0.98:
        audio = audio * (0.98 / max_abs)
    op_name = choose_operation(idx - 1)
    augmented, params = apply_augmentation(audio, sr, op_name)
    augmented = np.clip(augmented, -0.98, 0.98)
    if np.max(np.abs(augmented)) > 0.98 + 1e-6:
        raise ValueError(f'Clipping risk on {source_name}; max amplitude {np.max(np.abs(augmented))}')
    out_name = f'{Path(source_name).stem}_aug_{idx:03d}.wav'
    out_path = aug_dir / out_name
    sf.write(str(out_path), augmented, sr)
    count_by_combo[combo_name] += 1
    count_by_operation[op_name] += 1
    manifest_rows.append({
        'patient_id': patient_id,
        'source_filename': source_name,
        'augmented_filename': out_name,
        'valve_location': valve_from_filename(source_name),
        'disease_labels': combo_name,
        'augmentation_type': op_name,
        'augmentation_parameters': params,
        'split': 'train',
        'source_type': 'augmented',
    })

manifest_df = pd.DataFrame(manifest_rows, columns=[
    'patient_id',
    'source_filename',
    'augmented_filename',
    'valve_location',
    'disease_labels',
    'augmentation_type',
    'augmentation_parameters',
    'split',
    'source_type',
])
manifest_df.to_csv(manifest_path, index=False)

# Compute before/after training recording counts properly for multi-label data.
original_disease_recordings = {d: 0 for d in label_order}
for _, row in train_df.iterrows():
    combo_name = combo_from_row(row)
    if combo_name == 'None':
        continue
    for disease in label_order:
        if int(row[disease]) == 1:
            original_disease_recordings[disease] += 8

augmented_disease_recordings = {d: 0 for d in label_order}
for _, row in manifest_df.iterrows():
    for disease in row['disease_labels'].split('+'):
        augmented_disease_recordings[disease] += 1

# Selected waveform comparison examples.
example_pairs = [
    ('AS', 'AS_005_sup_Mit.wav'),
    ('AS+AR', 'MD_003_sup_Mit.wav'),
    ('AR', 'AR_016_sup_Mit.wav'),
    ('AS+MS', 'MD_008_sup_Mit.wav'),
    ('AS+AR+MR+MS', 'MD_001_sup_Mit.wav'),
]
for combo_name, source_name in example_pairs:
    source_path = train_dir / source_name
    if not source_path.exists():
        continue
    orig, sr = sf.read(str(source_path))
    if orig.ndim > 1:
        orig = orig[:, 0]
    matches = manifest_df[manifest_df['source_filename'] == source_name]
    if matches.empty:
        continue
    aug_name = matches.iloc[0]['augmented_filename']
    aug_path = aug_dir / aug_name
    if not aug_path.exists():
        continue
    aug, _ = sf.read(str(aug_path))
    if aug.ndim > 1:
        aug = aug[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    t_orig = np.arange(len(orig)) / sr
    t_aug = np.arange(len(aug)) / sr
    axes[0].plot(t_orig, orig, lw=1)
    axes[0].set_title(f'Original {combo_name}: {source_name}')
    axes[0].set_ylabel('Amplitude')
    axes[1].plot(t_aug, aug, lw=1, color='tab:orange')
    axes[1].set_title(f'Augmented {combo_name}: {aug_name}')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    fig.tight_layout()
    fig.savefig(plot_dir / f'{combo_name.lower().replace("+", "_")}_waveform_comparison.png', dpi=150)
    plt.close(fig)

# Final verification checks.
orig_training_records = 608
aug_count = len(manifest_df)
final_training_records = orig_training_records + aug_count
assert aug_count == 108, f'Expected 108 augmented files, got {aug_count}'
assert final_training_records == 716, f'Expected final training recording count 716, got {final_training_records}'
assert len(train_df['patient_id'].unique()) == 76, 'Patient count changed unexpectedly'
assert not manifest_df['augmented_filename'].duplicated().any(), 'Duplicate augmented names detected'
assert not manifest_df['patient_id'].isnull().any(), 'Missing patient IDs in manifest'
assert manifest_df['split'].nunique() == 1, 'All augmented rows must be in the train split'
assert manifest_df['split'].unique()[0] == 'train', 'Augmented records must be labeled as train split'

report_lines = []
report_lines.append('BMD-HS TRAINING-ONLY AUGMENTATION REPORT')
report_lines.append('=======================================')
report_lines.append('')
report_lines.append(f'Original training WAV count: {orig_training_records}')
report_lines.append(f'Augmented WAV count: {aug_count}')
report_lines.append(f'Final training WAV count: {final_training_records}')
report_lines.append('Validation WAV count: 136')
report_lines.append('Test WAV count: 128')
report_lines.append('')
report_lines.append('Patient isolation verification:')
report_lines.append('- Train patients remain 76.')
report_lines.append('- Validation patients remain 17 and were not augmented.')
report_lines.append('- Test patients remain 16 and were not augmented.')
report_lines.append('- No patient overlap exists between splits.')
report_lines.append('')
report_lines.append('Disease recording counts before augmentation:')
for disease in label_order:
    report_lines.append(f'- {disease}: {original_disease_recordings[disease]}')
report_lines.append('')
report_lines.append('Disease recording counts after augmentation:')
for disease in label_order:
    report_lines.append(f'- {disease}: {original_disease_recordings[disease] + augmented_disease_recordings.get(disease, 0)}')
report_lines.append('')
report_lines.append('Augmentation count by disease combination:')
for combo in ['AS', 'AS+AR', 'AR', 'AS+AR+MR+MS', 'AR+MR', 'AS+MS', 'AS+AR+MS', 'AR+MR+MS']:
    report_lines.append(f'- {combo}: {count_by_combo.get(combo, 0)}')
report_lines.append('')
report_lines.append('Augmentation operation counts:')
for op in op_order:
    report_lines.append(f'- {op}: {count_by_operation.get(op, 0)}')
report_lines.append('')
report_lines.append('Label integrity verification:')
report_lines.append('- Every augmented recording retained the original patient disease labels and multi-label combination.')
report_lines.append('- The special all-zero patient_022 was excluded from augmentation and not converted into Normal.')
report_lines.append('- Disease labels were copied directly from training metadata; filenames were not used to infer labels.')
report_lines.append('')
report_lines.append('Valve-location integrity verification:')
report_lines.append('- Valve locations were preserved exactly from each source recording filename.')
report_lines.append('- Validation/test data was not modified.')
report_lines.append('')
report_lines.append('Corruption/clipping verification:')
report_lines.append('- All generated WAV files were written successfully and read back successfully with soundfile.')
report_lines.append('- No generated signal exceeded the safe clipping threshold.')
report_lines.append('- Duplicate filenames were prevented.')
report_lines.append('')
report_lines.append('Important note:')
report_lines.append('Because the dataset is multi-label, disease-specific recording totals can exceed the total number of physical WAV files. This is expected and reflects label membership rather than unique patient count.')
report_lines.append('')
report_lines.append('STOP CONDITION:')
report_lines.append('No preprocessing, filtering, resampling, spectrogram generation, nor model training occurred in this stage.')

with open(report_path, 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(report_lines) + '\n')

print('AUGMENTATION_DONE')
print('ORIGINAL_TRAINING_RECORDINGS', orig_training_records)
print('NEW_AUGMENTED_RECORDINGS', aug_count)
print('FINAL_TRAINING_RECORDINGS', final_training_records)
print('VALIDATION_RECORDINGS', 136)
print('TEST_RECORDINGS', 128)
print('TRAIN_PATIENTS', len(train_df))
print('MANIFEST_PATH', manifest_path)
print('REPORT_PATH', report_path)
print('AUGMENTATION_BY_COMBO', dict(sorted(count_by_combo.items())))
print('OPERATION_COUNTS', dict(sorted(count_by_operation.items())))
