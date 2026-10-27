import os
import sys
from pathlib import Path

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

root = Path(__file__).resolve().parent
pre_dir = root / 'dataset' / 'preprocessed'
meta_dir = root / 'dataset' / 'metadata'
out_base = root / 'dataset' / 'spectrograms'
plot_dir = root / 'dataset' / 'plots' / 'mel_spectrogram_examples'
analysis_dir = root / 'dataset' / 'analysis'
support_dir = root / 'support_report'

manifest_path = meta_dir / 'spectrogram_manifest.csv'
analysis_report_path = analysis_dir / 'step_2_6_mel_spectrogram_report.txt'
support_report_path = support_dir / 'step_2_6_mel_spectrogram_report.txt'

SR = 2000
N_FFT = 256
HOP_LENGTH = 64
N_MELS = 64
FMIN = 20
FMAX = 1000
POWER = 2.0

for d in [
    out_base / 'train' / 'npy',
    out_base / 'validation' / 'npy',
    out_base / 'test' / 'npy',
    out_base / 'train' / 'png',
    out_base / 'validation' / 'png',
    out_base / 'test' / 'png',
    plot_dir,
    analysis_dir,
    support_dir,
]:
    d.mkdir(parents=True, exist_ok=True)

pre_manifest_file = meta_dir / 'preprocessing_manifest.csv'
if not pre_manifest_file.exists():
    raise FileNotFoundError(f"Preprocessing manifest not found at {pre_manifest_file}")

pre_manifest = pd.read_csv(pre_manifest_file)

def file_records():
    records = []
    for split_name in ['train', 'validation', 'test']:
        split_path = pre_dir / split_name
        for wav_path in sorted(split_path.glob('*.wav')):
            match = pre_manifest[pre_manifest['processed_filename'] == wav_path.name]
            if match.empty:
                raise ValueError(f'Metadata missing for processed WAV: {wav_path.name}')
            row = match.iloc[0].to_dict()
            records.append({
                'patient_id': str(row['patient_id']),
                'processed_filename': str(row['processed_filename']),
                'valve_location': str(row['valve_location']),
                'disease_labels': str(row['disease_labels']),
                'original_or_augmented': str(row['original_or_augmented']),
                'split': split_name,
                'source_path': wav_path,
            })
    return records


records = file_records()
print(f'Total preprocessed files to account for: {len(records)}')

manifest_rows = []
failures = []
generated_count = 0
reused_count = 0

for idx, item in enumerate(records, start=1):
    wav_path = item['source_path']
    stem = wav_path.stem
    npy_name = f'{stem}.npy'
    png_name = f'{stem}.png'
    split_dir = out_base / item['split']
    npy_path = split_dir / 'npy' / npy_name
    png_path = split_dir / 'png' / png_name

    # Check if valid npy and png already exist
    need_generation = True
    if npy_path.exists() and png_path.exists():
        try:
            arr = np.load(npy_path, allow_pickle=False)
            if (
                arr.ndim == 2
                and arr.shape[0] == N_MELS
                and np.all(np.isfinite(arr))
                and png_path.stat().st_size > 0
            ):
                need_generation = False
                mel_db = arr
                reused_count += 1
        except Exception:
            need_generation = True

    if need_generation:
        try:
            x, sr = sf.read(str(wav_path))
        except Exception as exc:
            failures.append((wav_path.name, item['patient_id'], f'read_error:{type(exc).__name__}:{exc}'))
            continue

        if x.ndim > 1:
            x = x[:, 0]
        x = np.asarray(x, dtype=np.float32)

        if x.size == 0 or not np.all(np.isfinite(x)):
            failures.append((wav_path.name, item['patient_id'], 'empty_or_nonfinite_input'))
            continue

        if sr != SR:
            failures.append((wav_path.name, item['patient_id'], f'wrong_sampling_rate:{sr}'))
            continue

        mel = librosa.feature.melspectrogram(
            y=x,
            sr=SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            fmin=FMIN,
            fmax=FMAX,
            power=POWER,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        if not np.all(np.isfinite(mel_db)):
            failures.append((wav_path.name, item['patient_id'], 'nonfinite_spectrogram'))
            continue

        np.save(npy_path, mel_db.astype(np.float32))
        
        # Verify saved array
        saved_arr = np.load(npy_path, allow_pickle=False)
        if not np.all(np.isfinite(saved_arr)):
            failures.append((wav_path.name, item['patient_id'], 'corrupt_npy'))
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        img = librosa.display.specshow(
            mel_db,
            sr=SR,
            hop_length=HOP_LENGTH,
            x_axis='time',
            y_axis='mel',
            fmin=FMIN,
            fmax=FMAX,
            ax=ax
        )
        ax.set_title(f"{item['patient_id']} | {item['disease_labels']} | {item['valve_location']} | {item['split']} | {item['original_or_augmented']}")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Mel Frequency')
        cbar = fig.colorbar(img, ax=ax)
        cbar.set_label('Log Power (dB)')
        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

        generated_count += 1
        print(f"[{idx}/{len(records)}] Generated: {item['split']}/{npy_name} (Shape: {mel_db.shape})")

    manifest_rows.append({
        'patient_id': item['patient_id'],
        'processed_filename': item['processed_filename'],
        'spectrogram_filename': npy_name,
        'valve_location': item['valve_location'],
        'disease_labels': item['disease_labels'],
        'original_or_augmented': item['original_or_augmented'],
        'split': item['split'],
        'sampling_rate': int(SR),
        'n_fft': int(N_FFT),
        'hop_length': int(HOP_LENGTH),
        'n_mels': int(N_MELS),
        'fmin': float(FMIN),
        'fmax': float(FMAX),
        'power': float(POWER),
        'spectrogram_shape': str(mel_db.shape),
        'numerical_path': str(npy_path.relative_to(root).as_posix()),
        'visualization_path': str(png_path.relative_to(root).as_posix()),
    })

print(f"Spectrogram processing finished. Reused: {reused_count}, Newly Generated: {generated_count}, Failures: {len(failures)}")

manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(manifest_path, index=False)
print(f"Manifest saved to {manifest_path} ({len(manifest_df)} records)")

# Generate representative visualization grid
examples = [
    ('patient_089', 'N_089_sup_Mit_prep.wav', 'train', 'N', 'Mitral', 'original'),
    ('patient_005', 'AS_005_sup_Mit_prep.wav', 'train', 'AS', 'Mitral', 'original'),
    ('patient_016', 'AR_016_sup_Mit_prep.wav', 'train', 'AR', 'Mitral', 'original'),
    ('patient_002', 'MR_002_sup_Mit_prep.wav', 'train', 'MR', 'Mitral', 'original'),
    ('patient_006', 'MS_006_sup_Mit_prep.wav', 'train', 'MS', 'Mitral', 'original'),
    ('patient_003', 'MD_003_sup_Mit_aug_001_prep.wav', 'train', 'AS+AR', 'Mitral', 'augmented'),
]
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()
for ax, (pid, fname, split_name, label, valve, kind) in zip(axes, examples):
    wav_path = pre_dir / split_name / fname
    if not wav_path.exists():
        continue
    y, sr = sf.read(str(wav_path))
    if y.ndim > 1:
        y = y[:, 0]
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=POWER)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX, ax=ax)
    ax.set_title(f'{pid} | {label} | {valve} | {kind}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel Frequency')
fig.tight_layout()
grid_path = plot_dir / 'mel_spectrogram_examples_grid.png'
fig.savefig(grid_path, dpi=180)
plt.close(fig)

# Single waveform + spectrogram verification plot
example_wav = pre_dir / 'train' / 'AS_005_sup_Mit_prep.wav'
if example_wav.exists():
    y, sr = sf.read(str(example_wav))
    if y.ndim > 1:
        y = y[:, 0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    t = np.arange(len(y)) / sr
    axes[0].plot(t, y, color='black', linewidth=1)
    axes[0].set_title('Preprocessed waveform: AS_005_sup_Mit_prep.wav')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=POWER)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    librosa.display.specshow(mel_db, sr=SR, hop_length=HOP_LENGTH, x_axis='time', y_axis='mel', fmin=FMIN, fmax=FMAX, ax=axes[1])
    axes[1].set_title('Mel spectrogram: AS_005_sup_Mit_prep.wav')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Mel Frequency')
    fig.tight_layout()
    verification_path = plot_dir / 'mel_verification_example.png'
    fig.savefig(verification_path, dpi=180)
    plt.close(fig)

# ==============================================================================
# Comprehensive Verification
# ==============================================================================
expected_counts = {'train': 716, 'validation': 136, 'test': 128}
actual_counts = {'train': 0, 'validation': 0, 'test': 0}

for row in manifest_rows:
    actual_counts[row['split']] += 1

missing_specs = sum(expected_counts[k] - actual_counts[k] for k in expected_counts)
duplicate_names = len(manifest_rows) - len(set(r['spectrogram_filename'] for r in manifest_rows))
corrupted_files = 0
nan_files = 0
inf_files = 0
wrong_shape_files = 0
label_mismatches = 0
patient_mismatches = 0
split_mismatches = 0

train_patients = set()
val_patients = set()
test_patients = set()

for row in manifest_rows:
    arr = np.load(root / row['numerical_path'], allow_pickle=False)
    if arr.ndim != 2 or arr.shape[0] != N_MELS:
        wrong_shape_files += 1
    if np.isnan(arr).any():
        nan_files += 1
    if np.isinf(arr).any():
        inf_files += 1

    if row['split'] == 'train':
        train_patients.add(row['patient_id'])
    elif row['split'] == 'validation':
        val_patients.add(row['patient_id'])
    elif row['split'] == 'test':
        test_patients.add(row['patient_id'])
    else:
        split_mismatches += 1

# Metadata consistency check against preprocessing manifest
for row in manifest_rows:
    match = pre_manifest[pre_manifest['processed_filename'] == row['processed_filename']]
    if match.empty:
        patient_mismatches += 1
        label_mismatches += 1
    else:
        orig = match.iloc[0]
        if str(orig['patient_id']) != row['patient_id']:
            patient_mismatches += 1
        if str(orig['disease_labels']) != row['disease_labels']:
            label_mismatches += 1
        if str(orig['valve_location']) != row['valve_location']:
            label_mismatches += 1
        if str(orig['original_or_augmented']) != row['original_or_augmented']:
            label_mismatches += 1

# Patient overlap check
train_val_overlap = len(train_patients.intersection(val_patients))
train_test_overlap = len(train_patients.intersection(test_patients))
val_test_overlap = len(val_patients.intersection(test_patients))
total_patient_overlap = train_val_overlap + train_test_overlap + val_test_overlap

# Overall directory scan check
disk_npy_counts = {
    'train': len(list((out_base / 'train' / 'npy').glob('*.npy'))),
    'validation': len(list((out_base / 'validation' / 'npy').glob('*.npy'))),
    'test': len(list((out_base / 'test' / 'npy').glob('*.npy'))),
}
disk_png_counts = {
    'train': len(list((out_base / 'train' / 'png').glob('*.png'))),
    'validation': len(list((out_base / 'validation' / 'png').glob('*.png'))),
    'test': len(list((out_base / 'test' / 'png').glob('*.png'))),
}

is_passed = (
    missing_specs == 0
    and duplicate_names == 0
    and corrupted_files == 0
    and nan_files == 0
    and inf_files == 0
    and wrong_shape_files == 0
    and label_mismatches == 0
    and patient_mismatches == 0
    and split_mismatches == 0
    and len(manifest_rows) == 980
    and actual_counts['train'] == 716
    and actual_counts['validation'] == 136
    and actual_counts['test'] == 128
    and len(train_patients) == 76
    and len(val_patients) == 17
    and len(test_patients) == 16
    and total_patient_overlap == 0
    and disk_npy_counts['train'] == 716
    and disk_npy_counts['validation'] == 136
    and disk_npy_counts['test'] == 128
    and disk_png_counts['train'] == 716
    and disk_png_counts['validation'] == 136
    and disk_png_counts['test'] == 128
)
status_str = "PASS" if is_passed else "FAIL"

report_text = f"""-----------------------------------------
MEL SPECTROGRAM REPORT
-----------------------------------------

Input Audio Preprocessing:
Train = 716 WAVs
Validation = 136 WAVs
Test = 128 WAVs
Total = 980 WAVs

Configuration:
Sampling rate: {SR} Hz
n_fft: {N_FFT}
hop_length: {HOP_LENGTH}
n_mels: {N_MELS}
fmin: {FMIN} Hz
fmax: {FMAX} Hz
power: {POWER}

-----------------------------------------
OUTPUT SPECTROGRAM COUNTS
-----------------------------------------
Train spectrograms: {actual_counts['train']} / 716
Validation spectrograms: {actual_counts['validation']} / 136
Test spectrograms: {actual_counts['test']} / 128
Total spectrograms: {len(manifest_rows)} / 980

Disk .npy files: Train={disk_npy_counts['train']}, Val={disk_npy_counts['validation']}, Test={disk_npy_counts['test']}
Disk .png files: Train={disk_png_counts['train']}, Val={disk_png_counts['validation']}, Test={disk_png_counts['test']}

-----------------------------------------
PATIENT ISOLATION & SPLIT CHECKS
-----------------------------------------
Train patients: {len(train_patients)} (Expected: 76)
Validation patients: {len(val_patients)} (Expected: 17)
Test patients: {len(test_patients)} (Expected: 16)
Total unique patients: {len(train_patients | val_patients | test_patients)} (Expected: 109)
Patient overlap: {total_patient_overlap}

-----------------------------------------
DATA INTEGRITY & QUALITY CHECKS
-----------------------------------------
Missing spectrograms: {missing_specs}
Duplicate spectrogram filenames: {duplicate_names}
Corrupted .npy files: {corrupted_files}
NaN values: {nan_files}
Infinite values: {inf_files}
Wrong shapes: {wrong_shape_files}
Label mismatches: {label_mismatches}
Patient mismatches: {patient_mismatches}
Split mismatches: {split_mismatches}

-----------------------------------------
STATUS
-----------------------------------------
{status_str}
"""

analysis_report_path.write_text(report_text, encoding='utf-8')
support_report_path.write_text(report_text, encoding='utf-8')

print("\n" + "=" * 50)
print(report_text)
print("=" * 50)

if failures:
    print("\nFAILURES ENCOUNTERED:")
    for f in failures:
        print(f"  Filename: {f[0]}, Patient ID: {f[1]}, Reason: {f[2]}")
else:
    print("\nNo failures encountered during spectrogram generation.")
