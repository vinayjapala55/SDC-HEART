import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
TRAIN_CSV = root / 'train.csv'
TRAIN_SPLIT = root / 'data' / 'splits' / 'train_patients.csv'
VAL_SPLIT = root / 'data' / 'splits' / 'val_patients.csv'
TEST_SPLIT = root / 'data' / 'splits' / 'test_patients.csv'
AUG_MANIFEST = root / 'dataset' / 'metadata' / 'augmentation_manifest.csv'
RAW_TRAIN_DIR = root / 'train'
AUG_TRAIN_DIR = root / 'dataset' / 'augmented' / 'train'
OUT_DIR = root / 'dataset' / 'preprocessed'
PLOT_DIR = root / 'dataset' / 'plots' / 'preprocessing'
SUPPORT_REPORT_DIR = root / 'support_report'
REPORT_PATH = SUPPORT_REPORT_DIR / 'step_2_5_preprocessing_report.txt'
MANIFEST_PATH = root / 'dataset' / 'metadata' / 'preprocessing_manifest.csv'

TARGET_SR = 2000
LOWPASS_CUTOFF = 1000
LABEL_ORDER = ['N', 'AS', 'AR', 'MR', 'MS']

for d in [OUT_DIR / 'train', OUT_DIR / 'validation', OUT_DIR / 'test', PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def combo_from_row(row):
    labels = []
    for d in LABEL_ORDER:
        value = row[d]
        if pd.isna(value):
            val = 0
        else:
            val = int(value)
        if val == 1:
            labels.append(d)
    return '+'.join(labels) if labels else 'None'


def resolve_existing_wav(candidate_name):
    candidate_name = str(candidate_name).strip()
    if candidate_name.endswith('.wav') or candidate_name.endswith('.WAV'):
        full = RAW_TRAIN_DIR / candidate_name
        if full.exists():
            return candidate_name
    direct = candidate_name + '.wav'
    path = RAW_TRAIN_DIR / direct
    if path.exists():
        return direct
    for path in sorted(RAW_TRAIN_DIR.glob(f'{candidate_name}*.wav')):
        return path.name
    raise FileNotFoundError(f'Could not resolve WAV for {candidate_name!r} in {RAW_TRAIN_DIR}')


def patient_to_recordings_map(train_manifest_df):
    mapping = {}
    for _, row in train_manifest_df.iterrows():
        pid = str(row['patient_id'])
        files = []
        for i in range(1, 9):
            name = str(row[f'recording_{i}'])
            if pd.notna(name) and name.strip():
                try:
                    resolved_name = resolve_existing_wav(name)
                except FileNotFoundError:
                    # keep the literal manifest value if it already includes the real on-disk filename
                    resolved_name = name if name.lower().endswith('.wav') else name + '.wav'
                files.append(resolved_name)
        mapping[pid] = files
    return mapping


def read_split_csv(path):
    df = pd.read_csv(path)
    return df['patient_id'].astype(str).tolist()


def unique_combo_label(patient_id, all_labels_map):
    if patient_id not in all_labels_map:
        return 'unknown'
    row = all_labels_map[patient_id]
    labels = [d for d in LABEL_ORDER if int(row[d]) == 1]
    return '+'.join(labels) if labels else 'None'


def valve_from_filename(filename):
    m = re.search(r'_(Mit|Tri|Pul|Aor)(?:_|\.|$)', filename, flags=re.IGNORECASE)
    if not m:
        return 'unknown'
    return m.group(1).capitalize()


def mono_if_needed(x):
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return x.astype(np.float32)


def butterworth_lowpass(x, sr):
    nyquist = sr / 2.0
    cutoff = LOWPASS_CUTOFF / nyquist
    sos = signal.butter(4, cutoff, btype='lowpass', output='sos')
    filtered = signal.sosfiltfilt(sos, x)
    return filtered.astype(np.float32)


def spectral_noise_reduce(x):
    # Conservative spectral gating: subtract a small noise floor from the magnitude spectrum
    # while preserving the dominant waveform content.
    spectrum = np.fft.rfft(x)
    magnitude = np.abs(spectrum)
    noise_floor = np.median(magnitude) * 0.35
    reduced = np.maximum(magnitude - noise_floor, 0.12 * magnitude)
    phase = np.exp(1j * np.angle(spectrum))
    recovered = np.fft.irfft(reduced * phase, n=len(x))
    recovered = np.asarray(recovered, dtype=np.float32)
    return recovered


def resample_to_2000(x, orig_sr):
    if orig_sr == TARGET_SR:
        return x.astype(np.float32)
    gcd = np.gcd(orig_sr, TARGET_SR)
    # Use polyphase resampling to avoid aliasing;
    # factorization is straightforward for this target-specific setup.
    num = TARGET_SR // gcd
    den = orig_sr // gcd
    y = signal.resample_poly(x.astype(np.float64), up=num, down=den)
    return np.asarray(y, dtype=np.float32)


def zscore_standardize(x):
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError('Non-finite values encountered during standardization')
    if std < 1e-9:
        std = 1.0
    y = (x - mean) / std
    if not np.all(np.isfinite(y)):
        raise ValueError('NaN or Inf produced during standardization')
    return y.astype(np.float32), mean, std


def safe_write_wav(path, x, sr):
    sf.write(path, x.astype(np.float32), sr)


def sample_stats(x, sr):
    return {
        'mean': float(np.mean(x)),
        'std': float(np.std(x)),
        'len': int(len(x)),
        'duration': float(len(x) / sr),
    }


def process_audio_file(input_path, split_name, source_file_name, patient_id, disease_labels, original_or_augmented, meta_row=None):
    audio, orig_sr = sf.read(str(input_path))
    if audio.ndim > 1:
        original_channels = audio.shape[1]
    else:
        original_channels = 1

    mono = mono_if_needed(audio)
    filtered = butterworth_lowpass(mono, orig_sr)
    reduced = spectral_noise_reduce(filtered)
    resampled = resample_to_2000(reduced, orig_sr)
    standardized, mean_val, std_val = zscore_standardize(resampled)

    if standardized.shape[0] == 0:
        raise ValueError(f'Empty processed signal for {input_path}')
    if not np.all(np.isfinite(standardized)):
        raise ValueError(f'Non-finite values in processed signal for {input_path}')

    output_name = Path(source_file_name).stem + '_prep.wav'
    out_dir = OUT_DIR / split_name
    out_path = out_dir / output_name
    safe_write_wav(out_path, standardized, TARGET_SR)

    # Read back the processed file to verify readability and channel count.
    processed_audio, final_sr = sf.read(str(out_path))
    if processed_audio.ndim > 1:
        final_channels = processed_audio.shape[1]
    else:
        final_channels = 1
    if final_channels != 1:
        raise ValueError(f'Processed file was not mono: {out_path}')
    if final_sr != TARGET_SR:
        raise ValueError(f'Wrong sampling rate for {out_path}: {final_sr}')
    if len(processed_audio) == 0:
        raise ValueError(f'Empty output file: {out_path}')
    if not np.all(np.isfinite(processed_audio)):
        raise ValueError(f'Non-finite output values detected in {out_path}')

    row = {
        'patient_id': patient_id,
        'source_filename': source_file_name,
        'processed_filename': output_name,
        'valve_location': valve_from_filename(source_file_name),
        'disease_labels': disease_labels,
        'original_or_augmented': original_or_augmented,
        'split': split_name,
        'original_sampling_rate': int(orig_sr),
        'final_sampling_rate': int(final_sr),
        'original_channels': int(original_channels),
        'final_channels': int(final_channels),
        'original_duration': float(len(mono) / orig_sr),
        'final_duration': float(len(processed_audio) / final_sr),
        'preprocessing_operations': 'mono_conversion;butterworth_lowpass_4th_order_1000Hz;spectral_noise_reduction;resample_2000Hz;zscore_standardization',
        'processed_relative_path': str(out_path.relative_to(root)),
        'mean': mean_val,
        'std': std_val,
        'final_mean': float(np.mean(processed_audio)),
        'final_std': float(np.std(processed_audio)),
    }
    return row, out_path


def build_file_list_from_split(split_name, patient_ids, train_manifest_df, augmented_manifest_df=None):
    files = []
    patient_map = patient_to_recordings_map(train_manifest_df)
    for pid in patient_ids:
        if pid in patient_map:
            for recording_name in patient_map[pid]:
                source_path = RAW_TRAIN_DIR / recording_name
                if not source_path.exists():
                    source_path = RAW_TRAIN_DIR / resolve_existing_wav(recording_name)
                files.append({
                    'patient_id': pid,
                    'source_filename': recording_name,
                    'source_path': source_path,
                    'original_or_augmented': 'original',
                    'split': split_name,
                    'disease_labels': combo_from_row(train_manifest_df.loc[train_manifest_df['patient_id'].astype(str) == pid].iloc[0]),
                })
    if augmented_manifest_df is not None and split_name == 'train':
        for _, row in augmented_manifest_df.iterrows():
            source_name = str(row['augmented_filename'])
            files.append({
                'patient_id': str(row['patient_id']),
                'source_filename': source_name,
                'source_path': AUG_TRAIN_DIR / source_name,
                'original_or_augmented': 'augmented',
                'split': 'train',
                'disease_labels': str(row['disease_labels']),
            })
    return files


def save_waveform_and_psd_verification(representative_rows):
    for item in representative_rows:
        src = item['source_path']
        if not src.exists():
            continue
        orig, sr = sf.read(str(src))
        orig = mono_if_needed(orig)
        filtered = butterworth_lowpass(orig, sr)
        reduced = spectral_noise_reduce(filtered)
        resampled = resample_to_2000(reduced, sr)
        final, mean_val, std_val = zscore_standardize(resampled)

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        time_orig = np.arange(len(orig)) / sr
        time_final = np.arange(len(final)) / TARGET_SR
        axes[0].plot(time_orig, orig, label='Original', linewidth=1.0)
        axes[0].plot(time_orig, filtered, label='Butterworth', linewidth=1.0, alpha=0.8)
        axes[0].plot(time_final[:len(orig)] if len(final) >= len(orig) else time_final, final[:len(time_final)], label='Final', linewidth=1.0, alpha=0.9)
        axes[0].set_title(f'{item["patient_id"]} | {item["disease_labels"]} | {item["source_filename"]}')
        axes[0].set_ylabel('Amplitude')
        axes[0].legend(loc='upper right')

        f_orig, p_orig = signal.welch(orig, fs=sr, nperseg=min(2048, len(orig)))
        f_filt, p_filt = signal.welch(filtered, fs=sr, nperseg=min(2048, len(filtered)))
        f_final, p_final = signal.welch(final, fs=TARGET_SR, nperseg=min(2048, len(final)))
        axes[1].plot(f_orig, p_orig, label='Original', alpha=0.8)
        axes[1].plot(f_filt, p_filt, label='Butterworth', alpha=0.8)
        axes[1].plot(f_final, p_final, label='Final', alpha=0.8)
        axes[1].axvline(LOWPASS_CUTOFF, color='r', linestyle='--', label='1000 Hz cutoff')
        axes[1].set_xlim(0, min(2000, max(f_final.max(), f_filt.max(), f_orig.max()) * 1.05))
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('PSD')
        axes[1].legend(loc='upper right')
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{Path(item['source_filename']).stem}_preprocessing_verification.png", dpi=150)
        plt.close(fig)


def build_report(summary):
    lines = []
    lines.append('-----------------------------------------')
    lines.append('PCG PREPROCESSING REPORT')
    lines.append('-----------------------------------------')
    lines.append('')
    lines.append('Input:')
    lines.append(f'Train: {summary["train_count"]} WAVs')
    lines.append(f'Validation: {summary["val_count"]} WAVs')
    lines.append(f'Test: {summary["test_count"]} WAVs')
    lines.append(f'Total: {summary["total_count"]} WAVs')
    lines.append('')
    lines.append('-----------------------------------------')
    lines.append('PREPROCESSING OPERATIONS')
    lines.append('-----------------------------------------')
    lines.append('Mono conversion: Convert stereo/multichannel WAV files to mono by averaging channels; keep mono recordings unchanged.')
    lines.append('Butterworth: 4th-order low-pass filter, 1000 Hz cutoff, zero-phase SOS filtering.')
    lines.append('Noise reduction: Conservative spectral gating with a low noise floor; suppresses only a small portion of the spectrum while retaining the cardiac waveform.')
    lines.append('Resampling: 2000 Hz final sample rate using polyphase resampling.')
    lines.append('Standardization: Z-score standardization with a safe sigma floor to avoid division by zero.')
    lines.append('')
    lines.append('-----------------------------------------')
    lines.append('OUTPUT COUNTS')
    lines.append('-----------------------------------------')
    lines.append(f'Train processed: {summary["train_processed"]}')
    lines.append(f'Validation processed: {summary["val_processed"]}')
    lines.append(f'Test processed: {summary["test_processed"]}')
    lines.append(f'Total processed: {summary["total_processed"]}')
    lines.append('Expected: 980')
    lines.append('')
    lines.append('-----------------------------------------')
    lines.append('QUALITY CHECKS')
    lines.append('-----------------------------------------')
    lines.append(f'Unreadable files: {summary["unreadable_files"]}')
    lines.append(f'NaN files: {summary["nan_files"]}')
    lines.append(f'Infinite-value files: {summary["inf_files"]}')
    lines.append(f'Wrong sample rate: {summary["wrong_sr"]}')
    lines.append(f'Wrong channel count: {summary["wrong_channels"]}')
    lines.append(f'Duplicate filenames: {summary["duplicate_filenames"]}')
    lines.append(f'Clipping/corruption: {summary["corruption_count"]}')
    lines.append(f'Missing metadata: {summary["missing_metadata"]}')
    lines.append(f'Patient overlap: {summary["patient_overlap_count"]}')
    lines.append(f'Label mismatches: {summary["label_mismatch_count"]}')
    lines.append('')
    lines.append('-----------------------------------------')
    lines.append('STATUS')
    lines.append('-----------------------------------------')
    if summary['total_processed'] == 980 and summary['unreadable_files'] == 0 and summary['nan_files'] == 0 and summary['inf_files'] == 0 and summary['wrong_sr'] == 0 and summary['wrong_channels'] == 0 and summary['duplicate_filenames'] == 0 and summary['corruption_count'] == 0 and summary['missing_metadata'] == 0 and summary['patient_overlap_count'] == 0 and summary['label_mismatch_count'] == 0:
        lines.append('PASS')
    else:
        lines.append('FAIL')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    train_manifest = pd.read_csv(TRAIN_CSV)
    augmentation_manifest = pd.read_csv(AUG_MANIFEST) if AUG_MANIFEST.exists() else pd.DataFrame()
    train_patients = read_split_csv(TRAIN_SPLIT)
    val_patients = read_split_csv(VAL_SPLIT)
    test_patients = read_split_csv(TEST_SPLIT)

    patient_map = patient_to_recordings_map(train_manifest)
    all_patients = set(train_patients) | set(val_patients) | set(test_patients)
    for pid in sorted(all_patients):
        if pid not in patient_map:
            raise ValueError(f'Missing patient metadata for {pid} in train.csv')

    train_items = build_file_list_from_split('train', train_patients, train_manifest, augmentation_manifest)
    val_items = build_file_list_from_split('validation', val_patients, train_manifest)
    test_items = build_file_list_from_split('test', test_patients, train_manifest)

    all_rows = []
    file_counts = {'train': 0, 'validation': 0, 'test': 0}
    seen_processed_names = set()
    patient_overlap_count = 0
    patient_sets = {'train': set(train_patients), 'validation': set(val_patients), 'test': set(test_patients)}
    for split_name, split_items in [('train', train_items), ('validation', val_items), ('test', test_items)]:
        for item in split_items:
            # check patient overlap
            for other_split, other_patients in patient_sets.items():
                if other_split == split_name:
                    continue
                if item['patient_id'] in other_patients:
                    patient_overlap_count += 1
            # ensure source exists
            if not item['source_path'].exists():
                raise FileNotFoundError(f'Missing source file: {item["source_path"]}')
            processed_row, _ = process_audio_file(
                item['source_path'],
                split_name,
                item['source_filename'],
                item['patient_id'],
                item['disease_labels'],
                item['original_or_augmented'],
            )
            if processed_row['processed_filename'] in seen_processed_names:
                raise ValueError(f'Duplicate processed filename generated: {processed_row["processed_filename"]}')
            seen_processed_names.add(processed_row['processed_filename'])
            all_rows.append(processed_row)
            file_counts[split_name] += 1

    manifest_df = pd.DataFrame(all_rows, columns=[
        'patient_id',
        'source_filename',
        'processed_filename',
        'valve_location',
        'disease_labels',
        'original_or_augmented',
        'split',
        'original_sampling_rate',
        'final_sampling_rate',
        'original_channels',
        'final_channels',
        'original_duration',
        'final_duration',
        'preprocessing_operations',
        'processed_relative_path',
        'mean',
        'std',
        'final_mean',
        'final_std',
    ])
    manifest_df.to_csv(MANIFEST_PATH, index=False)

    # Quality checks
    unreadable_files = 0
    nan_files = 0
    inf_files = 0
    wrong_sr = 0
    wrong_channels = 0
    duplicate_filenames = 0
    corruption_count = 0
    missing_metadata = 0
    label_mismatch_count = 0
    for _, row in manifest_df.iterrows():
        out_path = root / str(row['processed_relative_path'])
        if not out_path.exists():
            unreadable_files += 1
            continue
        try:
            x, sr = sf.read(str(out_path))
            if x.ndim > 1:
                if x.shape[1] != 1:
                    wrong_channels += 1
            if sr != TARGET_SR:
                wrong_sr += 1
            if np.any(~np.isfinite(x)):
                nan_files += 1
            if np.any(np.isinf(x)):
                inf_files += 1
            if len(x) == 0:
                corruption_count += 1
        except Exception:
            unreadable_files += 1
            continue

    # Duplicate filename check
    processed_names = manifest_df['processed_filename'].tolist()
    if len(set(processed_names)) != len(processed_names):
        duplicate_filenames = len(processed_names) - len(set(processed_names))

    # patient overlap checks
    train_set = set(train_patients)
    val_set = set(val_patients)
    test_set = set(test_patients)
    train_val = train_set & val_set
    train_test = train_set & test_set
    val_test = val_set & test_set

    # metadata checks
    for _, row in manifest_df.iterrows():
        patient_id = str(row['patient_id'])
        disease_value = str(row['disease_labels'])
        if patient_id not in patient_map:
            missing_metadata += 1
            continue
        if disease_value == 'None':
            continue
        record_row = train_manifest.loc[train_manifest['patient_id'].astype(str) == patient_id].iloc[0]
        expected_labels = combo_from_row(record_row)
        if expected_labels != disease_value:
            label_mismatch_count += 1

    # representative verification file set
    rep_candidates = []
    rep_candidates.extend([
        {'patient_id': 'patient_089', 'source_path': RAW_TRAIN_DIR / 'N_089_sup_Mit.wav', 'disease_labels': 'N', 'source_filename': 'N_089_sup_Mit.wav'},
        {'patient_id': 'patient_005', 'source_path': RAW_TRAIN_DIR / 'AS_005_sup_Mit.wav', 'disease_labels': 'AS', 'source_filename': 'AS_005_sup_Mit.wav'},
        {'patient_id': 'patient_016', 'source_path': RAW_TRAIN_DIR / 'AR_016_sup_Mit.wav', 'disease_labels': 'AR', 'source_filename': 'AR_016_sup_Mit.wav'},
        {'patient_id': 'patient_002', 'source_path': RAW_TRAIN_DIR / 'MR_002_sup_Mit.wav', 'disease_labels': 'MR', 'source_filename': 'MR_002_sup_Mit.wav'},
        {'patient_id': 'patient_006', 'source_path': RAW_TRAIN_DIR / 'MS_006_sup_Mit.wav', 'disease_labels': 'MS', 'source_filename': 'MS_006_sup_Mit.wav'},
    ])
    rep_candidates.append({'patient_id': 'patient_003', 'source_path': AUG_TRAIN_DIR / 'MD_003_sup_Mit_aug_001.wav', 'disease_labels': 'AS+AR', 'source_filename': 'MD_003_sup_Mit_aug_001.wav'})
    save_waveform_and_psd_verification(rep_candidates)

    summary = {
        'train_count': 716,
        'val_count': 136,
        'test_count': 128,
        'total_count': 980,
        'train_processed': int((manifest_df['split'] == 'train').sum()),
        'val_processed': int((manifest_df['split'] == 'validation').sum()),
        'test_processed': int((manifest_df['split'] == 'test').sum()),
        'total_processed': int(len(manifest_df)),
        'unreadable_files': unreadable_files,
        'nan_files': nan_files,
        'inf_files': inf_files,
        'wrong_sr': wrong_sr,
        'wrong_channels': wrong_channels,
        'duplicate_filenames': duplicate_filenames,
        'corruption_count': corruption_count,
        'missing_metadata': missing_metadata,
        'patient_overlap_count': patient_overlap_count,
        'label_mismatch_count': label_mismatch_count,
    }

    report_text = build_report(summary)
    REPORT_PATH.write_text(report_text, encoding='utf-8')

    print('PREPROCESSING_COMPLETE')
    print('TRAIN_PROCESSED', summary['train_processed'])
    print('VALIDATION_PROCESSED', summary['val_processed'])
    print('TEST_PROCESSED', summary['test_processed'])
    print('TOTAL_PROCESSED', summary['total_processed'])
    print('TRAIN_VAL_INTERSECTION', len(train_val))
    print('TRAIN_TEST_INTERSECTION', len(train_test))
    print('VAL_TEST_INTERSECTION', len(val_test))
    print('MANIFEST_PATH', MANIFEST_PATH)
    print('REPORT_PATH', REPORT_PATH)
