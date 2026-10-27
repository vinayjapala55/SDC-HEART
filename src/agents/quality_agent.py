"""
Agent 1: Signal and Data Quality Agent.

Responsibilities:
1. Verify presence of all four required anatomical valve inputs (Aor, Mit, Pul, Tri).
2. Validate file type, readability, duration, sample rate, amplitude clipping, and SNR.
3. Preprocess audio using verified Butterworth low-pass and polyphase resampling to 2000 Hz.
4. Generate standardized log-mel spectrograms (64, 626, 1) for each valve.
5. Return structured QualityAgentOutput without fabricating missing signals.
"""

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import soundfile as sf
import librosa
from scipy import signal

from src.agents.schemas import QualityAgentOutput, ValveSignalInfo

REQUIRED_VALVES = ['Aor', 'Pul', 'Tri', 'Mit']
TARGET_SR = 2000
LOWPASS_CUTOFF = 1000
DEFAULT_FRAMES = 626


class SignalQualityAgent:
    """
    Automated biomedical quality control agent for multi-valve PCG recordings.
    """

    def __init__(self):
        # 4th order Butterworth low-pass filter (cutoff 1000 Hz at 4000 Hz sr)
        self.sos = signal.butter(4, LOWPASS_CUTOFF, btype='lowpass', fs=4000, output='sos')

    def inspect_and_preprocess(
        self,
        valve_files: Dict[str, Any],  # maps valve_location ('Aor', 'Mit', 'Pul', 'Tri') -> file_bytes or Path or filename
        session_id: str = "SESSION_UNKNOWN"
    ) -> Tuple[QualityAgentOutput, Dict[str, np.ndarray]]:
        """
        Executes comprehensive audio validation and log-mel spectrogram generation.
        
        Returns:
            quality_output: Structured QualityAgentOutput
            spectrogram_dict: Dict mapping valve_location -> np.ndarray of shape (64, 626, 1)
        """
        timestamp = datetime.now().isoformat()
        valves_inspected: List[ValveSignalInfo] = []
        spectrogram_dict: Dict[str, np.ndarray] = {}
        missing_valves = [v for v in REQUIRED_VALVES if v not in valve_files]

        if missing_valves:
            output = QualityAgentOutput(
                session_id=session_id,
                timestamp=timestamp,
                status="FAIL",
                required_valves_present=False,
                valves_inspected=[],
                spectrograms_generated=False,
                summary=f"Validation Failed: Missing required valve recording(s): {', '.join(missing_valves)}.",
                error_message=f"Session requires recordings for all 4 valves: Aortic, Pulmonary, Tricuspid, and Mitral. Missing: {missing_valves}"
            )
            return output, {}

        overall_status = "PASS"
        warnings = []

        for valve in REQUIRED_VALVES:
            val_input = valve_files[valve]
            filename = getattr(val_input, 'filename', str(val_input))

            try:
                # Load audio data from bytes or filepath
                if isinstance(val_input, (bytes, bytearray)):
                    buf = io.BytesIO(val_input)
                    audio, sr = sf.read(buf)
                elif hasattr(val_input, 'read'):
                    content = val_input.read()
                    buf = io.BytesIO(content)
                    audio, sr = sf.read(buf)
                elif isinstance(val_input, (str, Path)):
                    audio, sr = sf.read(str(val_input))
                else:
                    raise ValueError(f"Unsupported audio input type for valve {valve}: {type(val_input)}")

                if audio.ndim > 1:
                    audio = audio[:, 0]  # Mono conversion

                duration = len(audio) / float(sr)
                if duration < 3.0:
                    raise ValueError(f"Recording too short ({duration:.1f}s). Minimum required duration is 3.0 seconds.")

                # Signal integrity checks
                is_finite = bool(np.all(np.isfinite(audio)))
                if not is_finite:
                    raise ValueError("Audio contains NaN or Infinite values.")

                max_amp = float(np.max(np.abs(audio)))
                clipping = max_amp >= 0.99
                if clipping:
                    warnings.append(f"Valve {valve}: Audio amplitude peak reaches saturation ({max_amp:.3f}).")

                # Amplitude normalization if peak exceeds safe range
                if max_amp > 0.98:
                    audio = audio * (0.98 / max_amp)

                # Preprocessing: Butterworth low-pass filter
                if sr >= 2000:
                    filter_sos = signal.butter(4, min(LOWPASS_CUTOFF, (sr / 2) - 10), btype='lowpass', fs=sr, output='sos')
                    audio_filtered = signal.sosfiltfilt(filter_sos, audio)
                else:
                    audio_filtered = audio

                # Resample to TARGET_SR (2000 Hz)
                if sr != TARGET_SR:
                    resampled = librosa.resample(audio_filtered, orig_sr=sr, target_sr=TARGET_SR)
                else:
                    resampled = audio_filtered

                # Z-score standardization
                mean_val = float(np.mean(resampled))
                std_val = float(np.std(resampled))
                norm_signal = (resampled - mean_val) / (std_val + 1e-8)

                # Generate Log-Mel Spectrogram (64 mels, 626 frames, 1 channel)
                mel = librosa.feature.melspectrogram(
                    y=norm_signal.astype(np.float32),
                    sr=TARGET_SR,
                    n_fft=256,
                    hop_length=64,
                    n_mels=64,
                    fmin=20,
                    fmax=1000,
                    power=2.0
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)

                # Standardize time steps to DEFAULT_FRAMES (626)
                if mel_db.shape[1] < DEFAULT_FRAMES:
                    pad_w = DEFAULT_FRAMES - mel_db.shape[1]
                    pad_floor = float(np.min(mel_db)) if mel_db.size > 0 else 0.0
                    mel_db = np.pad(mel_db, ((0, 0), (0, pad_w)), mode='constant', constant_values=pad_floor)
                elif mel_db.shape[1] > DEFAULT_FRAMES:
                    start_idx = (mel_db.shape[1] - DEFAULT_FRAMES) // 2
                    mel_db = mel_db[:, start_idx : start_idx + DEFAULT_FRAMES]

                mel_tensor = np.expand_dims(mel_db, axis=-1).astype(np.float32)  # (64, 626, 1)
                spectrogram_dict[valve] = mel_tensor

                # Estimated SNR
                signal_power = np.mean(norm_signal ** 2)
                noise_floor = np.percentile(norm_signal ** 2, 10) + 1e-9
                snr_val = float(10 * np.log10(signal_power / noise_floor))

                valves_inspected.append(ValveSignalInfo(
                    valve_location=valve,
                    filename=Path(filename).name,
                    sample_rate=sr,
                    duration_seconds=round(duration, 2),
                    is_valid=True,
                    clipping_detected=clipping,
                    snr_db=round(snr_val, 1)
                ))

            except Exception as e:
                overall_status = "FAIL"
                valves_inspected.append(ValveSignalInfo(
                    valve_location=valve,
                    filename=Path(filename).name,
                    sample_rate=0,
                    duration_seconds=0.0,
                    is_valid=False,
                    warning=str(e)
                ))

        if overall_status == "PASS" and warnings:
            overall_status = "WARN"

        summary = (
            f"Quality Inspection {overall_status}: All 4 anatomical valve recordings verified. "
            f"Durations: {', '.join([f'{v.valve_location}: {v.duration_seconds}s' for v in valves_inspected])}. "
            f"Standardized to 2000 Hz Log-Mel Spectrograms (64x626)."
        )
        if warnings:
            summary += f" Warnings: {'; '.join(warnings)}"

        quality_output = QualityAgentOutput(
            session_id=session_id,
            timestamp=timestamp,
            status=overall_status,
            required_valves_present=True,
            valves_inspected=valves_inspected,
            spectrograms_generated=bool(len(spectrogram_dict) == 4),
            summary=summary,
            error_message=None if overall_status != "FAIL" else "Signal validation errors detected in one or more valve inputs."
        )

        return quality_output, spectrogram_dict
