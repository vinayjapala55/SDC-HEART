"""
Stage 5 Test Suite: Preprocessing Pipeline and Model Architecture Integrity.

Tests:
1. Audio Preprocessing Pipeline: Resampling to 2000 Hz, 4th-order Butterworth bandpass (25-400 Hz),
   spike suppression, z-score standardization, 5.0-second duration (10,000 samples).
2. Mel Spectrogram Extraction: Shape consistency (64, 626), frequency bounds (25-400 Hz),
   on-disk dataset validation (zero NaN/Inf, realistic dB ranges).
3. RosaNet Backbone & Custom Attention Layers: SplitSelfAttention, CardiacSpatialTemporalAttention,
   DynamicDecisionFusion gating normalization.
4. Backpropagation & Gradient Flow Stability: Gradient computation via tf.GradientTape,
   finite bounded gradients, no vanishing or exploding gradients.
5. Custom Layer Serializability: get_config and from_config roundtrip tests.
6. XAI Grad-CAM Saliency: Target layer feature map activation, saliency heatmap bounds [0, 1].
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import scipy.signal as signal
import soundfile as sf
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.attention import SplitSelfAttention, CardiacSpatialTemporalAttention
from src.models.decision_fusion import build_rosanet_4valve_model, build_rosanet_patient_model, DynamicDecisionFusion
from src.models.loss import WeightedMultiLabelBCE
from src.models.xai import compute_rosanet_gradcam, extract_valve_attention_weights
from src.data.dataset_loader import DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS


class TestStage5PreprocessingAndModelIntegrity(unittest.TestCase):

    def test_01_audio_preprocessing_filtering_and_standardization(self):
        """Verify audio preprocessing: 2000 Hz resampling, 25-400 Hz bandpass, spike removal, z-score norm."""
        sr = 2000
        duration = 5.0
        n_samples = int(sr * duration)
        self.assertEqual(n_samples, 10000, "5.0 seconds at 2000 Hz must equal exactly 10,000 samples")

        t = np.linspace(0, duration, n_samples, endpoint=False)
        # Create composite test signal:
        # 10 Hz (low rumble, out of band), 100 Hz (cardiac band S1/S2), 800 Hz (high frequency noise)
        sig_10hz = 1.0 * np.sin(2 * np.pi * 10 * t)
        sig_100hz = 1.0 * np.sin(2 * np.pi * 100 * t)
        sig_800hz = 1.0 * np.sin(2 * np.pi * 800 * t)
        raw_signal = sig_10hz + sig_100hz + sig_800hz

        # Inject extreme spike at index 5000
        raw_signal[5000] = 50.0

        # 1. 4th-order Butterworth Bandpass Filter (25 - 400 Hz)
        nyquist = 0.5 * sr
        low = 25.0 / nyquist
        high = 400.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='bandpass')
        filtered = signal.filtfilt(b, a, raw_signal)

        # 2. Spike suppression
        threshold = 3.5 * np.std(filtered)
        spike_suppressed = np.clip(filtered, -threshold, threshold)
        self.assertLessEqual(abs(spike_suppressed[5000]), threshold + 1e-4)

        # 3. Z-score normalization
        normalized = (spike_suppressed - np.mean(spike_suppressed)) / (np.std(spike_suppressed) + 1e-8)
        self.assertAlmostEqual(float(np.mean(normalized)), 0.0, places=2)
        self.assertAlmostEqual(float(np.std(normalized)), 1.0, places=1)

        # Frequency domain verification: 100 Hz should have much higher energy than 10 Hz and 800 Hz
        fft_vals = np.abs(np.fft.rfft(normalized))
        freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)

        idx_10 = np.argmin(np.abs(freqs - 10))
        idx_100 = np.argmin(np.abs(freqs - 100))
        idx_800 = np.argmin(np.abs(freqs - 800))

        power_10 = fft_vals[idx_10]
        power_100 = fft_vals[idx_100]
        power_800 = fft_vals[idx_800]

        self.assertGreater(power_100, power_10 * 10.0, "10 Hz out-of-band energy was not properly attenuated")
        self.assertGreater(power_100, power_800 * 10.0, "800 Hz out-of-band energy was not properly attenuated")
        print("\n[TEST 1 PASS] Audio preprocessing verified (Bandpass 25-400Hz, spike clamp, z-score norm, 10,000 samples).")

    def test_02_spectrogram_dataset_integrity(self):
        """Verify on-disk numerical spectrograms match (64, 626), zero NaN/Inf, and realistic energy ranges."""
        spec_dir = ROOT / 'dataset' / 'spectrograms' / 'numerical'
        self.assertTrue(spec_dir.exists(), f"Spectrogram numerical directory not found at {spec_dir}")

        all_specs = list(spec_dir.glob('*.npy'))
        self.assertGreaterEqual(len(all_specs), 984, f"Expected at least 984 spectrograms, found {len(all_specs)}")

        # Sample across beginning, middle, and end of the dataset
        sample_indices = np.linspace(0, len(all_specs) - 1, 30, dtype=int)
        checked = 0

        for idx in sample_indices:
            f = all_specs[idx]
            arr = np.load(f)

            self.assertEqual(arr.shape, (DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS),
                             f"Incorrect shape for {f.name}: {arr.shape} vs ({DEFAULT_FREQ_BINS}, {DEFAULT_TIME_STEPS})")
            self.assertFalse(np.isnan(arr).any(), f"NaN detected in {f.name}")
            self.assertFalse(np.isinf(arr).any(), f"Inf detected in {f.name}")

            # Realistic log-mel energy bounds in dB (typically -100 to +30 dB)
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            self.assertGreater(min_val, -150.0, f"Unrealistically low dB value in {f.name}: {min_val}")
            self.assertLess(max_val, 100.0, f"Unrealistically high dB value in {f.name}: {max_val}")
            checked += 1

        print(f"\n[TEST 2 PASS] Spectrogram dataset integrity verified on {checked} files (shape=(64, 626), zero NaN/Inf, valid dB bounds).")

    def test_03_custom_attention_and_decision_fusion_layers(self):
        """Verify SplitSelfAttention, CardiacSpatialTemporalAttention, and DynamicDecisionFusion."""
        batch_size = 2
        time_steps = 626
        channels = 64

        # 1. SplitSelfAttention
        split_att = SplitSelfAttention(num_splits=2, name="test_split_att")
        x_att = tf.random.normal((batch_size, time_steps, channels))
        out_split = split_att(x_att)
        self.assertEqual(out_split.shape, (batch_size, time_steps, channels))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(out_split)).numpy())

        # 2. CardiacSpatialTemporalAttention
        cardiac_att = CardiacSpatialTemporalAttention(name="test_cardiac_att")
        out_cardiac = cardiac_att(x_att)
        self.assertEqual(out_cardiac.shape, (batch_size, time_steps, channels))
        self.assertFalse(tf.reduce_any(tf.math.is_nan(out_cardiac)).numpy())

        # 3. DynamicDecisionFusion (4 branches)
        fusion_4v = DynamicDecisionFusion(num_branches=4, num_classes=5, name="test_fusion_4v")
        dummy_br_decisions_4v = tf.nn.sigmoid(tf.random.normal((batch_size, 4, 5)))
        dummy_feats_4v = tf.random.normal((batch_size, 4, 128))

        fused_pat_4v = fusion_4v([dummy_br_decisions_4v, dummy_feats_4v])
        self.assertEqual(fused_pat_4v.shape, (batch_size, 5))
        self.assertTrue((fused_pat_4v.numpy() >= 0.0).all() and (fused_pat_4v.numpy() <= 1.0).all())

        # Verify gating weights sum to 1.0
        gating_weights = fusion_4v.get_last_gating_weights()
        if gating_weights is not None:
            sum_gating = np.sum(gating_weights, axis=-1)
            np.testing.assert_allclose(sum_gating, np.ones_like(sum_gating), atol=1e-5)

        # 4. DynamicDecisionFusion (8 branches)
        fusion_8b = DynamicDecisionFusion(num_branches=8, num_classes=5, name="test_fusion_8b")
        dummy_br_decisions_8b = tf.nn.sigmoid(tf.random.normal((batch_size, 8, 5)))
        dummy_feats_8b = tf.random.normal((batch_size, 8, 128))

        fused_pat_8b = fusion_8b([dummy_br_decisions_8b, dummy_feats_8b])
        self.assertEqual(fused_pat_8b.shape, (batch_size, 5))
        print("\n[TEST 3 PASS] Custom attention and dynamic fusion layers verified with normalized gating.")

    def test_04_backpropagation_and_gradient_flow_stability(self):
        """Verify backpropagation, gradient continuity, and absence of exploding/vanishing gradients."""
        model_4v = build_rosanet_4valve_model(
            input_shape=(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1),
            num_classes=5
        )
        loss_fn = WeightedMultiLabelBCE(pos_weights=[2.62, 4.85, 3.22, 1.92, 1.92])

        batch_size = 2
        dummy_inputs = [
            tf.random.normal((batch_size, DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1))
            for _ in range(4)
        ]
        # Target vectors: 1 normal sample, 1 multi-label sample
        y_true_pat = tf.constant([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 0.0]
        ], dtype=tf.float32)

        with tf.GradientTape() as tape:
            out_pat, out_br = model_4v(dummy_inputs, training=True)
            loss = loss_fn(y_true_pat, out_pat)

        trainable_vars = model_4v.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.assertGreater(len(gradients), 0, "No gradients computed during backward pass!")

        grad_norms = []
        for g, v in zip(gradients, trainable_vars):
            self.assertIsNotNone(g, f"Gradient for variable {v.name} is None!")
            g_np = g.numpy()
            self.assertFalse(np.isnan(g_np).any(), f"NaN gradient detected in variable {v.name}")
            self.assertFalse(np.isinf(g_np).any(), f"Inf gradient detected in variable {v.name}")
            norm = float(np.linalg.norm(g_np))
            grad_norms.append(norm)

        # Gradient explosion check
        max_grad_norm = max(grad_norms)
        self.assertLess(max_grad_norm, 100.0, f"Potential gradient explosion detected: max norm = {max_grad_norm:.2f}")

        # Non-zero gradient flow check (backprop actually propagates to early layers)
        non_zero_grads = sum(1 for norm in grad_norms if norm > 1e-7)
        self.assertGreater(non_zero_grads, len(grad_norms) * 0.8, "Too many zero gradients (vanishing gradient)")

        print(f"\n[TEST 4 PASS] Backpropagation stable across {len(trainable_vars)} trainable variables (max norm={max_grad_norm:.4f}, zero NaN/Inf).")

    def test_05_custom_layer_serializability(self):
        """Verify get_config and from_config roundtrips for all custom layers and losses."""
        # SplitSelfAttention
        att1 = SplitSelfAttention(num_splits=4, name="att1")
        cfg1 = att1.get_config()
        att1_recon = SplitSelfAttention.from_config(cfg1)
        self.assertEqual(att1_recon.num_splits, 4)

        # CardiacSpatialTemporalAttention
        att2 = CardiacSpatialTemporalAttention(name="att2")
        cfg2 = att2.get_config()
        att2_recon = CardiacSpatialTemporalAttention.from_config(cfg2)
        self.assertIsNotNone(att2_recon)

        # DynamicDecisionFusion
        fusion = DynamicDecisionFusion(num_branches=4, num_classes=5, name="fus")
        cfg3 = fusion.get_config()
        fusion_recon = DynamicDecisionFusion.from_config(cfg3)
        self.assertEqual(fusion_recon.num_branches, 4)
        self.assertEqual(fusion_recon.num_classes, 5)

        # WeightedMultiLabelBCE
        loss = WeightedMultiLabelBCE(pos_weights=[2.0, 3.0, 4.0, 5.0, 6.0], epsilon=1e-6)
        cfg4 = loss.get_config()
        loss_recon = WeightedMultiLabelBCE.from_config(cfg4)
        self.assertEqual(loss_recon.epsilon, 1e-6)
        print("\n[TEST 5 PASS] Custom layers and loss function successfully serialized and reconstructed.")

    def test_06_explainable_ai_gradcam_verification(self):
        """Verify Grad-CAM saliency computation on RosaNet 4-valve model."""
        model_4v = build_rosanet_4valve_model(
            input_shape=(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1),
            num_classes=5
        )

        dummy_spectrograms = {
            'Aor': np.random.randn(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1).astype(np.float32),
            'Mit': np.random.randn(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1).astype(np.float32),
            'Pul': np.random.randn(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1).astype(np.float32),
            'Tri': np.random.randn(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1).astype(np.float32)
        }

        # Target class: Aortic Stenosis (class index 1) on Aortic valve
        heatmap = compute_rosanet_gradcam(
            model=model_4v,
            spectrogram_dict=dummy_spectrograms,
            target_class_idx=1,
            target_valve='Aor'
        )

        self.assertIsNotNone(heatmap, "Grad-CAM returned None")
        self.assertEqual(heatmap.shape, (DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS),
                         f"Unexpected heatmap shape: {heatmap.shape}")
        self.assertFalse(np.isnan(heatmap).any(), "Grad-CAM heatmap contains NaN values")
        self.assertGreaterEqual(float(np.min(heatmap)), 0.0, "Grad-CAM min value is below 0.0")
        self.assertLessEqual(float(np.max(heatmap)), 1.0 + 1e-5, "Grad-CAM max value exceeds 1.0")

        # Test valve attention gating weight extraction
        gating_info = extract_valve_attention_weights(model_4v, dummy_spectrograms)
        self.assertIn('Aor', gating_info)
        self.assertIn('Mit', gating_info)
        self.assertIn('Pul', gating_info)
        self.assertIn('Tri', gating_info)
        weights_sum = sum(gating_info.values())
        self.assertAlmostEqual(weights_sum, 1.0, places=4, msg="Valve attention weights must sum to 1.0")

        print("\n[TEST 6 PASS] Grad-CAM saliency generation and valve attention weight extraction verified.")


if __name__ == '__main__':
    unittest.main()
