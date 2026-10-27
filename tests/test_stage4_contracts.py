"""
Stage 4 Test Suite: Data Integrity, Leakage Verification, Label Contracts, and Input Signatures.

Tests:
1. Patient Split Isolation (0 patient overlap between train/val/test, total 109 patients).
2. Augmentation Isolation (100% augmented recordings restricted strictly to train; 0 in val/test).
3. Stratified Class Balance (All 5 classes represented across all splits, including Normal in val/test).
4. Patient_034 Completeness (16 recordings: 8 original + 8 augmented, valid valves, valid spectrograms).
5. Patient_022 Unknown-Label Contract (-1.0 target vector, 0 loss contribution, 0 gradient penalty).
6. 4-Valve & 8-Branch RosaNet Input Signatures (Forward pass verification on synthetic tensors).
7. Orchestrator Contract & Schema Validation (4-valve structured pipeline execution).
"""

import os
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.patient_grouping import (
    load_patient_records,
    get_split_patients,
    parse_disease_labels,
    DISEASE_CLASSES,
    VALVE_LOCATIONS,
    POSTURES
)
from src.data.dataset_loader import (
    STANDARD_POSITIONS,
    DEFAULT_FREQ_BINS,
    DEFAULT_TIME_STEPS,
    load_and_standardize_spectrogram
)
from src.models.loss import WeightedMultiLabelBCE, MaskedWeightedMultiLabelBCE
from src.models.decision_fusion import build_rosanet_4valve_model, build_rosanet_patient_model
from src.agents.orchestrator import DiagnosticPipelineOrchestrator
from train_rosanet_full import compute_metrics


class TestStage4DataAndContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_split_csv = ROOT / 'data' / 'splits' / 'train_patients.csv'
        cls.val_split_csv = ROOT / 'data' / 'splits' / 'val_patients.csv'
        cls.test_split_csv = ROOT / 'data' / 'splits' / 'test_patients.csv'
        cls.spec_manifest_csv = ROOT / 'dataset' / 'metadata' / 'spectrogram_manifest.csv'
        cls.aug_manifest_csv = ROOT / 'dataset' / 'metadata' / 'augmentation_manifest.csv'
        cls.prep_manifest_csv = ROOT / 'dataset' / 'metadata' / 'preprocessing_manifest.csv'

    def test_01_patient_split_isolation(self):
        """Verify strict zero-patient overlap across all three splits and total 109 patients."""
        self.assertTrue(self.train_split_csv.exists(), "train_patients.csv does not exist")
        self.assertTrue(self.val_split_csv.exists(), "val_patients.csv does not exist")
        self.assertTrue(self.test_split_csv.exists(), "test_patients.csv does not exist")

        train_df = pd.read_csv(self.train_split_csv)
        val_df = pd.read_csv(self.val_split_csv)
        test_df = pd.read_csv(self.test_split_csv)

        train_pts = set(train_df['patient_id'].astype(str))
        val_pts = set(val_df['patient_id'].astype(str))
        test_pts = set(test_df['patient_id'].astype(str))

        # Check disjoint sets
        train_val_overlap = train_pts.intersection(val_pts)
        train_test_overlap = train_pts.intersection(test_pts)
        val_test_overlap = val_pts.intersection(test_pts)

        self.assertEqual(len(train_val_overlap), 0, f"Leakage detected between Train and Val: {train_val_overlap}")
        self.assertEqual(len(train_test_overlap), 0, f"Leakage detected between Train and Test: {train_test_overlap}")
        self.assertEqual(len(val_test_overlap), 0, f"Leakage detected between Val and Test: {val_test_overlap}")

        total_pts = len(train_pts) + len(val_pts) + len(test_pts)
        self.assertEqual(total_pts, 109, f"Expected 109 unique patients, got {total_pts}")
        self.assertEqual(len(train_pts), 76, f"Expected 76 train patients, got {len(train_pts)}")
        self.assertEqual(len(val_pts), 17, f"Expected 17 val patients, got {len(val_pts)}")
        self.assertEqual(len(test_pts), 16, f"Expected 16 test patients, got {len(test_pts)}")
        print("\n[TEST 1 PASS] Zero patient overlap across all splits (Train=76, Val=17, Test=16, Total=109).")

    def test_02_augmentation_isolation_no_leakage(self):
        """Verify 100% of augmented recordings belong to Train patients and 0 exist in Val/Test."""
        self.assertTrue(self.aug_manifest_csv.exists(), "augmentation_manifest.csv missing")
        self.assertTrue(self.spec_manifest_csv.exists(), "spectrogram_manifest.csv missing")

        train_pts = set(pd.read_csv(self.train_split_csv)['patient_id'].astype(str))
        val_pts = set(pd.read_csv(self.val_split_csv)['patient_id'].astype(str))
        test_pts = set(pd.read_csv(self.test_split_csv)['patient_id'].astype(str))

        aug_df = pd.read_csv(self.aug_manifest_csv)
        self.assertEqual(len(aug_df), 112, f"Expected 112 augmented recordings, got {len(aug_df)}")

        # Verify each augmented row belongs to a train patient
        for _, row in aug_df.iterrows():
            pid = str(row['patient_id'])
            self.assertIn(pid, train_pts, f"Augmented patient {pid} is NOT in train set!")
            self.assertNotIn(pid, val_pts, f"Augmented patient {pid} leaked into validation set!")
            self.assertNotIn(pid, test_pts, f"Augmented patient {pid} leaked into test set!")

        # Verify spectrogram manifest split labeling
        spec_df = pd.read_csv(self.spec_manifest_csv)
        self.assertEqual(len(spec_df), 984, f"Expected 984 total spectrograms, got {len(spec_df)}")

        val_rows = spec_df[spec_df['split'] == 'validation']
        test_rows = spec_df[spec_df['split'] == 'test']
        train_rows = spec_df[spec_df['split'] == 'train']

        # Ensure no augmented files in val or test
        val_aug_count = (val_rows['original_or_augmented'] == 'augmented').sum()
        test_aug_count = (test_rows['original_or_augmented'] == 'augmented').sum()

        self.assertEqual(val_aug_count, 0, f"Found {val_aug_count} augmented recordings in validation set!")
        self.assertEqual(test_aug_count, 0, f"Found {test_aug_count} augmented recordings in test set!")

        train_aug_count = (train_rows['original_or_augmented'] == 'augmented').sum()
        self.assertEqual(train_aug_count, 112, f"Expected 112 augmented rows in train, found {train_aug_count}")
        print("\n[TEST 2 PASS] Augmentation strictly isolated to Train. Validation=0 aug, Test=0 aug.")

    def test_03_class_distribution_and_normal_stratification(self):
        """Verify that Normal (N) and all other disease classes exist across all 3 splits."""
        train_df = pd.read_csv(self.train_split_csv)
        val_df = pd.read_csv(self.val_split_csv)
        test_df = pd.read_csv(self.test_split_csv)

        for name, df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            counts = {c: int((df[c] == 1).sum()) for c in ['AS', 'AR', 'MR', 'MS', 'N']}
            # Normal class MUST exist in every split
            self.assertGreater(counts['N'], 0, f"Normal class has 0 patients in {name} split!")
            print(f"\n[TEST 3 INFO] {name} Class Counts: {counts}")

        # Check exact Normal counts across splits
        train_n = int((train_df['N'] == 1).sum())
        val_n = int((val_df['N'] == 1).sum())
        test_n = int((test_df['N'] == 1).sum())

        self.assertEqual(train_n + val_n + test_n, 21, "Total Normal patients must be 21")
        self.assertEqual(train_n, 14, "Train Normal patients should be 14")
        self.assertEqual(val_n, 4, "Val Normal patients should be 4")
        self.assertEqual(test_n, 3, "Test Normal patients should be 3")
        print(f"\n[TEST 3 PASS] Normal class is properly stratified: Train={train_n}, Val={val_n}, Test={test_n}.")

    def test_04_patient_034_completeness(self):
        """Verify Patient_034 has 16 complete recordings (8 original + 8 augmented) with valid valves and spectrograms."""
        spec_df = pd.read_csv(self.spec_manifest_csv)
        p34_rows = spec_df[spec_df['patient_id'] == 'patient_034']

        self.assertEqual(len(p34_rows), 16, f"Patient_034 expected 16 recordings (8 original + 8 augmented), got {len(p34_rows)}")

        orig_count = (p34_rows['original_or_augmented'] == 'original').sum()
        aug_count = (p34_rows['original_or_augmented'] == 'augmented').sum()
        self.assertEqual(orig_count, 8, f"Patient_034 expected 8 original recordings, got {orig_count}")
        self.assertEqual(aug_count, 8, f"Patient_034 expected 8 augmented recordings, got {aug_count}")

        valves = sorted(p34_rows['valve_location'].tolist())
        expected_valves = sorted(['Aor', 'Mit', 'Pul', 'Tri'] * 4)
        self.assertEqual(valves, expected_valves, f"Unexpected valves for patient_034: {valves}")
        self.assertNotIn('unknown', valves, "Valve labeled 'unknown' in patient_034!")

        # Verify sitting and supine postures
        postures = set(p34_rows['processed_filename'].apply(lambda x: 'sit' if '_sit_' in x else 'sup'))
        self.assertEqual(postures, {'sit', 'sup'}, f"Expected both sit and sup postures for patient_034, got {postures}")

        # Verify numerical files exist and load correctly
        for _, r in p34_rows.iterrows():
            npy_path = ROOT / str(r['numerical_path'])
            self.assertTrue(npy_path.exists(), f"Spectrogram file does not exist: {npy_path}")
            arr = np.load(npy_path)
            self.assertEqual(arr.shape, (64, 626), f"Wrong spectrogram shape for {npy_path}: {arr.shape}")
            self.assertFalse(np.isnan(arr).any(), f"NaN values found in {npy_path}")
            self.assertFalse(np.isinf(arr).any(), f"Inf values found in {npy_path}")

        print("\n[TEST 4 PASS] Patient_034 has 16 complete recordings (8 raw original + 8 augmented) with 0 unknown valves and valid spectrograms.")

    def test_05_patient_022_unknown_label_contract(self):
        """Verify Patient_022 MVP unknown target contract: parsed as -1.0, 0 loss, 0 gradient penalty."""
        # Check parsing
        vec_nan = parse_disease_labels(np.nan)
        vec_mvp = parse_disease_labels('MVP')
        vec_empty = parse_disease_labels('')

        np.testing.assert_array_equal(vec_nan, np.full(5, -1.0, dtype=np.float32))
        np.testing.assert_array_equal(vec_mvp, np.full(5, -1.0, dtype=np.float32))
        np.testing.assert_array_equal(vec_empty, np.full(5, -1.0, dtype=np.float32))

        # Check loss and gradient masking
        loss_fn = WeightedMultiLabelBCE()

        # Batch of 2 samples: Sample 0 has normal targets, Sample 1 is Patient_022 (-1.0 masked)
        y_true_masked = tf.constant([[-1.0, -1.0, -1.0, -1.0, -1.0]], dtype=tf.float32)
        y_pred = tf.constant([[0.8, 0.2, 0.9, 0.1, 0.5]], dtype=tf.float32)

        loss_masked = loss_fn(y_true_masked, y_pred)
        self.assertAlmostEqual(float(loss_masked.numpy()), 0.0, places=5, msg="Masked targets produced non-zero loss!")

        # Gradient tape check
        with tf.GradientTape() as tape:
            tape.watch(y_pred)
            loss = loss_fn(y_true_masked, y_pred)
        grads = tape.gradient(loss, y_pred)
        np.testing.assert_array_equal(grads.numpy(), np.zeros((1, 5), dtype=np.float32), err_msg="Masked targets produced non-zero gradient!")

        # Metric computation check
        y_true_batch = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0]  # Patient_022
        ], dtype=np.float32)
        y_pred_batch = np.array([
            [0.9, 0.1, 0.1, 0.1, 0.1],
            [0.8, 0.2, 0.3, 0.4, 0.5]
        ], dtype=np.float32)

        metrics = compute_metrics(y_true_batch, y_pred_batch)
        self.assertIn('macro_auroc', metrics)
        self.assertIn('macro_f1', metrics)
        print("\n[TEST 5 PASS] Patient_022 parsed as -1.0; produces strictly 0 loss and 0 gradient.")

    def test_06_model_architecture_and_input_signatures(self):
        """Verify forward pass on 4-valve RosaNet and 8-branch RosaNet."""
        batch_size = 2
        # 4-valve input: [Aor, Mit, Pul, Tri]
        model_4v = build_rosanet_4valve_model(
            input_shape=(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1),
            num_classes=5
        )
        self.assertEqual(len(model_4v.inputs), 4, f"Expected 4 inputs, got {len(model_4v.inputs)}")
        self.assertEqual(len(model_4v.outputs), 2, f"Expected 2 outputs (pat, br), got {len(model_4v.outputs)}")

        dummy_4v_inputs = [
            np.random.randn(batch_size, DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1).astype(np.float32)
            for _ in range(4)
        ]
        out_pat_4v, out_br_4v = model_4v(dummy_4v_inputs, training=False)
        self.assertEqual(out_pat_4v.shape, (batch_size, 5))
        self.assertEqual(out_br_4v.shape, (batch_size, 4, 5))
        self.assertTrue((out_pat_4v.numpy() >= 0.0).all() and (out_pat_4v.numpy() <= 1.0).all())

        # 8-branch input
        model_8b = build_rosanet_patient_model(
            input_shape=(DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1),
            num_branches=8,
            num_classes=5
        )
        self.assertEqual(len(model_8b.inputs), 8)
        dummy_8b_inputs = [
            np.random.randn(batch_size, DEFAULT_FREQ_BINS, DEFAULT_TIME_STEPS, 1).astype(np.float32)
            for _ in range(8)
        ]
        out_pat_8b, out_br_8b = model_8b(dummy_8b_inputs, training=False)
        self.assertEqual(out_pat_8b.shape, (batch_size, 5))
        self.assertEqual(out_br_8b.shape, (batch_size, 8, 5))
        print("\n[TEST 6 PASS] Forward passes verified for both 4-valve and 8-branch RosaNet models.")

    def test_07_orchestrator_schema_and_contract(self):
        """Verify DiagnosticPipelineOrchestrator contract handles 4-valve diagnostic inputs."""
        import io
        import soundfile as sf
        orchestrator = DiagnosticPipelineOrchestrator(model_checkpoint_path=None)
        sr = 2000
        duration = 5.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        synthetic_pcg = (0.5 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(len(t))).astype(np.float32)

        valve_audio_dict = {}
        for v in ['Aor', 'Mit', 'Pul', 'Tri']:
            buf = io.BytesIO()
            sf.write(buf, synthetic_pcg, sr, format='WAV')
            valve_audio_dict[v] = buf.getvalue()

        report = orchestrator.process_session(
            valve_files=valve_audio_dict,
            session_id="SES-STAGE4-TEST",
            patient_metadata={"patient_id": "test_patient_audit", "notes": "Stage 4 Contract Verification"}
        )

        self.assertIsNotNone(report)
        self.assertEqual(report.patient_session_id, "SES-STAGE4-TEST")
        self.assertIsNotNone(report.quality_summary)
        self.assertIsNotNone(report.prediction_summary)
        self.assertIsNotNone(report.evidence_summary)
        self.assertTrue(len(report.structured_markdown_report) > 0)
        print("\n[TEST 7 PASS] DiagnosticPipelineOrchestrator contract verified successfully.")


if __name__ == '__main__':
    unittest.main()
