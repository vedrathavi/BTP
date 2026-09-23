import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts.analyze_thresholds import (
    analyze_arrays,
    calculate_binary_metrics,
    summarize_round,
    validate_prediction_file,
)


class ThresholdAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 0, 1, 1])
        self.y_prob = np.array([0.10, 0.40, 0.60, 0.90])

    def test_metrics_at_threshold(self):
        metrics = calculate_binary_metrics(self.y_true, self.y_prob, 0.5)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["specificity"], 1.0)
        self.assertEqual(metrics["sensitivity"], 1.0)
        self.assertEqual(metrics["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["tn"], 2)
        self.assertEqual(metrics["fp"], 0)
        self.assertEqual(metrics["fn"], 0)
        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["auc"], 1.0)

    def test_sweep_contains_requested_thresholds(self):
        sweep = analyze_arrays(self.y_true, self.y_prob, thresholds=[0.05, 0.25, 0.95])
        self.assertEqual(sweep["threshold"].tolist(), [0.05, 0.25, 0.95])
        self.assertEqual(len(sweep), 3)

    def test_round_summary_reports_best_and_fixed_threshold(self):
        sweep = analyze_arrays(self.y_true, self.y_prob, thresholds=[0.25, 0.5, 0.75])
        summary = summarize_round(1, sweep)
        self.assertEqual(summary["round"], 1)
        self.assertEqual(summary["best_f1"]["threshold"], 0.5)
        self.assertEqual(summary["best_balanced_accuracy"]["threshold"], 0.5)
        self.assertEqual(summary["threshold_0.25"]["threshold"], 0.25)

    def test_saved_prediction_file_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "round_01.npz"
            np.savez_compressed(path, round=np.array(1), y_true=self.y_true, y_prob=self.y_prob)
            y_true, y_prob = validate_prediction_file(path, expected_samples=4)
            self.assertEqual(y_true.shape[0], 4)
            self.assertEqual(y_prob.shape[0], 4)

    def test_saved_prediction_file_validation_rejects_wrong_sample_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "round_01.npz"
            np.savez_compressed(path, y_true=self.y_true, y_prob=self.y_prob)
            with self.assertRaisesRegex(ValueError, "Expected 485 samples"):
                validate_prediction_file(path, expected_samples=485)


if __name__ == "__main__":
    unittest.main()