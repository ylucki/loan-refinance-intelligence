import unittest

import numpy as np

from src.evaluation.metrics import (
    classification_metrics_from_proba,
    regression_metrics,
    select_threshold,
    threshold_sweep,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_classification_metrics_shape(self):
        y_true = np.array([0, 1, 0, 1, 1, 0])
        proba = np.array([0.1, 0.9, 0.3, 0.7, 0.6, 0.2])
        out = classification_metrics_from_proba(y_true, proba, threshold=0.5)
        self.assertTrue("roc_auc" in out)
        self.assertTrue("log_loss" in out)
        self.assertGreater(out["roc_auc"], 0.5)

    def test_regression_metrics_shape(self):
        y_true = np.array([10.0, 12.0, 15.0, 17.0])
        pred = np.array([10.5, 11.8, 14.5, 16.9])
        out = regression_metrics(y_true, pred)
        self.assertTrue("rmse" in out)
        self.assertTrue("r2" in out)

    def test_threshold_selection(self):
        y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        proba = np.array([0.05, 0.9, 0.2, 0.8, 0.7, 0.4, 0.65, 0.1])
        table = threshold_sweep(y_true, proba, thresholds=[0.2, 0.3, 0.4, 0.5])
        sel = select_threshold(table, objective="f1", min_precision=0.5)
        self.assertTrue(0.2 <= sel.threshold <= 0.5)


if __name__ == "__main__":
    unittest.main()
