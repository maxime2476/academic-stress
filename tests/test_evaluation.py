"""
Tests for src.evaluation — metrics, plots, calibration, exports.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    calibration_analysis,
    compute_all_metrics,
    export_full_scores,
    export_metrics_json,
    export_predictions,
    plot_calibration,
    plot_confusion_matrix,
    plot_odds_ratios,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_threshold_comparison,
)
from src.modeling import ThresholdCandidate


# ── Helpers ───────────────────────────────────────────────────────────────────

def _perfect_probs(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Labels and probabilities that agree perfectly (AUC = 1.0)."""
    y     = np.array([0] * (n // 2) + [1] * (n // 2))
    probs = np.array([0.1] * (n // 2) + [0.9] * (n // 2), dtype=float)
    return y, probs


def _random_probs(n: int = 80, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    probs = np.clip(y * 0.55 + rng.uniform(0, 0.45, size=n), 0, 1)
    return y, probs


# ── compute_all_metrics ───────────────────────────────────────────────────────

class TestComputeAllMetrics:

    def test_perfect_model_auc_is_one(self):
        y, probs = _perfect_probs()
        m = compute_all_metrics(y, probs, threshold=0.5)
        assert m["roc_auc"] == pytest.approx(1.0)

    def test_all_required_keys_present(self):
        y, probs = _random_probs()
        m = compute_all_metrics(y, probs, threshold=0.5)
        required = (
            "roc_auc", "pr_auc", "brier_score", "accuracy",
            "sensitivity", "specificity", "ppv", "npv", "f1",
            "tp", "tn", "fp", "fn", "threshold", "n", "prevalence",
            "classification_report",
        )
        for key in required:
            assert key in m, f"Missing key: {key}"

    def test_metrics_in_valid_range(self):
        y, probs = _random_probs()
        m = compute_all_metrics(y, probs, threshold=0.5)
        for key in ("roc_auc", "pr_auc", "accuracy", "sensitivity",
                    "specificity", "ppv", "npv", "f1"):
            assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of [0,1]"

    def test_brier_score_range(self):
        y, probs = _random_probs()
        m = compute_all_metrics(y, probs, threshold=0.5)
        assert 0.0 <= m["brier_score"] <= 1.0

    def test_confusion_matrix_counts_sum_to_n(self):
        y, probs = _random_probs()
        m = compute_all_metrics(y, probs, threshold=0.5)
        total = m["tp"] + m["tn"] + m["fp"] + m["fn"]
        assert total == len(y)

    def test_sensitivity_is_tp_over_tp_fn(self):
        y, probs = _random_probs()
        m = compute_all_metrics(y, probs, threshold=0.5)
        tp, fn = m["tp"], m["fn"]
        if tp + fn > 0:
            expected = tp / (tp + fn)
            assert m["sensitivity"] == pytest.approx(expected)

    def test_threshold_affects_sensitivity_specificity(self):
        y, probs = _random_probs()
        m_low  = compute_all_metrics(y, probs, threshold=0.2)
        m_high = compute_all_metrics(y, probs, threshold=0.8)
        # Lower threshold → higher sensitivity (recall), lower specificity
        assert m_low["sensitivity"] >= m_high["sensitivity"] - 0.1
        assert m_high["specificity"] >= m_low["specificity"] - 0.1

    def test_label_stored_in_output(self):
        y, probs = _random_probs()
        m = compute_all_metrics(y, probs, threshold=0.5, label="train")
        assert m["label"] == "train"


# ── calibration_analysis ──────────────────────────────────────────────────────

class TestCalibrationAnalysis:

    def test_brier_scores_are_positive(self, fitted_model, encoded_data):
        X_tr, X_te, y_tr, y_te = encoded_data
        probs_test = fitted_model.predict_proba(X_te)[:, 1]
        result = calibration_analysis(
            fitted_model, X_tr, X_te,
            y_tr.values, y_te.values, probs_test,
        )
        assert result["brier_score_raw"]        >= 0
        assert result["brier_score_calibrated"] >= 0

    def test_expected_keys_in_result(self, fitted_model, encoded_data):
        X_tr, X_te, y_tr, y_te = encoded_data
        probs_test = fitted_model.predict_proba(X_te)[:, 1]
        result = calibration_analysis(
            fitted_model, X_tr, X_te,
            y_tr.values, y_te.values, probs_test,
        )
        for key in ("brier_score_raw", "brier_score_calibrated",
                    "brier_improvement", "calibration_raw", "calibration_isotonic"):
            assert key in result


# ── Plot functions (smoke tests — they should not raise) ─────────────────────

class TestPlots:
    """Verify plots generate without exceptions; do not check aesthetics."""

    def test_plot_roc_curve(self, tmp_path):
        y, probs = _random_probs()
        fig = plot_roc_curve(y, probs, threshold=0.5,
                             save_path=tmp_path / "roc.png")
        assert (tmp_path / "roc.png").exists()

    def test_plot_pr_curve(self, tmp_path):
        y, probs = _random_probs()
        fig = plot_precision_recall_curve(y, probs, threshold=0.5,
                                          save_path=tmp_path / "pr.png")
        assert (tmp_path / "pr.png").exists()

    def test_plot_confusion_matrix(self, tmp_path):
        y, probs = _random_probs()
        y_pred = (probs >= 0.5).astype(int)
        fig = plot_confusion_matrix(y, y_pred, threshold=0.5,
                                    save_path=tmp_path / "cm.png")
        assert (tmp_path / "cm.png").exists()

    def test_plot_calibration(self, tmp_path):
        y, probs = _random_probs()
        fig = plot_calibration(y, probs, save_path=tmp_path / "cal.png")
        assert (tmp_path / "cal.png").exists()

    def test_plot_odds_ratios(self, tmp_path):
        import pandas as pd
        bor = pd.DataFrame({
            "feature": ["a", "b", "c"],
            "odds_ratio": [1.5, 0.8, 2.1],
            "ci_lower_95": [1.0, 0.4, 1.3],
            "ci_upper_95": [2.0, 1.2, 3.0],
        })
        fig = plot_odds_ratios(bor, save_path=tmp_path / "or.png")
        assert (tmp_path / "or.png").exists()

    def test_plot_threshold_comparison(self, tmp_path):
        y, probs = _random_probs()
        candidates = [
            ThresholdCandidate("a", 0.3, 0.8, 0.5, 0.65, 0.70, "desc"),
            ThresholdCandidate("b", 0.5, 0.6, 0.7, 0.60, 0.66, "desc"),
        ]
        fig = plot_threshold_comparison(y, probs, candidates,
                                        save_path=tmp_path / "thresh.png")
        assert (tmp_path / "thresh.png").exists()

    def test_no_file_when_save_path_is_none(self):
        y, probs = _random_probs()
        fig = plot_roc_curve(y, probs, save_path=None)
        assert fig is not None   # figure object returned


# ── Export functions ──────────────────────────────────────────────────────────

class TestExportFunctions:

    def test_export_predictions_creates_csv(self, tmp_path):
        rng = np.random.default_rng(0)
        n = 30
        X = pd.DataFrame({"a": rng.integers(1, 5, n), "b": rng.integers(1, 5, n)})
        y = pd.Series(rng.integers(0, 2, n), name="HighStress")
        probs = rng.random(n)
        path = tmp_path / "preds.csv"
        export_predictions(X, y, probs, threshold=0.5, path=path)
        assert path.exists()
        df_out = pd.read_csv(path)
        assert "y_true" in df_out.columns
        assert "prob_high_stress" in df_out.columns
        assert "y_pred" in df_out.columns

    def test_export_full_scores_creates_csv(self, tmp_path):
        rng = np.random.default_rng(1)
        n = 50
        X = pd.DataFrame({"x": rng.integers(1, 5, n)})
        y = pd.Series(rng.integers(0, 2, n), name="HighStress")
        probs = rng.random(n)
        path = tmp_path / "full.csv"
        export_full_scores(X, y, probs, threshold=0.5, path=path)
        assert path.exists()
        df_out = pd.read_csv(path)
        assert len(df_out) == n

    def test_export_metrics_json(self, tmp_path):
        metrics = {"roc_auc": 0.75, "n": 49, "array": np.array([1, 2, 3])}
        path = tmp_path / "metrics.json"
        export_metrics_json(metrics, path)
        assert path.exists()
        with open(path) as fh:
            loaded = json.load(fh)
        assert loaded["roc_auc"] == pytest.approx(0.75)
        assert loaded["array"] == [1, 2, 3]

    def test_predictions_y_pred_matches_threshold(self, tmp_path):
        rng = np.random.default_rng(2)
        n = 20
        X = pd.DataFrame({"f": rng.integers(1, 5, n)})
        y = pd.Series(rng.integers(0, 2, n))
        probs = rng.random(n)
        t = 0.6
        path = tmp_path / "p.csv"
        export_predictions(X, y, probs, threshold=t, path=path)
        df_out = pd.read_csv(path)
        expected_pred = (probs >= t).astype(int)
        np.testing.assert_array_equal(df_out["y_pred"].values, expected_pred)
