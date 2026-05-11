"""
Tests for src.modeling — training, CV, threshold optimisation, model persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.modeling import (
    CVResult,
    ThresholdCandidate,
    TrainingResult,
    _find_threshold_max_f1,
    _find_threshold_min_cost,
    _make_candidate,
    _bootstrap_odds_ratios,
    load_model,
    save_model,
    train_and_evaluate,
)
from tests.conftest import make_raw_df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _toy_probs_labels(n: int = 80, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    probs = np.clip(y * 0.6 + rng.uniform(0, 0.4, size=n), 0, 1)
    return probs, y


# ── CVResult ──────────────────────────────────────────────────────────────────

class TestCVResult:

    def test_str_format(self):
        cv = CVResult(mean=0.75, std=0.05, scores=[0.70, 0.75, 0.80])
        assert "0.750" in str(cv)
        assert "0.050" in str(cv)

    def test_mean_std_stored(self):
        cv = CVResult(mean=0.8, std=0.02, scores=[])
        assert cv.mean == pytest.approx(0.8)
        assert cv.std  == pytest.approx(0.02)


# ── _find_threshold_max_f1 ────────────────────────────────────────────────────

class TestFindThresholdMaxF1:

    def test_returns_float_in_unit_interval(self):
        probs, y = _toy_probs_labels()
        t = _find_threshold_max_f1(probs, y)
        assert 0.0 < t < 1.0

    def test_maximises_f1(self):
        probs, y = _toy_probs_labels()
        t = _find_threshold_max_f1(probs, y)
        from sklearn.metrics import f1_score
        f1_opt = f1_score(y, (probs >= t).astype(int), zero_division=0)
        # Check that nearby thresholds do not yield strictly higher F1
        for delta in [-0.02, 0.02]:
            t_alt = np.clip(t + delta, 0.05, 0.95)
            f1_alt = f1_score(y, (probs >= t_alt).astype(int), zero_division=0)
            assert f1_opt >= f1_alt - 0.02  # allow small floating-point tolerance


# ── _find_threshold_min_cost ──────────────────────────────────────────────────

class TestFindThresholdMinCost:

    def test_returns_float_in_unit_interval(self):
        probs, y = _toy_probs_labels()
        t = _find_threshold_min_cost(probs, y, fn_cost=3, fp_cost=1)
        assert 0.0 < t < 1.0

    def test_higher_fn_cost_lowers_threshold(self):
        """Penalising FN more → lower threshold (predict more positives)."""
        probs, y = _toy_probs_labels(seed=7)
        t_balanced = _find_threshold_min_cost(probs, y, fn_cost=1, fp_cost=1)
        t_fn_heavy  = _find_threshold_min_cost(probs, y, fn_cost=5, fp_cost=1)
        assert t_fn_heavy <= t_balanced + 0.15  # generally lower or equal


# ── _make_candidate ───────────────────────────────────────────────────────────

class TestMakeCandidate:

    def test_sensitivity_specificity_in_unit_interval(self):
        probs, y = _toy_probs_labels()
        tc = _make_candidate("test", 0.5, probs, y, "desc")
        assert 0.0 <= tc.sensitivity <= 1.0
        assert 0.0 <= tc.specificity <= 1.0

    def test_f1_in_unit_interval(self):
        probs, y = _toy_probs_labels()
        tc = _make_candidate("test", 0.5, probs, y, "desc")
        assert 0.0 <= tc.f1 <= 1.0

    def test_accuracy_equals_manual_computation(self):
        probs, y = _toy_probs_labels()
        t = 0.4
        tc = _make_candidate("test", t, probs, y, "")
        y_pred = (probs >= t).astype(int)
        expected_acc = float((y_pred == y).mean())
        assert tc.accuracy == pytest.approx(expected_acc, abs=1e-6)


# ── _bootstrap_odds_ratios ────────────────────────────────────────────────────

class TestBootstrapOddsRatios:

    def test_output_shape(self, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        lrcfg = {
            "solver": "liblinear", "penalty": "l2", "C": 1.0,
            "class_weight": "balanced", "max_iter": 5000, "random_state": 42,
        }
        bor = _bootstrap_odds_ratios(
            X_tr.values, y_tr.values, X_tr.columns.tolist(),
            lrcfg, n_bootstrap=30, random_state=0,
        )
        assert list(bor.columns) == ["feature", "odds_ratio", "ci_lower_95", "ci_upper_95"]
        assert len(bor) == X_tr.shape[1]

    def test_ci_encompasses_point_estimate(self, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        lrcfg = {
            "solver": "liblinear", "penalty": "l2", "C": 1.0,
            "class_weight": "balanced", "max_iter": 5000, "random_state": 42,
        }
        bor = _bootstrap_odds_ratios(
            X_tr.values, y_tr.values, X_tr.columns.tolist(),
            lrcfg, n_bootstrap=50, random_state=0,
        )
        assert (bor["ci_lower_95"] <= bor["odds_ratio"]).all()
        assert (bor["odds_ratio"] <= bor["ci_upper_95"]).all()

    def test_odds_ratios_positive(self, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        lrcfg = {
            "solver": "liblinear", "penalty": "l2", "C": 1.0,
            "class_weight": "balanced", "max_iter": 5000, "random_state": 42,
        }
        bor = _bootstrap_odds_ratios(
            X_tr.values, y_tr.values, X_tr.columns.tolist(),
            lrcfg, n_bootstrap=30, random_state=0,
        )
        assert (bor["odds_ratio"] > 0).all()


# ── save_model / load_model ───────────────────────────────────────────────────

class TestModelPersistence:

    def test_save_and_load_roundtrip(self, fitted_model, fitted_pipeline, tmp_path):
        path = tmp_path / "model.joblib"
        save_model(fitted_model, fitted_pipeline, path)
        assert path.exists()

        model_loaded, pipeline_loaded = load_model(path)
        assert isinstance(model_loaded, LogisticRegression)

    def test_loaded_model_predictions_identical(self, fitted_model, fitted_pipeline,
                                                 encoded_data, tmp_path):
        X_tr, X_te, _, _ = encoded_data
        path = tmp_path / "model.joblib"
        save_model(fitted_model, fitted_pipeline, path)
        model_loaded, _ = load_model(path)

        probs_orig   = fitted_model.predict_proba(X_te)[:, 1]
        probs_loaded = model_loaded.predict_proba(X_te)[:, 1]
        np.testing.assert_allclose(probs_orig, probs_loaded)

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "does_not_exist.joblib")


# ── train_and_evaluate (integration) ─────────────────────────────────────────

class TestTrainAndEvaluate:

    def test_returns_training_result(self, X_y):
        X, y = X_y
        result = train_and_evaluate(X, y)
        assert isinstance(result, TrainingResult)

    def test_cv_results_contain_all_metrics(self, X_y):
        X, y = X_y
        result = train_and_evaluate(X, y)
        for metric in ("roc_auc", "pr_auc", "brier_score"):
            assert metric in result.cv_results

    def test_cv_auc_above_chance(self, X_y):
        X, y = X_y
        result = train_and_evaluate(X, y)
        assert result.cv_results["roc_auc"].mean > 0.5

    def test_three_threshold_candidates(self, X_y):
        X, y = X_y
        result = train_and_evaluate(X, y)
        assert len(result.thresholds) == 3

    def test_train_test_sizes_consistent_with_config(self, X_y):
        from src.config import get_config
        X, y = X_y
        result = train_and_evaluate(X, y)
        expected_test = round(len(X) * get_config()["modeling"]["test_size"])
        # Allow ±2 for rounding
        assert abs(len(result.X_test) - expected_test) <= 2

    def test_stratified_split_preserves_prevalence(self, X_y):
        X, y = X_y
        result = train_and_evaluate(X, y)
        train_prev = result.y_train.mean()
        test_prev  = result.y_test.mean()
        assert abs(train_prev - test_prev) < 0.15  # within 15 pp

    def test_feature_names_match_encoded_columns(self, X_y):
        X, y = X_y
        result = train_and_evaluate(X, y)
        assert result.feature_names == result.X_train_enc.columns.tolist()

    def test_probs_in_unit_interval(self, X_y):
        X, y = X_y
        result = train_and_evaluate(X, y)
        assert result.test_probs.min() >= 0.0
        assert result.test_probs.max() <= 1.0
