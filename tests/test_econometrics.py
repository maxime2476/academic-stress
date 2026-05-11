"""
Tests for src.econometrics — all 13 statistical/econometric tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.econometrics import (
    TestResult,
    compute_information_criteria,
    compute_pseudo_r2,
    compute_vif,
    cramers_v,
    cramers_v_with_ci,
    hosmer_lemeshow_test,
    kruskal_wallis_test,
    likelihood_ratio_test,
    marginal_effects_at_mean,
    pairwise_mann_whitney,
    pearson_deviance_tests,
    spearman_correlation_matrix,
    wald_tests,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _simple_lr(X: pd.DataFrame, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(solver="liblinear", max_iter=5000,
                               class_weight="balanced", random_state=42)
    model.fit(X, y)
    return model


def _binary_data(n: int = 100, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.uniform(-2, 2, n),
        "x2": rng.uniform(-1, 1, n),
        "x3": rng.integers(0, 2, n).astype(float),
    })
    log_odds = 0.5 * X["x1"] - 0.3 * X["x2"] + 0.8 * X["x3"]
    p = 1 / (1 + np.exp(-log_odds))
    y = rng.binomial(1, p).astype(int)
    return X, y


# ── TestResult ─────────────────────────────────────────────────────────────────

class TestTestResult:

    def test_is_significant_below_alpha(self):
        r = TestResult("test", statistic=10.0, p_value=0.01, df=1)
        assert r.is_significant(alpha=0.05) is True

    def test_is_not_significant_above_alpha(self):
        r = TestResult("test", statistic=1.0, p_value=0.3, df=1)
        assert r.is_significant(alpha=0.05) is False

    def test_str_contains_name(self):
        r = TestResult("MyTest", statistic=5.0, p_value=0.02, df=2)
        assert "MyTest" in str(r)

    def test_str_shows_significance_marker(self):
        r = TestResult("T", statistic=5.0, p_value=0.01, df=1)
        assert "*" in str(r)


# ── compute_vif ────────────────────────────────────────────────────────────────

class TestComputeVIF:

    def test_returns_dataframe(self):
        X, _ = _binary_data()
        result = compute_vif(X)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        X, _ = _binary_data()
        result = compute_vif(X)
        for col in ("feature", "VIF", "concern"):
            assert col in result.columns

    def test_all_features_represented(self):
        X, _ = _binary_data()
        result = compute_vif(X)
        assert set(result["feature"]) == set(X.columns)

    def test_vif_positive(self):
        X, _ = _binary_data()
        result = compute_vif(X)
        assert (result["VIF"].dropna() > 0).all()

    def test_high_collinearity_detected(self):
        rng = np.random.default_rng(0)
        n = 200
        x1 = rng.normal(0, 1, n)
        x2 = x1 + rng.normal(0, 0.01, n)   # near-perfect collinearity
        X = pd.DataFrame({"x1": x1, "x2": x2})
        result = compute_vif(X)
        assert (result["VIF"] > 10).any()


# ── hosmer_lemeshow_test ───────────────────────────────────────────────────────

class TestHosmerLemeshow:

    def test_returns_test_result(self):
        X, y = _binary_data()
        model = _simple_lr(X, y)
        probs = model.predict_proba(X)[:, 1]
        result = hosmer_lemeshow_test(y, probs, n_groups=10)
        assert isinstance(result, TestResult)

    def test_statistic_is_non_negative(self):
        X, y = _binary_data()
        model = _simple_lr(X, y)
        probs = model.predict_proba(X)[:, 1]
        result = hosmer_lemeshow_test(y, probs)
        assert result.statistic >= 0

    def test_p_value_in_unit_interval(self):
        X, y = _binary_data()
        model = _simple_lr(X, y)
        probs = model.predict_proba(X)[:, 1]
        result = hosmer_lemeshow_test(y, probs)
        assert 0.0 <= result.p_value <= 1.0

    def test_well_calibrated_model_high_pvalue(self):
        """A perfectly calibrated model (observed ≈ predicted) should not show poor fit."""
        rng = np.random.default_rng(99)
        n = 300
        probs = rng.uniform(0.05, 0.95, n)
        # Generate y such that P(y=1|p) = p exactly → well-calibrated
        y = rng.binomial(1, probs).astype(int)
        result = hosmer_lemeshow_test(y, probs, n_groups=10)
        # Well-calibrated model should have p > 0.05 (fail to reject good fit)
        assert result.p_value >= 0.05

    def test_degrees_of_freedom_formula(self):
        X, y = _binary_data(n=200)
        model = _simple_lr(X, y)
        probs = model.predict_proba(X)[:, 1]
        groups = 8
        result = hosmer_lemeshow_test(y, probs, n_groups=groups)
        assert result.df == result.details["n_groups_actual"] - 2


# ── likelihood_ratio_test ──────────────────────────────────────────────────────

class TestLikelihoodRatioTest:

    def test_returns_test_result(self):
        X, y = _binary_data()
        result = likelihood_ratio_test(X, y)
        assert isinstance(result, TestResult)

    def test_statistic_non_negative(self):
        X, y = _binary_data()
        result = likelihood_ratio_test(X, y)
        assert result.statistic >= 0

    def test_p_value_significant_for_predictive_model(self):
        """When X is predictive of y, LR test should be significant."""
        rng = np.random.default_rng(1)
        n = 150
        X = pd.DataFrame({"x": rng.normal(0, 1, n)})
        y = (1 / (1 + np.exp(-2 * X["x"].values)) > 0.5).astype(int)
        result = likelihood_ratio_test(X, y)
        assert result.p_value < 0.05

    def test_ll_full_greater_than_ll_null(self):
        X, y = _binary_data(n=200)
        result = likelihood_ratio_test(X, y)
        assert result.details["ll_full"] > result.details["ll_null"]


# ── wald_tests ────────────────────────────────────────────────────────────────

class TestWaldTests:

    def test_returns_dataframe(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = wald_tests(fitted_model, X_tr, y_tr.values)
        assert isinstance(result, pd.DataFrame)

    def test_all_features_in_output(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = wald_tests(fitted_model, X_tr, y_tr.values)
        assert set(result["feature"]) == set(X_tr.columns)

    def test_p_values_in_unit_interval(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = wald_tests(fitted_model, X_tr, y_tr.values)
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_odds_ratios_positive(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = wald_tests(fitted_model, X_tr, y_tr.values)
        assert (result["odds_ratio"] > 0).all()

    def test_sorted_by_p_value(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = wald_tests(fitted_model, X_tr, y_tr.values)
        p_vals = result["p_value"].tolist()
        assert p_vals == sorted(p_vals)


# ── compute_pseudo_r2 ─────────────────────────────────────────────────────────

class TestPseudoR2:

    def test_returns_all_three_r2(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = compute_pseudo_r2(fitted_model, X_tr, y_tr.values)
        for key in ("mcfadden_r2", "cox_snell_r2", "nagelkerke_r2"):
            assert key in result

    def test_mcfadden_between_0_and_1(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = compute_pseudo_r2(fitted_model, X_tr, y_tr.values)
        assert 0 <= result["mcfadden_r2"] <= 1

    def test_nagelkerke_geq_cox_snell(self, fitted_model, encoded_data):
        """Nagelkerke is the scaled version of Cox-Snell, so NK ≥ CS."""
        X_tr, _, y_tr, _ = encoded_data
        result = compute_pseudo_r2(fitted_model, X_tr, y_tr.values)
        assert result["nagelkerke_r2"] >= result["cox_snell_r2"] - 1e-9

    def test_r2_higher_for_better_model(self):
        """A model with informative features should have higher McFadden R² than noise."""
        rng = np.random.default_rng(3)
        n = 200
        x_info = rng.normal(0, 1, n)
        y = (1 / (1 + np.exp(-2 * x_info)) > 0.5).astype(int)

        X_good  = pd.DataFrame({"x": x_info})
        X_noise = pd.DataFrame({"x": rng.normal(0, 1, n)})

        m_good  = _simple_lr(X_good, y)
        m_noise = _simple_lr(X_noise, y)

        r2_good  = compute_pseudo_r2(m_good,  X_good,  y)["mcfadden_r2"]
        r2_noise = compute_pseudo_r2(m_noise, X_noise, y)["mcfadden_r2"]
        assert r2_good >= r2_noise


# ── compute_information_criteria ─────────────────────────────────────────────

class TestInformationCriteria:

    def test_returns_aic_bic(self, fitted_model, encoded_data):
        X_tr, _, y_tr, _ = encoded_data
        result = compute_information_criteria(fitted_model, X_tr, y_tr.values)
        assert "aic" in result
        assert "bic" in result

    def test_bic_penalises_more_than_aic_for_large_n(self, fitted_model, encoded_data):
        """For n > e² ≈ 7.4, BIC penalty (k ln n) > AIC penalty (2k), so BIC > AIC."""
        X_tr, _, y_tr, _ = encoded_data
        result = compute_information_criteria(fitted_model, X_tr, y_tr.values)
        assert result["n"] > 7
        assert result["bic"] > result["aic"]


# ── cramers_v ──────────────────────────────────────────────────────────────────

class TestCramersV:

    def test_perfect_association(self):
        x = pd.Series(["a", "b", "c"] * 30)
        y = x.copy()   # identical → V = 1
        v = cramers_v(x, y)
        assert v == pytest.approx(1.0, abs=1e-6)

    def test_no_association(self):
        rng = np.random.default_rng(0)
        x = pd.Series(rng.choice(["a", "b"], 200))
        y = pd.Series(rng.choice(["c", "d"], 200))
        v = cramers_v(x, y)
        assert v < 0.2   # near zero for independent variables

    def test_v_in_unit_interval(self):
        x = pd.Series(["a", "b", "a", "b", "c"])
        y = pd.Series(["x", "y", "x", "x", "y"])
        v = cramers_v(x, y)
        assert 0.0 <= v <= 1.0

    def test_with_ci_returns_correct_keys(self):
        x = pd.Series(["a", "b"] * 50)
        y = pd.Series(["x", "y"] * 50)
        result = cramers_v_with_ci(x, y, n_bootstrap=50)
        for key in ("cramers_v", "ci_lower_95", "ci_upper_95", "bootstrap_std"):
            assert key in result

    def test_ci_lower_leq_ci_upper(self):
        x = pd.Series(["a", "b"] * 50)
        y = pd.Series(["x", "y"] * 50)
        result = cramers_v_with_ci(x, y, n_bootstrap=50)
        assert result["ci_lower_95"] <= result["ci_upper_95"]


# ── kruskal_wallis_test ───────────────────────────────────────────────────────

class TestKruskalWallis:

    def _make_group_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        return pd.DataFrame({
            "group": (["A"] * 30 + ["B"] * 30 + ["C"] * 30),
            "target": np.concatenate([
                rng.integers(1, 3, 30),   # low
                rng.integers(2, 4, 30),   # medium
                rng.integers(3, 6, 30),   # high
            ]),
        })

    def test_returns_test_result(self):
        df = self._make_group_df()
        result = kruskal_wallis_test(df, "group", "target")
        assert isinstance(result, TestResult)

    def test_significant_for_distinct_groups(self):
        df = self._make_group_df()
        result = kruskal_wallis_test(df, "group", "target")
        assert result.p_value < 0.05

    def test_not_significant_for_equal_groups(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "group": (["X"] * 40 + ["Y"] * 40),
            "target": rng.integers(1, 6, 80),   # same distribution
        })
        result = kruskal_wallis_test(df, "group", "target")
        assert result.p_value > 0.01   # usually not significant

    def test_raises_on_single_group(self):
        df = pd.DataFrame({"g": ["A"] * 10, "t": [1] * 10})
        with pytest.raises(ValueError):
            kruskal_wallis_test(df, "g", "t")


# ── pairwise_mann_whitney ──────────────────────────────────────────────────────

class TestPairwiseMannWhitney:

    def test_returns_dataframe(self):
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20,
            "target": np.concatenate([
                np.ones(20), np.full(20, 3)
            ]),
        })
        result = pairwise_mann_whitney(df, "group", "target")
        assert isinstance(result, pd.DataFrame)

    def test_bonferroni_columns_present(self):
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20 + ["C"] * 20,
            "target": np.random.randint(1, 6, 60),
        })
        result = pairwise_mann_whitney(df, "group", "target")
        assert "p_value_bonferroni" in result.columns
        assert "significant_bonferroni" in result.columns

    def test_bonferroni_p_values_geq_raw(self):
        df = pd.DataFrame({
            "group": ["A"] * 20 + ["B"] * 20,
            "target": np.concatenate([np.ones(20), np.full(20, 5)]),
        })
        result = pairwise_mann_whitney(df, "group", "target")
        assert (result["p_value_bonferroni"] >= result["p_value_raw"]).all()


# ── spearman_correlation_matrix ───────────────────────────────────────────────

class TestSpearmanCorrelations:

    def test_returns_two_dataframes(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        })
        rho, pval = spearman_correlation_matrix(df, ["a", "b"])
        assert isinstance(rho, pd.DataFrame)
        assert isinstance(pval, pd.DataFrame)

    def test_diagonal_is_one(self):
        df = pd.DataFrame({"a": range(10), "b": range(10, 20)})
        rho, _ = spearman_correlation_matrix(df, ["a", "b"])
        assert rho.loc["a", "a"] == pytest.approx(1.0)
        assert rho.loc["b", "b"] == pytest.approx(1.0)

    def test_perfect_negative_correlation(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [5, 4, 3, 2, 1]})
        rho, _ = spearman_correlation_matrix(df, ["x", "y"])
        assert rho.loc["x", "y"] == pytest.approx(-1.0, abs=1e-6)


# ── marginal_effects_at_mean ──────────────────────────────────────────────────

class TestMarginalEffects:

    def test_returns_dataframe(self, fitted_model, encoded_data):
        X_tr, _, _, _ = encoded_data
        result = marginal_effects_at_mean(fitted_model, X_tr)
        assert isinstance(result, pd.DataFrame)

    def test_all_features_represented(self, fitted_model, encoded_data):
        X_tr, _, _, _ = encoded_data
        result = marginal_effects_at_mean(fitted_model, X_tr)
        assert set(result["feature"]) == set(X_tr.columns)

    def test_ame_magnitude_less_than_coef(self, fitted_model, encoded_data):
        """AME = β × avg_scale where avg_scale = mean(p̂(1-p̂)) ≤ 0.25 → |AME| ≤ |β|/4."""
        X_tr, _, _, _ = encoded_data
        result = marginal_effects_at_mean(fitted_model, X_tr)
        assert (result["AME"].abs() <= result["coef"].abs() + 1e-6).all()

    def test_sorted_by_absolute_ame(self, fitted_model, encoded_data):
        X_tr, _, _, _ = encoded_data
        result = marginal_effects_at_mean(fitted_model, X_tr)
        ames = result["AME"].abs().tolist()
        assert ames == sorted(ames, reverse=True)


# ── pearson_deviance_tests ────────────────────────────────────────────────────

class TestPearsonDeviance:

    def test_returns_both_tests(self, fitted_model, encoded_data):
        X_tr, X_te, y_tr, y_te = encoded_data
        result = pearson_deviance_tests(fitted_model, X_te, y_te.values)
        assert "pearson"  in result
        assert "deviance" in result

    def test_statistics_are_positive(self, fitted_model, encoded_data):
        X_tr, X_te, y_tr, y_te = encoded_data
        result = pearson_deviance_tests(fitted_model, X_te, y_te.values)
        assert result["pearson"].statistic  >= 0
        assert result["deviance"].statistic >= 0

    def test_p_values_in_unit_interval(self, fitted_model, encoded_data):
        X_tr, X_te, y_tr, y_te = encoded_data
        result = pearson_deviance_tests(fitted_model, X_te, y_te.values)
        for name, t in result.items():
            assert 0.0 <= t.p_value <= 1.0, f"{name}: p={t.p_value}"
