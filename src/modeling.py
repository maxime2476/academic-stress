"""
Model training, cross-validation, threshold optimisation, and persistence.

The main function is ``train_and_evaluate`` which returns a fully trained
``LogisticRegression`` model along with CV results and threshold candidates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    average_precision_score,
)

from src.config import get_config, resolve_path
from src.preprocessing import build_preprocessing_pipeline, transform_to_dataframe

log = logging.getLogger(__name__)


# ?? Data classes ???????????????????????????????????????????????????????????????

@dataclass
class CVResult:
    """Cross-validation result for a single metric."""
    mean: float
    std: float
    scores: list[float]

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.std:.3f}"


@dataclass
class ThresholdCandidate:
    """A decision threshold with its associated metrics."""
    name: str
    threshold: float
    sensitivity: float   # recall for positive class
    specificity: float   # recall for negative class
    f1: float
    accuracy: float
    description: str


@dataclass
class TrainingResult:
    """Complete output of a model training run."""
    model: LogisticRegression
    pipeline: Any                             # fitted sklearn Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_enc: pd.DataFrame                # encoded training features
    X_test_enc: pd.DataFrame                 # encoded test features
    feature_names: list[str]
    cv_results: dict[str, CVResult]
    thresholds: list[ThresholdCandidate]
    bootstrap_odds_ratios: pd.DataFrame
    train_probs: np.ndarray
    test_probs: np.ndarray


# ?? Main training function ????????????????????????????????????????????????????

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    cfg: dict | None = None,
) -> TrainingResult:
    """Train logistic regression, run CV, optimise thresholds, bootstrap CIs.

    Parameters
    ----------
    X   : raw (not yet encoded) feature DataFrame
    y   : binary target Series (0/1)
    cfg : config dict (default: loaded from config.yaml)

    Returns
    -------
    TrainingResult dataclass with all artefacts.
    """
    if cfg is None:
        cfg = get_config()

    mcfg = cfg["modeling"]
    lrcfg = mcfg["logistic_regression"]

    # ?? Train / test split ????????????????????????????????????????????????????
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=mcfg["test_size"],
        random_state=mcfg["random_state"],
        stratify=y,
    )
    log.info(
        "Split -- train: %d, test: %d | positive rate train: %.2f test: %.2f",
        len(X_train), len(X_test), y_train.mean(), y_test.mean(),
    )

    # ?? Preprocessing (fit on train only) ?????????????????????????????????????
    pipeline = build_preprocessing_pipeline()
    pipeline.fit(X_train)

    X_train_enc = transform_to_dataframe(pipeline, X_train)
    X_test_enc  = transform_to_dataframe(pipeline, X_test)
    feature_names = X_train_enc.columns.tolist()

    # ?? Logistic Regression ???????????????????????????????????????????????????
    model = LogisticRegression(
        solver=lrcfg["solver"],
        penalty=lrcfg["penalty"],
        C=lrcfg["C"],
        class_weight=lrcfg["class_weight"],
        max_iter=lrcfg["max_iter"],
        random_state=lrcfg["random_state"],
    )
    model.fit(X_train_enc, y_train)
    log.info("Model fitted. Coefficients shape: %s", model.coef_.shape)

    # ?? Cross-validation ??????????????????????????????????????????????????????
    cv_results = _cross_validate(model, pipeline, X, y, mcfg)

    # ?? Predicted probabilities ???????????????????????????????????????????????
    train_probs = model.predict_proba(X_train_enc)[:, 1]
    test_probs  = model.predict_proba(X_test_enc)[:, 1]

    # ?? Threshold analysis ????????????????????????????????????????????????????
    thresholds = _compute_threshold_candidates(test_probs, y_test.values, cfg)

    # ?? Bootstrap odds ratios ?????????????????????????????????????????????????
    bor = _bootstrap_odds_ratios(
        X_train_enc.values, y_train.values,
        feature_names, lrcfg,
        n_bootstrap=mcfg["bootstrap_samples"],
        random_state=mcfg["bootstrap_random_state"],
    )

    return TrainingResult(
        model=model,
        pipeline=pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_enc=X_train_enc,
        X_test_enc=X_test_enc,
        feature_names=feature_names,
        cv_results=cv_results,
        thresholds=thresholds,
        bootstrap_odds_ratios=bor,
        train_probs=train_probs,
        test_probs=test_probs,
    )


# ?? Cross-validation ??????????????????????????????????????????????????????????

def _cross_validate(
    model: LogisticRegression,
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    mcfg: dict,
) -> dict[str, CVResult]:
    """Stratified k-fold CV on the full dataset (preprocessing inside fold)."""
    from sklearn.pipeline import Pipeline as SKPipeline

    full_pipe = SKPipeline(
        steps=[
            ("preprocessing", pipeline),
            ("classifier", model),
        ]
    )

    cv = StratifiedKFold(
        n_splits=mcfg["cv_folds"],
        shuffle=True,
        random_state=mcfg["random_state"],
    )

    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "brier_score": "neg_brier_score",
    }

    scores = cross_validate(
        full_pipe, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    cv_results: dict[str, CVResult] = {}
    for metric, key in [
        ("roc_auc", "test_roc_auc"),
        ("pr_auc", "test_average_precision"),
        ("brier_score", "test_brier_score"),
    ]:
        vals = scores[key]
        if metric == "brier_score":
            vals = -vals  # neg_brier_score -> positive Brier score
        cv_results[metric] = CVResult(
            mean=float(np.mean(vals)),
            std=float(np.std(vals)),
            scores=vals.tolist(),
        )

    log.info(
        "CV -- AUC: %s | PR-AUC: %s | Brier: %s",
        cv_results["roc_auc"],
        cv_results["pr_auc"],
        cv_results["brier_score"],
    )
    return cv_results


# ?? Threshold candidates ??????????????????????????????????????????????????????

def _compute_threshold_candidates(
    probs: np.ndarray,
    y_true: np.ndarray,
    cfg: dict,
) -> list[ThresholdCandidate]:
    """Compute three operationally meaningful decision thresholds."""
    tcfg = cfg["thresholds"]
    candidates: list[ThresholdCandidate] = []

    # 1 - F1-maximising threshold
    t_f1 = _find_threshold_max_f1(probs, y_true)
    candidates.append(_make_candidate("f1_optimised", t_f1, probs, y_true,
        "Maximises F1-score. Best for balanced precision/recall."))

    # 2 - Operational threshold from config (manually validated)
    t_op = tcfg["operational"]
    candidates.append(_make_candidate("operational", t_op, probs, y_true,
        f"Manually validated threshold (t={t_op}). "
        "Balances sensitivity and specificity for screening use."))

    # 3 - Cost-sensitive threshold (minimise FN_cost·FN + FP_cost·FP)
    fn_cost = tcfg["fn_cost"]
    fp_cost = tcfg["fp_cost"]
    t_cost = _find_threshold_min_cost(probs, y_true, fn_cost, fp_cost)
    candidates.append(_make_candidate("cost_minimised", t_cost, probs, y_true,
        f"Minimises weighted cost with FN_cost={fn_cost}, FP_cost={fp_cost}. "
        "Prioritises avoiding missed high-stress cases."))

    for c in candidates:
        log.info(
            "Threshold [%s] t=%.3f | Sens=%.3f Spec=%.3f F1=%.3f Acc=%.3f",
            c.name, c.threshold, c.sensitivity, c.specificity, c.f1, c.accuracy,
        )
    return candidates


def _find_threshold_max_f1(probs: np.ndarray, y_true: np.ndarray) -> float:
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, (probs >= t).astype(int), zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def _find_threshold_min_cost(
    probs: np.ndarray,
    y_true: np.ndarray,
    fn_cost: float,
    fp_cost: float,
) -> float:
    thresholds = np.linspace(0.01, 0.99, 199)
    costs = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        costs.append(fn_cost * fn + fp_cost * fp)
    return float(thresholds[int(np.argmin(costs))])


def _make_candidate(
    name: str,
    t: float,
    probs: np.ndarray,
    y_true: np.ndarray,
    description: str,
) -> ThresholdCandidate:
    y_pred = (probs >= t).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = (tp + tn) / len(y_true)
    return ThresholdCandidate(
        name=name,
        threshold=t,
        sensitivity=sensitivity,
        specificity=specificity,
        f1=f1,
        accuracy=accuracy,
        description=description,
    )


# ?? Bootstrap odds ratios ??????????????????????????????????????????????????????

def _bootstrap_odds_ratios(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    lrcfg: dict,
    n_bootstrap: int = 500,
    random_state: int = 0,
) -> pd.DataFrame:
    """Estimate 95% CI for odds ratios via non-parametric bootstrap."""
    rng = np.random.default_rng(random_state)
    n = len(y)
    coef_samples = np.zeros((n_bootstrap, len(feature_names)))

    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b, y_b = X[idx], y[idx]
        if len(np.unique(y_b)) < 2:
            coef_samples[i] = np.nan
            continue
        lr = LogisticRegression(
            solver=lrcfg["solver"],
            penalty=lrcfg["penalty"],
            C=lrcfg["C"],
            class_weight=lrcfg["class_weight"],
            max_iter=lrcfg["max_iter"],
            random_state=lrcfg["random_state"],
        )
        lr.fit(X_b, y_b)
        coef_samples[i] = lr.coef_[0]

    coef_samples = coef_samples[~np.isnan(coef_samples[:, 0])]
    or_point = np.exp(coef_samples.mean(axis=0))
    ci_low   = np.exp(np.percentile(coef_samples, 2.5, axis=0))
    ci_high  = np.exp(np.percentile(coef_samples, 97.5, axis=0))

    result = pd.DataFrame({
        "feature": feature_names,
        "odds_ratio": or_point,
        "ci_lower_95": ci_low,
        "ci_upper_95": ci_high,
    })
    log.info("Bootstrap OR computed over %d valid samples", len(coef_samples))
    return result.sort_values("odds_ratio", ascending=False).reset_index(drop=True)


# ?? Model persistence ??????????????????????????????????????????????????????????

def save_model(model: LogisticRegression, pipeline: Any, path: str | Path) -> None:
    """Persist model + fitted pipeline to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "pipeline": pipeline}, path)
    log.info("Model saved to %s", path)


def load_model(path: str | Path) -> tuple[LogisticRegression, Any]:
    """Load model + pipeline from a joblib file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    bundle = joblib.load(path)
    log.info("Model loaded from %s", path)
    return bundle["model"], bundle["pipeline"]
