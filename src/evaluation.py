"""
Model evaluation: metrics computation, plots, and result exports.

All plot functions accept a ``save_path`` argument.  When provided the figure is
saved to disk (PNG, 150 dpi) AND returned.  When ``None`` the figure is only
returned (useful in notebooks or tests).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)

plt.style.use("seaborn-v0_8-whitegrid")
PALETTE = sns.color_palette("muted")


# ?? Metrics ????????????????????????????????????????????????????????????????????

def compute_all_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float,
    label: str = "",
) -> dict[str, Any]:
    """Return a comprehensive metrics dictionary for a given threshold.

    Includes AUC-ROC, PR-AUC, Brier score, accuracy, sensitivity,
    specificity, PPV, NPV, F1, and the full classification report.
    """
    y_pred = (probs >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics: dict[str, Any] = {
        "label": label,
        "threshold": threshold,
        "n": int(len(y_true)),
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier_score": float(brier_score_loss(y_true, probs)),
        "accuracy": float((tp + tn) / len(y_true)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=["No High Stress (0)", "High Stress (1)"],
            output_dict=True,
        ),
    }
    log.info(
        "[%s] AUC=%.3f PR-AUC=%.3f Brier=%.3f Acc=%.3f Sens=%.3f Spec=%.3f",
        label or "test",
        metrics["roc_auc"], metrics["pr_auc"], metrics["brier_score"],
        metrics["accuracy"], metrics["sensitivity"], metrics["specificity"],
    )
    return metrics


def calibration_analysis(
    model: LogisticRegression,
    X_train_enc: pd.DataFrame,
    X_test_enc: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    probs_test_raw: np.ndarray,
) -> dict[str, Any]:
    """Compare raw vs. isotonic-calibrated probabilities on the test set."""
    # Isotonic calibration fitted on training data
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    calibrated.fit(X_train_enc, y_train)
    probs_calibrated = calibrated.predict_proba(X_test_enc)[:, 1]

    brier_raw  = brier_score_loss(y_test, probs_test_raw)
    brier_cal  = brier_score_loss(y_test, probs_calibrated)

    frac_raw, mean_raw = calibration_curve(y_test, probs_test_raw, n_bins=6)
    frac_cal, mean_cal = calibration_curve(y_test, probs_calibrated, n_bins=6)

    result = {
        "brier_score_raw": float(brier_raw),
        "brier_score_calibrated": float(brier_cal),
        "brier_improvement": float(brier_raw - brier_cal),
        "calibration_raw": {"mean_predicted": mean_raw.tolist(), "fraction_positive": frac_raw.tolist()},
        "calibration_isotonic": {"mean_predicted": mean_cal.tolist(), "fraction_positive": frac_cal.tolist()},
    }
    log.info("Calibration -- raw Brier=%.4f, isotonic Brier=%.4f", brier_raw, brier_cal)
    return result


# ?? Plots ??????????????????????????????????????????????????????????????????????

def plot_roc_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot ROC curve with AUC annotation and optional threshold marker."""
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    auc = roc_auc_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    if threshold is not None:
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(fpr[idx], tpr[idx], s=80, zorder=5, color=PALETTE[3],
                   label=f"Operational threshold (t={threshold:.3f})")

    ax.set_xlabel("False Positive Rate (1 ? Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("ROC Curve -- Logistic Regression")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot Precision-Recall curve with AP annotation."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    ap = average_precision_score(y_true, probs)
    baseline = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color=PALETTE[1], lw=2, label=f"AP = {ap:.3f}")
    ax.axhline(baseline, color="k", ls="--", lw=1, alpha=0.5,
               label=f"Baseline prevalence ({baseline:.2f})")

    if threshold is not None:
        # Find the index in thresholds array (len = len(precision) - 1)
        idx = np.argmin(np.abs(thresholds - threshold))
        ax.scatter(recall[idx], precision[idx], s=80, zorder=5, color=PALETTE[3],
                   label=f"Operational threshold (t={threshold:.3f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot labelled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No High Stress", "High Stress"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.3f})")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_calibration(
    y_true: np.ndarray,
    probs_raw: np.ndarray,
    probs_calibrated: np.ndarray | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot reliability diagram (calibration curve)."""
    frac_raw, mean_raw = calibration_curve(y_true, probs_raw, n_bins=6)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.plot(mean_raw, frac_raw, "o-", color=PALETTE[0], label="Raw logistic regression")

    if probs_calibrated is not None:
        frac_cal, mean_cal = calibration_curve(y_true, probs_calibrated, n_bins=6)
        ax.plot(mean_cal, frac_cal, "s--", color=PALETTE[2], label="Isotonic calibrated")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve (Reliability Diagram)")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_odds_ratios(
    odds_ratios: pd.DataFrame,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Forest plot for bootstrapped odds ratios with 95% CI."""
    df = odds_ratios.copy().sort_values("odds_ratio")

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.55)))
    y_pos = np.arange(len(df))

    ax.barh(y_pos, df["odds_ratio"], color=PALETTE[0], alpha=0.7, height=0.5)
    ax.errorbar(
        df["odds_ratio"], y_pos,
        xerr=[df["odds_ratio"] - df["ci_lower_95"], df["ci_upper_95"] - df["odds_ratio"]],
        fmt="none", color="black", capsize=4, lw=1.5,
    )
    ax.axvline(1.0, color="red", ls="--", lw=1.5, label="OR = 1 (no effect)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("Odds Ratio (95% Bootstrap CI)")
    ax.set_title("Odds Ratios -- Logistic Regression Coefficients")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


def plot_threshold_comparison(
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: list,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Bar chart comparing sensitivity, specificity, F1 across threshold candidates."""
    records = []
    for tc in thresholds:
        records.append({
            "Threshold": f"{tc.name}\nt={tc.threshold:.3f}",
            "Sensitivity": tc.sensitivity,
            "Specificity": tc.specificity,
            "F1": tc.f1,
            "Accuracy": tc.accuracy,
        })
    df = pd.DataFrame(records)

    metrics_cols = ["Sensitivity", "Specificity", "F1", "Accuracy"]
    df_melt = df.melt(id_vars="Threshold", value_vars=metrics_cols,
                      var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=df_melt, x="Threshold", y="Score", hue="Metric",
                palette=PALETTE[:4], ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title("Threshold Candidates -- Performance Comparison")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, save_path)
    return fig


# ?? Result export ??????????????????????????????????????????????????????????????

def export_predictions(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    probs: np.ndarray,
    threshold: float,
    path: str | Path,
) -> None:
    """Export test-set predictions to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = X_test.copy().reset_index(drop=True)
    out["y_true"] = y_test.values
    out["prob_high_stress"] = probs.round(4)
    out["y_pred"] = (probs >= threshold).astype(int)
    out.to_csv(path, index=False)
    log.info("Predictions exported to %s", path)


def export_full_scores(
    X: pd.DataFrame,
    y: pd.Series,
    probs: np.ndarray,
    threshold: float,
    path: str | Path,
) -> None:
    """Export predicted probabilities for the full dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = X.copy().reset_index(drop=True)
    out["y_true"] = y.values
    out["prob_high_stress"] = probs.round(4)
    out["y_pred"] = (probs >= threshold).astype(int)
    out.to_csv(path, index=False)
    log.info("Full scores exported to %s", path)


def export_metrics_json(metrics: dict, path: str | Path) -> None:
    """Persist metrics dict as JSON, converting numpy types for serialisation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serialisable = json.loads(json.dumps(metrics, default=_convert))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)
    log.info("Metrics exported to %s", path)


# ?? Private helpers ????????????????????????????????????????????????????????????

def _save_fig(fig: plt.Figure, path: str | Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved to %s", path)
