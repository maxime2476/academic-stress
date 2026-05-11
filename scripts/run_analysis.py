"""
Main analysis pipeline -- run end-to-end.

Usage (from project root):
    python scripts/run_analysis.py

What this script does
---------------------
1.  Load and validate raw data          (src.data_loader)
2.  Extract features and binary target  (src.preprocessing)
3.  Train logistic regression + CV      (src.modeling)
4.  Evaluate on test set                (src.evaluation)
5.  Export all artefacts to outputs/    (figures, metrics JSON, CSV predictions)
6.  Save fitted model + pipeline        (outputs/models/)

All parameters are read from config.yaml -- no hardcoded values in this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ?? Allow running from the project root without installing the package ?????????
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.config import get_config, resolve_path
from src.data_loader import load_and_clean_data, get_data_summary
from src.preprocessing import extract_features_target, transform_to_dataframe
from src.modeling import train_and_evaluate, save_model
from src.evaluation import (
    compute_all_metrics,
    calibration_analysis,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_calibration,
    plot_odds_ratios,
    plot_threshold_comparison,
    export_predictions,
    export_full_scores,
    export_metrics_json,
)
from src.utils import setup_logging, ensure_dirs, save_json, print_section


def main() -> None:
    cfg = get_config()
    setup_logging(log_file=resolve_path(cfg["outputs"]["logs_dir"]) / "run_analysis.log")

    fig_dir     = resolve_path(cfg["outputs"]["figures_dir"])
    results_dir = resolve_path(cfg["outputs"]["results_dir"])
    models_dir  = resolve_path(cfg["outputs"]["models_dir"])
    ensure_dirs(fig_dir, results_dir, models_dir)

    # ?? 1. Data loading ????????????????????????????????????????????????????????
    print_section("1 -- Data Loading & Validation")
    data_path = resolve_path(cfg["data"]["path"])
    df, report = load_and_clean_data(data_path, encoding=cfg["data"]["encoding"])
    print(report.summary())

    data_summary = get_data_summary(df)
    save_json(data_summary, results_dir / "data_summary.json")

    # ?? 2. Feature / target extraction ????????????????????????????????????????
    print_section("2 -- Feature & Target Extraction")
    X, y = extract_features_target(df, cfg)
    print(f"Features : {X.shape[1]} columns, {X.shape[0]} rows")
    print(f"Target   : HighStress=1 -> {y.sum()} ({100*y.mean():.1f}%)")

    # ?? 3. Training ????????????????????????????????????????????????????????????
    print_section("3 -- Model Training")
    result = train_and_evaluate(X, y, cfg)

    print("\nCross-validation results:")
    for metric, cv in result.cv_results.items():
        print(f"  {metric:<20}: {cv}")

    # ?? 4. Evaluation ?????????????????????????????????????????????????????????
    print_section("4 -- Evaluation")

    # Pick the operational threshold for primary reporting
    op_threshold = cfg["thresholds"]["operational"]
    op_tc = next((t for t in result.thresholds if t.name == "operational"), result.thresholds[1])

    metrics_test = compute_all_metrics(
        result.y_test.values,
        result.test_probs,
        threshold=op_tc.threshold,
        label="test_set",
    )
    metrics_train = compute_all_metrics(
        result.y_train.values,
        result.train_probs,
        threshold=op_tc.threshold,
        label="train_set",
    )

    _print_metrics(metrics_test, label="Test set")
    _print_metrics(metrics_train, label="Train set")

    # Calibration
    cal_result = calibration_analysis(
        result.model,
        result.X_train_enc,
        result.X_test_enc,
        result.y_train.values,
        result.y_test.values,
        result.test_probs,
    )
    print(f"\nCalibration -- raw Brier: {cal_result['brier_score_raw']:.4f} "
          f"| isotonic Brier: {cal_result['brier_score_calibrated']:.4f}")

    # ?? 5. Figures ????????????????????????????????????????????????????????????
    print_section("5 -- Generating Figures")

    plot_roc_curve(
        result.y_test.values, result.test_probs,
        threshold=op_tc.threshold,
        save_path=fig_dir / "roc_curve.png",
    )
    plot_precision_recall_curve(
        result.y_test.values, result.test_probs,
        threshold=op_tc.threshold,
        save_path=fig_dir / "pr_curve.png",
    )
    plot_confusion_matrix(
        result.y_test.values,
        (result.test_probs >= op_tc.threshold).astype(int),
        threshold=op_tc.threshold,
        save_path=fig_dir / "confusion_matrix.png",
    )
    plot_calibration(
        result.y_test.values, result.test_probs,
        save_path=fig_dir / "calibration_curve.png",
    )
    plot_odds_ratios(
        result.bootstrap_odds_ratios,
        save_path=fig_dir / "odds_ratios.png",
    )
    plot_threshold_comparison(
        result.y_test.values, result.test_probs,
        result.thresholds,
        save_path=fig_dir / "threshold_comparison.png",
    )
    print(f"  Figures saved to {fig_dir}")

    # ?? 6. Result exports ?????????????????????????????????????????????????????
    print_section("6 -- Exporting Results")

    export_predictions(
        result.X_test, result.y_test, result.test_probs,
        threshold=op_tc.threshold,
        path=results_dir / "predictions_test.csv",
    )
    export_full_scores(
        X, y,
        probs=result.model.predict_proba(
            transform_to_dataframe(result.pipeline, X)
        )[:, 1],
        threshold=op_tc.threshold,
        path=results_dir / "scores_full.csv",
    )

    all_results = {
        "metrics_test":   metrics_test,
        "metrics_train":  metrics_train,
        "cv_results":     {k: {"mean": v.mean, "std": v.std} for k, v in result.cv_results.items()},
        "calibration":    cal_result,
        "thresholds":     [
            {
                "name": t.name, "threshold": t.threshold,
                "sensitivity": t.sensitivity, "specificity": t.specificity,
                "f1": t.f1, "accuracy": t.accuracy, "description": t.description,
            }
            for t in result.thresholds
        ],
        "odds_ratios":    result.bootstrap_odds_ratios.to_dict(orient="records"),
        "feature_names":  result.feature_names,
        "data_report":    {
            "n_raw": report.n_rows_raw,
            "n_clean": report.n_rows_after_drop,
            "missing": report.missing_per_column,
        },
    }
    export_metrics_json(all_results, results_dir / "full_results.json")
    print(f"  Results saved to {results_dir}")

    # ?? 7. Model persistence ??????????????????????????????????????????????????
    print_section("7 -- Saving Model")
    save_model(result.model, result.pipeline, models_dir / "logistic_regression.joblib")
    print(f"  Model saved to {models_dir}")

    print_section("Done")
    print(f"  AUC-ROC (test)  : {metrics_test['roc_auc']:.3f}")
    print(f"  PR-AUC  (test)  : {metrics_test['pr_auc']:.3f}")
    print(f"  Accuracy (test) : {metrics_test['accuracy']:.3f}")
    print(f"  Sensitivity     : {metrics_test['sensitivity']:.3f}")
    print(f"  Specificity     : {metrics_test['specificity']:.3f}")
    print(f"  Brier score     : {metrics_test['brier_score']:.4f}")


def _print_metrics(m: dict, label: str) -> None:
    print(f"\n  [{label}]")
    print(f"    AUC-ROC     : {m['roc_auc']:.3f}")
    print(f"    PR-AUC      : {m['pr_auc']:.3f}")
    print(f"    Brier score : {m['brier_score']:.4f}")
    print(f"    Accuracy    : {m['accuracy']:.3f}")
    print(f"    Sensitivity : {m['sensitivity']:.3f}  (recall for positive class)")
    print(f"    Specificity : {m['specificity']:.3f}  (recall for negative class)")
    print(f"    PPV         : {m['ppv']:.3f}  (precision)")
    print(f"    F1          : {m['f1']:.3f}")
    print(f"    TP={m['tp']}  TN={m['tn']}  FP={m['fp']}  FN={m['fn']}")


if __name__ == "__main__":
    main()
