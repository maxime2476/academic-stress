"""
Econometric test battery -- run as a standalone script.

Usage (from project root):
    python scripts/run_econometrics.py

Prerequisites
-------------
Run ``scripts/run_analysis.py`` first so that the fitted model is available
at ``outputs/models/logistic_regression.joblib``.  If not found the script
will re-train a fresh model before running the tests.

What this script does
---------------------
1.  Load data and re-encode features
2.  Load (or retrain) the fitted model
3.  Run all 13 econometric/statistical tests via
    ``src.econometrics.run_full_econometric_battery``
4.  Print a structured console report
5.  Export detailed results to ``outputs/results/econometrics_report.json``
6.  Generate diagnostic figures to ``outputs/figures/``

Tests covered
-------------
1.  VIF -- multicollinearity
2.  Hosmer-Lemeshow -- goodness-of-fit
3.  Likelihood Ratio Test -- vs. null model
4.  Wald tests -- per-coefficient significance
5.  McFadden R2
6.  Cox-Snell R2
7.  Nagelkerke R2
8.  AIC / BIC
9.  Pearson / Deviance goodness-of-fit
10. Cramer's V with 95% CI -- categorical associations
11. Kruskal-Wallis + pairwise Mann-Whitney -- group differences
12. Spearman correlations -- ordinal relationships
13. Average Marginal Effects (AME) at the mean
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.config import get_config, resolve_path
from src.data_loader import load_and_clean_data
from src.preprocessing import (
    extract_features_target,
    build_preprocessing_pipeline,
    transform_to_dataframe,
)
from src.modeling import train_and_evaluate, load_model
from src.econometrics import (
    run_full_econometric_battery,
    cramers_v_matrix,
    spearman_correlation_matrix,
)
from src.utils import setup_logging, ensure_dirs, save_json, print_section


def main() -> None:
    cfg = get_config()
    setup_logging(
        log_file=resolve_path(cfg["outputs"]["logs_dir"]) / "run_econometrics.log"
    )

    fig_dir     = resolve_path(cfg["outputs"]["figures_dir"])
    results_dir = resolve_path(cfg["outputs"]["results_dir"])
    models_dir  = resolve_path(cfg["outputs"]["models_dir"])
    ensure_dirs(fig_dir, results_dir, models_dir)

    # ?? Load data ??????????????????????????????????????????????????????????????
    print_section("Loading Data")
    data_path = resolve_path(cfg["data"]["path"])
    df, report = load_and_clean_data(data_path, encoding=cfg["data"]["encoding"])
    print(f"  n = {len(df)} observations")

    X, y = extract_features_target(df, cfg)

    # ?? Load or retrain model ?????????????????????????????????????????????????
    model_path = models_dir / "logistic_regression.joblib"
    if model_path.exists():
        print_section("Loading Saved Model")
        model, pipeline = load_model(model_path)
        pipeline.fit(X)  # refit pipeline on full X to get feature names
        X_enc = transform_to_dataframe(pipeline, X)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=cfg["modeling"]["test_size"],
            random_state=cfg["modeling"]["random_state"],
            stratify=y,
        )
        pipeline_train = build_preprocessing_pipeline()
        pipeline_train.fit(X_train)
        X_train_enc = transform_to_dataframe(pipeline_train, X_train)
        X_test_enc  = transform_to_dataframe(pipeline_train, X_test)
        model.fit(X_train_enc, y_train)
    else:
        print_section("No Saved Model Found -- Retraining")
        result = train_and_evaluate(X, y, cfg)
        model      = result.model
        pipeline   = result.pipeline
        X_train    = result.X_train
        X_test     = result.X_test
        y_train    = result.y_train
        y_test     = result.y_test
        X_train_enc = result.X_train_enc
        X_test_enc  = result.X_test_enc

    # ?? Run full econometric battery ??????????????????????????????????????????
    print_section("Running Econometric Battery (13 tests)")
    eco_report = run_full_econometric_battery(
        model=model,
        X_train_enc=X_train_enc,
        X_test_enc=X_test_enc,
        y_train=y_train.values if hasattr(y_train, "values") else y_train,
        y_test=y_test.values  if hasattr(y_test,  "values") else y_test,
        df_raw=df,
        cfg=cfg,
    )

    # ?? Console report ????????????????????????????????????????????????????????
    _print_console_report(eco_report, cfg)

    # ?? Figures ???????????????????????????????????????????????????????????????
    print_section("Generating Econometric Figures")

    _plot_vif(eco_report["vif"], save_path=fig_dir / "vif_barplot.png")
    _plot_wald(eco_report["wald_tests"], save_path=fig_dir / "wald_coefficients.png")
    _plot_marginal_effects(eco_report["marginal_effects"],
                           save_path=fig_dir / "marginal_effects.png")
    _plot_spearman(eco_report["spearman_correlations"],
                   cfg["columns"]["ordinal"] + [cfg["data"]["target_column"]],
                   save_path=fig_dir / "spearman_heatmap.png")
    _plot_cramers_v(eco_report["cramers_v"],
                    save_path=fig_dir / "cramers_v_associations.png")

    print(f"  Figures saved to {fig_dir}")

    # ?? JSON export ???????????????????????????????????????????????????????????
    save_json(eco_report, results_dir / "econometrics_report.json")
    print(f"  Econometrics report saved to {results_dir / 'econometrics_report.json'}")
    print_section("Done")


# ?? Console printing ???????????????????????????????????????????????????????????

def _print_console_report(report: dict, cfg: dict) -> None:
    alpha = cfg["econometrics"]["significance_level"]

    print_section("1 -- Multicollinearity (VIF)")
    vif_df = pd.DataFrame(report["vif"])
    print(vif_df.to_string(index=False))

    print_section("2 -- Hosmer-Lemeshow Test")
    hl = report["hosmer_lemeshow"]
    _print_test(hl["statistic"], hl["p_value"], hl["df"], hl["conclusion"], alpha)

    print_section("3 -- Likelihood Ratio Test")
    lr = report["likelihood_ratio_test"]
    _print_test(lr["statistic"], lr["p_value"], lr["df"], lr["conclusion"], alpha)
    print(f"  LL_full = {lr['ll_full']:.3f} | LL_null = {lr['ll_null']:.3f}")

    print_section("4 -- Wald Tests (per coefficient)")
    wald_df = pd.DataFrame(report["wald_tests"])
    cols_show = ["feature", "coef", "std_error", "z_stat", "p_value", "odds_ratio", "significant"]
    print(wald_df[cols_show].to_string(index=False))

    print_section("5-7 -- Pseudo R2")
    pr2 = report["pseudo_r2"]
    print(f"  McFadden  R2  : {pr2['mcfadden_r2']:.4f}")
    print(f"  Cox-Snell R2  : {pr2['cox_snell_r2']:.4f}")
    print(f"  Nagelkerke R2 : {pr2['nagelkerke_r2']:.4f}")
    print(f"  (McFadden > 0.20 indicates good model fit)")

    print_section("8 -- Information Criteria")
    ic = report["information_criteria"]
    print(f"  AIC : {ic['aic']:.3f}")
    print(f"  BIC : {ic['bic']:.3f}")
    print(f"  (k={ic['k']} parameters, n={ic['n']})")

    print_section("9 -- Pearson & Deviance Tests")
    for name, t in report["pearson_deviance"].items():
        _print_test(t["statistic"], t["p_value"], t["df"], t["conclusion"], alpha,
                    label=name.capitalize())

    print_section("10 -- Cramer's V (Categorical <-> Target)")
    for col, stats in report["cramers_v"].items():
        sig_marker = "*" if stats["ci_lower_95"] > 0.10 else ""
        print(f"  {col[:55]:<55} V={stats['cramers_v']:.3f} "
              f"[{stats['ci_lower_95']:.3f}, {stats['ci_upper_95']:.3f}] {sig_marker}")

    print_section("11 -- Kruskal-Wallis (Group Differences in Stress Level)")
    for col, t in report["kruskal_wallis"].items():
        _print_test(t["statistic"], t["p_value"], t["df"], t["conclusion"], alpha,
                    label=col[:40])

    print_section("12 -- Spearman Correlations (Ordinal Features <-> Target)")
    rho_dict  = report["spearman_correlations"]["rho"]
    pval_dict = report["spearman_correlations"]["pvalue"]
    target_col = list(rho_dict.keys())[-1]
    print(f"  {'Feature':<55} rho       p-value")
    for feat in list(rho_dict.keys())[:-1]:
        rho  = rho_dict[feat].get(target_col, "N/A")
        pval = pval_dict[feat].get(target_col, "N/A")
        try:
            sig = "*" if float(pval) < alpha else ""
            print(f"  {feat[:55]:<55} {float(rho):+.3f}   {float(pval):.4f} {sig}")
        except (TypeError, ValueError):
            print(f"  {feat[:55]:<55} {rho}   {pval}")

    print_section("13 -- Average Marginal Effects (AME)")
    me_df = pd.DataFrame(report["marginal_effects"])
    print(me_df[["feature", "coef", "AME"]].to_string(index=False))
    print("\n  AME interpretation: a one-unit increase in feature X changes")
    print("  P(HighStress=1) by AME on average, holding others constant.")


def _print_test(stat, pval, df, conclusion, alpha, label="") -> None:
    sig = "*** SIGNIFICANT ***" if pval < alpha else "not significant"
    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}stat={stat:.4f}  p={pval:.4f}  df={df}  -> {sig}")
    print(f"  {conclusion}")


# ?? Diagnostic figures ?????????????????????????????????????????????????????????

def _plot_vif(vif_records: list, save_path: Path) -> None:
    df = pd.DataFrame(vif_records).sort_values("VIF", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 0.5)))
    colors = ["#d62728" if v >= 10 else "#ff7f0e" if v >= 5 else "#2ca02c"
              for v in df["VIF"]]
    ax.barh(df["feature"], df["VIF"], color=colors, alpha=0.85)
    ax.axvline(5,  color="orange", ls="--", lw=1.5, label="VIF=5 (moderate)")
    ax.axvline(10, color="red",    ls="--", lw=1.5, label="VIF=10 (severe)")
    ax.set_xlabel("Variance Inflation Factor (VIF)")
    ax.set_title("Multicollinearity Check -- VIF per Feature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_wald(wald_records: list, save_path: Path) -> None:
    df = pd.DataFrame(wald_records).sort_values("coef")
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.55)))
    y_pos = range(len(df))
    colors = ["#d62728" if sig else "#1f77b4" for sig in df["significant"]]
    ax.barh(list(y_pos), df["coef"], color=colors, alpha=0.8, height=0.55)
    ax.errorbar(
        df["coef"], list(y_pos),
        xerr=[df["coef"] - df["ci_lower_95"], df["ci_upper_95"] - df["coef"]],
        fmt="none", color="black", capsize=3, lw=1.2,
    )
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(df["feature"], fontsize=9)
    ax.set_xlabel("Coefficient (log-odds)")
    ax.set_title("Wald Test -- Logistic Regression Coefficients with 95% CI\n"
                 "(red = significant at alpha=0.05)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_marginal_effects(me_records: list, save_path: Path) -> None:
    df = pd.DataFrame(me_records).sort_values("AME")
    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.55)))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in df["AME"]]
    ax.barh(df["feature"], df["AME"], color=colors, alpha=0.8)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Average Marginal Effect -- DP(HighStress=1)")
    ax.set_title("Average Marginal Effects at the Mean\n"
                 "(red = increases stress probability)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_spearman(spearman_dict: dict, cols: list, save_path: Path) -> None:
    rho_df = pd.DataFrame(spearman_dict["rho"])
    rho_df = rho_df.loc[
        [c for c in cols if c in rho_df.index],
        [c for c in cols if c in rho_df.columns],
    ].astype(float)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        rho_df, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1,
        linewidths=0.5, ax=ax,
    )
    ax.set_title("Spearman Correlation Matrix -- Ordinal Features & Stress Index")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cramers_v(cramers_dict: dict, save_path: Path) -> None:
    features = list(cramers_dict.keys())
    vals     = [cramers_dict[f]["cramers_v"]    for f in features]
    ci_low   = [cramers_dict[f]["cramers_v"] - cramers_dict[f]["ci_lower_95"] for f in features]
    ci_high  = [cramers_dict[f]["ci_upper_95"] - cramers_dict[f]["cramers_v"] for f in features]

    order = sorted(range(len(vals)), key=lambda i: vals[i])
    features_s = [features[i] for i in order]
    vals_s     = [vals[i]     for i in order]
    ci_low_s   = [ci_low[i]   for i in order]
    ci_high_s  = [ci_high[i]  for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, len(features) * 0.6)))
    ax.barh(features_s, vals_s, color="#2ca02c", alpha=0.75)
    ax.errorbar(vals_s, features_s, xerr=[ci_low_s, ci_high_s],
                fmt="none", color="black", capsize=4, lw=1.5)
    ax.axvline(0.1, color="orange", ls="--", lw=1.2, label="V=0.10 (weak)")
    ax.axvline(0.3, color="red",    ls="--", lw=1.2, label="V=0.30 (moderate)")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Cramer's V (95% Bootstrap CI)")
    ax.set_title("Categorical Association with Stress Level (HighStress target)\nCramer's V")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
