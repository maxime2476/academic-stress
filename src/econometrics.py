"""
Econometric and statistical test battery for logistic regression.

Tests implemented
-----------------
1.  VIF (Variance Inflation Factor)          -- multicollinearity
2.  Hosmer-Lemeshow goodness-of-fit test     -- model calibration
3.  Likelihood Ratio Test                    -- nested model comparison
4.  Wald test on individual coefficients     -- significance of each predictor
5.  McFadden R2                              -- pseudo R2 (log-likelihood based)
6.  Cox-Snell R2                             -- pseudo R2
7.  Nagelkerke R2                            -- pseudo R2 (scaled Cox-Snell)
8.  AIC / BIC                                -- information criteria
9.  Pearson & Deviance goodness-of-fit       -- overall fit
10. Cramer's V with 95 % bootstrap CI       -- pairwise categorical association
11. Kruskal-Wallis + pairwise Mann-Whitney   -- ordinal group comparisons
12. Spearman correlations                    -- ordinal/ordinal relationships
13. Marginal effects at the mean (MEM)       -- dP/dX at X?

All tests return a ``TestResult`` dataclass or a plain ``dict`` for easy JSON
serialisation.  The function ``run_full_econometric_battery`` aggregates all
tests into a single report.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from statsmodels.stats.outliers_influence import variance_inflation_factor

log = logging.getLogger(__name__)


# ?? Data class ?????????????????????????????????????????????????????????????????

@dataclass
class TestResult:
    """Standardised container for a single statistical test."""
    name: str
    statistic: float
    p_value: float
    df: int | None = None
    conclusion: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def __str__(self) -> str:
        sig = "*" if self.is_significant() else ""
        stat_str = f"stat={self.statistic:.4f}"
        pval_str = f"p={self.p_value:.4f}"
        df_str   = f"df={self.df}" if self.df is not None else ""
        parts    = [s for s in [stat_str, pval_str, df_str] if s]
        return f"{self.name}: {', '.join(parts)} {sig} | {self.conclusion}"


# ?? 1. VIF ?????????????????????????????????????????????????????????????????????

def compute_vif(X: pd.DataFrame, vif_threshold: float = 5.0) -> pd.DataFrame:
    """Compute Variance Inflation Factor for each column of X.

    VIF_j = 1 / (1 - R2_j) where R2_j is the R2 from regressing feature j
    on all other features (OLS).

    Interpretation
    --------------
    VIF < 5   -> no concerning multicollinearity
    5 <= VIF < 10 -> moderate multicollinearity
    VIF >= 10  -> severe multicollinearity

    Note: VIF is computed on the numeric design matrix.  Binary dummies from
    one-hot encoding (drop='first') have bounded VIF by construction.
    """
    X_arr = X.values.astype(float)
    vifs = []
    for i in range(X_arr.shape[1]):
        try:
            v = variance_inflation_factor(X_arr, i)
        except Exception:
            v = np.nan
        vifs.append(v)

    result = pd.DataFrame({
        "feature": X.columns.tolist(),
        "VIF": vifs,
        "concern": ["none" if v < 5 else ("moderate" if v < 10 else "severe")
                    for v in vifs],
    })
    high_vif = result[result["VIF"] >= vif_threshold]
    if not high_vif.empty:
        log.warning("High VIF features (>= %.1f): %s", vif_threshold, high_vif["feature"].tolist())
    return result.sort_values("VIF", ascending=False).reset_index(drop=True)


# ?? 2. Hosmer-Lemeshow test ????????????????????????????????????????????????????

def hosmer_lemeshow_test(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_groups: int = 10,
) -> TestResult:
    """Hosmer-Lemeshow goodness-of-fit test for logistic regression.

    Procedure
    ---------
    1. Rank observations by predicted probability.
    2. Divide into ``n_groups`` equal-frequency groups (deciles by default).
    3. Within each group compute:
       - O_1 (observed positives),  E_1 (expected positives = ? p?)
       - O_0 (observed negatives),  E_0 (expected negatives = ? (1-p?))
    4. HL statistic = ?_g [ (O_1g ? E_1g)2 / E_1g  +  (O_0g ? E_0g)2 / E_0g ]
    5. Under H0 (good fit): HL ~ chi2(n_groups ? 2).

    H0: the model fits the data well (p > 0.05 -> good fit).
    """
    df = pd.DataFrame({"y": y_true, "p": probs})
    df["group"] = pd.qcut(df["p"], q=n_groups, duplicates="drop", labels=False)

    g_actual = df["group"].nunique()
    table = df.groupby("group").agg(
        O1=("y", "sum"),
        n=("y", "count"),
        E1=("p", "sum"),
    ).reset_index()
    table["O0"] = table["n"] - table["O1"]
    table["E0"] = table["n"] - table["E1"]

    # Avoid division by zero for near-empty expected cells
    hl_stat = (
        ((table["O1"] - table["E1"]) ** 2 / table["E1"].clip(lower=1e-6)).sum()
        + ((table["O0"] - table["E0"]) ** 2 / table["E0"].clip(lower=1e-6)).sum()
    )
    df_stat = g_actual - 2
    p_value = 1 - stats.chi2.cdf(hl_stat, df=df_stat) if df_stat > 0 else np.nan

    conclusion = (
        "Model fits data well (fail to reject H0)" if p_value > 0.05
        else "Evidence of poor fit (reject H0 at alpha=0.05)"
    )
    return TestResult(
        name="Hosmer-Lemeshow",
        statistic=float(hl_stat),
        p_value=float(p_value),
        df=int(df_stat),
        conclusion=conclusion,
        details={"n_groups_actual": g_actual, "group_table": table.round(3).to_dict()},
    )


# ?? 3. Likelihood Ratio Test ???????????????????????????????????????????????????

def likelihood_ratio_test(
    X_full: pd.DataFrame,
    y: np.ndarray,
    X_null: pd.DataFrame | None = None,
    model_params: dict | None = None,
) -> TestResult:
    """Compare full model against null (intercept-only) model.

    LR = ?2 × (LL_null ? LL_full)  ~  chi2(df = p_full ? p_null)

    H0: the restricted (null) model fits as well as the full model.
    """
    if model_params is None:
        model_params = {}

    # Full model
    lr_full = LogisticRegression(max_iter=5000, **model_params)
    lr_full.fit(X_full, y)
    ll_full = -log_loss(y, lr_full.predict_proba(X_full), normalize=False)

    # Null model (intercept only -- predict prevalence)
    p_null = float(y.mean())
    p_null = np.clip(p_null, 1e-9, 1 - 1e-9)
    ll_null = float(y.sum() * np.log(p_null) + (len(y) - y.sum()) * np.log(1 - p_null))

    lr_stat = -2.0 * (ll_null - ll_full)
    df_stat = X_full.shape[1]  # number of predictors added vs. null
    p_value = stats.chi2.sf(lr_stat, df=df_stat)

    conclusion = (
        f"Full model significantly better than null (reject H0) at alpha=0.05"
        if p_value < 0.05
        else "Full model does NOT significantly improve over null model"
    )
    return TestResult(
        name="Likelihood Ratio Test (full vs. null)",
        statistic=float(lr_stat),
        p_value=float(p_value),
        df=int(df_stat),
        conclusion=conclusion,
        details={"ll_full": float(ll_full), "ll_null": float(ll_null)},
    )


# ?? 4. Wald tests ??????????????????????????????????????????????????????????????

def wald_tests(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: np.ndarray,
) -> pd.DataFrame:
    """Approximate Wald z-test for each coefficient.

    z_j = beta_j / SE_j  where SE is estimated from the diagonal of the
    inverse observed Fisher information matrix: I(beta) = X? diag(p?(1-p?)) X.

    Note: these are large-sample approximations; with n=139 they provide
    indicative rather than definitive inference.
    """
    probs = model.predict_proba(X)[:, 1]
    X_arr = X.values.astype(float)

    # Observed Fisher information (Hessian of log-likelihood, negated)
    W = probs * (1 - probs)                   # shape (n,)
    info_matrix = (X_arr.T * W) @ X_arr       # shape (p, p)

    try:
        cov_matrix = np.linalg.inv(info_matrix)
        se = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        log.warning("Fisher information matrix is singular; using pseudo-inverse")
        cov_matrix = np.linalg.pinv(info_matrix)
        se = np.sqrt(np.abs(np.diag(cov_matrix)))

    coef = model.coef_[0]
    z_stats = coef / se
    p_values = 2 * stats.norm.sf(np.abs(z_stats))

    ci_low  = coef - 1.96 * se
    ci_high = coef + 1.96 * se

    result = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "coef":       coef.round(4),
        "std_error":  se.round(4),
        "z_stat":     z_stats.round(4),
        "p_value":    p_values.round(4),
        "ci_lower_95": ci_low.round(4),
        "ci_upper_95": ci_high.round(4),
        "odds_ratio":  np.exp(coef).round(4),
        "significant": p_values < 0.05,
    })
    return result.sort_values("p_value").reset_index(drop=True)


# ?? 5-7. Pseudo R2 ?????????????????????????????????????????????????????????????

def compute_pseudo_r2(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, float]:
    """Compute McFadden, Cox-Snell, and Nagelkerke pseudo-R2 values.

    McFadden  : 1 ? LL_full / LL_null       (range [0, 1])
    Cox-Snell : 1 ? exp(2/n (LL_null?LL_full))  (does not reach 1 for discrete y)
    Nagelkerke: CS / CS_max                  (scaled to [0, 1])

    Rough benchmarks for McFadden: 0.2-0.4 -> good fit; > 0.4 -> excellent.
    """
    n = len(y)
    p_full = model.predict_proba(X)[:, 1]
    ll_full = -log_loss(y, p_full, normalize=False)

    p_null = float(np.clip(y.mean(), 1e-9, 1 - 1e-9))
    ll_null = float(y.sum() * np.log(p_null) + (n - y.sum()) * np.log(1 - p_null))

    mcfadden = 1.0 - (ll_full / ll_null)

    cox_snell = 1.0 - np.exp((ll_null - ll_full) * 2.0 / n)
    cs_max    = 1.0 - np.exp(ll_null * 2.0 / n)
    nagelkerke = cox_snell / cs_max if cs_max > 0 else np.nan

    result = {
        "mcfadden_r2":   float(mcfadden),
        "cox_snell_r2":  float(cox_snell),
        "nagelkerke_r2": float(nagelkerke),
        "ll_full":       float(ll_full),
        "ll_null":       float(ll_null),
        "n":             int(n),
    }
    log.info(
        "Pseudo R2 -- McFadden=%.3f, Cox-Snell=%.3f, Nagelkerke=%.3f",
        mcfadden, cox_snell, nagelkerke,
    )
    return result


# ?? 8. AIC / BIC ???????????????????????????????????????????????????????????????

def compute_information_criteria(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, float]:
    """AIC and BIC for the fitted logistic regression.

    AIC = ?2 LL + 2k
    BIC = ?2 LL + k ln(n)

    where k = number of estimated parameters (coefficients + intercept).
    Lower is better; BIC penalises complexity more heavily.
    """
    n = len(y)
    k = X.shape[1] + 1  # predictors + intercept
    p_full = model.predict_proba(X)[:, 1]
    ll_full = -log_loss(y, p_full, normalize=False)

    aic = -2.0 * ll_full + 2.0 * k
    bic = -2.0 * ll_full + k * np.log(n)

    result = {"aic": float(aic), "bic": float(bic), "k": int(k), "n": int(n)}
    log.info("AIC=%.3f, BIC=%.3f (k=%d, n=%d)", aic, bic, k, n)
    return result


# ?? 9. Pearson & Deviance residual tests ??????????????????????????????????????

def pearson_deviance_tests(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict[str, TestResult]:
    """Pearson chi-square and deviance goodness-of-fit tests.

    Pearson  : ? (y ? p?)2 / [p?(1-p?)]   ~  chi2(n ? k ? 1)
    Deviance : ?2 ? [y ln(p?) + (1-y) ln(1-p?)]  ~  chi2(n ? k ? 1)

    These tests compare the fitted model to a saturated model (one parameter
    per observation).  With binary data and no repeated covariate patterns,
    both tests have low power -- treat as supplementary diagnostics.
    """
    probs = np.clip(model.predict_proba(X)[:, 1], 1e-10, 1 - 1e-10)
    y_arr = y.astype(float)
    n, k = len(y), X.shape[1]
    df = n - k - 1

    pearson_stat = float(np.sum((y_arr - probs) ** 2 / (probs * (1 - probs))))
    pearson_p    = float(stats.chi2.sf(pearson_stat, df=df))

    deviance_stat = float(-2.0 * np.sum(
        y_arr * np.log(probs) + (1 - y_arr) * np.log(1 - probs)
    ))
    deviance_p = float(stats.chi2.sf(deviance_stat, df=df))

    return {
        "pearson": TestResult(
            name="Pearson goodness-of-fit",
            statistic=pearson_stat,
            p_value=pearson_p,
            df=df,
            conclusion=(
                "Good fit (p > 0.05)" if pearson_p > 0.05
                else "Poor fit indicated (p <= 0.05)"
            ),
        ),
        "deviance": TestResult(
            name="Deviance goodness-of-fit",
            statistic=deviance_stat,
            p_value=deviance_p,
            df=df,
            conclusion=(
                "Good fit (p > 0.05)" if deviance_p > 0.05
                else "Poor fit indicated (p <= 0.05)"
            ),
        ),
    }


# ?? 10. Cramer's V with bootstrap CI ?????????????????????????????????????????

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramer's V -- symmetric measure of association between two categoricals.

    Range: [0, 1] where 0 = no association, 1 = perfect association.
    """
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    min_dim = min(ct.shape) - 1
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def cramers_v_matrix(
    df: pd.DataFrame,
    categorical_cols: list[str],
    n_bootstrap: int = 500,
    random_state: int = 0,
) -> pd.DataFrame:
    """Return pairwise Cramer's V matrix for all categorical column pairs."""
    cols = categorical_cols
    matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)
    for c1 in cols:
        for c2 in cols:
            matrix.loc[c1, c2] = cramers_v(df[c1], df[c2])
    return matrix.round(3)


def cramers_v_with_ci(
    x: pd.Series,
    y: pd.Series,
    n_bootstrap: int = 500,
    random_state: int = 0,
) -> dict[str, float]:
    """Cramer's V with 95% bootstrap confidence interval."""
    point = cramers_v(x, y)
    rng = np.random.default_rng(random_state)
    n = len(x)
    x_arr = x.reset_index(drop=True)
    y_arr = y.reset_index(drop=True)
    boot_vs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        # reset_index avoids duplicate-label errors in pd.crosstab
        boot_vs.append(cramers_v(
            x_arr.iloc[idx].reset_index(drop=True),
            y_arr.iloc[idx].reset_index(drop=True),
        ))
    return {
        "cramers_v":      point,
        "ci_lower_95":    float(np.percentile(boot_vs, 2.5)),
        "ci_upper_95":    float(np.percentile(boot_vs, 97.5)),
        "bootstrap_std":  float(np.std(boot_vs)),
    }


# ?? 11. Kruskal-Wallis + pairwise Mann-Whitney ????????????????????????????????

def kruskal_wallis_test(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
) -> TestResult:
    """Kruskal-Wallis H-test: non-parametric one-way ANOVA.

    H0: all group distributions are equal (location).
    Used here to test whether stress levels differ across categorical groups.
    """
    groups = [g[target_col].values for _, g in df.groupby(group_col)]
    if len(groups) < 2:
        raise ValueError(f"Column '{group_col}' has fewer than 2 unique values.")
    h_stat, p_value = stats.kruskal(*groups)
    conclusion = (
        f"Stress levels differ significantly across {group_col} groups (H0 rejected)"
        if p_value < 0.05
        else f"No significant difference in stress levels across {group_col} groups"
    )
    return TestResult(
        name=f"Kruskal-Wallis ({group_col})",
        statistic=float(h_stat),
        p_value=float(p_value),
        df=len(groups) - 1,
        conclusion=conclusion,
    )


def pairwise_mann_whitney(
    df: pd.DataFrame,
    group_col: str,
    target_col: str,
) -> pd.DataFrame:
    """Pairwise Mann-Whitney U tests with Bonferroni correction.

    Returns a DataFrame with every pair, raw p-values, and adjusted p-values.
    """
    groups = df.groupby(group_col)[target_col].apply(list)
    group_names = list(groups.index)
    records = []
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g1, g2 = groups.iloc[i], groups.iloc[j]
            u_stat, p_raw = stats.mannwhitneyu(g1, g2, alternative="two-sided")
            records.append({
                "group_1": group_names[i],
                "group_2": group_names[j],
                "u_stat": round(u_stat, 2),
                "p_value_raw": round(p_raw, 4),
            })
    result = pd.DataFrame(records)
    if not result.empty:
        # Bonferroni correction
        m = len(result)
        result["p_value_bonferroni"] = (result["p_value_raw"] * m).clip(upper=1.0).round(4)
        result["significant_bonferroni"] = result["p_value_bonferroni"] < 0.05
    return result


# ?? 12. Spearman correlations ?????????????????????????????????????????????????

def spearman_correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise Spearman rho and p-value matrices.

    Spearman is appropriate for ordinal / non-normal variables.
    """
    n = len(cols)
    rho_mat  = pd.DataFrame(np.ones((n, n)), index=cols, columns=cols)
    pval_mat = pd.DataFrame(np.zeros((n, n)), index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i == j:
                continue
            rho, pv = stats.spearmanr(df[c1].dropna(), df[c2].dropna())
            rho_mat.loc[c1, c2] = round(float(rho), 3)
            pval_mat.loc[c1, c2] = round(float(pv), 4)
    return rho_mat, pval_mat


# ?? 13. Marginal effects at the mean ?????????????????????????????????????????

def marginal_effects_at_mean(
    model: LogisticRegression,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Average Marginal Effects (AME) for each predictor.

    For a logistic regression:
       dP/dX_j = beta_j × (1/n) ?_i [p?_i (1 ? p?_i)]

    This gives the average change in P(Y=1) per one-unit increase in X_j,
    averaged over the observed distribution of all other predictors.

    For binary dummies, the AME approximates the discrete change in probability
    when the dummy switches from 0 to 1, holding others at observed values.
    """
    probs = model.predict_proba(X)[:, 1]
    avg_scale = float(np.mean(probs * (1 - probs)))
    coef = model.coef_[0]

    result = pd.DataFrame({
        "feature": X.columns.tolist(),
        "coef": coef,
        "AME": coef * avg_scale,
        "avg_scale_factor": avg_scale,
    })
    result["AME"] = result["AME"].round(4)
    result["coef"] = result["coef"].round(4)
    return result.sort_values("AME", key=abs, ascending=False).reset_index(drop=True)


# ?? Full battery ???????????????????????????????????????????????????????????????

def run_full_econometric_battery(
    model: LogisticRegression,
    X_train_enc: pd.DataFrame,
    X_test_enc: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    df_raw: pd.DataFrame,
    cfg: dict,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Run all econometric tests and return a single structured report.

    Parameters
    ----------
    model         : fitted LogisticRegression (trained on X_train_enc)
    X_train_enc   : encoded training features (DataFrame)
    X_test_enc    : encoded test features (DataFrame)
    y_train       : training target array
    y_test        : test target array
    df_raw        : original cleaned DataFrame (for Cramer's V, KW tests)
    cfg           : loaded config dict
    alpha         : significance level for binary conclusions

    Returns
    -------
    Nested dict suitable for JSON export.
    """
    train_probs = model.predict_proba(X_train_enc)[:, 1]
    test_probs  = model.predict_proba(X_test_enc)[:, 1]

    ordinal_cols  = cfg["columns"]["ordinal"]
    nominal_cols  = cfg["columns"]["nominal"]
    target_col    = cfg["data"]["target_column"]
    vif_threshold = cfg["econometrics"]["vif_threshold"]
    hl_groups     = cfg["econometrics"]["hosmer_lemeshow_groups"]

    log.info("=== Starting econometric battery ===")
    report: dict[str, Any] = {}

    # 1. VIF on training encoded features
    log.info("1/13 -- VIF")
    report["vif"] = compute_vif(X_train_enc, vif_threshold).to_dict(orient="records")

    # 2. Hosmer-Lemeshow (test set)
    log.info("2/13 -- Hosmer-Lemeshow")
    hl = hosmer_lemeshow_test(y_test, test_probs, n_groups=hl_groups)
    report["hosmer_lemeshow"] = {"statistic": hl.statistic, "p_value": hl.p_value,
                                  "df": hl.df, "conclusion": hl.conclusion}

    # 3. Likelihood Ratio Test (training set)
    log.info("3/13 -- Likelihood Ratio Test")
    lr_test = likelihood_ratio_test(X_train_enc, y_train)
    report["likelihood_ratio_test"] = {"statistic": lr_test.statistic,
                                        "p_value": lr_test.p_value,
                                        "df": lr_test.df,
                                        "conclusion": lr_test.conclusion,
                                        **lr_test.details}

    # 4. Wald tests (training set)
    log.info("4/13 -- Wald tests")
    wald = wald_tests(model, X_train_enc, y_train)
    report["wald_tests"] = wald.to_dict(orient="records")

    # 5-7. Pseudo R2 (training set)
    log.info("5-7/13 -- Pseudo R2")
    report["pseudo_r2"] = compute_pseudo_r2(model, X_train_enc, y_train)

    # 8. AIC / BIC (training set)
    log.info("8/13 -- AIC / BIC")
    report["information_criteria"] = compute_information_criteria(model, X_train_enc, y_train)

    # 9. Pearson / Deviance (test set)
    log.info("9/13 -- Pearson & Deviance")
    pdev = pearson_deviance_tests(model, X_test_enc, y_test)
    report["pearson_deviance"] = {
        k: {"statistic": v.statistic, "p_value": v.p_value,
            "df": v.df, "conclusion": v.conclusion}
        for k, v in pdev.items()
    }

    # 10. Cramer's V -- categorical associations with target
    log.info("10/13 -- Cramer's V")
    df_cat = df_raw[nominal_cols + [target_col]].copy()
    df_cat[target_col] = df_cat[target_col].astype(str)
    report["cramers_v"] = {
        col: cramers_v_with_ci(df_raw[col], df_cat[target_col])
        for col in nominal_cols
    }

    # 11. Kruskal-Wallis + pairwise Mann-Whitney for each nominal feature
    log.info("11/13 -- Kruskal-Wallis")
    report["kruskal_wallis"] = {}
    report["mann_whitney"] = {}
    for col in nominal_cols:
        try:
            kw = kruskal_wallis_test(df_raw, group_col=col, target_col=target_col)
            report["kruskal_wallis"][col] = {
                "statistic": kw.statistic, "p_value": kw.p_value,
                "df": kw.df, "conclusion": kw.conclusion,
            }
            mw = pairwise_mann_whitney(df_raw, group_col=col, target_col=target_col)
            report["mann_whitney"][col] = mw.to_dict(orient="records")
        except Exception as exc:
            log.warning("KW/MW test failed for '%s': %s", col, exc)

    # 12. Spearman correlations between ordinal features and target
    log.info("12/13 -- Spearman correlations")
    spearman_cols = ordinal_cols + [target_col]
    spearman_cols_present = [c for c in spearman_cols if c in df_raw.columns]
    rho_mat, pval_mat = spearman_correlation_matrix(df_raw, spearman_cols_present)
    report["spearman_correlations"] = {
        "rho":    rho_mat.to_dict(),
        "pvalue": pval_mat.to_dict(),
    }

    # 13. Marginal effects (training set)
    log.info("13/13 -- Marginal effects")
    me = marginal_effects_at_mean(model, X_train_enc)
    report["marginal_effects"] = me.to_dict(orient="records")

    log.info("=== Econometric battery complete ===")
    return report
