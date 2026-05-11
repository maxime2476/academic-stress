"""
Feature engineering and sklearn preprocessing pipeline.

Design principles
-----------------
- The pipeline is **fit only on training data** to prevent data leakage.
- Ordinal features are binned (1-2 / 3 / 4-5) then ordinally encoded.
- Nominal features are one-hot encoded with ``drop='first'`` to avoid
  perfect multicollinearity in the design matrix.
- All transformers preserve column names for interpretability.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.config import get_config

log = logging.getLogger(__name__)


# ?? Constants ??????????????????????????????????????????????????????????????????

# Ordered bin labels for ordinal encoded Likert groups
ORDINAL_LEVELS = ["1-2", "3", "4-5"]


# ?? Custom transformer ?????????????????????????????????????????????????????????

class LikertBinner(BaseEstimator, TransformerMixin):
    """Bin integer Likert (1-5) values into three ordered groups.

    Groups
    ------
    1-2  -> low
    3    -> medium
    4-5  -> high

    Parameters
    ----------
    bin_edges  : cut edges passed to ``pd.cut`` (right-inclusive).
    bin_labels : string labels corresponding to each interval.
    """

    def __init__(
        self,
        bin_edges: list[float] | None = None,
        bin_labels: list[str] | None = None,
    ) -> None:
        cfg = get_config()["preprocessing"]
        self.bin_edges = bin_edges or cfg["bin_edges"]
        self.bin_labels = bin_labels or cfg["bin_labels"]

    def fit(self, X: pd.DataFrame, y: Any = None) -> "LikertBinner":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in X.columns:
            X[col] = pd.cut(
                X[col].astype(float),
                bins=self.bin_edges,
                labels=self.bin_labels,
                right=True,
            ).astype(str)
        return X

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        return list(input_features) if input_features is not None else []


# ?? Pipeline factory ??????????????????????????????????????????????????????????

def build_preprocessing_pipeline(
    ordinal_cols: list[str] | None = None,
    nominal_cols: list[str] | None = None,
) -> Pipeline:
    """Build a column-transformer pipeline for the academic stress dataset.

    The pipeline applies:
      1. ``LikertBinner``  - groups 1-5 ratings into three labelled bands
      2. ``OrdinalEncoder`` - encodes the ordered string labels as integers (0,1,2)
      3. ``OneHotEncoder``  - encodes nominal categoricals (drop first category)

    The pipeline is intentionally *not* yet fitted here; call ``.fit()`` only
    on training data.

    Returns
    -------
    sklearn Pipeline with a single ``ColumnTransformer`` step named
    ``"feature_engineering"``.
    """
    from sklearn.compose import ColumnTransformer

    cfg = get_config()
    ordinal_cols = ordinal_cols or cfg["columns"]["ordinal"]
    nominal_cols = nominal_cols or cfg["columns"]["nominal"]

    ordinal_categories = [ORDINAL_LEVELS] * len(ordinal_cols)

    ordinal_pipe = Pipeline(
        steps=[
            ("binner", LikertBinner()),
            (
                "encoder",
                OrdinalEncoder(
                    categories=ordinal_categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    nominal_pipe = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(
                    drop="first",
                    sparse_output=False,
                    handle_unknown="ignore",
                ),
            )
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("ordinal", ordinal_pipe, ordinal_cols),
            ("nominal", nominal_pipe, nominal_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[("feature_engineering", transformer)])
    return pipeline


# ?? Feature / target extraction ??????????????????????????????????????????????

def extract_features_target(
    df: pd.DataFrame,
    cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into X (features) and y (binary target).

    The target is binarised at the ``high_stress_cutoff`` defined in config:
    ``y = 1`` if stress index >= cutoff, else ``0``.
    """
    if cfg is None:
        cfg = get_config()

    target_col = cfg["data"]["target_column"]
    cutoff = cfg["target"]["high_stress_cutoff"]

    ordinal_cols = cfg["columns"]["ordinal"]
    nominal_cols = cfg["columns"]["nominal"]
    feature_cols = ordinal_cols + nominal_cols

    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from DataFrame: {missing}")

    X = df[feature_cols].copy()
    y = (df[target_col] >= cutoff).astype(int).rename("HighStress")

    log.info(
        "Target distribution -- HighStress=1: %d (%.1f%%), HighStress=0: %d (%.1f%%)",
        y.sum(), 100 * y.mean(),
        (y == 0).sum(), 100 * (1 - y.mean()),
    )
    return X, y


def get_feature_names_after_fit(pipeline: Pipeline) -> list[str]:
    """Extract column names from a fitted pipeline.

    Works with the ColumnTransformer + Pipeline structure built by
    ``build_preprocessing_pipeline``.
    """
    ct = pipeline.named_steps["feature_engineering"]
    cfg = get_config()
    ordinal_cols = cfg["columns"]["ordinal"]
    nominal_cols = cfg["columns"]["nominal"]

    # Ordinal: names unchanged after binner + ordinal encoder
    ordinal_names = ordinal_cols

    # Nominal: OneHotEncoder generates "<col>_<value>" names (drop='first')
    ohe: OneHotEncoder = ct.named_transformers_["nominal"].named_steps["encoder"]
    try:
        nominal_names = ohe.get_feature_names_out(nominal_cols).tolist()
    except AttributeError:
        nominal_names = ohe.get_feature_names(nominal_cols).tolist()

    return ordinal_names + nominal_names


def transform_to_dataframe(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply fitted pipeline and return a named DataFrame."""
    X_arr = pipeline.transform(X)
    names = get_feature_names_after_fit(pipeline)
    return pd.DataFrame(X_arr, columns=names, index=X.index)
