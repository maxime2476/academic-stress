"""
Shared pytest fixtures for the academic stress project test suite.

All fixtures are scope='function' by default (fresh per test) unless noted.
The synthetic datasets created here intentionally mirror the real dataset's
schema so that every test is completely independent of the CSV file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# ── Ensure project root is importable ─────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import reset_config


# ── Column name constants (mirror the real CSV) ────────────────────────────────
COL_PEER       = "Peer pressure"
COL_FAMILY     = "Academic pressure from your home"
COL_COMP       = "What would you rate the academic  competition in your student life"
COL_STAGE      = "Your Academic Stage"
COL_ENV        = "Study Environment"
COL_COPING     = "What coping strategy you use as a student?"
COL_HABITS     = "Do you have any bad habits like smoking or drinking on a daily basis?"
COL_TARGET     = "Rate your academic stress index"
COL_TIMESTAMP  = "Timestamp"

ORDINAL_COLS  = [COL_PEER, COL_FAMILY, COL_COMP]
NOMINAL_COLS  = [COL_STAGE, COL_ENV, COL_COPING, COL_HABITS]
ALL_FEAT_COLS = ORDINAL_COLS + NOMINAL_COLS

RNG = np.random.default_rng(42)


# ── Synthetic dataset factory ──────────────────────────────────────────────────

def make_raw_df(n: int = 100, seed: int = 42, missing_frac: float = 0.0) -> pd.DataFrame:
    """Generate a synthetic DataFrame that matches the real dataset schema."""
    rng = np.random.default_rng(seed)
    stages   = rng.choice(["undergraduate", "high school", "post-graduate"], n)
    envs     = rng.choice(["Peaceful", "Noisy", "disrupted"], n)
    copings  = rng.choice(
        [
            "Analyze the situation and handle it with intellect",
            "Social support (friends or family)",
            "Emotional breakdown (crying a lot)",
        ],
        n,
    )
    habits   = rng.choice(["No", "Yes", "prefer not to say"], n)
    peer     = rng.integers(1, 6, size=n)
    family   = rng.integers(1, 6, size=n)
    comp     = rng.integers(1, 6, size=n)
    stress   = rng.integers(1, 6, size=n)
    ts       = pd.date_range("2025-07-24", periods=n, freq="h")

    df = pd.DataFrame({
        COL_TIMESTAMP: ts.strftime("%d/%m/%Y %H:%M:%S"),
        COL_STAGE:    stages,
        COL_PEER:     peer,
        COL_FAMILY:   family,
        COL_ENV:      envs,
        COL_COPING:   copings,
        COL_HABITS:   habits,
        COL_COMP:     comp,
        COL_TARGET:   stress,
    })

    if missing_frac > 0:
        n_missing = max(1, int(n * missing_frac))
        miss_rows = rng.choice(n, size=n_missing, replace=False)
        miss_cols = rng.choice(ALL_FEAT_COLS, size=n_missing)
        for row, col in zip(miss_rows, miss_cols):
            df.loc[row, col] = np.nan

    return df


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def raw_df() -> pd.DataFrame:
    """100-row synthetic raw DataFrame (no missing values)."""
    return make_raw_df(n=100, seed=42)


@pytest.fixture()
def raw_df_with_missing() -> pd.DataFrame:
    """100-row synthetic DataFrame with ~5% missing values."""
    return make_raw_df(n=100, seed=42, missing_frac=0.05)


@pytest.fixture()
def clean_df() -> pd.DataFrame:
    """Cleaned (no missing) synthetic DataFrame."""
    from src.data_loader import _drop_missing_rows, _parse_timestamp, ValidationReport
    df = make_raw_df(n=100, seed=42)
    report = ValidationReport()
    df = _drop_missing_rows(df, report)
    df = _parse_timestamp(df, report)
    return df.reset_index(drop=True)


@pytest.fixture()
def X_y(clean_df) -> tuple[pd.DataFrame, pd.Series]:
    """Feature matrix and binary target from the synthetic clean DataFrame."""
    from src.preprocessing import extract_features_target
    return extract_features_target(clean_df)


@pytest.fixture()
def fitted_pipeline(X_y):
    """Preprocessing pipeline fitted on X (full synthetic data)."""
    from src.preprocessing import build_preprocessing_pipeline, transform_to_dataframe
    X, y = X_y
    pipe = build_preprocessing_pipeline()
    pipe.fit(X)
    return pipe


@pytest.fixture()
def encoded_data(X_y):
    """Encoded feature DataFrames and split indices."""
    from src.preprocessing import build_preprocessing_pipeline, transform_to_dataframe
    from sklearn.model_selection import train_test_split
    X, y = X_y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    pipe = build_preprocessing_pipeline()
    pipe.fit(X_train)
    X_train_enc = transform_to_dataframe(pipe, X_train)
    X_test_enc  = transform_to_dataframe(pipe, X_test)
    return X_train_enc, X_test_enc, y_train, y_test


@pytest.fixture()
def fitted_model(encoded_data):
    """Simple logistic regression trained on synthetic encoded training data."""
    X_train_enc, _, y_train, _ = encoded_data
    model = LogisticRegression(
        solver="liblinear", max_iter=5000,
        class_weight="balanced", random_state=42,
    )
    model.fit(X_train_enc, y_train)
    return model


@pytest.fixture(autouse=True)
def reset_cfg():
    """Reset the cached config singleton between tests."""
    reset_config()
    yield
    reset_config()
