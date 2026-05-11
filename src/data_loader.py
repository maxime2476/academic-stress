"""
Data loading, validation and cleaning.

Entry point
-----------
``load_and_clean_data(path)`` returns a validated, cleaned DataFrame ready for
preprocessing.  Intermediate helpers are public so unit tests can target them
individually.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ?? Expected schema ????????????????????????????????????????????????????????????
EXPECTED_COLUMNS: list[str] = [
    "Timestamp",
    "Your Academic Stage",
    "Peer pressure",
    "Academic pressure from your home",
    "Study Environment",
    "What coping strategy you use as a student?",
    "Do you have any bad habits like smoking or drinking on a daily basis?",
    "What would you rate the academic  competition in your student life",
    "Rate your academic stress index",
]

LIKERT_COLUMNS: list[str] = [
    "Peer pressure",
    "Academic pressure from your home",
    "What would you rate the academic  competition in your student life",
    "Rate your academic stress index",
]

VALID_LIKERT_RANGE = (1, 5)

VALID_CATEGORIES: dict[str, set[str]] = {
    "Your Academic Stage": {"undergraduate", "high school", "post-graduate"},
    "Study Environment": {"Peaceful", "Noisy", "disrupted"},
    "What coping strategy you use as a student?": {
        "Analyze the situation and handle it with intellect",
        "Social support (friends or family)",
        "Emotional breakdown (crying a lot)",
    },
    "Do you have any bad habits like smoking or drinking on a daily basis?": {
        "No",
        "Yes",
        "prefer not to say",
    },
}


@dataclass
class ValidationReport:
    """Summary of data quality checks performed during loading."""

    n_rows_raw: int = 0
    n_rows_after_drop: int = 0
    missing_per_column: dict[str, int] = field(default_factory=dict)
    out_of_range_per_column: dict[str, int] = field(default_factory=dict)
    unexpected_categories: dict[str, list[str]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def is_clean(self) -> bool:
        return (
            sum(self.out_of_range_per_column.values()) == 0
            and all(len(v) == 0 for v in self.unexpected_categories.values())
        )

    def summary(self) -> str:
        lines = [
            f"Rows raw          : {self.n_rows_raw}",
            f"Rows after drop   : {self.n_rows_after_drop}",
            f"Dropped (missing) : {self.n_rows_raw - self.n_rows_after_drop}",
            "Missing per column:",
        ]
        for col, n in self.missing_per_column.items():
            lines.append(f"  {col}: {n}")
        if self.out_of_range_per_column:
            lines.append("Out-of-range Likert values:")
            for col, n in self.out_of_range_per_column.items():
                lines.append(f"  {col}: {n}")
        if self.unexpected_categories:
            lines.append("Unexpected category values:")
            for col, vals in self.unexpected_categories.items():
                lines.append(f"  {col}: {vals}")
        for w in self.warnings:
            lines.append(f"[WARNING] {w}")
        return "\n".join(lines)


# ?? Public API ?????????????????????????????????????????????????????????????????

def load_and_clean_data(
    path: str | Path,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, ValidationReport]:
    """Load CSV, validate schema, drop missing rows, return clean DataFrame.

    Returns
    -------
    df      : cleaned DataFrame (index reset)
    report  : ValidationReport describing what was found and dropped
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    log.info("Loading data from %s", path)
    df_raw = pd.read_csv(path, encoding=encoding)
    # Strip leading/trailing whitespace from column names to handle CSV inconsistencies
    df_raw.columns = [c.strip() for c in df_raw.columns]
    log.info("Raw shape: %s", df_raw.shape)

    report = ValidationReport(n_rows_raw=len(df_raw))
    report.missing_per_column = _count_missing(df_raw)

    _check_schema(df_raw, report)
    _check_likert_ranges(df_raw, report)
    _check_categories(df_raw, report)

    df_clean = _drop_missing_rows(df_raw, report)
    df_clean = _parse_timestamp(df_clean, report)
    df_clean = df_clean.reset_index(drop=True)

    report.n_rows_after_drop = len(df_clean)
    log.info("Clean shape: %s | Report: %s", df_clean.shape, report.is_clean())
    return df_clean, report


def get_data_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Return a descriptive summary dict suitable for JSON export."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    summary: dict[str, Any] = {
        "n_observations": len(df),
        "n_features": df.shape[1],
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "numeric_stats": df[num_cols].describe().round(3).to_dict(),
        "categorical_distributions": {
            col: df[col].value_counts().to_dict() for col in cat_cols
        },
    }
    return summary


# ?? Private helpers ?????????????????????????????????????????????????????????????

def _count_missing(df: pd.DataFrame) -> dict[str, int]:
    counts = df.isnull().sum()
    return {col: int(n) for col, n in counts.items() if n > 0}


def _check_schema(df: pd.DataFrame, report: ValidationReport) -> None:
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        msg = f"Missing expected columns: {missing_cols}"
        report.warnings.append(msg)
        log.warning(msg)

    extra_cols = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra_cols:
        log.info("Extra columns (ignored): %s", extra_cols)


def _check_likert_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    lo, hi = VALID_LIKERT_RANGE
    for col in LIKERT_COLUMNS:
        if col not in df.columns:
            continue
        out = df[col].dropna()
        bad = int(((out < lo) | (out > hi)).sum())
        if bad:
            report.out_of_range_per_column[col] = bad
            log.warning("Column '%s' has %d values outside [%d, %d]", col, bad, lo, hi)


def _check_categories(df: pd.DataFrame, report: ValidationReport) -> None:
    for col, valid_set in VALID_CATEGORIES.items():
        if col not in df.columns:
            continue
        actual = set(df[col].dropna().unique())
        unexpected = sorted(actual - valid_set)
        if unexpected:
            report.unexpected_categories[col] = unexpected
            log.warning("Column '%s' has unexpected values: %s", col, unexpected)


def _drop_missing_rows(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    cols_to_check = [c for c in EXPECTED_COLUMNS if c != "Timestamp" and c in df.columns]
    before = len(df)
    df_clean = df.dropna(subset=cols_to_check)
    dropped = before - len(df_clean)
    if dropped:
        log.info("Dropped %d row(s) with missing values in analysis columns", dropped)
    return df_clean


def _parse_timestamp(df: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    if "Timestamp" not in df.columns:
        return df
    try:
        df = df.copy()
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True, errors="coerce")
        n_bad = int(df["Timestamp"].isnull().sum())
        if n_bad:
            report.warnings.append(f"{n_bad} Timestamp(s) could not be parsed.")
    except Exception as exc:  # noqa: BLE001
        report.warnings.append(f"Timestamp parse error: {exc}")
    return df
