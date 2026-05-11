"""
Tests for src.data_loader — loading, validation, and cleaning.
"""

from __future__ import annotations

import textwrap
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data_loader import (
    ValidationReport,
    _check_categories,
    _check_likert_ranges,
    _check_schema,
    _count_missing,
    _drop_missing_rows,
    _parse_timestamp,
    get_data_summary,
    load_and_clean_data,
    LIKERT_COLUMNS,
    VALID_LIKERT_RANGE,
)
from tests.conftest import (
    COL_TARGET, COL_TIMESTAMP, COL_PEER, COL_FAMILY, COL_COMP,
    COL_STAGE, COL_ENV, COL_COPING, COL_HABITS, make_raw_df,
)


# ── load_and_clean_data ────────────────────────────────────────────────────────

class TestLoadAndClean:

    def test_loads_real_csv(self, tmp_path):
        """Writing the synthetic dataset to disk and loading it succeeds."""
        df_orig = make_raw_df(n=50, seed=0)
        csv_path = tmp_path / "test.csv"
        df_orig.to_csv(csv_path, index=False)

        df, report = load_and_clean_data(csv_path)

        assert len(df) <= 50
        assert isinstance(df, pd.DataFrame)
        assert isinstance(report, ValidationReport)

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_and_clean_data(tmp_path / "nonexistent.csv")

    def test_index_is_reset_after_load(self, tmp_path):
        df_orig = make_raw_df(n=30, missing_frac=0.1, seed=1)
        csv_path = tmp_path / "data.csv"
        df_orig.to_csv(csv_path, index=False)

        df, _ = load_and_clean_data(csv_path)
        assert list(df.index) == list(range(len(df)))

    def test_report_counts_are_consistent(self, tmp_path):
        df_orig = make_raw_df(n=40, missing_frac=0.1, seed=2)
        csv_path = tmp_path / "data.csv"
        df_orig.to_csv(csv_path, index=False)

        df, report = load_and_clean_data(csv_path)
        assert report.n_rows_raw == 40
        assert report.n_rows_after_drop == len(df)
        assert report.n_rows_raw >= report.n_rows_after_drop


# ── _count_missing ─────────────────────────────────────────────────────────────

class TestCountMissing:

    def test_no_missing(self, raw_df):
        result = _count_missing(raw_df)
        assert result == {}

    def test_detects_missing(self, raw_df_with_missing):
        result = _count_missing(raw_df_with_missing)
        assert isinstance(result, dict)
        assert all(v > 0 for v in result.values())

    def test_only_reports_columns_with_missing(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        result = _count_missing(df)
        assert "a" in result
        assert "b" not in result
        assert result["a"] == 1


# ── _check_schema ──────────────────────────────────────────────────────────────

class TestCheckSchema:

    def test_no_warnings_on_complete_schema(self, raw_df):
        report = ValidationReport()
        _check_schema(raw_df, report)
        assert report.warnings == []

    def test_warns_on_missing_column(self):
        df = pd.DataFrame({"col_a": [1]})
        report = ValidationReport()
        _check_schema(df, report)
        assert any("Missing expected columns" in w for w in report.warnings)


# ── _check_likert_ranges ───────────────────────────────────────────────────────

class TestCheckLikertRanges:

    def test_valid_likert_passes(self, raw_df):
        report = ValidationReport()
        _check_likert_ranges(raw_df, report)
        assert report.out_of_range_per_column == {}

    def test_detects_out_of_range_low(self, raw_df):
        df = raw_df.copy()
        df.loc[0, COL_PEER] = 0   # below valid range
        report = ValidationReport()
        _check_likert_ranges(df, report)
        assert COL_PEER in report.out_of_range_per_column
        assert report.out_of_range_per_column[COL_PEER] == 1

    def test_detects_out_of_range_high(self, raw_df):
        df = raw_df.copy()
        df.loc[5, COL_COMP] = 6   # above valid range
        report = ValidationReport()
        _check_likert_ranges(df, report)
        assert COL_COMP in report.out_of_range_per_column

    def test_valid_boundary_values_accepted(self, raw_df):
        df = raw_df.copy()
        df.loc[0, COL_PEER] = 1
        df.loc[1, COL_PEER] = 5
        report = ValidationReport()
        _check_likert_ranges(df, report)
        assert COL_PEER not in report.out_of_range_per_column


# ── _check_categories ─────────────────────────────────────────────────────────

class TestCheckCategories:

    def test_valid_categories_pass(self, raw_df):
        report = ValidationReport()
        _check_categories(raw_df, report)
        assert report.unexpected_categories == {}

    def test_detects_unexpected_category(self, raw_df):
        df = raw_df.copy()
        df.loc[0, COL_STAGE] = "unknown_stage"
        report = ValidationReport()
        _check_categories(df, report)
        assert COL_STAGE in report.unexpected_categories
        assert "unknown_stage" in report.unexpected_categories[COL_STAGE]


# ── _drop_missing_rows ────────────────────────────────────────────────────────

class TestDropMissingRows:

    def test_no_rows_dropped_when_complete(self, raw_df):
        report = ValidationReport()
        df_clean = _drop_missing_rows(raw_df, report)
        assert len(df_clean) == len(raw_df)

    def test_drops_rows_with_missing_feature(self, raw_df):
        df = raw_df.copy()
        df.loc[0, COL_PEER] = np.nan
        df.loc[1, COL_ENV]  = np.nan
        report = ValidationReport()
        df_clean = _drop_missing_rows(df, report)
        assert len(df_clean) <= len(df) - 2

    def test_does_not_drop_rows_with_only_timestamp_missing(self, raw_df):
        df = raw_df.copy()
        df.loc[0, COL_TIMESTAMP] = np.nan
        report = ValidationReport()
        df_clean = _drop_missing_rows(df, report)
        assert len(df_clean) == len(raw_df)


# ── _parse_timestamp ───────────────────────────────────────────────────────────

class TestParseTimestamp:

    def test_parses_day_first_timestamps(self, raw_df):
        report = ValidationReport()
        df_parsed = _parse_timestamp(raw_df, report)
        assert pd.api.types.is_datetime64_any_dtype(df_parsed[COL_TIMESTAMP])

    def test_warns_on_unparseable_timestamp(self, raw_df):
        df = raw_df.copy()
        df.loc[0, COL_TIMESTAMP] = "not_a_date"
        report = ValidationReport()
        df_parsed = _parse_timestamp(df, report)
        assert df_parsed.loc[0, COL_TIMESTAMP] is pd.NaT

    def test_no_warning_when_all_valid(self, raw_df):
        report = ValidationReport()
        _parse_timestamp(raw_df, report)
        timestamp_warnings = [w for w in report.warnings if "Timestamp" in w]
        assert len(timestamp_warnings) == 0


# ── get_data_summary ──────────────────────────────────────────────────────────

class TestGetDataSummary:

    def test_summary_keys_present(self, clean_df):
        summary = get_data_summary(clean_df)
        for key in ("n_observations", "n_features", "numeric_features",
                    "categorical_features", "numeric_stats", "categorical_distributions"):
            assert key in summary

    def test_n_observations_matches(self, clean_df):
        summary = get_data_summary(clean_df)
        assert summary["n_observations"] == len(clean_df)

    def test_categorical_distributions_non_empty(self, clean_df):
        summary = get_data_summary(clean_df)
        for col, dist in summary["categorical_distributions"].items():
            assert len(dist) > 0


# ── ValidationReport ─────────────────────────────────────────────────────────

class TestValidationReport:

    def test_is_clean_true_when_no_issues(self):
        r = ValidationReport()
        assert r.is_clean() is True

    def test_is_clean_false_when_out_of_range(self):
        r = ValidationReport()
        r.out_of_range_per_column["x"] = 1
        assert r.is_clean() is False

    def test_is_clean_false_when_unexpected_categories(self):
        r = ValidationReport()
        r.unexpected_categories["y"] = ["bad_val"]
        assert r.is_clean() is False

    def test_summary_contains_row_counts(self):
        r = ValidationReport(n_rows_raw=140, n_rows_after_drop=139)
        summary = r.summary()
        assert "140" in summary
        assert "139" in summary
