"""
Tests for src.preprocessing — feature engineering, binning, encoding, pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.preprocessing import (
    LikertBinner,
    build_preprocessing_pipeline,
    extract_features_target,
    get_feature_names_after_fit,
    transform_to_dataframe,
    ORDINAL_LEVELS,
)
from tests.conftest import (
    COL_PEER, COL_FAMILY, COL_COMP,
    COL_STAGE, COL_ENV, COL_COPING, COL_HABITS, COL_TARGET,
    make_raw_df,
)


# ── LikertBinner ──────────────────────────────────────────────────────────────

class TestLikertBinner:

    def _make_series_df(self, values: list[int]) -> pd.DataFrame:
        return pd.DataFrame({"x": values})

    def test_bins_low_values(self):
        binner = LikertBinner()
        df = self._make_series_df([1, 2])
        result = binner.transform(df)
        assert all(result["x"] == "1-2")

    def test_bins_medium_values(self):
        binner = LikertBinner()
        df = self._make_series_df([3])
        result = binner.transform(df)
        assert result["x"].iloc[0] == "3"

    def test_bins_high_values(self):
        binner = LikertBinner()
        df = self._make_series_df([4, 5])
        result = binner.transform(df)
        assert all(result["x"] == "4-5")

    def test_all_three_bins_present(self):
        binner = LikertBinner()
        df = self._make_series_df([1, 3, 5])
        result = binner.transform(df)
        assert set(result["x"].tolist()) == {"1-2", "3", "4-5"}

    def test_fit_returns_self(self):
        binner = LikertBinner()
        df = self._make_series_df([1, 2, 3])
        returned = binner.fit(df)
        assert returned is binner

    def test_does_not_mutate_input(self):
        binner = LikertBinner()
        df = self._make_series_df([2, 4])
        original = df.copy()
        binner.transform(df)
        pd.testing.assert_frame_equal(df, original)

    def test_handles_float_inputs(self):
        binner = LikertBinner()
        df = pd.DataFrame({"x": [1.0, 3.0, 5.0]})
        result = binner.transform(df)
        assert set(result["x"].tolist()) == {"1-2", "3", "4-5"}


# ── build_preprocessing_pipeline ─────────────────────────────────────────────

class TestBuildPreprocessingPipeline:

    def test_returns_sklearn_pipeline(self):
        pipe = build_preprocessing_pipeline()
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_feature_engineering_step(self):
        pipe = build_preprocessing_pipeline()
        assert "feature_engineering" in pipe.named_steps

    def test_pipeline_fits_without_error(self, X_y):
        X, y = X_y
        pipe = build_preprocessing_pipeline()
        pipe.fit(X)   # should not raise

    def test_pipeline_transforms_correct_shape(self, X_y):
        X, y = X_y
        pipe = build_preprocessing_pipeline()
        pipe.fit(X)
        X_enc = pipe.transform(X)
        assert X_enc.shape[0] == len(X)
        assert X_enc.shape[1] > 0

    def test_no_nan_in_transformed_output(self, X_y):
        X, y = X_y
        pipe = build_preprocessing_pipeline()
        pipe.fit(X)
        X_enc = pipe.transform(X)
        assert not np.isnan(X_enc).any()

    def test_pipeline_refuses_leakage(self, X_y):
        """Fit only on train; transform train and test separately."""
        from sklearn.model_selection import train_test_split
        X, y = X_y
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)

        pipe = build_preprocessing_pipeline()
        pipe.fit(X_train)

        enc_train = pipe.transform(X_train)
        enc_test  = pipe.transform(X_test)

        assert enc_train.shape[1] == enc_test.shape[1]


# ── extract_features_target ───────────────────────────────────────────────────

class TestExtractFeaturesTarget:

    def test_returns_correct_shapes(self, clean_df):
        X, y = extract_features_target(clean_df)
        assert X.shape[0] == len(clean_df)
        assert len(y) == len(clean_df)

    def test_target_is_binary(self, clean_df):
        _, y = extract_features_target(clean_df)
        assert set(y.unique()).issubset({0, 1})

    def test_target_name_is_highstress(self, clean_df):
        _, y = extract_features_target(clean_df)
        assert y.name == "HighStress"

    def test_feature_columns_match_expected(self, clean_df):
        X, _ = extract_features_target(clean_df)
        from src.config import get_config
        cfg = get_config()
        expected = set(cfg["columns"]["ordinal"] + cfg["columns"]["nominal"])
        assert set(X.columns) == expected

    def test_target_cutoff_correct(self, clean_df):
        from src.config import get_config
        cutoff = get_config()["target"]["high_stress_cutoff"]
        X, y = extract_features_target(clean_df)
        raw_stress = clean_df[COL_TARGET]
        expected_y = (raw_stress >= cutoff).astype(int)
        pd.testing.assert_series_equal(
            y.reset_index(drop=True),
            expected_y.reset_index(drop=True).rename("HighStress"),
        )

    def test_raises_on_missing_column(self, clean_df):
        df = clean_df.drop(columns=[COL_TARGET])
        with pytest.raises(ValueError, match="Columns missing"):
            extract_features_target(df)


# ── get_feature_names_after_fit ───────────────────────────────────────────────

class TestGetFeatureNamesAfterFit:

    def test_returns_list_of_strings(self, X_y):
        X, _ = X_y
        pipe = build_preprocessing_pipeline()
        pipe.fit(X)
        names = get_feature_names_after_fit(pipe)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_names_match_transformed_columns(self, X_y):
        X, _ = X_y
        pipe = build_preprocessing_pipeline()
        pipe.fit(X)
        X_df = transform_to_dataframe(pipe, X)
        names = get_feature_names_after_fit(pipe)
        assert names == X_df.columns.tolist()

    def test_ordinal_names_unchanged(self, X_y):
        from src.config import get_config
        X, _ = X_y
        pipe = build_preprocessing_pipeline()
        pipe.fit(X)
        names = get_feature_names_after_fit(pipe)
        ordinal_cols = get_config()["columns"]["ordinal"]
        for col in ordinal_cols:
            assert col in names


# ── transform_to_dataframe ────────────────────────────────────────────────────

class TestTransformToDataFrame:

    def test_returns_dataframe(self, X_y, fitted_pipeline):
        X, _ = X_y
        result = transform_to_dataframe(fitted_pipeline, X)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_row_index(self, X_y, fitted_pipeline):
        X, _ = X_y
        X_indexed = X.reset_index(drop=False)   # keeps old index as column
        X_sub = X.iloc[10:30]
        result = transform_to_dataframe(fitted_pipeline, X_sub)
        assert len(result) == 20

    def test_columns_consistent_train_test(self, X_y):
        from sklearn.model_selection import train_test_split
        X, y = X_y
        X_tr, X_te = train_test_split(X, test_size=0.3, random_state=0)
        pipe = build_preprocessing_pipeline()
        pipe.fit(X_tr)
        tr_enc = transform_to_dataframe(pipe, X_tr)
        te_enc = transform_to_dataframe(pipe, X_te)
        assert tr_enc.columns.tolist() == te_enc.columns.tolist()
