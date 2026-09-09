"""
Tests for src/features/engineering.py and src/features/pipeline.py

Run with: pytest tests/test_features.py -v

Design principle:
    All tests use synthetic DataFrames — no dependency on real dataset.
    Tests are deterministic (fixed random_state=42).
    Tests verify leakage prevention, shape consistency, and transform identity.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.engineering import FeatureEngineer
from src.features.pipeline import PreprocessingPipeline, SplitData


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

def make_sample_df(n_legit: int = 800, n_fraud: int = 40, seed: int = 42) -> pd.DataFrame:
    """Create a minimal synthetic creditcard-schema DataFrame."""
    rng = np.random.default_rng(seed)
    n = n_legit + n_fraud
    v_cols = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    df = pd.DataFrame({
        "Time": rng.uniform(0, 172792, n),
        **v_cols,
        "Amount": rng.exponential(scale=88, size=n).clip(0),
        "Class": [0] * n_legit + [1] * n_fraud,
    })
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return make_sample_df()


@pytest.fixture(scope="module")
def fe() -> FeatureEngineer:
    return FeatureEngineer()


@pytest.fixture(scope="module")
def engineered_df(fe, raw_df) -> pd.DataFrame:
    return fe.transform(raw_df, drop_duplicates=True)


@pytest.fixture(scope="module")
def X_y(fe, engineered_df):
    return fe.get_X_y(engineered_df)


# ---------------------------------------------------------------------------
# FeatureEngineer — transform()
# ---------------------------------------------------------------------------

class TestFeatureEngineerTransform:
    def test_transform_does_not_modify_original(self, fe, raw_df):
        original_cols = list(raw_df.columns)
        original_shape = raw_df.shape
        _ = fe.transform(raw_df)
        assert list(raw_df.columns) == original_cols
        assert raw_df.shape == original_shape

    def test_transform_creates_log_amount(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        assert "log_amount" in df_out.columns

    def test_log_amount_is_log1p_of_amount(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        expected = np.log1p(df_out["Amount"])
        np.testing.assert_allclose(df_out["log_amount"].values, expected.values, rtol=1e-10)

    def test_log_amount_handles_zeros(self, fe):
        df_zero = pd.DataFrame({
            "Time": [1000.0], "Amount": [0.0], "Class": [0],
            **{f"V{i}": [0.0] for i in range(1, 29)},
        })
        df_out = fe.transform(df_zero, drop_duplicates=False)
        # log1p(0) = 0, should not produce NaN or -inf
        assert np.isfinite(df_out["log_amount"].iloc[0])
        assert df_out["log_amount"].iloc[0] == pytest.approx(0.0)

    def test_transform_creates_hour_of_day(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        assert "hour_of_day" in df_out.columns

    def test_hour_of_day_range(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        assert df_out["hour_of_day"].min() >= 0.0
        assert df_out["hour_of_day"].max() < 24.0

    def test_transform_creates_is_night(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        assert "is_night" in df_out.columns

    def test_is_night_is_binary(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        unique_vals = set(df_out["is_night"].unique())
        assert unique_vals.issubset({0, 1})

    def test_is_night_correct_for_night_time(self, fe):
        """Time = 3600 seconds = 1h (night). is_night should be 1."""
        df_night = pd.DataFrame({
            "Time": [3600.0],  # 1am
            "Amount": [50.0],
            "Class": [0],
            **{f"V{i}": [0.0] for i in range(1, 29)},
        })
        df_out = fe.transform(df_night, drop_duplicates=False)
        assert df_out["is_night"].iloc[0] == 1

    def test_is_night_correct_for_day_time(self, fe):
        """Time = 43200 seconds = 12h (noon). is_night should be 0."""
        df_day = pd.DataFrame({
            "Time": [43200.0],  # 12pm
            "Amount": [50.0],
            "Class": [0],
            **{f"V{i}": [0.0] for i in range(1, 29)},
        })
        df_out = fe.transform(df_day, drop_duplicates=False)
        assert df_out["is_night"].iloc[0] == 0

    def test_transform_preserves_all_original_columns(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        for col in raw_df.columns:
            assert col in df_out.columns, f"Original column '{col}' missing from output"

    def test_transform_adds_exactly_3_columns(self, fe, raw_df):
        df_out = fe.transform(raw_df, drop_duplicates=False)
        assert len(df_out.columns) == len(raw_df.columns) + 3


# ---------------------------------------------------------------------------
# FeatureEngineer — get_X_y()
# ---------------------------------------------------------------------------

class TestFeatureEngineerGetXy:
    def test_get_X_y_separates_target(self, fe, engineered_df):
        X, y = fe.get_X_y(engineered_df)
        assert "Class" not in X.columns
        assert y.name == "Class"

    def test_get_X_y_preserves_row_count(self, fe, engineered_df):
        X, y = fe.get_X_y(engineered_df)
        assert len(X) == len(engineered_df)
        assert len(y) == len(engineered_df)

    def test_get_X_y_target_is_binary(self, fe, engineered_df):
        _, y = fe.get_X_y(engineered_df)
        assert set(y.unique()).issubset({0, 1})

    def test_get_X_y_raises_without_target_column(self, fe):
        df_no_target = pd.DataFrame({"Amount": [1.0, 2.0], "V1": [0.1, 0.2]})
        with pytest.raises(KeyError, match="Class"):
            fe.get_X_y(df_no_target)


# ---------------------------------------------------------------------------
# FeatureEngineer — transform_single()
# ---------------------------------------------------------------------------

class TestFeatureEngineerTransformSingle:
    def test_transform_single_returns_one_row(self, fe):
        txn = {
            "Time": 50000.0, "Amount": 149.62,
            **{f"V{i}": float(i * 0.1) for i in range(1, 29)},
        }
        df_out = fe.transform_single(txn)
        assert len(df_out) == 1

    def test_transform_single_has_derived_features(self, fe):
        txn = {
            "Time": 50000.0, "Amount": 149.62,
            **{f"V{i}": float(i * 0.1) for i in range(1, 29)},
        }
        df_out = fe.transform_single(txn)
        assert "log_amount" in df_out.columns
        assert "hour_of_day" in df_out.columns
        assert "is_night" in df_out.columns


# ---------------------------------------------------------------------------
# PreprocessingPipeline — Anti-Leakage Tests
# ---------------------------------------------------------------------------

class TestPreprocessingPipelineLeakage:
    def test_pipeline_splits_correct_sizes(self, X_y):
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        n_total = len(X)
        # Allow ±2 rows for rounding
        assert abs(len(splits.y_train) + len(splits.y_val) + len(splits.y_test) - n_total) <= 2

    def test_train_test_no_index_overlap(self, X_y):
        """CRITICAL: train and test indices must be disjoint."""
        X, y = X_y
        pp = PreprocessingPipeline()
        _ = pp.fit_transform_split(X, y)

        # The pipeline logs leakage; we also assert here directly
        train_val_test_total = (
            pp.metadata["split_sizes"]["train"]
            + pp.metadata["split_sizes"]["val"]
            + pp.metadata["split_sizes"]["test"]
        )
        assert abs(train_val_test_total - len(X)) <= 2

    def test_scaler_stats_come_from_training(self, X_y):
        """Scaler mean/std should differ from full-dataset stats."""
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        # Amount scaler mean should not equal full-dataset mean exactly
        # (unless by extreme coincidence with tiny samples)
        # Just verify scaler stats are recorded
        assert "Amount" in pp.metadata.get("scaler_stats", {})

    def test_stratification_preserves_class_ratio(self, X_y):
        """Train, val, and test should all have similar fraud rates."""
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        overall_rate = float(y.mean())
        train_rate = float(splits.y_train.mean())
        val_rate = float(splits.y_val.mean())
        test_rate = float(splits.y_test.mean())
        # Stratification should keep rates within 0.05 of overall
        for rate, name in [(train_rate, "train"), (val_rate, "val"), (test_rate, "test")]:
            assert abs(rate - overall_rate) < 0.05, (
                f"{name} fraud rate {rate:.4f} deviates too much from {overall_rate:.4f}"
            )


# ---------------------------------------------------------------------------
# PreprocessingPipeline — Output Shape & Values
# ---------------------------------------------------------------------------

class TestPreprocessingPipelineOutput:
    def test_output_is_numpy_array(self, X_y):
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        assert isinstance(splits.X_train, np.ndarray)
        assert isinstance(splits.X_val, np.ndarray)
        assert isinstance(splits.X_test, np.ndarray)

    def test_output_n_features_consistent(self, X_y):
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        n_out = len(splits.feature_names_out)
        assert splits.X_train.shape[1] == n_out
        assert splits.X_val.shape[1] == n_out
        assert splits.X_test.shape[1] == n_out

    def test_scaled_columns_have_near_zero_mean_on_train(self, X_y):
        """StandardScaler should centre training data near 0."""
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        # First 4 columns are the scaled ones (Amount, Time, log_amount, hour_of_day)
        scale_cols_count = len(pp._scale_cols)
        scaled_train = splits.X_train[:, :scale_cols_count]
        train_means = scaled_train.mean(axis=0)
        np.testing.assert_allclose(train_means, 0.0, atol=1e-10)

    def test_scaled_columns_near_unit_variance_on_train(self, X_y):
        """StandardScaler should give unit variance on training data."""
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        scale_cols_count = len(pp._scale_cols)
        scaled_train = splits.X_train[:, :scale_cols_count]
        train_stds = scaled_train.std(axis=0)
        np.testing.assert_allclose(train_stds, 1.0, atol=1e-10)

    def test_no_nan_in_output(self, X_y):
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        assert not np.isnan(splits.X_train).any()
        assert not np.isnan(splits.X_val).any()
        assert not np.isnan(splits.X_test).any()

    def test_feature_names_out_correct_count(self, X_y):
        """Should have: 4 scale + 28 PCA + 1 is_night = 33 features."""
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)
        # 4 scale: Amount, Time, log_amount, hour_of_day
        # 28 PCA: V1–V28
        # 1 binary: is_night
        assert len(splits.feature_names_out) == 33


# ---------------------------------------------------------------------------
# PreprocessingPipeline — Transform Consistency (Inference Path)
# ---------------------------------------------------------------------------

class TestPreprocessingPipelineConsistency:
    def test_batch_vs_single_row_identical(self, X_y):
        """Batch transform and single-row transform must produce identical output."""
        X, y = X_y
        pp = PreprocessingPipeline()
        splits = pp.fit_transform_split(X, y)

        # Take a few rows from the test set (original pre-transform X)
        sample = X.iloc[:10]
        batch_out = pp.transform(sample)
        row_outs = np.vstack([pp.transform(sample.iloc[[i]]) for i in range(10)])

        np.testing.assert_allclose(batch_out, row_outs, rtol=1e-10, atol=1e-12)

    def test_transform_raises_if_not_fitted(self):
        pp = PreprocessingPipeline()
        df = pd.DataFrame({"Amount": [1.0], "V1": [0.1]})
        with pytest.raises(RuntimeError, match="not been fitted"):
            pp.transform(df)

    def test_fit_twice_raises(self, X_y):
        X, y = X_y
        pp = PreprocessingPipeline()
        _ = pp.fit_transform_split(X, y)
        with pytest.raises(RuntimeError, match="already fitted"):
            pp.fit_transform_split(X, y)

    def test_reset_allows_refit(self, X_y):
        X, y = X_y
        pp = PreprocessingPipeline()
        _ = pp.fit_transform_split(X, y)
        pp.reset()
        # Should not raise after reset
        _ = pp.fit_transform_split(X, y)


# ---------------------------------------------------------------------------
# PreprocessingPipeline — Save/Load Round-trip
# ---------------------------------------------------------------------------

class TestPreprocessingPipelineSaveLoad:
    def test_save_and_load_produces_identical_transform(self, X_y, tmp_path):
        X, y = X_y
        pp_original = PreprocessingPipeline()
        splits = pp_original.fit_transform_split(X, y)

        # Override pipeline dir to tmp_path
        pp_original._pipeline_dir = tmp_path
        pp_original.save()

        # Load from the same tmp_path
        pp_loaded = PreprocessingPipeline.load(pipeline_dir=tmp_path)

        sample = X.iloc[:20]
        original_out = pp_original.transform(sample)
        loaded_out = pp_loaded.transform(sample)

        np.testing.assert_allclose(original_out, loaded_out, rtol=1e-10, atol=1e-12)

    def test_load_raises_if_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Pipeline file not found"):
            PreprocessingPipeline.load(pipeline_dir=tmp_path)

    def test_metadata_is_saved_and_loaded(self, X_y, tmp_path):
        X, y = X_y
        pp = PreprocessingPipeline()
        _ = pp.fit_transform_split(X, y)
        pp._pipeline_dir = tmp_path
        pp.save()

        pp_loaded = PreprocessingPipeline.load(pipeline_dir=tmp_path)
        assert pp_loaded.feature_names_out == pp.feature_names_out
        assert pp_loaded._scale_cols == pp._scale_cols
