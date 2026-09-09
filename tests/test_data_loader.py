"""
Tests for src/data/loader.py and src/data/validator.py

Run with: pytest tests/test_data_loader.py -v
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import DataLoader, DatasetValidationError
from src.data.validator import DataValidator, ValidationReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """
    Minimal valid DataFrame that mimics the creditcard.csv schema.
    Uses 600 rows (492 legit + 108 fraud) — sufficient for all checks.
    Does NOT use the real dataset so tests run offline/quickly.
    """
    rng = np.random.default_rng(42)
    n_legit = 492
    n_fraud = 108

    v_cols = {f"V{i}": rng.standard_normal(n_legit + n_fraud) for i in range(1, 29)}
    df = pd.DataFrame({
        "Time": rng.uniform(0, 172792, n_legit + n_fraud),
        **v_cols,
        "Amount": rng.exponential(scale=88, size=n_legit + n_fraud),
        "Class": [0] * n_legit + [1] * n_fraud,
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@pytest.fixture(scope="module")
def validator() -> DataValidator:
    return DataValidator()


# ---------------------------------------------------------------------------
# DataLoader tests (filesystem-independent)
# ---------------------------------------------------------------------------

class TestDataLoaderInit:
    def test_loader_initialises_without_error(self):
        loader = DataLoader()
        assert loader is not None

    def test_loader_accepts_custom_path(self, tmp_path):
        custom = tmp_path / "custom.csv"
        loader = DataLoader(data_path=custom)
        assert loader._data_path == custom

    def test_loader_raises_file_not_found(self, tmp_path):
        loader = DataLoader(data_path=tmp_path / "nonexistent.csv")
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            loader.load()


# ---------------------------------------------------------------------------
# DataLoader.get_statistics
# ---------------------------------------------------------------------------

class TestDataLoaderStatistics:
    def test_statistics_returns_correct_keys(self, sample_df):
        loader = DataLoader()
        stats = loader.get_statistics(sample_df)
        required_keys = [
            "n_rows", "n_columns", "n_fraud", "n_legitimate",
            "fraud_rate", "total_missing_values", "duplicate_rows",
            "column_names", "dtypes", "class_balance",
        ]
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

    def test_statistics_row_count(self, sample_df):
        loader = DataLoader()
        stats = loader.get_statistics(sample_df)
        assert stats["n_rows"] == len(sample_df)

    def test_statistics_class_balance(self, sample_df):
        loader = DataLoader()
        stats = loader.get_statistics(sample_df)
        assert stats["n_fraud"] == 108
        assert stats["n_legitimate"] == 492
        assert stats["fraud_rate"] > 0

    def test_statistics_no_missing(self, sample_df):
        loader = DataLoader()
        stats = loader.get_statistics(sample_df)
        assert stats["total_missing_values"] == 0

    def test_statistics_detects_missing(self, sample_df):
        df_with_missing = sample_df.copy()
        df_with_missing.loc[0, "Amount"] = float("nan")
        loader = DataLoader()
        stats = loader.get_statistics(df_with_missing)
        assert stats["total_missing_values"] == 1
        assert "Amount" in stats["missing_by_column"]

    def test_statistics_detects_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        loader = DataLoader()
        stats = loader.get_statistics(df_dup)
        assert stats["duplicate_rows"] == 5


# ---------------------------------------------------------------------------
# DataLoader column helpers
# ---------------------------------------------------------------------------

class TestDataLoaderColumnHelpers:
    def test_get_pca_columns(self, sample_df):
        loader = DataLoader()
        pca_cols = loader.get_pca_columns(sample_df)
        assert len(pca_cols) == 28
        assert all(c.startswith("V") for c in pca_cols)
        assert "V1" in pca_cols
        assert "V28" in pca_cols

    def test_get_feature_columns_excludes_target(self, sample_df):
        loader = DataLoader()
        feat_cols = loader.get_feature_columns(sample_df)
        assert "Class" not in feat_cols
        assert "Time" in feat_cols
        assert "Amount" in feat_cols
        assert len(feat_cols) == 30  # 31 columns - 1 target


# ---------------------------------------------------------------------------
# DataValidator — Hard Checks
# ---------------------------------------------------------------------------

class TestDataValidatorHardChecks:
    def test_valid_df_passes(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        # Should pass (or only have soft warnings about row count below 200k)
        # Hard errors should not include target or type issues
        hard = [e for e in report.hard_errors if "Target" in e or "null" in e]
        assert len(hard) == 0

    def test_missing_target_column_raises(self, validator):
        df_no_target = pd.DataFrame({
            "Time": [1.0], "V1": [0.1], "Amount": [100.0]
        })
        with pytest.raises(Exception):
            validator.validate(df_no_target, raise_on_error=True)

    def test_invalid_target_values_raises(self, sample_df, validator):
        df_bad_target = sample_df.copy()
        df_bad_target["Class"] = df_bad_target["Class"].map({0: 0, 1: 2})
        with pytest.raises(Exception):
            validator.validate(df_bad_target, raise_on_error=True)

    def test_all_null_column_raises(self, sample_df, validator):
        df_null_col = sample_df.copy()
        df_null_col["V1"] = float("nan")
        with pytest.raises(Exception):
            validator.validate(df_null_col, raise_on_error=True)


# ---------------------------------------------------------------------------
# DataValidator — Report fields
# ---------------------------------------------------------------------------

class TestDataValidatorReport:
    def test_report_is_ValidationReport(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        assert isinstance(report, ValidationReport)

    def test_report_fraud_stats(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        assert report.n_fraud == 108
        assert report.n_legitimate == 492
        assert 0 < report.fraud_rate_pct < 100

    def test_report_flags_duplicates(self, sample_df, validator):
        df_dup = pd.concat([sample_df, sample_df.iloc[:3]], ignore_index=True)
        report = validator.validate(df_dup, raise_on_error=False)
        assert report.duplicate_rows == 3
        dup_warnings = [w for w in report.warnings if "duplicate" in w.lower()]
        assert len(dup_warnings) > 0

    def test_report_all_numeric_true(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        # All columns in sample_df are numeric
        assert report.all_numeric is True

    def test_report_non_numeric_detected(self, sample_df, validator):
        df_with_str = sample_df.copy()
        df_with_str["merchant_name"] = "ShopA"
        report = validator.validate(df_with_str, raise_on_error=False)
        assert report.all_numeric is False
        assert "merchant_name" in report.non_numeric_columns

    def test_report_top_correlated_features(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        # Should compute some correlations
        assert isinstance(report.top_correlated_features, dict)

    def test_report_summary_string(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        summary = report.summary()
        assert "Rows" in summary
        assert "Fraud" in summary

    def test_report_amount_stats(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        assert "mean" in report.amount_stats
        assert "max" in report.amount_stats
        assert report.amount_stats["min"] >= 0

    def test_report_time_range_hours(self, sample_df, validator):
        report = validator.validate(sample_df, raise_on_error=False)
        # Our sample has Time up to 172792 → ≈48h
        assert report.time_range_hours > 0
