"""
Fraud Shield — Data Validator
==============================
Column-level schema enforcement and quality checks.
Runs AFTER the loader and BEFORE feature engineering.

Key design principle:
    All checks are declarative — driven by config.yaml.
    The validator never modifies data; it only reports problems.
    Hard failures raise DataValidationError.
    Soft warnings are logged.

Usage:
    from src.data.validator import DataValidator
    validator = DataValidator()
    report = validator.validate(df)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when a hard validation failure is detected."""


# ---------------------------------------------------------------------------
# Validation Report — structured result returned by validate()
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured validation report for a dataset."""
    passed: bool = True
    n_rows: int = 0
    n_columns: int = 0
    n_fraud: int = 0
    n_legitimate: int = 0
    fraud_rate_pct: float = 0.0
    imbalance_ratio: str = ""
    total_missing: int = 0
    missing_by_column: dict = field(default_factory=dict)
    duplicate_rows: int = 0
    all_numeric: bool = True
    non_numeric_columns: list = field(default_factory=list)
    target_classes: list = field(default_factory=list)
    hard_errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    column_dtypes: dict = field(default_factory=dict)
    amount_stats: dict = field(default_factory=dict)
    time_range_hours: float = 0.0
    top_correlated_features: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Validation {status}",
            f"  Rows:          {self.n_rows:>10,}",
            f"  Columns:       {self.n_columns:>10}",
            f"  Fraud:         {self.n_fraud:>10,}  ({self.fraud_rate_pct:.4f}%)",
            f"  Legitimate:    {self.n_legitimate:>10,}",
            f"  Imbalance:     {self.imbalance_ratio}",
            f"  Missing:       {self.total_missing:>10,}",
            f"  Duplicates:    {self.duplicate_rows:>10,}",
        ]
        if self.hard_errors:
            lines.append(f"  Hard Errors:   {len(self.hard_errors)}")
            for e in self.hard_errors:
                lines.append(f"    [ERROR] {e}")
        if self.warnings:
            lines.append(f"  Warnings:      {len(self.warnings)}")
            for w in self.warnings:
                lines.append(f"    [WARN]  {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DataValidator:
    """
    Validates a DataFrame against the project's expected schema.

    Checks performed:
        Hard (raise on failure):
            - Minimum row count
            - Target column present
            - Target column contains only {0, 1}
            - No all-null columns

        Soft (warn only):
            - Missing values in any column
            - Duplicate rows
            - Non-numeric columns present
            - Amount/Time columns present
            - Fraud rate deviates from expected range
    """

    # Expected fraud rate range for the ULB dataset (± generous tolerance)
    _EXPECTED_FRAUD_RATE_MIN = 0.001   # 0.1%
    _EXPECTED_FRAUD_RATE_MAX = 0.005   # 0.5%

    def __init__(self) -> None:
        self._cfg = get_config()
        self._ds_cfg = self._cfg["dataset"]
        self._target_col = self._ds_cfg["target_column"]
        self._min_rows = self._ds_cfg["expected_min_rows"]
        self._scale_cols = self._ds_cfg["scale_columns"]

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate(self, df: pd.DataFrame, raise_on_error: bool = True) -> ValidationReport:
        """
        Validate a DataFrame and return a structured ValidationReport.

        Args:
            df: The DataFrame to validate (raw, unmodified).
            raise_on_error: If True, raise DataValidationError on hard failures.

        Returns:
            ValidationReport: Detailed validation results.

        Raises:
            DataValidationError: If raise_on_error=True and hard errors exist.
        """
        report = ValidationReport()
        report.n_rows = len(df)
        report.n_columns = len(df.columns)
        report.column_dtypes = {c: str(t) for c, t in df.dtypes.items()}

        # --- Hard checks ---
        self._check_min_rows(df, report)
        self._check_target_column(df, report)
        self._check_no_all_null_columns(df, report)

        # Compute target-dependent stats only if target column exists
        if self._target_col in df.columns:
            self._compute_class_stats(df, report)

        # --- Soft checks ---
        self._check_missing_values(df, report)
        self._check_duplicates(df, report)
        self._check_numeric_only(df, report)
        self._check_scale_columns(df, report)
        self._check_fraud_rate(report)
        self._compute_amount_stats(df, report)
        self._compute_time_stats(df, report)
        self._compute_top_correlations(df, report)

        # Final pass/fail
        report.passed = len(report.hard_errors) == 0

        if report.warnings:
            for w in report.warnings:
                logger.warning("Validation warning: %s", w)

        if not report.passed:
            error_msg = (
                f"Dataset validation FAILED with {len(report.hard_errors)} error(s):\n"
                + "\n".join(f"  [{i+1}] {e}" for i, e in enumerate(report.hard_errors))
            )
            logger.error(error_msg)
            if raise_on_error:
                raise DataValidationError(error_msg)
        else:
            logger.info(
                "Dataset validation PASSED | rows=%d | fraud=%d (%.4f%%) | "
                "missing=%d | duplicates=%d",
                report.n_rows, report.n_fraud, report.fraud_rate_pct,
                report.total_missing, report.duplicate_rows,
            )

        return report

    # ------------------------------------------------------------------
    # Hard checks
    # ------------------------------------------------------------------

    def _check_min_rows(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if len(df) < self._min_rows:
            report.hard_errors.append(
                f"Row count {len(df):,} is below minimum {self._min_rows:,}. "
                f"Ensure the full dataset is loaded."
            )

    def _check_target_column(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if self._target_col not in df.columns:
            report.hard_errors.append(
                f"Target column '{self._target_col}' not found. "
                f"Available: {list(df.columns)}"
            )
            return

        unique = sorted(df[self._target_col].dropna().unique().tolist())
        report.target_classes = unique
        unexpected = set(unique) - {0, 1}
        if unexpected:
            report.hard_errors.append(
                f"Target column '{self._target_col}' contains unexpected values "
                f"{unexpected}. Expected only 0 and 1."
            )

    def _check_no_all_null_columns(self, df: pd.DataFrame, report: ValidationReport) -> None:
        all_null = [c for c in df.columns if df[c].isnull().all()]
        if all_null:
            report.hard_errors.append(
                f"Columns with all-null values: {all_null}"
            )

    # ------------------------------------------------------------------
    # Soft checks
    # ------------------------------------------------------------------

    def _compute_class_stats(self, df: pd.DataFrame, report: ValidationReport) -> None:
        report.n_fraud = int(df[self._target_col].sum())
        report.n_legitimate = int(len(df) - report.n_fraud)
        report.fraud_rate_pct = round(report.n_fraud / len(df) * 100, 6)
        ratio = int(report.n_legitimate / max(report.n_fraud, 1))
        report.imbalance_ratio = f"1:{ratio}"

    def _check_missing_values(self, df: pd.DataFrame, report: ValidationReport) -> None:
        missing_per_col = df.isnull().sum()
        report.total_missing = int(missing_per_col.sum())
        report.missing_by_column = {
            c: int(v) for c, v in missing_per_col.items() if v > 0
        }
        if report.total_missing > 0:
            report.warnings.append(
                f"Dataset has {report.total_missing:,} missing values in "
                f"{len(report.missing_by_column)} column(s): {report.missing_by_column}"
            )

    def _check_duplicates(self, df: pd.DataFrame, report: ValidationReport) -> None:
        report.duplicate_rows = int(df.duplicated().sum())
        if report.duplicate_rows > 0:
            report.warnings.append(
                f"Dataset has {report.duplicate_rows:,} duplicate rows. "
                f"These will be dropped during preprocessing."
            )

    def _check_numeric_only(self, df: pd.DataFrame, report: ValidationReport) -> None:
        non_numeric = [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c]) and c != self._target_col
        ]
        report.non_numeric_columns = non_numeric
        report.all_numeric = len(non_numeric) == 0
        if non_numeric:
            report.warnings.append(
                f"Non-numeric columns found (will need encoding or dropping): {non_numeric}"
            )

    def _check_scale_columns(self, df: pd.DataFrame, report: ValidationReport) -> None:
        for col in self._scale_cols:
            if col not in df.columns:
                report.warnings.append(
                    f"Expected scale column '{col}' not found in dataset."
                )

    def _check_fraud_rate(self, report: ValidationReport) -> None:
        if report.n_rows > 0:
            rate = report.n_fraud / report.n_rows
            if not (self._EXPECTED_FRAUD_RATE_MIN <= rate <= self._EXPECTED_FRAUD_RATE_MAX):
                report.warnings.append(
                    f"Fraud rate {rate:.4%} is outside expected range "
                    f"[{self._EXPECTED_FRAUD_RATE_MIN:.4%}, "
                    f"{self._EXPECTED_FRAUD_RATE_MAX:.4%}]. "
                    f"Verify dataset is the correct ULB dataset."
                )

    def _compute_amount_stats(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if "Amount" in df.columns:
            report.amount_stats = {
                "mean": float(df["Amount"].mean()),
                "std": float(df["Amount"].std()),
                "min": float(df["Amount"].min()),
                "max": float(df["Amount"].max()),
                "median": float(df["Amount"].median()),
                "skewness": float(df["Amount"].skew()),
                "zeros": int((df["Amount"] == 0).sum()),
            }

    def _compute_time_stats(self, df: pd.DataFrame, report: ValidationReport) -> None:
        if "Time" in df.columns:
            report.time_range_hours = float(df["Time"].max() / 3600)

    def _compute_top_correlations(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Compute absolute Pearson correlation of each feature with target."""
        if self._target_col not in df.columns:
            return
        numeric_df = df.select_dtypes(include=[np.number])
        corr = (
            numeric_df.corr()[self._target_col]
            .drop(self._target_col, errors="ignore")
            .abs()
            .sort_values(ascending=False)
        )
        report.top_correlated_features = {
            k: round(float(v), 6) for k, v in corr.head(10).items()
        }
