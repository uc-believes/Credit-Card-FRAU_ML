"""
Fraud Shield — Dataset Loader & Validator
==========================================
Responsible for loading the raw creditcard.csv and performing
schema validation, quality checks, and basic statistics reporting.

IMPORTANT: This module never modifies the raw data — it reads,
validates, and reports only. Preprocessing is handled by src/features/.

Usage:
    from src.data.loader import DataLoader
    loader = DataLoader()
    df = loader.load()
    stats = loader.get_statistics(df)
"""

from __future__ import annotations

import logging
import pathlib
from typing import Optional

import pandas as pd

from src.utils.config import get_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetValidationError(Exception):
    """Raised when the loaded dataset fails schema or quality validation."""
    pass


class DataLoader:
    """
    Loads and validates the raw credit card transaction dataset.

    The dataset is expected to be the ULB Kaggle creditcard.csv (or a
    compatible schema). Schema expectations are read from config.yaml
    and verified at load time — never assumed.
    """

    def __init__(self, data_path: Optional[pathlib.Path] = None) -> None:
        """
        Args:
            data_path: Override path to CSV. If None, uses config paths.data.raw.
        """
        self._cfg = get_config()
        self._project_root = get_project_root()

        if data_path is not None:
            self._data_path = pathlib.Path(data_path)
        else:
            self._data_path = self._project_root / self._cfg["paths"]["data"]["raw"]

        self._target_col = self._cfg["dataset"]["target_column"]
        self._expected_min_rows = self._cfg["dataset"]["expected_min_rows"]
        self._expected_cols = self._cfg["dataset"]["expected_columns"]
        self._scale_columns = self._cfg["dataset"]["scale_columns"]
        self._drop_columns = self._cfg["dataset"]["drop_columns"] or []

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def load(self, validate: bool = True) -> pd.DataFrame:
        """
        Load the raw CSV into a DataFrame.

        Args:
            validate: If True, run schema and quality checks after loading.

        Returns:
            pd.DataFrame: Raw dataset (unmodified).

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            DatasetValidationError: If validation fails.
        """
        logger.info("Loading dataset from: %s", self._data_path)

        if not self._data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self._data_path}\n"
                f"Please download creditcard.csv from:\n"
                f"  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
                f"And place it at: data/raw/creditcard.csv"
            )

        try:
            df = pd.read_csv(self._data_path)
        except Exception as e:
            raise DatasetValidationError(f"Failed to read CSV: {e}") from e

        logger.info(
            "Dataset loaded: %d rows × %d columns", len(df), len(df.columns)
        )

        if validate:
            self._validate(df)

        return df

    def get_statistics(self, df: pd.DataFrame) -> dict:
        """
        Compute and return basic dataset statistics.

        Args:
            df: Loaded DataFrame.

        Returns:
            dict with keys: n_rows, n_columns, n_fraud, n_legitimate,
                            fraud_rate, missing_values, duplicates,
                            column_names, dtypes, amount_stats, class_balance
        """
        target = self._target_col
        n_fraud = int(df[target].sum())
        n_legit = int(len(df) - n_fraud)
        fraud_rate = float(n_fraud / len(df))

        missing_per_col = df.isnull().sum()
        total_missing = int(missing_per_col.sum())

        amount_col = "Amount" if "Amount" in df.columns else None
        amount_stats = (
            df["Amount"].describe().to_dict() if amount_col else {}
        )

        stats = {
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "n_fraud": n_fraud,
            "n_legitimate": n_legit,
            "fraud_rate": round(fraud_rate * 100, 4),
            "fraud_rate_ratio": f"1:{int(n_legit / max(n_fraud, 1))}",
            "total_missing_values": total_missing,
            "missing_by_column": missing_per_col[missing_per_col > 0].to_dict(),
            "duplicate_rows": int(df.duplicated().sum()),
            "column_names": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "amount_stats": amount_stats,
            "class_balance": {
                "fraud (1)": n_fraud,
                "legitimate (0)": n_legit,
            },
        }

        self._log_statistics(stats)
        return stats

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Return the list of feature columns (excluding target and drop columns).

        Args:
            df: Loaded DataFrame.

        Returns:
            List of feature column names.
        """
        exclude = set(self._drop_columns + [self._target_col])
        return [c for c in df.columns if c not in exclude]

    def get_pca_columns(self, df: pd.DataFrame) -> list[str]:
        """
        Return PCA-transformed feature columns (V1 through V28 or equivalent).

        Args:
            df: Loaded DataFrame.

        Returns:
            List of V-feature column names detected in the dataset.
        """
        return [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Run schema and quality checks on the loaded DataFrame.

        Raises:
            DatasetValidationError: On any validation failure.
        """
        errors: list[str] = []

        # Check minimum row count
        if len(df) < self._expected_min_rows:
            errors.append(
                f"Row count {len(df):,} is below minimum expected "
                f"{self._expected_min_rows:,}. Is this the full dataset?"
            )

        # Check target column exists
        if self._target_col not in df.columns:
            errors.append(
                f"Target column '{self._target_col}' not found in dataset. "
                f"Available columns: {list(df.columns)}"
            )

        # Check scale columns exist
        for col in self._scale_columns:
            if col not in df.columns:
                errors.append(f"Expected scale column '{col}' not found in dataset.")

        # Check target is binary
        if self._target_col in df.columns:
            unique_targets = set(df[self._target_col].unique())
            if not unique_targets.issubset({0, 1}):
                errors.append(
                    f"Target column '{self._target_col}' contains unexpected values: "
                    f"{unique_targets}. Expected only 0 and 1."
                )

        # Check no all-null columns
        all_null_cols = [c for c in df.columns if df[c].isnull().all()]
        if all_null_cols:
            errors.append(f"Columns with all null values: {all_null_cols}")

        if errors:
            error_msg = "Dataset validation failed:\n" + "\n".join(
                f"  [{i+1}] {e}" for i, e in enumerate(errors)
            )
            logger.error(error_msg)
            raise DatasetValidationError(error_msg)

        # Non-fatal warnings
        missing_total = int(df.isnull().sum().sum())
        if missing_total > 0:
            logger.warning("Dataset has %d missing values.", missing_total)

        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            logger.warning("Dataset has %d duplicate rows.", dup_count)

        logger.info("Dataset validation passed.")

    def _log_statistics(self, stats: dict) -> None:
        """Log key statistics to the logger."""
        logger.info(
            "Dataset stats | rows=%d | fraud=%d (%.4f%%) | legitimate=%d | "
            "missing=%d | duplicates=%d",
            stats["n_rows"],
            stats["n_fraud"],
            stats["fraud_rate"],
            stats["n_legitimate"],
            stats["total_missing_values"],
            stats["duplicate_rows"],
        )
