"""
Fraud Shield — Feature Engineering
====================================
Creates derived features from raw columns.

VERIFIED AGAINST ACTUAL DATASET (2026-09-09):
    - Amount: highly right-skewed (skewness=16.98, kurtosis=845.09)
      → log1p(Amount) normalises this significantly
    - Time: spans exactly 48 hours (0 – 172792 seconds)
      → hour_of_day captures known time-of-day fraud patterns
      → is_night: 8.40% of transactions occur between 0–6h
    - V1–V28: PCA-transformed, mean≈0, std≈1–2 (already scaled by ULB)
      → passed through WITHOUT additional scaling

Top fraud-correlated features from inspection:
    V17 (0.326), V14 (0.303), V12 (0.261), V10 (0.217), V16 (0.197)

DESIGN PRINCIPLE:
    This module only CREATES features from the raw columns.
    It does NOT scale, encode, or split.
    Scaling is handled exclusively inside the sklearn Pipeline
    and is fit ONLY on the training split.

    The same FeatureEngineer instance must be used for both
    training and inference to guarantee consistent column order.

Usage:
    from src.features.engineering import FeatureEngineer
    fe = FeatureEngineer()
    df_engineered = fe.transform(df_raw)
    X, y = fe.get_X_y(df_engineered)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants — derived from actual dataset inspection (2026-09-09)
# ---------------------------------------------------------------------------
# These are documented facts about the raw dataset, NOT assumptions.
_SECONDS_PER_DAY: int = 86_400
_NIGHT_HOUR_END: int = 6        # 0h–6h is classified as "night"
_NIGHT_FRACTION_ACTUAL: float = 0.0840  # 8.40% of all transactions (verified)


class FeatureEngineer:
    """
    Transforms raw transaction DataFrame into a feature-engineered DataFrame.

    Derived features created:
        log_amount  : log1p(Amount) — reduces extreme right skew (skewness=16.98)
        hour_of_day : fractional hour within a 24h cycle from Time column
        is_night    : 1 if hour_of_day < 6 else 0

    No features are dropped here. Target separation is done via get_X_y().

    Thread-safe: stateless transform (no fitted state).
    """

    def __init__(self) -> None:
        self._cfg = get_config()
        self._ds_cfg = self._cfg["dataset"]
        self._target_col = self._ds_cfg["target_column"]
        self._drop_cols = self._ds_cfg.get("drop_columns") or []
        self._scale_cols = self._ds_cfg["scale_columns"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform(self, df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Apply feature engineering to a raw DataFrame.

        Steps (in order):
            1. Optionally drop duplicate rows.
            2. Drop columns listed in config.dataset.drop_columns.
            3. Create log_amount from Amount.
            4. Create hour_of_day and is_night from Time.

        Args:
            df: Raw DataFrame from DataLoader (NEVER modifies the original).
            drop_duplicates: Drop duplicate rows (default True for training).
                             Set False for inference on single transactions.

        Returns:
            New DataFrame with all original columns plus derived features.
            The raw DataFrame is NOT modified.
        """
        df_out = df.copy()

        if drop_duplicates:
            before = len(df_out)
            df_out = df_out.drop_duplicates()
            dropped = before - len(df_out)
            if dropped > 0:
                logger.info("Dropped %d duplicate rows (%d -> %d)", dropped, before, len(df_out))

        if self._drop_cols:
            existing_drop = [c for c in self._drop_cols if c in df_out.columns]
            if existing_drop:
                df_out = df_out.drop(columns=existing_drop)
                logger.debug("Dropped config-listed columns: %s", existing_drop)

        df_out = self._create_log_amount(df_out)
        df_out = self._create_time_features(df_out)

        logger.debug(
            "Feature engineering complete: %d rows, %d columns (was %d)",
            len(df_out), len(df_out.columns), len(df.columns),
        )
        return df_out

    def get_X_y(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target from an engineered DataFrame.

        Args:
            df: Output of transform().

        Returns:
            (X, y): Feature matrix and target Series.

        Raises:
            KeyError: If target column is not present.
        """
        if self._target_col not in df.columns:
            raise KeyError(
                f"Target column '{self._target_col}' not found. "
                f"Was transform() called first?"
            )
        y = df[self._target_col].copy().astype(int)
        X = df.drop(columns=[self._target_col])
        logger.debug(
            "X/y split: X=%s, y=%s (fraud=%d, legit=%d)",
            X.shape, y.shape, int(y.sum()), int((y == 0).sum()),
        )
        return X, y

    def get_feature_names(self, df_raw: pd.DataFrame) -> dict[str, list[str]]:
        """
        Return categorised feature name lists for use by the preprocessing pipeline.

        Called ONCE before pipeline construction so that pipeline knows
        which columns to scale and which to pass through.

        Args:
            df_raw: Raw DataFrame (pre-transform).

        Returns:
            dict with keys:
                'scale'       — columns to StandardScale
                'passthrough' — columns to pass as-is
                'target'      — the target column (not a feature)
                'derived'     — newly created feature names
        """
        pca_cols = [c for c in df_raw.columns if c.startswith("V") and c[1:].isdigit()]
        scale_cols = self._scale_cols + ["log_amount", "hour_of_day"]
        passthrough_cols = pca_cols
        derived_cols = ["log_amount", "hour_of_day", "is_night"]

        # is_night is binary (0/1), so it does NOT need scaling
        return {
            "scale": scale_cols,
            "passthrough": passthrough_cols + ["is_night"],
            "target": [self._target_col],
            "derived": derived_cols,
        }

    def transform_single(self, transaction: dict) -> pd.DataFrame:
        """
        Engineer features for a single transaction dict (inference path).

        Args:
            transaction: Dict with the same keys as the raw CSV columns
                         (excluding 'Class').

        Returns:
            Single-row DataFrame with all engineered features.
        """
        df_single = pd.DataFrame([transaction])
        return self.transform(df_single, drop_duplicates=False)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_log_amount(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create log_amount = log1p(Amount).

        Rationale (from actual data):
            Amount has skewness=16.98, kurtosis=845.09.
            log1p handles Amount=0 (1,825 such transactions verified) safely.
            After transform: skewness is significantly reduced.
        """
        if "Amount" not in df.columns:
            logger.warning("'Amount' column not found — log_amount not created.")
            return df
        df = df.copy()
        df["log_amount"] = np.log1p(df["Amount"])
        return df

    def _create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create hour_of_day and is_night from Time.

        Rationale (from actual data):
            Time spans 0–172,792 seconds (exactly 48 hours).
            We extract time-of-day using modulo 86400 / 3600.
            8.40% of transactions occur between 0–6h (night window).
            Fraud transactions have mean Time=80,747s (earlier) vs 94,838s (legit).
            Night flag may capture low-activity fraud windows.
        """
        if "Time" not in df.columns:
            logger.warning("'Time' column not found — time features not created.")
            return df
        df = df.copy()
        df["hour_of_day"] = (df["Time"] % _SECONDS_PER_DAY) / 3600.0
        df["is_night"] = (df["hour_of_day"] < _NIGHT_HOUR_END).astype(int)
        return df
