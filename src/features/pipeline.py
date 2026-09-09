"""
Fraud Shield — Preprocessing Pipeline
=======================================
Builds, fits, and serialises the leakage-safe sklearn Pipeline.

ANTI-LEAKAGE GUARANTEE:
    - Pipeline.fit() is called ONLY on the training split.
    - All scalers learn statistics (mean, std) from training data only.
    - The same fitted pipeline is applied to val and test sets.
    - Inference uses the same serialised pipeline — never re-fits.

PIPELINE STRUCTURE:
    ColumnTransformer:
        StandardScaler  → ['Amount', 'Time', 'log_amount', 'hour_of_day']
        passthrough     → ['V1'–'V28', 'is_night']

    Verified from inspection (2026-09-09):
        Amount: mean=88.35, std=250.12 — needs scaling
        Time:   mean=94813, std=47488  — needs scaling
        V1–V28: mean≈0, std≈1–2        — already PCA-scaled by ULB; passthrough
        log_amount: derived, not yet scaled
        hour_of_day: 0–24 range, unscaled
        is_night: binary {0,1} — no scaling

SPLITTING STRATEGY:
    Stratified split preserving class ratio across all three sets.
    Default: 60% train / 20% val / 20% test (config-driven).
    Random state fixed at 42 for reproducibility.

Usage:
    from src.features.pipeline import PreprocessingPipeline
    pp = PreprocessingPipeline()
    splits = pp.fit_transform_split(X, y)
    pp.save()
    # --- inference ---
    pp2 = PreprocessingPipeline.load()
    X_scaled = pp2.transform(X_new)
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.utils.config import get_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Typed return type for split data
# ---------------------------------------------------------------------------

class SplitData:
    """Container for the stratified train/val/test split."""

    def __init__(
        self,
        X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray,
        y_train: pd.Series, y_val: pd.Series, y_test: pd.Series,
        feature_names_out: list[str],
    ) -> None:
        self.X_train = X_train
        self.X_val = X_val
        self.X_test = X_test
        self.y_train = y_train
        self.y_val = y_val
        self.y_test = y_test
        self.feature_names_out = feature_names_out

    def summary(self) -> str:
        lines = [
            "Split Summary:",
            f"  Train : X{self.X_train.shape}  fraud={self.y_train.sum()} "
            f"({self.y_train.mean()*100:.3f}%)",
            f"  Val   : X{self.X_val.shape}  fraud={self.y_val.sum()} "
            f"({self.y_val.mean()*100:.3f}%)",
            f"  Test  : X{self.X_test.shape}  fraud={self.y_test.sum()} "
            f"({self.y_test.mean()*100:.3f}%)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Preprocessing Pipeline
# ---------------------------------------------------------------------------

class PreprocessingPipeline:
    """
    Leakage-safe preprocessing pipeline for credit card fraud detection.

    Wraps a sklearn ColumnTransformer + StandardScaler inside a Pipeline.
    Handles train/val/test splitting internally to ensure the scaler
    is always fitted exclusively on training data.

    Attributes:
        pipeline: The fitted sklearn Pipeline (None before fit).
        feature_names_in: Column names that were passed to fit().
        feature_names_out: Column names of the transformed output.
        metadata: Dict of fitting metadata (shapes, dates, feature lists).
    """

    _PIPELINE_FILENAME = "preprocessing_pipeline.joblib"
    _METADATA_FILENAME = "preprocessing_metadata.json"

    def __init__(self) -> None:
        self._cfg = get_config()
        self._ds_cfg = self._cfg["dataset"]
        self._split_cfg = self._cfg["splitting"]
        self._root = get_project_root()
        self._pipeline_dir = self._root / self._cfg["paths"]["artifacts"]["pipelines"]

        self.pipeline: Optional[Pipeline] = None
        self.feature_names_in: list[str] = []
        self.feature_names_out: list[str] = []
        self.metadata: dict = {}

        self._scale_cols: list[str] = []   # set during _build_pipeline
        self._pass_cols: list[str] = []    # set during _build_pipeline

    # ------------------------------------------------------------------
    # Primary public API
    # ------------------------------------------------------------------

    def fit_transform_split(self, X: pd.DataFrame, y: pd.Series) -> SplitData:
        """
        Split data into train/val/test, fit the pipeline on training data only,
        and transform all three splits.

        Steps:
            1. Stratified train+val / test split (20% test holdout)
            2. Stratified train / val split from the train+val portion
            3. Determine scale vs passthrough columns from X
            4. Build sklearn Pipeline
            5. Pipeline.fit() on X_train ONLY
            6. Pipeline.transform() on train, val, test
            7. Store metadata

        Args:
            X: Feature DataFrame (output of FeatureEngineer.transform(), target removed).
            y: Target Series (0/1 integers).

        Returns:
            SplitData: Dataclass with X_train, X_val, X_test, y_train, y_val, y_test arrays.
        """
        if self.pipeline is not None:
            raise RuntimeError(
                "Pipeline is already fitted. Create a new PreprocessingPipeline instance "
                "or call .reset() before fitting again."
            )

        self._validate_X_y(X, y)

        # --- Step 1: train+val / test split ---
        test_size = self._split_cfg["test_size"]
        random_state = self._split_cfg["random_state"]
        stratify = self._split_cfg["stratify"]

        stratify_col = y if stratify else None

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_col,
        )
        logger.info(
            "Test split: %d train+val, %d test (%.0f%% / %.0f%%)",
            len(X_trainval), len(X_test),
            (1 - test_size) * 100, test_size * 100,
        )

        # --- Step 2: train / val split ---
        val_size = self._split_cfg["validation_size"]
        # val_size is proportion of the FULL dataset; adjust for current subset
        val_fraction_of_trainval = val_size / (1.0 - test_size)

        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=val_fraction_of_trainval,
            random_state=random_state,
            stratify=y_trainval if stratify else None,
        )
        logger.info(
            "Train/val split: %d train, %d val",
            len(X_train), len(X_val),
        )

        # --- Step 3 & 4: Build pipeline from training column names ---
        self._build_pipeline(X_train)

        # --- Step 5: Fit on training data ONLY ---
        logger.info("Fitting preprocessing pipeline on training data (%d rows)...", len(X_train))
        self.pipeline.fit(X_train)
        logger.info("Pipeline fitted. Scaler learned from training data only.")

        # --- Step 6: Transform all splits ---
        X_train_t = self.pipeline.transform(X_train)
        X_val_t = self.pipeline.transform(X_val)
        X_test_t = self.pipeline.transform(X_test)

        # --- Store metadata ---
        self._store_metadata(X, y, X_train, X_val, X_test)

        split = SplitData(
            X_train=X_train_t, X_val=X_val_t, X_test=X_test_t,
            y_train=y_train, y_val=y_val, y_test=y_test,
            feature_names_out=self.feature_names_out,
        )

        logger.info("%s", split.summary())
        self._log_leakage_check(X_train, X_test)

        return split

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform a new DataFrame using the already-fitted pipeline.

        Used for inference — ensures identical transformation as training.

        Args:
            X: Feature DataFrame with the same columns as training X.

        Returns:
            np.ndarray: Transformed feature matrix.

        Raises:
            RuntimeError: If pipeline has not been fitted yet.
        """
        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline has not been fitted. Call fit_transform_split() "
                "or load a saved pipeline with PreprocessingPipeline.load()."
            )
        return self.pipeline.transform(X)

    def save(self) -> tuple[pathlib.Path, pathlib.Path]:
        """
        Serialise the fitted pipeline and metadata to disk.

        Returns:
            (pipeline_path, metadata_path): Paths to saved files.

        Raises:
            RuntimeError: If pipeline has not been fitted.
        """
        if self.pipeline is None:
            raise RuntimeError("Cannot save an unfitted pipeline.")

        self._pipeline_dir.mkdir(parents=True, exist_ok=True)
        pipeline_path = self._pipeline_dir / self._PIPELINE_FILENAME
        metadata_path = self._pipeline_dir / self._METADATA_FILENAME

        joblib.dump(self.pipeline, pipeline_path)
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, default=str)

        logger.info("Pipeline saved to: %s", pipeline_path)
        logger.info("Metadata saved to: %s", metadata_path)
        return pipeline_path, metadata_path

    @classmethod
    def load(cls, pipeline_dir: Optional[pathlib.Path] = None) -> "PreprocessingPipeline":
        """
        Load a previously saved pipeline from disk.

        Args:
            pipeline_dir: Directory containing saved pipeline files.
                          Defaults to config value.

        Returns:
            PreprocessingPipeline: Instance with fitted pipeline loaded.

        Raises:
            FileNotFoundError: If pipeline file does not exist.
        """
        instance = cls()

        if pipeline_dir is None:
            pipeline_dir = instance._pipeline_dir

        pipeline_path = pipeline_dir / cls._PIPELINE_FILENAME
        metadata_path = pipeline_dir / cls._METADATA_FILENAME

        if not pipeline_path.exists():
            raise FileNotFoundError(
                f"Pipeline file not found: {pipeline_path}\n"
                f"Run `python scripts/preprocess.py` to create it."
            )

        instance.pipeline = joblib.load(pipeline_path)

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                instance.metadata = json.load(f)
            instance.feature_names_in = instance.metadata.get("feature_names_in", [])
            instance.feature_names_out = instance.metadata.get("feature_names_out", [])
            instance._scale_cols = instance.metadata.get("scale_columns", [])
            instance._pass_cols = instance.metadata.get("passthrough_columns", [])

        logger.info("Pipeline loaded from: %s", pipeline_path)
        return instance

    def reset(self) -> None:
        """Reset the pipeline to unfitted state."""
        self.pipeline = None
        self.feature_names_in = []
        self.feature_names_out = []
        self.metadata = {}
        self._scale_cols = []
        self._pass_cols = []

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self, X_ref: pd.DataFrame) -> None:
        """
        Build the ColumnTransformer Pipeline from actual column names.

        Column categorisation (verified from dataset inspection):
            Scale: Amount, Time, log_amount, hour_of_day
            Passthrough: V1–V28, is_night

        Args:
            X_ref: The training DataFrame (used to determine actual columns present).
        """
        cfg_scale_cols = self._ds_cfg["scale_columns"]  # ['Amount', 'Time']

        # Columns that need StandardScaler — must exist in X_ref
        scale_cols = [c for c in cfg_scale_cols + ["log_amount", "hour_of_day"]
                      if c in X_ref.columns]

        # Passthrough: V-features + is_night
        pca_cols = [c for c in X_ref.columns if c.startswith("V") and c[1:].isdigit()]
        pass_cols = [c for c in pca_cols + ["is_night"] if c in X_ref.columns]

        # Verify no column overlap
        overlap = set(scale_cols) & set(pass_cols)
        if overlap:
            raise ValueError(f"Column assignment overlap: {overlap}")

        # Verify all X columns are accounted for
        assigned = set(scale_cols) | set(pass_cols)
        unassigned = [c for c in X_ref.columns if c not in assigned]
        if unassigned:
            logger.warning(
                "Unassigned columns (not scaled, not passed): %s. "
                "They will be DROPPED by ColumnTransformer.",
                unassigned,
            )

        self._scale_cols = scale_cols
        self._pass_cols = pass_cols

        logger.info(
            "Pipeline columns | scale=%d %s | passthrough=%d",
            len(scale_cols), scale_cols, len(pass_cols),
        )

        transformer = ColumnTransformer(
            transformers=[
                ("scaler", StandardScaler(), scale_cols),
                ("passthrough", "passthrough", pass_cols),
            ],
            remainder="drop",       # Drop any unassigned columns
            verbose_feature_names_out=False,
        )

        self.pipeline = Pipeline([("preprocessor", transformer)])

        # Store feature column names for reference
        self.feature_names_in = list(X_ref.columns)

        # Compute output feature names (scale_cols first, then pass_cols)
        self.feature_names_out = scale_cols + pass_cols

    def _store_metadata(
        self,
        X: pd.DataFrame, y: pd.Series,
        X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame,
    ) -> None:
        """Store pipeline fitting metadata."""
        cfg = self._split_cfg
        self.metadata = {
            "fitted_at": datetime.now(timezone.utc).isoformat(),
            "total_rows_used": len(X),
            "n_features_in": len(self.feature_names_in),
            "n_features_out": len(self.feature_names_out),
            "feature_names_in": self.feature_names_in,
            "feature_names_out": self.feature_names_out,
            "scale_columns": self._scale_cols,
            "passthrough_columns": self._pass_cols,
            "split_config": {
                "test_size": cfg["test_size"],
                "validation_size": cfg["validation_size"],
                "random_state": cfg["random_state"],
                "stratified": cfg["stratify"],
            },
            "split_sizes": {
                "train": len(X_train),
                "val": len(X_val),
                "test": len(X_test),
            },
            "class_distribution": {
                "train": {"fraud": int(y.loc[X_train.index].sum()),
                          "legitimate": int((y.loc[X_train.index] == 0).sum())},
                "val": {"fraud": int(y.loc[X_val.index].sum()),
                        "legitimate": int((y.loc[X_val.index] == 0).sum())},
                "test": {"fraud": int(y.loc[X_test.index].sum()),
                         "legitimate": int((y.loc[X_test.index] == 0).sum())},
            },
            "scaler_stats": self._extract_scaler_stats(),
            "pipeline_version": "1.0.0",
        }

    def _extract_scaler_stats(self) -> dict:
        """Extract mean/std from the fitted StandardScaler for documentation."""
        if self.pipeline is None:
            return {}
        try:
            scaler = self.pipeline.named_steps["preprocessor"].named_transformers_["scaler"]
            return {
                col: {"mean": float(scaler.mean_[i]), "std": float(scaler.scale_[i])}
                for i, col in enumerate(self._scale_cols)
            }
        except Exception:
            return {}

    def _validate_X_y(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Basic pre-flight checks before splitting."""
        if len(X) != len(y):
            raise ValueError(
                f"X and y have different lengths: X={len(X)}, y={len(y)}"
            )
        if y.nunique() != 2:
            raise ValueError(
                f"y must be binary (2 classes), found: {sorted(y.unique())}"
            )
        min_samples = 10
        if y.sum() < min_samples:
            raise ValueError(
                f"Fewer than {min_samples} fraud samples ({y.sum()}) — "
                f"cannot stratify reliably."
            )
        logger.debug("X/y validation passed: X%s y=%d fraud=%d", X.shape, len(y), int(y.sum()))

    def _log_leakage_check(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        """Assert no index overlap between train and test sets."""
        train_idx = set(X_train.index)
        test_idx = set(X_test.index)
        overlap = train_idx & test_idx
        if overlap:
            # This should never happen with train_test_split, but we verify.
            logger.error(
                "LEAKAGE DETECTED: %d indices appear in both train and test sets!",
                len(overlap),
            )
        else:
            logger.info("Leakage check passed: zero index overlap between train and test.")
