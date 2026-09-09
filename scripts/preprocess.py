"""
Fraud Shield — Preprocessing Pipeline Script
=============================================
End-to-end data loading, validation, feature engineering, splitting,
pipeline fitting, and artifact saving.

Run this script BEFORE training any models.

Usage:
    python scripts/preprocess.py
    python scripts/preprocess.py --no-save   (dry-run, no files written)
    python scripts/preprocess.py --sample 5000  (use a sample for testing)

Outputs (saved to artifacts/pipelines/):
    preprocessing_pipeline.joblib    — fitted sklearn Pipeline
    preprocessing_metadata.json      — split sizes, feature names, scaler stats

Also saves processed splits to data/processed/:
    train.parquet, validation.parquet, test.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
import pathlib
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logging, get_logger
from src.utils.config import get_config, get_project_root
from src.data.loader import DataLoader, DatasetValidationError
from src.data.validator import DataValidator
from src.features.engineering import FeatureEngineer
from src.features.pipeline import PreprocessingPipeline

setup_logging()
logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fraud Shield — Run preprocessing pipeline"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Dry run — do not save any artifacts to disk.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N rows (for quick testing). Default: use full dataset.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = get_config()
    root = get_project_root()
    t_start = time.time()

    print("=" * 70)
    print("  Fraud Shield - Preprocessing Pipeline")
    print("=" * 70)
    if args.sample:
        print(f"  [!] SAMPLE MODE: using {args.sample:,} rows only")
    if args.no_save:
        print("  [!] DRY RUN: artifacts will NOT be saved")
    print()

    # ------------------------------------------------------------------
    # Step 1: Load Dataset
    # ------------------------------------------------------------------
    print("[1/6] Loading dataset...")
    loader = DataLoader()
    try:
        df = loader.load(validate=True)
    except FileNotFoundError as e:
        print(f"\n  [!] {e}")
        return 1
    except DatasetValidationError as e:
        print(f"\n  [!] Validation failed: {e}")
        return 1

    if args.sample:
        # Stratified sample to preserve class ratio
        fraud_df = df[df[cfg["dataset"]["target_column"]] == 1]
        legit_df = df[df[cfg["dataset"]["target_column"]] == 0]
        n_fraud = max(1, int(args.sample * len(fraud_df) / len(df)))
        n_legit = args.sample - n_fraud
        df = (
            fraud_df.sample(n=min(n_fraud, len(fraud_df)), random_state=42)
            ._append(legit_df.sample(n=min(n_legit, len(legit_df)), random_state=42))
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )
        print(f"  Sampled to {len(df):,} rows (fraud={df[cfg['dataset']['target_column']].sum()})")

    print(f"  Loaded: {len(df):,} rows x {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # Step 2: Deep Validation
    # ------------------------------------------------------------------
    print("\n[2/6] Validating dataset...")
    validator = DataValidator()
    report = validator.validate(df, raise_on_error=True)
    print(f"  {report.summary()}")

    # ------------------------------------------------------------------
    # Step 3: Feature Engineering
    # ------------------------------------------------------------------
    print("\n[3/6] Engineering features...")
    fe = FeatureEngineer()
    df_eng = fe.transform(df, drop_duplicates=True)
    X, y = fe.get_X_y(df_eng)

    feature_cats = fe.get_feature_names(df)
    print(f"  Rows after dedup:     {len(df_eng):,}")
    print(f"  Feature columns:      {X.shape[1]}")
    print(f"  Scale columns:        {feature_cats['scale']}")
    print(f"  Passthrough columns:  {len(feature_cats['passthrough'])} columns")
    print(f"  Derived features:     {feature_cats['derived']}")
    print(f"  Target distribution:  fraud={y.sum()} ({y.mean()*100:.4f}%)")

    # ------------------------------------------------------------------
    # Step 4: Split + Fit Pipeline (CRITICAL: no leakage)
    # ------------------------------------------------------------------
    print("\n[4/6] Splitting and fitting preprocessing pipeline...")
    print("  Strategy: Stratified 60/20/20 (train/val/test)")
    print("  ANTI-LEAKAGE: Scaler will be fitted on training data only")

    pp = PreprocessingPipeline()
    splits = pp.fit_transform_split(X, y)
    print(f"\n  {splits.summary()}")

    # ------------------------------------------------------------------
    # Step 5: Verify training/inference consistency
    # ------------------------------------------------------------------
    print("\n[5/6] Consistency check (training vs inference transform)...")
    _verify_transform_consistency(pp, X, splits)

    # ------------------------------------------------------------------
    # Step 6: Save artifacts
    # ------------------------------------------------------------------
    if not args.no_save:
        print("\n[6/6] Saving artifacts...")

        # Save pipeline
        pipe_path, meta_path = pp.save()
        print(f"  Pipeline:  {pipe_path}")
        print(f"  Metadata:  {meta_path}")

        # Save processed splits to parquet
        processed_dir = root / "data" / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        import pandas as pd
        import numpy as np

        feature_cols = splits.feature_names_out

        train_df = pd.DataFrame(splits.X_train, columns=feature_cols)
        train_df["Class"] = splits.y_train.values
        val_df = pd.DataFrame(splits.X_val, columns=feature_cols)
        val_df["Class"] = splits.y_val.values
        test_df = pd.DataFrame(splits.X_test, columns=feature_cols)
        test_df["Class"] = splits.y_test.values

        train_df.to_parquet(processed_dir / "train.parquet", index=False)
        val_df.to_parquet(processed_dir / "validation.parquet", index=False)
        test_df.to_parquet(processed_dir / "test.parquet", index=False)

        print(f"  Train:     {processed_dir / 'train.parquet'}")
        print(f"  Val:       {processed_dir / 'validation.parquet'}")
        print(f"  Test:      {processed_dir / 'test.parquet'}")
    else:
        print("\n[6/6] Skipping save (dry-run mode).")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Preprocessing complete in {elapsed:.1f}s")
    if not args.no_save:
        print("  Ready for Phase 3: Model Training")
        print("  Next: python scripts/train.py")
    print(f"{'=' * 70}\n")
    return 0


def _verify_transform_consistency(
    pp: PreprocessingPipeline,
    X: "pd.DataFrame",
    splits: "SplitData",
) -> None:
    """
    Verify that the pipeline produces identical output when called
    on a single row vs the full batch (inference consistency test).
    """
    import numpy as np

    # Pick one row from the test set (post-split) by position
    # We need the original X row — find it by index
    import pandas as pd

    # Get a fraud row if possible (more interesting edge case)
    target_col = get_config()["dataset"]["target_column"]
    # Use the first 5 rows of X as test samples
    sample_X = X.iloc[:5].copy()

    # Transform via batch pipeline
    batch_out = pp.transform(sample_X)

    # Transform row by row (simulates inference path)
    row_outs = []
    for i in range(len(sample_X)):
        row = sample_X.iloc[[i]]
        row_out = pp.transform(row)
        row_outs.append(row_out)

    row_stacked = np.vstack(row_outs)

    max_diff = np.abs(batch_out - row_stacked).max()
    if max_diff < 1e-10:
        print(f"  [OK] Consistency check PASSED: max diff = {max_diff:.2e}")
    else:
        print(f"  [!] WARNING: max diff = {max_diff:.6f} between batch and row-by-row transform")
        print("      This may indicate a stateful transform issue.")

    # Verify output shape
    n_features = len(splits.feature_names_out)
    assert batch_out.shape[1] == n_features, (
        f"Output shape mismatch: got {batch_out.shape[1]}, expected {n_features}"
    )
    print(f"  [OK] Output shape: {batch_out.shape[1]} features per transaction")
    print(f"       Features: {splits.feature_names_out[:4]} ... {splits.feature_names_out[-2:]}")


if __name__ == "__main__":
    sys.exit(main())
