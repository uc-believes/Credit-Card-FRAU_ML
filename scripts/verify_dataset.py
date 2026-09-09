"""
Fraud Shield — Dataset Verification Script
==========================================
Run this script after placing creditcard.csv to verify it loaded correctly.

Usage:
    python scripts/verify_dataset.py
"""

import sys
import pathlib

# Ensure project root is on path
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.logger import setup_logging
from src.data.loader import DataLoader, DatasetValidationError

setup_logging(level="INFO")


def main() -> int:
    print("=" * 60)
    print("  Fraud Shield — Dataset Verification")
    print("=" * 60)

    loader = DataLoader()

    # 1. Load and validate
    print("\n[1/3] Loading dataset...")
    try:
        df = loader.load(validate=True)
        print(f"  [OK] Loaded {len(df):,} rows x {len(df.columns)} columns")
    except FileNotFoundError as e:
        print(f"\n  [!] Dataset not found:\n{e}")
        print("\n  ACTION REQUIRED:")
        print("  Place creditcard.csv in data/raw/ then re-run this script.")
        print("  See data/README.md for download instructions.")
        return 1
    except DatasetValidationError as e:
        print(f"\n  [!] Validation failed:\n{e}")
        return 1

    # 2. Print statistics
    print("\n[2/3] Computing statistics...")
    stats = loader.get_statistics(df)

    print(f"\n  Dataset Summary")
    print(f"  {'-' * 40}")
    print(f"  Rows:               {stats['n_rows']:>12,}")
    print(f"  Columns:            {stats['n_columns']:>12}")
    print(f"  Fraud (Class=1):    {stats['n_fraud']:>12,}  ({stats['fraud_rate']:.4f}%)")
    print(f"  Legitimate (Class=0): {stats['n_legitimate']:>10,}  ({100 - stats['fraud_rate']:.4f}%)")
    print(f"  Imbalance ratio:    {stats['fraud_rate_ratio']:>12}")
    print(f"  Missing values:     {stats['total_missing_values']:>12,}")
    print(f"  Duplicate rows:     {stats['duplicate_rows']:>12,}")

    # 3. Show column overview
    print(f"\n[3/3] Column overview")
    print(f"  {'-' * 40}")
    pca_cols = loader.get_pca_columns(df)
    feature_cols = loader.get_feature_columns(df)
    print(f"  PCA columns (V*):   {len(pca_cols)} ({pca_cols[0]} ... {pca_cols[-1]})")
    print(f"  Scale columns:      Amount, Time")
    print(f"  Target column:      Class")
    print(f"  Total features:     {len(feature_cols)}")

    print(f"\n{'=' * 60}")
    print("  [OK] Dataset verification PASSED - ready for Phase 1 (EDA)")
    print(f"{'=' * 60}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
