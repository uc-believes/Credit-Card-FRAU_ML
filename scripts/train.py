"""
Fraud Shield — Model Training & Evaluation Pipeline
===================================================
Phase 2 Experimentation script.

Loads preprocessed parquet splits, trains all enabled models, performs
threshold optimization on validation data, ranks models via composite score,
evaluates the selected winner on the held-out test set, and saves all artifacts.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import pathlib
import sys
import time

# Ensure project root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.evaluation.metrics import build_comparison_table
from src.models.trainer import ModelTrainer
from src.utils.config import get_config, get_project_root
from src.utils.logger import get_logger
from src.utils.serialization import save_report

logger = get_logger("scripts.train")


def load_processed_splits(root: pathlib.Path, cfg: dict) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    list[str]
]:
    """
    Load preprocessed train, validation, and test splits from parquet.
    """
    train_path = root / cfg["paths"]["data"]["train"]
    val_path = root / cfg["paths"]["data"]["validation"]
    test_path = root / cfg["paths"]["data"]["test"]
    target_col = cfg["dataset"]["target_column"]

    print(f"[*] Loading processed parquet splits from {train_path.parent}...")

    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)

    feature_names = [c for c in df_train.columns if c != target_col]

    X_train = df_train[feature_names].values
    y_train = df_train[target_col].values.astype(int)

    X_val = df_val[feature_names].values
    y_val = df_val[target_col].values.astype(int)

    X_test = df_test[feature_names].values
    y_test = df_test[target_col].values.astype(int)

    print(f"    Train split: {len(y_train):,} samples ({int(y_train.sum()):,} fraud, {100 * y_train.mean():.3f}%)")
    print(f"    Val split:   {len(y_val):,} samples ({int(y_val.sum()):,} fraud, {100 * y_val.mean():.3f}%)")
    print(f"    Test split:  {len(y_test):,} samples ({int(y_test.sum()):,} fraud, {100 * y_test.mean():.3f}%)")
    print(f"    Features:    {len(feature_names)} features")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def main() -> None:
    start_time = time.time()
    root = get_project_root()
    cfg = get_config()

    print("=" * 80)
    print(" FRAUD SHIELD -- PHASE 2: MODEL TRAINING & EXPERIMENTATION")
    print("=" * 80)

    # 1. Load data
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_processed_splits(root, cfg)

    # 2. Train models
    print("\n[*] Initializing Model Trainer...")
    trainer = ModelTrainer(config=cfg)

    print("\n[*] Training all enabled models...")
    train_results = trainer.train_all(X_train, y_train, X_val, y_val)

    # 3. Model Comparison Table
    val_reports = [report for _, report in train_results.values()]
    print("\n" + "=" * 80)
    print(" VALIDATION SET MODEL COMPARISON (Optimized Threshold)")
    print("=" * 80)
    print(build_comparison_table(val_reports))
    print("=" * 80)

    # 4. Select Winner
    print("\n[*] Selecting best model according to composite score & safety gates...")
    best_key, best_model, best_val_report = trainer.select_best_model(train_results)
    print(f"    Selected Winner: {best_val_report.model_name} ({best_key})")
    print(f"    Validation Optimal Threshold: {best_val_report.threshold:.4f}")
    print(f"    Validation F2-Score:          {best_val_report.f2_score:.4f}")
    print(f"    Validation Recall:            {best_val_report.recall:.4f}")
    print(f"    Validation Precision:         {best_val_report.precision:.4f}")
    print(f"    Validation PR-AUC:            {best_val_report.pr_auc:.4f}")
    print(f"    Validation ROC-AUC:           {best_val_report.roc_auc:.4f}")
    print(f"    Validation Composite Score:   {best_val_report.composite_score:.4f}")

    # 5. Evaluate on held-out test split (NO leakage, fixed threshold)
    print("\n[*] Evaluating winning model on held-out TEST set...")
    print(f"    Applying validation threshold ({best_val_report.threshold:.4f}) without re-tuning.")
    test_report = trainer.evaluate_test(
        model=best_model,
        model_key=best_key,
        X_test=X_test,
        y_test=y_test,
        threshold=best_val_report.threshold,
    )

    print("\n" + "=" * 80)
    print(f" FINAL TEST EVALUATION: {test_report.model_name}")
    print("=" * 80)
    print(f"  Test Samples:      {test_report.n_samples:,} (Fraud: {test_report.n_fraud:,})")
    print(f"  Applied Threshold: {test_report.threshold:.4f}")
    print(f"  F2-Score:          {test_report.f2_score:.4f}")
    print(f"  Recall:            {test_report.recall:.4f} ({test_report.tp}/{test_report.n_fraud} fraud caught)")
    print(f"  Precision:         {test_report.precision:.4f}")
    print(f"  F1-Score:          {test_report.f1_score:.4f}")
    print(f"  PR-AUC:            {test_report.pr_auc:.4f}")
    print(f"  ROC-AUC:           {test_report.roc_auc:.4f}")
    print(f"  Accuracy:          {test_report.accuracy:.4f}")
    print(f"  False Positives:   {test_report.fp:,}")
    print(f"  False Negatives:   {test_report.fn:,} (missed fraud)")
    print("=" * 80)

    # 6. Save comprehensive comparison report
    summary_report = {
        "execution_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": round(time.time() - start_time, 2),
        "selected_model": best_key,
        "selected_model_name": best_val_report.model_name,
        "validation_optimal_threshold": best_val_report.threshold,
        "validation_results": {
            k: rep.to_dict(include_curves=False)
            for k, (_, rep) in train_results.items()
        },
        "test_results": test_report.to_dict(include_curves=False),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
    }
    save_report(summary_report, "model_comparison.json")
    print("\n[OK] Model comparison and test report saved to artifacts/reports/model_comparison.json")
    print(f"[OK] Best model saved to artifacts/models/best_model.joblib")
    print(f"[OK] Total time elapsed: {time.time() - start_time:.2f}s\n")


if __name__ == "__main__":
    main()
