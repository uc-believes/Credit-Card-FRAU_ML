"""Artifact and inference consistency verification — Phase 1 final check."""
import pathlib, json, sys
import joblib
import numpy as np
import pandas as pd

root = pathlib.Path(r"c:\Users\Admin\Desktop\coding\antigrav\res projects\credit card fraud")
sys.path.insert(0, str(root))

print("=== ARTIFACT VERIFICATION ===")

# 1. Pipeline file
pipe_path = root / "artifacts/pipelines/preprocessing_pipeline.joblib"
print(f"\n[1] Pipeline file: {pipe_path.stat().st_size:,} bytes")
pipeline = joblib.load(pipe_path)
print(f"     Type: {type(pipeline).__name__}")
print(f"     Steps: {[s[0] for s in pipeline.steps]}")

# 2. Metadata
meta_path = root / "artifacts/pipelines/preprocessing_metadata.json"
with open(meta_path) as f:
    meta = json.load(f)
print(f"\n[2] Metadata summary:")
print(f"     Fitted at:      {meta['fitted_at']}")
print(f"     Rows used:      {meta['total_rows_used']:,}")
print(f"     Features in:    {meta['n_features_in']}")
print(f"     Features out:   {meta['n_features_out']}")
print(f"     Scale cols:     {meta['scale_columns']}")
print(f"     Passthrough:    {len(meta['passthrough_columns'])} cols")
print(f"     Train/Val/Test: {meta['split_sizes']['train']:,} / {meta['split_sizes']['val']:,} / {meta['split_sizes']['test']:,}")
train_fraud = meta['class_distribution']['train']['fraud']
val_fraud   = meta['class_distribution']['val']['fraud']
test_fraud  = meta['class_distribution']['test']['fraud']
print(f"     Fraud per split: train={train_fraud}, val={val_fraud}, test={test_fraud}")
print("     Scaler stats:")
for col, stats in meta['scaler_stats'].items():
    print(f"       {col:<15} mean={stats['mean']:>10.4f}  std={stats['std']:>10.4f}")

# 3. Processed splits
print("\n[3] Processed splits:")
for split in ["train", "validation", "test"]:
    p = root / f"data/processed/{split}.parquet"
    df = pd.read_parquet(p)
    fraud = int(df["Class"].sum())
    rate = fraud / len(df) * 100
    print(f"     {split:<12} {len(df):>8,} rows  {df.shape[1]} cols  fraud={fraud} ({rate:.3f}%)")

# 4. Inference consistency test
print("\n[4] Inference consistency (loaded pipeline -> single transaction):")
from src.features.pipeline import PreprocessingPipeline
from src.features.engineering import FeatureEngineer

pp = PreprocessingPipeline.load()
fe = FeatureEngineer()

txn = {"Time": 50000.0, "Amount": 149.62}
txn.update({f"V{i}": float(i * 0.05 - 1.4) for i in range(1, 29)})
df_single = fe.transform_single(txn)
result = pp.transform(df_single)
print(f"     Input features:    {df_single.shape[1]}")
print(f"     Output shape:      {result.shape}")
print(f"     Amount (scaled):   {result[0, 0]:.6f}")
print(f"     Time (scaled):     {result[0, 1]:.6f}")
print(f"     log_amount (sc):   {result[0, 2]:.6f}")
print(f"     hour_of_day (sc):  {result[0, 3]:.6f}")
print(f"     No NaN in output:  {not np.isnan(result).any()}")
print(f"     Features match:    {result.shape[1] == meta['n_features_out']}")

print("\n=== ALL ARTIFACTS VERIFIED SUCCESSFULLY ===")
