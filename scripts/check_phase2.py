"""Phase 2 pre-flight check."""
import pathlib, sys
root = pathlib.Path(r"c:\Users\Admin\Desktop\coding\antigrav\res projects\credit card fraud")
sys.path.insert(0, str(root))

import pandas as pd

print("=== LOCAL STATE CHECK ===\n")

for split in ["train", "validation", "test"]:
    p = root / f"data/processed/{split}.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        fraud = int(df["Class"].sum())
        print(f"  {split:<12} {len(df):>8,} rows  fraud={fraud}  ({p.stat().st_size//1024} KB)")
    else:
        print(f"  {split:<12} MISSING")

pp = root / "artifacts/pipelines/preprocessing_pipeline.joblib"
print(f"\n  Pipeline:   {'EXISTS (' + str(pp.stat().st_size) + ' bytes)' if pp.exists() else 'MISSING'}")

raw = root / "data/raw/creditcard.csv"
print(f"  Raw CSV:    {'EXISTS' if raw.exists() else 'MISSING'}")

from src.utils.config import get_config
cfg = get_config()
print(f"\n  Config:     {cfg['project']['name']} v{cfg['project']['version']}")
print(f"  Models in config: {list(cfg['models'].keys())}")
print(f"  Random state: {cfg['splitting']['random_state']}")
