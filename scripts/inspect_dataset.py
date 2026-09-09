"""Deep dataset inspection — generates real statistics before any preprocessing code is written."""
import sys, pathlib
ROOT = pathlib.Path(r"c:\Users\Admin\Desktop\coding\antigrav\res projects\credit card fraud")
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

df = pd.read_csv(ROOT / "data/raw/creditcard.csv")

print("=" * 70)
print("DATASET DEEP INSPECTION")
print("=" * 70)

# 1. Basic shape
print(f"\n[1] Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

# 2. Data types
print("\n[2] Data Types:")
for col, dtype in df.dtypes.items():
    print(f"    {col:<8} {str(dtype):<12}")

# 3. Missing values
missing = df.isnull().sum()
print(f"\n[3] Missing Values: {missing.sum()} total")
if missing.sum() > 0:
    print(missing[missing > 0])

# 4. Duplicates
dups = df.duplicated().sum()
print(f"\n[4] Duplicate Rows: {dups:,}")

# 5. Target distribution
print("\n[5] Target Column (Class):")
vc = df["Class"].value_counts().sort_index()
for k, v in vc.items():
    label = "Fraud" if k == 1 else "Legit"
    pct = v / len(df) * 100
    print(f"    Class={k} ({label}): {v:>7,}  ({pct:.4f}%)")
print(f"    Imbalance ratio: 1:{int(vc[0]/vc[1])}")

# 6. Amount stats
print("\n[6] Amount Statistics:")
amt = df["Amount"].describe()
for stat, val in amt.items():
    print(f"    {stat:<10} {val:>12.4f}")
print(f"    skewness:  {df['Amount'].skew():>12.4f}")
print(f"    kurtosis:  {df['Amount'].kurt():>12.4f}")
print(f"    zeros:     {(df['Amount'] == 0).sum():>12,}")

# 7. Time stats
print("\n[7] Time Statistics:")
t = df["Time"].describe()
for stat, val in t.items():
    print(f"    {stat:<10} {val:>12.4f}")
print(f"    max hours: {df['Time'].max()/3600:>12.2f}h  (dataset spans {df['Time'].max()/3600:.1f} hours)")
hours = (df["Time"] % 86400) / 3600
print(f"    night txns (0-6h): {(hours < 6).sum():>8,}  ({(hours < 6).mean()*100:.2f}%)")

# 8. V features range
print("\n[8] PCA Feature (V1-V28) Ranges:")
v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
v_stats = df[v_cols].agg(["min","max","mean","std"])
print(f"    {'Feature':<8} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
for col in v_cols[:5]:
    row = v_stats[col]
    print(f"    {col:<8} {row['min']:>10.3f} {row['max']:>10.3f} {row['mean']:>10.4f} {row['std']:>10.4f}")
print(f"    ... (showing first 5 of 28)")
print(f"    Global V-feature range: [{df[v_cols].min().min():.3f}, {df[v_cols].max().max():.3f}]")

# 9. By-class stats for Amount
print("\n[9] Amount by Class:")
for cls in [0, 1]:
    sub = df[df["Class"] == cls]["Amount"]
    print(f"    Class={cls}: mean={sub.mean():.2f}  median={sub.median():.2f}  max={sub.max():.2f}  std={sub.std():.2f}")

# 10. Time by class
print("\n[10] Time by Class:")
for cls in [0, 1]:
    sub = df[df["Class"] == cls]["Time"]
    print(f"     Class={cls}: mean={sub.mean():.0f}s  min={sub.min():.0f}  max={sub.max():.0f}")

# 11. Correlation of features with target
print("\n[11] Top 10 Features Correlated with Class (absolute Pearson):")
corr = df.corr(numeric_only=True)["Class"].drop("Class").abs().sort_values(ascending=False)
for feat, val in corr.head(10).items():
    print(f"    {feat:<8} {val:.6f}")

# 12. Negative V features for fraud
print("\n[12] V-feature means: Fraud vs Legitimate (top 5 by difference):")
fraud = df[df["Class"]==1][v_cols].mean()
legit = df[df["Class"]==0][v_cols].mean()
diff = (fraud - legit).abs().sort_values(ascending=False)
print(f"    {'Feature':<8} {'Fraud Mean':>12} {'Legit Mean':>12} {'Abs Diff':>12}")
for col in diff.head(5).index:
    print(f"    {col:<8} {fraud[col]:>12.4f} {legit[col]:>12.4f} {diff[col]:>12.4f}")

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
