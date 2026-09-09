# Fraud Shield — Environment Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.13.x | Tested on 3.13.5 |
| pip | 25.x+ | `python -m pip install --upgrade pip` |
| Disk space | ≥ 1 GB free | Dataset + models + artifacts |

---

## Step 1: Clone / Open the Project

Ensure you are in the project root directory:
```
c:\Users\Admin\Desktop\coding\antigrav\res projects\credit card fraud\
```

---

## Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Verify key packages installed:
```bash
python -c "import sklearn, xgboost, shap, imblearn, flask, yaml; print('All imports OK')"
```

---

## Step 3: Get the Dataset

### Option A — Manual Download (Recommended)
1. Go to: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Sign in / create a free Kaggle account
3. Click **Download** → extracts `creditcard.csv`
4. Place the file at: `data/raw/creditcard.csv`

### Option B — Kaggle CLI
```bash
# Step 1: Get API token
# → Go to https://www.kaggle.com/settings
# → Scroll to "API" section → "Create New API Token"
# → This downloads kaggle.json

# Step 2: Place credentials
# Windows: C:\Users\<YourName>\.kaggle\kaggle.json
# Make sure the file exists and is readable

# Step 3: Download
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p data/raw/
```

### Verify Dataset
```bash
python -c "
import pandas as pd
df = pd.read_csv('data/raw/creditcard.csv')
print(f'Rows: {len(df):,}')
print(f'Columns: {list(df.columns)}')
print(f'Fraud: {df.Class.sum()} ({df.Class.mean()*100:.3f}%)')
"
```

Expected output:
```
Rows: 284,807
Columns: ['Time', 'V1', ..., 'V28', 'Amount', 'Class']
Fraud: 492 (0.173%)
```

---

## Step 4: Train Models

```bash
python scripts/train.py
```

This will:
- Load and validate the dataset
- Run feature engineering
- Split data (60% train / 20% val / 20% test)
- Train all 4 models + dummy baseline
- Evaluate and compare on validation set
- Select best model
- Save models and pipeline to `artifacts/`
- Print evaluation report

Training typically takes 2–5 minutes on CPU for the full dataset.

---

## Step 5: Run the Dashboard

```bash
python run.py
```

Open your browser at: **http://localhost:5000**

---

## Step 6: Verify Everything Works

```bash
# Run unit tests
pytest tests/ -v

# Check artifacts were created
python -c "
import pathlib
artifacts = list(pathlib.Path('artifacts').rglob('*'))
print(f'Artifacts: {len(artifacts)} files')
for f in artifacts:
    if f.is_file():
        print(f'  {f}')
"
```

---

## Configuration

All configurable parameters are in `configs/config.yaml`. Key settings:

| Setting | Location | Default |
|---------|----------|---------|
| Risk thresholds | `risk_engine.thresholds` | LOW≤30, MED≤70, HIGH>70 |
| Model selection metric weights | `model_selection.metric_weights` | F2=0.35, PRAUC=0.30 |
| Train/val/test split | `splitting.*` | 60/20/20 |
| SMOTE enabled | `imbalance.use_smote` | false |
| Threshold optimization | `threshold.optimize` | true |

To change any parameter, edit `configs/config.yaml` and re-run `scripts/train.py`.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: creditcard.csv` | See Step 3 above |
| `DatasetValidationError` | Check the CSV has 31 columns and a `Class` column |
| Flask port in use | Change `app.port` in `configs/config.yaml` |
| SHAP slow on large data | SHAP uses a sample of 500 rows for background — configurable |
| `numpy` version conflicts | Ensure numpy==2.2.6; SHAP 0.52+ supports numpy 2.x |

---

*Setup Guide v1.0 | Fraud Shield | Code Masala*
