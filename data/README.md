# Fraud Shield — Data Directory
# ================================
# This directory stores raw and processed transaction data.
#
# IMPORTANT: Raw data files (creditcard.csv) are NOT committed to version control.
# Add data/raw/*.csv to your .gitignore file.

## Getting the Dataset

The project uses the **ULB Credit Card Fraud Detection dataset** from Kaggle.

### Dataset Details
| Property | Value |
|----------|-------|
| Source | Kaggle — ULB Machine Learning Group |
| URL | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| File | `creditcard.csv` |
| Size | ~150 MB |
| Rows | 284,807 transactions |
| Columns | 31 (Time, V1–V28, Amount, Class) |
| Fraud rate | ~0.173% (492 fraud / 284,315 legitimate) |

### Download Instructions

#### Option A — Kaggle Website (Recommended)
1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Click **Download** (requires free Kaggle account)
3. Unzip the downloaded file
4. Place `creditcard.csv` in this directory: `data/raw/creditcard.csv`

#### Option B — Kaggle CLI
```bash
# 1. Install kaggle CLI (already installed in this project)
pip install kaggle

# 2. Download your API token from https://www.kaggle.com/settings
#    Click "Create New API Token" → downloads kaggle.json
#    Place kaggle.json at: C:\Users\<YourName>\.kaggle\kaggle.json

# 3. Download the dataset
kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p data/raw/
```

### Column Descriptions

| Column | Type | Description |
|--------|------|-------------|
| `Time` | float64 | Seconds elapsed between this transaction and the first transaction in the dataset |
| `V1`–`V28` | float64 | Principal components from PCA transformation (anonymized for privacy) |
| `Amount` | float64 | Transaction amount in Euros |
| `Class` | int64 | Target label: **0** = legitimate, **1** = fraudulent |

> **Note**: V1–V28 are PCA-transformed. Original feature names are confidential and cannot be recovered. This is intentional by the dataset provider.

### Directory Structure
```
data/
├── raw/
│   └── creditcard.csv      ← Place downloaded file here
├── processed/
│   ├── creditcard_processed.parquet   ← Auto-generated after preprocessing
│   ├── train.parquet                  ← Auto-generated after splitting
│   ├── validation.parquet             ← Auto-generated after splitting
│   └── test.parquet                   ← Auto-generated after splitting
└── README.md               ← This file
```

### Academic Use Notice
This dataset is provided under the Open Database License (ODbL).
Attribution: Worldline and the Machine Learning Group of ULB (Université Libre de Bruxelles).

Reference paper:
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
> *Calibrating Probability with Undersampling for Unbalanced Classification.*
> 2015 IEEE Symposium Series on Computational Intelligence.
