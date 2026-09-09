# Fraud Shield — Data Dictionary
## Dataset: ULB Credit Card Fraud Detection

> **Status**: Template — will be populated with exact statistics after dataset inspection.
> Run `python scripts/train.py` (or the EDA notebook) to auto-generate real statistics.

---

## Dataset Overview

| Property | Value (Expected) | Value (Verified) |
|----------|-----------------|-----------------|
| Source | Kaggle — ULB MLG | Kaggle — ULB MLG |
| Filename | `creditcard.csv` | `creditcard.csv` |
| Total rows | 284,807 | **284,807** |
| Total columns | 31 | **31** |
| Missing values | 0 | **0** |
| Duplicate rows | 1,081 | **1,081** |
| Fraud transactions | 492 | **492** |
| Legitimate transactions | 284,315 | **284,315** |
| Fraud rate | 0.173% | **0.1727%** |
| Imbalance ratio | ~1:578 | **1:577** |

---

## Column Descriptions

### Time
| Property | Value |
|----------|-------|
| Type | float64 |
| Description | Seconds elapsed between this transaction and the first transaction in the dataset |
| Range | 0 – 172,792 (approx. 48 hours of data) |
| Role | Feature → engineered to `hour_of_day` and `is_night` |
| Notes | Does NOT represent a real timestamp; used to extract time-of-day patterns |

### V1 – V28 (28 columns)
| Property | Value |
|----------|-------|
| Type | float64 |
| Description | Principal components from PCA transformation applied by ULB to anonymize cardholder data |
| Range | Varies per component (roughly –30 to +30) |
| Role | Primary features — passed through the preprocessing pipeline without rescaling |
| Notes | Original feature names are confidential and cannot be recovered. PCA transformation was already applied. |

### Amount
| Property | Value |
|----------|-------|
| Type | float64 |
| Description | Transaction amount in Euros |
| Range | 0.00 – 25,691.16 (approx.) |
| Typical value | ~88 EUR (median) |
| Role | Feature → scaled with StandardScaler + log1p transformation |
| Notes | Highly right-skewed; log transformation reduces outlier impact |

### Class (Target)
| Property | Value |
|----------|-------|
| Type | int64 |
| Values | 0 = Legitimate, 1 = Fraudulent |
| Distribution | 0: 284,315 (99.827%), 1: 492 (0.173%) |
| Role | **Target variable** — what all models predict |

---

## Engineered Features

These features are **created** by `src/features/engineering.py` and are NOT in the raw CSV:

| Feature | Formula | Description |
|---------|---------|-------------|
| `log_amount` | `log1p(Amount)` | Log-transformed amount; reduces skewness and outlier influence |
| `hour_of_day` | `(Time % 86400) / 3600` | Hour within a 24h cycle extracted from Time |
| `is_night` | `1 if hour_of_day < 6 else 0` | Binary flag: transactions between midnight and 6am |

---

## Class Imbalance Analysis

The target column is severely imbalanced:

```
Legitimate (Class=0): 284,315  ████████████████████████████████████  99.827%
Fraud      (Class=1):     492  ▌                                      0.173%
```

**Implication**: A model that always predicts "legitimate" achieves 99.827% accuracy — making accuracy a meaningless metric. This is why we use Precision, Recall, F2-score, PR-AUC, and ROC-AUC.

**Strategies employed**:
1. `class_weight='balanced'` — adjusts loss function to penalize minority class misclassification more heavily
2. Optional SMOTE — synthetic oversampling of minority class in training set only

---

## Feature Statistics

*(Auto-populated after dataset inspection — placeholder shown)*

| Feature | Min | Max | Mean | Std | Fraud Mean | Legit Mean |
|---------|-----|-----|------|-----|-----------|-----------|
| Time | 0 | ~172792 | *TBD* | *TBD* | *TBD* | *TBD* |
| Amount | 0 | ~25691 | *TBD* | *TBD* | *TBD* | *TBD* |
| V1 | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* | *TBD* |
| ... | ... | ... | ... | ... | ... | ... |

---

## Known Dataset Limitations

1. **Anonymized features**: V1–V28 cannot be interpreted in business terms — SHAP explanations reference these by feature name only
2. **Historical snapshot**: Data covers ~48 hours of European cardholder transactions — not a continuous stream
3. **No merchant/location data**: The dataset does not include merchant category, geolocation, or cardholder ID
4. **No temporal context per cardholder**: We cannot compute per-customer velocity features from this dataset alone
5. **Synthetic limitation**: A real fraud system would have richer contextual features; this dataset is intentionally limited for privacy

---

*Data Dictionary v1.0 | Fraud Shield | Code Masala | To be finalized after EDA*
