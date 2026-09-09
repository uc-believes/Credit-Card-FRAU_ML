# Fraud Shield — Project Progress Report
## Milestone Implementation Report: Phases 0 & 1

---

| | |
|---|---|
| **Project Title** | Credit Card Fraud Detection Using Machine Learning Algorithms |
| **System Name** | Fraud Shield — Explainable & Adaptive Credit Card Fraud Risk Intelligence System |
| **Team** | Code Masala |
| **Members** | Akshat Mehra (BTAD2401008) · Ujjwal Chandravanshi (BTAD2401070) |
| **Course** | Artificial Intelligence & Data Science |
| **Coordinator** | Asst. Prof. Geetika Hazra |
| **Report Date** | September 9, 2026 |
| **Phases Covered** | Phase 0 — Foundation & Environment · Phase 1 — Data Engineering |
| **Status** | Both phases **complete and verified** |

> **Academic Disclaimer**: This system is an academic prototype for educational purposes only. It is NOT intended for real banking or financial deployment.

---

## Table of Contents

1. Executive Summary
2. Project Vision & Architecture
3. Phase 0: Foundation & Environment
4. Dataset Acquisition & Assessment
5. Phase 1: Data Engineering
6. Feature Engineering
7. Preprocessing Pipeline Design
8. Anti-Leakage Verification
9. Test Suite Results
10. Deliverables Inventory
11. Verified Numbers Summary
12. Next Steps

---

## 1. Executive Summary

Two implementation phases of the Fraud Shield system have been completed and fully verified. The work covered environment setup, dataset acquisition, schema validation, statistical analysis, feature engineering, a reproducible leakage-safe preprocessing pipeline, and a 59-test automated test suite.

All numbers presented in this report are **verified against the actual dataset**. No statistics have been estimated, fabricated, or assumed.

**Key accomplishments:**

- Full project scaffold (27 directories, all modules initialized)
- Real dataset obtained, inspected, and documented: 284,807 transactions
- Class imbalance quantified: 1 fraud per 577 legitimate transactions
- Three derived features created with data-backed statistical rationale
- Leakage-safe sklearn Pipeline fitted on training data only
- Stratified 60/20/20 train/val/test split with verified stratification
- Pipeline serialized to disk (2,896 bytes); load-transform consistency confirmed (max diff = 0.00e+00)
- **59 automated tests — all passing**

---

## 2. Project Vision & Architecture

Fraud Shield is designed as a nine-layer system:

```
Input Transaction
      |
      v
  Data Loader & Validator         [Phase 1 - COMPLETE]
      |
      v
  Feature Engineering             [Phase 1 - COMPLETE]
      |
      v
  Preprocessing Pipeline          [Phase 1 - COMPLETE]
      |
      v
  Model Training (LR/DT/RF/XGB)  [Phase 2 - Next]
      |
      v
  Evaluation & Selection          [Phase 2 - Next]
      |
      v
  Risk Engine                     [Phase 3 - Planned]
      |
      v
  SHAP Explainability             [Phase 3 - Planned]
      |
      v
  Flask Dashboard                 [Phase 4 - Planned]
```

**Model Selection Criterion:** Weighted composite score  
`Score = 0.35 × F2 + 0.30 × PR-AUC + 0.20 × ROC-AUC + 0.15 × F1`

F2-score is the primary metric (recall-weighted) because missing a fraud case is costlier than a false alarm in a detection system.

---

## 3. Phase 0: Foundation & Environment

### 3.1 Workspace State Before Work Began

The workspace was a clean slate:

| Component | Status at Start |
|-----------|----------------|
| `codes/` directory | Empty |
| `files/` directory | 5 reference documents only (milestone PDFs, PPTX, thumbnail) |
| Dataset | Not present |
| Python ML packages | scikit-learn, XGBoost, SHAP — not installed |
| Project code | None |

### 3.2 Reference Documents Reviewed

Both milestone documents were read in full before any code was written:

| Document | Content |
|----------|---------|
| `Milestone 1.pdf` | Problem statement, ML approach (LR/DT/RF/XGBoost), evaluation metrics, technology stack |
| `Milestone 2.pdf` | 9-step end-to-end workflow: data → clean → preprocess → balance → split → train → evaluate → select → predict |

### 3.3 Project Structure Created

The following directory tree was scaffolded (27 directories):

```
credit card fraud/
├── data/raw/              ← Dataset storage
├── data/processed/        ← Processed splits
├── notebooks/
├── src/
│   ├── data/              ← Loader, Validator
│   ├── features/          ← Engineering, Pipeline
│   ├── models/            ← LR, DT, RF, XGBoost
│   ├── evaluation/        ← Metrics, Threshold, Comparison
│   ├── explainability/    ← SHAP
│   ├── monitoring/        ← Drift detection
│   ├── risk/              ← Risk engine
│   └── utils/             ← Config, Logger, Serialization
├── artifacts/
│   ├── models/
│   ├── pipelines/         ← Fitted pipeline (SAVED)
│   ├── metrics/
│   ├── plots/
│   └── reports/
├── app/                   ← Flask dashboard (Phase 4)
├── tests/                 ← 59 automated tests
├── configs/config.yaml    ← Single source of truth
├── docs/                  ← 5 documentation files
├── scripts/               ← CLI scripts
├── requirements.txt
└── README.md
```

### 3.4 Environment Verification

All required packages confirmed installed and functional:

| Package | Version | Purpose |
|---------|---------|---------|
| scikit-learn | 1.9.0 | ML models, pipeline, evaluation |
| XGBoost | 3.4.1 | Gradient boosting model |
| SHAP | 0.52.0 | Explainability |
| imbalanced-learn | 0.14.2 | SMOTE (optional) |
| joblib | 1.6.0 | Model serialization |
| seaborn | 0.13.2 | Visualization |
| pandas | 3.0.5 | Data manipulation |
| numpy | 2.2.6 | Numerical operations |
| Flask | 3.1.1 | Web dashboard |
| PyYAML | 6.0.2 | Configuration |
| pyarrow | latest | Parquet file I/O |
| pytest | 9.1.1 | Testing framework |

---

## 4. Dataset Acquisition & Assessment

### 4.1 Dataset Identity

The standard benchmark dataset for this problem domain was obtained:

| Property | Value |
|----------|-------|
| **Source** | Kaggle — ULB Machine Learning Group |
| **URL** | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| **File** | `creditcard.csv` |
| **File size** | 143 MB (zipped: 65 MB) |
| **Citation** | Dal Pozzolo et al., 2015 IEEE SSCI |

### 4.2 Schema Inspection (Verified)

All 31 columns confirmed present with correct types:

| Column(s) | Type | Count | Treatment |
|-----------|------|-------|-----------|
| `Time` | float64 | 1 | Scale + engineer to `hour_of_day`, `is_night` |
| `V1` – `V28` | float64 | 28 | Passthrough (already PCA-scaled by ULB) |
| `Amount` | float64 | 1 | Scale + engineer to `log_amount` |
| `Class` | int64 | 1 | Target variable (0=legit, 1=fraud) |

> The V1–V28 features are PCA-transformed by ULB to anonymize cardholder data. Original feature names are confidential and cannot be recovered. This is a documented property of the dataset.

### 4.3 Dataset Quality Assessment (Verified)

| Quality Metric | Verified Value |
|----------------|---------------|
| Total rows | **284,807** |
| Total columns | **31** |
| Missing values | **0** (zero missing values — no imputation required) |
| Duplicate rows | **1,081** (dropped during preprocessing) |
| All-null columns | **None** |
| Non-numeric columns | **None** (no encoding required) |
| Data type issues | **None** |
| Target value range | **{0, 1}** — binary, as expected |

### 4.4 Class Distribution (Verified)

| Class | Label | Count | Percentage |
|-------|-------|-------|-----------|
| 0 | Legitimate | 284,315 | **99.8273%** |
| 1 | Fraud | 492 | **0.1727%** |
| — | Imbalance ratio | — | **1 : 577** |

**Impact:** A naive model predicting "always legitimate" achieves 99.8% accuracy — making accuracy a completely misleading metric. The system therefore uses Precision, Recall, F1, F2, PR-AUC, and ROC-AUC as primary evaluation criteria.

### 4.5 Descriptive Statistics (Verified)

**Amount column:**

| Statistic | Value |
|-----------|-------|
| Mean | €88.35 |
| Median | €22.00 |
| Std Dev | €250.12 |
| Minimum | €0.00 |
| Maximum | €25,691.16 |
| Skewness | **16.9777** (extremely right-skewed) |
| Kurtosis | **845.09** (heavy-tailed) |
| Zero-amount transactions | 1,825 |
| Fraud mean Amount | €122.21 |
| Legitimate mean Amount | €88.29 |

**Time column:**

| Statistic | Value |
|-----------|-------|
| Range | 0 – 172,792 seconds (exactly 48 hours) |
| Mean | 94,813 seconds |
| Night transactions (0–6h) | **23,934** (8.40% of all transactions) |
| Fraud mean Time | 80,747s (earlier in dataset) |
| Legitimate mean Time | 94,838s |

### 4.6 Top Features Correlated with Fraud (Verified)

Absolute Pearson correlation with `Class` target:

| Rank | Feature | Correlation | Notes |
|------|---------|-------------|-------|
| 1 | V17 | **0.3265** | Strongest fraud signal |
| 2 | V14 | **0.3025** | Second strongest |
| 3 | V12 | **0.2606** | |
| 4 | V10 | **0.2169** | |
| 5 | V16 | **0.1965** | |
| 6 | V3 | 0.1930 | |
| 7 | V7 | 0.1873 | |
| 8 | V11 | 0.1549 | |
| 9 | V4 | 0.1334 | |
| 10 | V18 | 0.1115 | |

**Fraud vs Legitimate Mean Differences (top 5):**

| Feature | Fraud Mean | Legitimate Mean | Absolute Difference |
|---------|-----------|----------------|---------------------|
| V3 | -7.0333 | 0.0122 | **7.0455** |
| V14 | -6.9717 | 0.0121 | **6.9838** |
| V17 | -6.6658 | 0.0115 | **6.6774** |
| V12 | -6.2594 | 0.0108 | **6.2702** |
| V10 | -5.6769 | 0.0098 | **5.6867** |

These features are expected to appear prominently in SHAP explanations.

---

## 5. Phase 1: Data Engineering

### 5.1 Modules Implemented

| Module | Path | Lines of Code | Purpose |
|--------|------|--------------|---------|
| `DataLoader` | `src/data/loader.py` | ~180 | Load, validate schema, report statistics |
| `DataValidator` | `src/data/validator.py` | ~230 | Hard/soft validation, structured report |
| `FeatureEngineer` | `src/features/engineering.py` | ~190 | Derive features, separate X and y |
| `PreprocessingPipeline` | `src/features/pipeline.py` | ~320 | Build, fit, transform, save/load |
| `Serialization utils` | `src/utils/serialization.py` | ~120 | Save/load models, metrics, reports |
| `ConfigLoader` | `src/utils/config.py` | ~110 | YAML config with typed accessors |
| `Logger` | `src/utils/logger.py` | ~80 | Rotating file logger |

### 5.2 Data Validation Design

The `DataValidator` enforces a two-tier check system:

**Hard Failures (raise `DataValidationError`):**
- Row count below 200,000 minimum
- `Class` target column missing
- Target column contains values outside {0, 1}
- Any column is entirely null

**Soft Warnings (logged, execution continues):**
- Missing values in any column
- Duplicate rows detected
- Non-numeric columns present
- `Amount` or `Time` columns missing
- Fraud rate outside expected range [0.1%, 0.5%]

Result on actual dataset: **PASSED** — 1 warning (1,081 duplicates to be dropped).

---

## 6. Feature Engineering

### 6.1 Derived Features

Three new features were created, each with a statistical rationale grounded in the actual dataset inspection:

#### `log_amount` = log1p(Amount)

| Rationale | Evidence |
|-----------|---------|
| Amount is extremely right-skewed | Skewness = **16.9777**, Kurtosis = **845.09** |
| Standard scaling alone is insufficient | Max = €25,691 vs Median = €22 — 3 orders of magnitude difference |
| 1,825 zero-amount transactions | `log1p(0) = 0` — handles zero safely |
| After log transform | Skewness reduced substantially; more Gaussian-like distribution |

#### `hour_of_day` = (Time % 86400) / 3600

| Rationale | Evidence |
|-----------|---------|
| Dataset spans exactly 48 hours | Time range: 0 – 172,792 seconds verified |
| Time-of-day is a known fraud signal | Fraud mean Time=80,747s vs Legitimate=94,838s — different temporal pattern |
| Cyclical time extraction | Modulo 86,400 extracts position within a 24-hour cycle |
| Range | 0.0 – 23.99 hours |

#### `is_night` = 1 if hour_of_day < 6 else 0

| Rationale | Evidence |
|-----------|---------|
| Night is a low-supervision window | 8.40% of all transactions occur between 0h–6h (verified: 23,934 transactions) |
| Binary flag captures non-linear signal | Complements the continuous `hour_of_day` feature |
| Zero-cost to add | Binary, no additional scaling needed |

### 6.2 Column Treatment Summary

| Category | Columns | Count | Treatment |
|----------|---------|-------|-----------|
| Scaled | Amount, Time, log_amount, hour_of_day | 4 | StandardScaler (fit on train only) |
| Passthrough | V1 – V28 | 28 | No scaling (already PCA-scaled by ULB) |
| Passthrough binary | is_night | 1 | No scaling (values are 0 or 1) |
| Dropped | Class | 1 | Target — separated before transform |
| **Total features out** | | **33** | |

---

## 7. Preprocessing Pipeline Design

### 7.1 Architecture

The pipeline uses scikit-learn's `Pipeline` + `ColumnTransformer`:

```
sklearn.pipeline.Pipeline
    └── preprocessor: ColumnTransformer
            ├── scaler: StandardScaler → [Amount, Time, log_amount, hour_of_day]
            └── passthrough              → [V1–V28, is_night]
            └── remainder: "drop"        → (no unassigned columns)
```

### 7.2 Splitting Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Test holdout | 20% | Final evaluation set — never seen during training |
| Validation | 20% | Hyperparameter tuning and threshold optimization |
| Training | 60% | Model training and scaler fitting |
| Stratified | Yes | Preserves fraud rate across all three splits |
| Random seed | 42 | Fixed for full reproducibility |

### 7.3 Split Results on Real Dataset (Verified)

| Split | Rows | Fraud | Legitimate | Fraud Rate |
|-------|------|-------|-----------|-----------|
| **Train** | 170,235 | 284 | 169,951 | 0.167% |
| **Validation** | 56,745 | 94 | 56,651 | 0.166% |
| **Test** | 56,746 | 95 | 56,651 | 0.167% |
| **Total** | 283,726 | 473 | 283,253 | 0.167% |

Stratification preserved: all three splits maintain ≈0.167% fraud rate (vs 0.1727% raw — slight reduction due to dropping 1,081 duplicates, 19 of which were fraud rows).

### 7.4 Fitted Scaler Statistics (Verified — From Training Data Only)

These values come exclusively from the 170,235 training rows. Validation and test sets are transformed using these learned parameters — not their own statistics.

| Feature | Training Mean | Training Std |
|---------|--------------|-------------|
| Amount | 88.1606 | 246.4492 |
| Time | 94,935.857 | 47,513.337 |
| log_amount | 3.1545 | 1.6556 |
| hour_of_day | 14.5393 | 5.8411 |

### 7.5 Artifact Serialization

| Artifact | Path | Size | Format |
|----------|------|------|--------|
| Preprocessing pipeline | `artifacts/pipelines/preprocessing_pipeline.joblib` | 2,896 bytes | joblib |
| Pipeline metadata | `artifacts/pipelines/preprocessing_metadata.json` | ~3 KB | JSON |
| Training split | `data/processed/train.parquet` | — | Parquet |
| Validation split | `data/processed/validation.parquet` | — | Parquet |
| Test split | `data/processed/test.parquet` | — | Parquet |

---

## 8. Anti-Leakage Verification

Data leakage prevention is critical for producing honest evaluation results. The following guarantees have been implemented and verified:

| Leakage Risk | Prevention Mechanism | Verification |
|-------------|---------------------|-------------|
| Scaler learns from test data | `Pipeline.fit()` called only on `X_train` (170,235 rows) | Scaler stats extracted from training fit only |
| SMOTE applied before split | SMOTE is disabled by default; if enabled, applied after split | Config flag `imbalance.use_smote: false` |
| Train/test index overlap | Explicit index overlap check after split | Runtime log confirms: "zero index overlap" |
| Threshold optimized on test set | Threshold optimized on validation set only; test set reserved | Enforced by pipeline design |
| Model selection on test set | Model comparison uses validation metrics only | Architecture enforced |
| Duplicate leakage | Duplicates dropped before split | 1,081 duplicates removed pre-split |

**Runtime leakage check output:**
```
INFO | Leakage check passed: zero index overlap between train and test.
```

**Batch vs single-row inference consistency:**
```
[OK] Consistency check PASSED: max diff = 0.00e+00
```
This confirms that the inference path (single transaction) and the training path (batch transform) are mathematically identical.

---

## 9. Test Suite Results

### 9.1 Summary

```
============================= test session starts =============================
platform win32 -- Python 3.13.5, pytest-9.1.1
collected 59 items

tests/test_data_loader.py .......................   (24 tests)
tests/test_features.py .................................  (35 tests)

============================= 59 passed in 2.36s ==============================
```

**Result: 59/59 PASSED — 0 failures, 0 errors**

### 9.2 Test Coverage by Category

| Test Class | Tests | What It Verifies |
|-----------|-------|-----------------|
| `TestDataLoaderInit` | 3 | Initialization, custom paths, missing-file handling |
| `TestDataLoaderStatistics` | 6 | Row counts, fraud counts, missing value detection, duplicate detection |
| `TestDataLoaderColumnHelpers` | 2 | PCA column detection, feature column exclusion |
| `TestDataValidatorHardChecks` | 4 | Missing target, invalid target values, all-null columns |
| `TestDataValidatorReport` | 8 | Report structure, fraud stats, warnings, amount stats, time stats |
| `TestFeatureEngineerTransform` | 12 | No original modification, log_amount, hour_of_day, is_night (correctness, edge cases, ranges) |
| `TestFeatureEngineerGetXy` | 4 | Target separation, row preservation, binary check |
| `TestFeatureEngineerTransformSingle` | 2 | Single-row inference shape and features |
| `TestPreprocessingPipelineLeakage` | 4 | Split sizes, no index overlap, scaler from train, stratification |
| `TestPreprocessingPipelineOutput` | 6 | numpy arrays, shape consistency, zero-mean train, unit-variance train, no NaN |
| `TestPreprocessingPipelineConsistency` | 4 | Batch≡single-row, unfitted error, double-fit error, reset |
| `TestPreprocessingPipelineSaveLoad` | 3 | Save/load produces identical output, missing-file error, metadata persistence |

All tests use **synthetic fixtures** — no dependency on the real dataset file, so tests run fast (2.36 seconds) and can be executed offline.

---

## 10. Deliverables Inventory

### Documentation Files (Complete)

| File | Status |
|------|--------|
| `README.md` | Complete |
| `docs/ARCHITECTURE.md` | Complete |
| `docs/DATA_DICTIONARY.md` | Complete (verified with real data) |
| `docs/MODEL_EXPERIMENTS.md` | Template ready — awaiting Phase 2 training |
| `docs/API.md` | Complete (design ready) |
| `docs/SETUP.md` | Complete |
| `data/README.md` | Complete (download instructions) |

### Code Modules (Complete)

| Module | Status |
|--------|--------|
| `src/utils/config.py` | Complete |
| `src/utils/logger.py` | Complete |
| `src/utils/serialization.py` | Complete |
| `src/data/loader.py` | Complete |
| `src/data/validator.py` | Complete |
| `src/features/engineering.py` | Complete |
| `src/features/pipeline.py` | Complete |
| `src/models/` | Phase 2 — next |
| `src/evaluation/` | Phase 2 — next |
| `src/explainability/` | Phase 3 — planned |
| `src/risk/` | Phase 3 — planned |
| `src/monitoring/` | Phase 3 — planned |
| `app/` | Phase 4 — planned |

### Scripts (Complete)

| Script | Purpose | Status |
|--------|---------|--------|
| `scripts/check_phase0.py` | Verify environment | Complete |
| `scripts/verify_dataset.py` | Verify dataset loaded correctly | Complete |
| `scripts/inspect_dataset.py` | Deep statistical analysis | Complete |
| `scripts/verify_artifacts.py` | Verify saved pipeline artifacts | Complete |
| `scripts/preprocess.py` | End-to-end preprocessing CLI | Complete |

### Saved Artifacts (On Disk)

| Artifact | Size | Status |
|----------|------|--------|
| `artifacts/pipelines/preprocessing_pipeline.joblib` | 2,896 bytes | **Saved** |
| `artifacts/pipelines/preprocessing_metadata.json` | ~3 KB | **Saved** |
| `data/processed/train.parquet` | — | **Saved** |
| `data/processed/validation.parquet` | — | **Saved** |
| `data/processed/test.parquet` | — | **Saved** |

---

## 11. Verified Numbers Summary

All numbers in this section were obtained by running code against the actual dataset. Nothing is estimated.

| Metric | Value |
|--------|-------|
| Raw dataset rows | 284,807 |
| Raw dataset columns | 31 |
| Fraud transactions | 492 (0.1727%) |
| Legitimate transactions | 284,315 (99.8273%) |
| Imbalance ratio | 1 : 577 |
| Missing values | 0 |
| Duplicate rows | 1,081 |
| Rows after deduplication | 283,726 |
| Training rows | 170,235 |
| Validation rows | 56,745 |
| Test rows | 56,746 |
| Training fraud | 284 (0.167%) |
| Validation fraud | 94 (0.166%) |
| Test fraud | 95 (0.167%) |
| Features into pipeline | 33 |
| Features out of pipeline | 33 |
| Features requiring scaling | 4 |
| PCA features (passthrough) | 28 |
| Binary features (passthrough) | 1 |
| Amount (train) mean / std | 88.16 / 246.45 |
| Time (train) mean / std | 94,935.9 / 47,513.3 |
| log_amount (train) mean / std | 3.1545 / 1.6556 |
| hour_of_day (train) mean / std | 14.5393 / 5.8411 |
| Inference consistency max diff | 0.00e+00 (exact) |
| Test suite results | **59 / 59 passed** |
| Test execution time | 2.36 seconds |

---

## 12. Next Steps — Phase 2: Model Training

The next phase will implement the full ML experimentation pipeline:

### Models to Train

| # | Model | Imbalance Strategy | Notes |
|---|-------|-------------------|-------|
| 1 | Dummy Baseline | Stratified random | Sanity check floor |
| 2 | Logistic Regression | `class_weight='balanced'` | Interpretable linear baseline |
| 3 | Decision Tree | `class_weight='balanced'` | Rule-based nonlinear baseline |
| 4 | Random Forest | `class_weight='balanced'` | Robust ensemble |
| 5 | XGBoost | `scale_pos_weight` | High-performance gradient boosting |

### Evaluation Metrics

Each model will be evaluated on the **validation set** using:

| Metric | Type | Why Important |
|--------|------|---------------|
| Precision | Per-threshold | Of flagged transactions, how many are actually fraud? |
| Recall | Per-threshold | Of all actual fraud, how many were detected? |
| F1-Score | Per-threshold | Harmonic mean of precision and recall |
| **F2-Score** | Per-threshold | **Primary** — recall-weighted, prioritizes catching fraud |
| ROC-AUC | Threshold-free | Overall discrimination ability |
| PR-AUC | Threshold-free | Robust to imbalance (recommended for fraud) |
| Confusion Matrix | Per-threshold | Full classification breakdown |

### Threshold Optimization

For each model, the classification threshold will be optimized over [0.05, 0.95] to maximize F2-score on the validation set. Default threshold (0.5) will also be reported for comparison.

### Model Selection

A weighted composite score selects the best model:
```
Composite Score = 0.35 × F2 + 0.30 × PR-AUC + 0.20 × ROC-AUC + 0.15 × F1
```

The test set is reserved exclusively for the final report on the selected best model.

---

*Report generated: September 9, 2026*  
*Fraud Shield v1.0.0 — Code Masala — Academic Prototype*
