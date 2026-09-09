# Fraud Shield — System Architecture
## Explainable & Adaptive Credit Card Fraud Risk Intelligence System
> **Version**: 1.0 | **Team**: Code Masala | **Academic Prototype**

---

## Overview

Fraud Shield is a multi-layer fraud risk intelligence system. It ingests historical credit card transaction data, trains and compares multiple ML models, selects the best performer using a multi-metric criterion, and produces calibrated fraud probabilities with explainable SHAP-based feature contributions. A Flask dashboard provides six operational views.

This is an academic prototype. It is NOT intended for real banking deployment.

---

## System Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           FRAUD SHIELD SYSTEM                              │
│                                                                            │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐   ┌───────────────────┐  │
│  │  RAW DATA│──▶│  LOADER  │──▶│  FEATURE ENG │──▶│  SKLEARN PIPELINE │  │
│  │ (CSV)    │   │ validator│   │  engineering │   │  (fit on train)   │  │
│  └──────────┘   └──────────┘   └──────────────┘   └────────┬──────────┘  │
│                                                             │              │
│                              ┌──────────────────────────────┘              │
│                              ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      TRAINING LAYER                                  │  │
│  │                                                                      │  │
│  │  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐  │  │
│  │  │  Logistic   │  │  Decision    │  │   Random    │  │ XGBoost  │  │  │
│  │  │ Regression  │  │    Tree      │  │   Forest    │  │          │  │  │
│  │  │(class_weight│  │(class_weight)│  │(class_weight│  │(scale_   │  │  │
│  │  │ =balanced)  │  │              │  │ =balanced)  │  │pos_weight│  │  │
│  │  └──────┬──────┘  └──────┬───────┘  └──────┬──────┘  └────┬─────┘  │  │
│  └─────────┼───────────────┼─────────────────┼──────────────┼──────────┘  │
│            └───────────────┴──────────────────┘──────────────┘             │
│                                      │                                     │
│                                      ▼                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     EVALUATION LAYER                                 │  │
│  │  Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Confusion Matrix │  │
│  │  Threshold Optimization → Best Operating Point per model             │  │
│  │  Composite Score → Best Model Selection                              │  │
│  └──────────────────────────┬───────────────────────────────────────────┘  │
│                             │                                              │
│                    ┌────────┘                                              │
│                    ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │                     INFERENCE PIPELINE                              │  │
│  │  (load preprocessing pipeline + best model from artifacts/)         │  │
│  │  Raw Transaction → Preprocess → Predict → Calibrate (if needed)     │  │
│  │              ↓                                                       │  │
│  │         Fraud Probability (0.0 – 1.0)                               │  │
│  └────────────────────────────┬────────────────────────────────────────┘  │
│                               │                                            │
│            ┌──────────────────┼──────────────────┐                        │
│            ▼                  ▼                  ▼                        │
│  ┌─────────────────┐  ┌───────────────┐  ┌──────────────────────────┐   │
│  │   RISK ENGINE   │  │ EXPLAINABILITY│  │     MONITORING           │   │
│  │  Probability →  │  │   (SHAP)      │  │  Prediction logging      │   │
│  │  0-100 score    │  │  Top-K feature│  │  Distribution tracking   │   │
│  │  LOW/MED/HIGH   │  │  contributions│  │  PSI drift indicators    │   │
│  └────────┬────────┘  └───────┬───────┘  └──────────────────────────┘   │
│           │                   │                                           │
│           └────────┬──────────┘                                          │
│                    ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    FLASK DASHBOARD (6 panels)                        │  │
│  │  Overview | Prediction | Investigation | Comparison | Explain | Mon  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer Descriptions

### 1. Data Layer (`src/data/`)
- **`loader.py`**: Reads raw CSV, validates schema and row count against config expectations, logs statistics. Never modifies raw data.
- **`validator.py`**: Column-level quality checks (types, ranges, missing values, duplicates).

### 2. Feature Engineering Layer (`src/features/`)
- **`engineering.py`**: Derives new features:
  - `log_amount`: log1p(Amount) — handles skewness
  - `hour_of_day`: Time % 86400 / 3600 — cyclical time signal
  - `is_night`: binary flag for transactions 0–6h
- **`pipeline.py`**: Assembles a scikit-learn `Pipeline` with `ColumnTransformer`:
  - `StandardScaler` on `Amount`, `Time`, and derived features
  - Passthrough for `V1`–`V28` (already PCA-scaled by ULB)
  - Pipeline is **fit only on training data**, preventing leakage

### 3. Model Layer (`src/models/`)
- Each model is a self-contained module with config-driven hyperparameters
- **Imbalance handling**:
  - Primary: `class_weight='balanced'` for LR/DT/RF; `scale_pos_weight` for XGBoost
  - Secondary (optional): SMOTE via `imbalanced-learn`, applied **after** train/test split
- All models trained with `random_state=42` for reproducibility
- A `DummyClassifier` baseline is always included for sanity checking

### 4. Evaluation Layer (`src/evaluation/`)
- **`metrics.py`**: Computes precision, recall, F1, F2 (β=2), ROC-AUC, PR-AUC, confusion matrix
- **`threshold.py`**: Grid-searches threshold in [0.05, 0.95] to maximize F2-score (configurable)
- **`comparison.py`**: Builds ranked comparison table using weighted composite score from config

### 5. Risk Engine (`src/risk/engine.py`)
- Converts raw probability to integer risk score (0–100)
- Maps score to category using config thresholds (LOW/MEDIUM/HIGH)
- Returns structured decision object with label, color, and recommended action

### 6. Explainability Layer (`src/explainability/explainer.py`)
- **TreeExplainer**: Random Forest, XGBoost (exact Shapley values)
- **LinearExplainer**: Logistic Regression (linear SHAP)
- Returns top-K features with contribution direction and magnitude
- Includes standard disclaimer: correlational, not causal

### 7. Monitoring Layer (`src/monitoring/monitor.py`)
- Appends each prediction to a JSONL log
- Tracks prediction score distribution and class distribution over time
- Computes Population Stability Index (PSI) as a simple drift indicator
- Prototype-level: not a production monitoring platform

### 8. Application Layer (`app/`)
- Flask app factory pattern (`app/factory.py`)
- Six routes served by `app/routes/`
- Jinja2 HTML templates in `app/templates/`
- Dark financial-theme CSS + Chart.js in `app/static/`

---

## Data Flow: Training vs Inference

### Training Flow
```
creditcard.csv
    → DataLoader.load()           # Schema + quality validation
    → FeatureEngineer.transform() # Derive log_amount, hour_of_day, is_night
    → stratified_split()          # 60% train / 20% val / 20% test (config-driven)
    → Pipeline.fit(X_train)       # Fit scaler ON TRAINING DATA ONLY
    → Pipeline.transform(X_*)     # Apply to all splits
    → ModelTrainer.train(X_train, y_train)  # Per-model training
    → Evaluator.evaluate(models, X_val, y_val)   # Val metrics
    → ThresholdOptimizer.optimize(model, X_val)  # Per-model threshold
    → ModelSelector.select(comparison_table)     # Best model by composite score
    → Final evaluation on X_test, y_test         # True holdout
    → serialize: pipeline.joblib + best_model.joblib + metadata.json
```

### Inference Flow
```
raw_transaction (dict)
    → load pipeline.joblib        # Same transformer as training
    → Pipeline.transform()        # Identical preprocessing
    → model.predict_proba()       # Fraud probability
    → RiskEngine.score()          # 0-100 risk score + category
    → SHAPExplainer.explain()     # Top-K feature contributions
    → MonitoringLogger.log()      # Append to JSONL log
    → return PredictionResult     # Structured output dict
```

---

## Anti-Leakage Guarantees

| Potential Leakage Point | Mitigation |
|------------------------|------------|
| Scaler fitted on all data | `Pipeline.fit()` called **only** on training split |
| SMOTE applied before split | SMOTE is applied **after** stratified split, on training set only |
| Test statistics used during EDA | EDA is on full dataset for understanding; no statistics from EDA feed into training parameters |
| Threshold optimized on test set | Threshold optimized on **validation** set; final report uses test set |
| Model selection on test set | Model comparison uses **validation** set metrics; test set used only for final report |

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| ML core | scikit-learn 1.9.0 | Mature, well-tested, Pipeline support |
| Gradient boosting | XGBoost 3.4.1 | High-performance; excellent with imbalanced data |
| Imbalance | imbalanced-learn 0.14.2 | Standard SMOTE implementation |
| Explainability | SHAP 0.52.0 | State-of-the-art model-agnostic explanations |
| Serialization | joblib 1.6.0 | Efficient sklearn object serialization |
| Web app | Flask 3.1.1 | Already installed; full HTML/CSS control |
| Templates | Jinja2 3.1.6 | Native Flask templating |
| Charts | Chart.js (CDN) | No server-side Plotly overhead |
| Server-side plots | matplotlib 3.10.6 | EDA + training plots |
| Configuration | PyYAML 6.0.2 | Human-readable, version-control-friendly |
| Data | pandas 3.0.5, numpy 2.2.6 | Standard |

---

## Security & Privacy Notes

- Raw data is NOT committed to version control
- No cardholder PII is exposed (V1–V28 are already anonymized PCA components)
- No API keys or credentials are stored in code
- The Flask `secret_key` must be changed before any deployment
- Monitoring logs contain only transformed feature values, not raw financial details

---

*Architecture Document v1.0 | Fraud Shield | Code Masala | Academic Prototype*
