# Fraud Shield
## Explainable & Adaptive Credit Card Fraud Risk Intelligence System

> **Team**: Code Masala | **Members**: Akshat Mehra (BTAD2401008), Ujjwal Chandravanshi (BTAD2401070)
> **Course**: Artificial Intelligence & Data Science | **Coordinator**: Asst. Prof. Geetika Hazra
> **Status**: Academic Prototype — NOT suitable for production banking deployment

---

## What This System Does

Fraud Shield is an end-to-end ML system that:
1. Trains and compares four ML algorithms on historical transaction data
2. Selects the best model using a multi-metric criterion (F2, PR-AUC, ROC-AUC, F1)
3. Produces a calibrated **fraud probability** for each transaction
4. Converts probability → **Risk Score (0–100)** → **Risk Level (LOW/MEDIUM/HIGH)**
5. Explains *why* a transaction was flagged using **SHAP feature contributions**
6. Provides a professional **Flask dashboard** with 6 operational views

### Example Output
```
Transaction ID:   TXN-2024-001
Amount:           €2,847.50
Fraud Probability: 91.4%
Risk Score:        91/100
Risk Level:        HIGH
Decision:          REVIEW

Top Contributing Factors:
  [+] V14: -4.21  (strongly associated with fraud pattern)
  [+] V4:  +3.88  (amount-velocity correlation)
  [+] Amount: +2.14 (unusually high for account)
  [-] V17: -1.93  (mitigating factor)

⚠ Note: Contributions indicate statistical associations, not causal reasons.
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Get the Dataset
Download `creditcard.csv` from Kaggle:
```
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```
Place it at: `data/raw/creditcard.csv`

See [`data/README.md`](data/README.md) for full instructions.

### 3. Train Models
```bash
python scripts/train.py
```

### 4. Run the Dashboard
```bash
python run.py
```
Open: http://localhost:5000

---

## Project Structure

```
credit card fraud/
├── data/               # Raw and processed datasets (not committed to git)
├── notebooks/          # EDA and experiment notebooks
├── src/
│   ├── data/           # Data loading and validation
│   ├── features/       # Feature engineering and preprocessing pipeline
│   ├── models/         # Model training (LR, DT, RF, XGBoost)
│   ├── evaluation/     # Metrics, threshold optimization, model comparison
│   ├── explainability/ # SHAP-based explanations
│   ├── monitoring/     # Prediction logging and drift detection
│   ├── risk/           # Risk score engine
│   └── utils/          # Config loader, logger, serialization
├── artifacts/          # Trained models, pipelines, metrics, plots
├── app/                # Flask web application
├── tests/              # Unit tests (pytest)
├── configs/            # config.yaml — all configurable parameters
├── docs/               # Architecture, API, setup, data dictionary
├── scripts/            # CLI training and evaluation scripts
├── requirements.txt
└── run.py              # Flask entry point
```

---

## ML Models Compared

| Model | Imbalance Strategy | Notes |
|-------|-------------------|-------|
| Logistic Regression | `class_weight='balanced'` | Baseline; interpretable |
| Decision Tree | `class_weight='balanced'` | Rule-based patterns |
| Random Forest | `class_weight='balanced'` | Robust ensemble |
| XGBoost | `scale_pos_weight` | High-performance boosting |
| Dummy Baseline | Stratified | Sanity check floor |

**Model selection criterion**: Weighted composite of F2-score (35%), PR-AUC (30%), ROC-AUC (20%), F1 (15%).

---

## Key Engineering Decisions

- **No data leakage**: Preprocessing pipeline fitted on training data only
- **SMOTE after split**: If SMOTE is used, it is applied only to the training set
- **Threshold optimization**: Classification threshold is optimized on validation set, not test set
- **Configurable thresholds**: All risk/model thresholds are in `configs/config.yaml` — not hard-coded
- **Reproducible**: `random_state=42` everywhere; stratified splits

---

## Documentation

| Document | Description |
|----------|-------------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System architecture and layer descriptions |
| [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md) | Dataset column definitions (populated after data inspection) |
| [`docs/MODEL_EXPERIMENTS.md`](docs/MODEL_EXPERIMENTS.md) | Training results and model comparison (populated after training) |
| [`docs/API.md`](docs/API.md) | Flask API endpoint reference |
| [`docs/SETUP.md`](docs/SETUP.md) | Full environment setup guide |

---

## Academic Disclaimer

This system is built as an academic prototype for a semester project in Artificial Intelligence & Data Science. It demonstrates ML concepts for educational purposes only.

**It is NOT suitable for real banking or financial deployment.** Do not use this system to make real fraud decisions.
