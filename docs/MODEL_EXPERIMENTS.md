# Fraud Shield — Model Experiments Log
## Training Results & Model Comparison

> **Status**: Pending — this document will be populated automatically after `python scripts/train.py` completes.
> All tables and numbers below are placeholders. Do NOT cite them as results.

---

## Experiment Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | creditcard.csv (ULB) |
| Split | 60% train / 20% val / 20% test |
| Random state | 42 |
| Imbalance strategy | class_weight='balanced' (primary) |
| SMOTE | Disabled (configurable) |
| Threshold optimization | F2-score on validation set |

---

## Models Trained

1. **Dummy Baseline** (stratified random) — sanity check floor
2. **Logistic Regression** — interpretable linear baseline
3. **Decision Tree** — rule-based nonlinear baseline
4. **Random Forest** — ensemble; primary tree-based candidate
5. **XGBoost** — gradient boosting; expected top performer

---

## Validation Set Results *(To Be Populated)*

| Model | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC | Threshold | Composite Score |
|-------|-----------|--------|----|----|---------|--------|-----------|----------------|
| Dummy Baseline | TBD | TBD | TBD | TBD | TBD | TBD | 0.50 | TBD |
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Composite Score Formula
```
Score = 0.35 × F2 + 0.30 × PR-AUC + 0.20 × ROC-AUC + 0.15 × F1
```
(Weights configured in `configs/config.yaml → model_selection.metric_weights`)

---

## Selected Best Model

**Winner**: *(To Be Determined)*
**Rationale**: *(To Be Populated)*

---

## Final Test Set Results *(To Be Populated)*

> Test set metrics are computed ONLY on the selected best model.
> All other model evaluation was done on the validation set.

| Metric | Value |
|--------|-------|
| Precision | TBD |
| Recall | TBD |
| F1-Score | TBD |
| F2-Score | TBD |
| ROC-AUC | TBD |
| PR-AUC | TBD |
| Operating Threshold | TBD |

### Confusion Matrix *(To Be Populated)*
```
              Predicted Legit   Predicted Fraud
Actual Legit       TN                FP
Actual Fraud       FN                TP
```

---

## Imbalance Strategy Comparison *(To Be Populated)*

| Strategy | F2-Score | PR-AUC | Notes |
|---------|---------|--------|-------|
| class_weight='balanced' | TBD | TBD | Primary |
| SMOTE + class_weight | TBD | TBD | Secondary experiment |
| No imbalance handling | TBD | TBD | Baseline comparison |

---

## Threshold Analysis *(To Be Populated)*

For the best model, precision and recall at different thresholds:

| Threshold | Precision | Recall | F1 | F2 |
|-----------|-----------|--------|----|----|
| 0.10 | TBD | TBD | TBD | TBD |
| 0.20 | TBD | TBD | TBD | TBD |
| 0.30 | TBD | TBD | TBD | TBD |
| **Optimal** | **TBD** | **TBD** | **TBD** | **TBD** |
| 0.50 | TBD | TBD | TBD | TBD |

---

## Observations & Analysis *(To Be Populated)*

*(This section will be written after real results are available.)*

---

*Model Experiments Log v1.0 | Fraud Shield | Code Masala | Pending dataset & training*
