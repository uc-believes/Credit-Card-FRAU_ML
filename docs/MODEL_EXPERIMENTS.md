# Fraud Shield — Model Experiments & Evaluation Log
## Phase 2: Model Training, Threshold Optimization & Comparative Analysis

> **Status**: Completed  
> **Date**: September 2026  
> **Hardware & Environment**: Python 3.13.5 on Windows 64-bit  
> **Total Pipeline Execution Time**: 16.70 seconds  
> **Artifacts Saved**: `artifacts/models/`, `artifacts/metrics/`, `artifacts/reports/model_comparison.json`  

---

## 1. Experiment Configuration & Methodology

All models were evaluated strictly using leakage-safe principles established during Phase 1:
- **No Data Leakage**: The preprocessing pipeline (StandardScaler fitted exclusively on the 60% training split) transformed the features. The validation split (20%) was reserved for model comparison and threshold optimization. The held-out test split (20%) was reserved solely for evaluating the final selected winner.
- **Reproducibility**: Random seed fixed at `42` across all splits, baseline generators, and stochastic model initializations.
- **No Accuracy-Only Decisions**: Accuracy is mathematically uninformative in severe class imbalance (e.g., predicting 100% legitimate transactions yields 99.83% accuracy while catching 0% of fraud). Primary metric is $F_2$-score (recall-biased, $\beta=2$), supported by PR-AUC, ROC-AUC, and $F_1$-score.

| Parameter | Configuration Value | Rationale |
|:---|:---|:---|
| **Raw Dataset** | ULB European Cardholders (`creditcard.csv`) | 284,807 transactions, 492 frauds (0.172% imbalance) |
| **Splitting Strategy** | Stratified Train (60%) / Val (20%) / Test (20%) | Preserves exact class ratio across all splits |
| **Train Split Size** | 170,235 samples (284 fraud, 0.167%) | Primary fitting partition |
| **Val Split Size** | 56,745 samples (94 fraud, 0.166%) | Model selection and threshold search partition |
| **Test Split Size** | 56,746 samples (95 fraud, 0.167%) | Unseen evaluation partition |
| **Input Features** | 33 engineered & scaled features | $V_1$–$V_{28}$, Amount, Time, $\log(1+\text{Amount})$, hour_of_day, is_night |
| **Primary Imbalance Strategy** | Algorithmic Class Weighting (`class_weight='balanced'`, `scale_pos_weight`) | Eliminates synthetic distortion risks; penalizes fraud misclassification |
| **Threshold Tuning** | Grid search over $[0.05, 0.95]$ in 100 steps maximizing $F_2$ | Optimized exclusively on Validation split |

---

## 2. Models Tested

1. **Dummy Baseline (Stratified)**:
   - Randomly generates predictions according to empirical class distribution.
   - Purpose: Establishes the uninformative baseline floor.
2. **Logistic Regression**:
   - Linear classification with L2 regularization ($C=1.0$, solver=`lbfgs`, `class_weight='balanced'`).
   - Purpose: Highly interpretable linear benchmark.
3. **Decision Tree Classifier**:
   - Single CART tree (`max_depth=10`, `min_samples_split=10`, `min_samples_leaf=5`, `class_weight='balanced'`).
   - Purpose: Non-linear, single-rule-tree benchmark.
4. **Random Forest Classifier**:
   - Bagged ensemble of 200 trees (`max_depth=15`, `min_samples_split=10`, `min_samples_leaf=5`, `class_weight='balanced'`).
   - Purpose: High-capacity bagging model with out-of-bag variance reduction.
5. **XGBoost (Extreme Gradient Boosting)**:
   - Gradient boosted decision trees (`n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method='hist'`).
   - Imbalance treatment: `scale_pos_weight = 598.42` ($N_{\text{legitimate}} / N_{\text{fraud}}$ on training split).
   - Purpose: State-of-the-art gradient boosting for tabular data with asymmetric positive weighting.

---

## 3. Validation Set Results (Actual Measured Performance)

Each model was evaluated on the **56,745 validation samples (94 frauds)**. For each model, the threshold was swept across $[0.05, 0.95]$ to identify the operating point maximizing the $F_2$-score.

### Comparison Table

| Rank | Model | Threshold | Recall | Precision | F1-Score | F2-Score | ROC-AUC | PR-AUC | Composite Score | Minimum Gates |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | **XGBoost** | **0.2773** | **0.8617** | **0.9101** | **0.8852** | **0.8710** | **0.9734** | **0.8811** | **0.8966** | **PASSED [OK]** |
| 2 | Random Forest | 0.4864 | 0.8617 | 0.8438 | 0.8526 | 0.8581 | 0.9689 | 0.8383 | 0.8735 | PASSED [OK] |
| 3 | Logistic Regression | 0.9500 | 0.8830 | 0.3374 | 0.4882 | 0.6672 | 0.9828 | 0.7882 | 0.7398 | PASSED [OK] |
| 4 | Decision Tree | 0.0500 | 0.8298 | 0.1818 | 0.2983 | 0.4845 | 0.9139 | 0.4955 | 0.5458 | PASSED [OK] |
| 5 | Dummy Baseline | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4992 | 0.0017 | 0.1003 | FAILED (Floor) |

### Composite Ranking Formula
$$\text{Composite Score} = 0.35 \times F_2 + 0.30 \times \text{PR-AUC} + 0.20 \times \text{ROC-AUC} + 0.15 \times F_1$$

### Validation Confusion Matrices

#### 1. XGBoost (Operating Threshold = 0.2773)
```
                    Predicted Legit    Predicted Fraud
Actual Legit            56,643               8          (Specificity: 99.986%)
Actual Fraud                13              81          (Recall: 86.17%)
```
- **False Positives**: Only 8 customer transactions incorrectly flagged out of 56,651 legitimate transactions.
- **Precision**: 91.01% (81 true frauds / 89 alerts).

#### 2. Random Forest (Operating Threshold = 0.4864)
```
                    Predicted Legit    Predicted Fraud
Actual Legit            56,636              15          (Specificity: 99.974%)
Actual Fraud                13              81          (Recall: 86.17%)
```
- Achieves identical recall to XGBoost (81/94), but with almost double the false positives (15 vs. 8).

#### 3. Logistic Regression (Operating Threshold = 0.9500)
```
                    Predicted Legit    Predicted Fraud
Actual Legit            56,488             163          (Specificity: 99.712%)
Actual Fraud                11              83          (Recall: 88.30%)
```
- Catches 2 additional frauds (83/94), but at the cost of 163 false alarms (Precision drops to 33.74%).

#### 4. Decision Tree (Operating Threshold = 0.0500)
```
                    Predicted Legit    Predicted Fraud
Actual Legit            56,300             351          (Specificity: 99.380%)
Actual Fraud                16              78          (Recall: 82.98%)
```
- Suffers from 351 false alarms (Precision: 18.18%).

---

## 4. Model Selection Rationale

**Selected Winner**: **XGBoost (Extreme Gradient Boosting)**

### Why XGBoost Won:
1. **Highest Composite Score (0.8966)**:
   Outperforms all other candidates across the weighted objective combining fraud recall, area metrics, and precision balance.
2. **Dominant PR-AUC (0.8811)**:
   PR-AUC is the gold standard for highly imbalanced fraud classification because it evaluates true positive detection against false alarms without being inflated by the massive volume of true negatives. XGBoost achieved **0.8811**, a significant margin over Random Forest (**0.8383**) and Logistic Regression (**0.7882**).
3. **Lowest Operational Friction (Only 8 False Positives on Validation)**:
   In fraud detection, false alarms trigger card blocks, SMS alerts, or costly manual agent reviews. XGBoost maintains **91.01% precision** while still capturing **86.17% of all fraud instances**.
4. **Generalization Stability**:
   Gradient boosting combined with feature subsampling (`colsample_bytree=0.8`) and sample subsampling (`subsample=0.8`) proved resilient against the high-dimensional noise in PCA components.

---

## 5. Held-Out Test Set Evaluation (Unseen Data Verification)

To verify the absence of data leakage or overfitting during threshold optimization, the selected **XGBoost** model was evaluated on the **56,746 held-out test transactions (95 frauds)** using the **fixed operating threshold of 0.2773** determined during validation.

### Test Metrics Summary

| Metric | Measured Test Value | Interpretation / Operational Impact |
|:---|:---:|:---|
| **Operating Threshold** | `0.2773` | Pre-tuned on validation set (zero leakage) |
| **Precision** | **92.59%** | 75 out of 81 triggered alerts were genuine fraud |
| **Recall (Sensitivity)** | **78.95%** | Caught 75 of 95 fraud attempts |
| **F1-Score** | **0.8523** | Harmonic mean of precision and recall |
| **F2-Score** | **0.8134** | Primary recall-weighted benchmark |
| **PR-AUC** | **0.8178** | Precision-Recall curve area |
| **ROC-AUC** | **0.9683** | True Positive vs False Positive trade-off |
| **Accuracy** | **99.95%** | Overall correct classification rate |
| **Specificity** | **99.99%** | 56,645 out of 56,651 legitimate transactions cleared smoothly |

### Test Confusion Matrix
```
                    Predicted Legit    Predicted Fraud
Actual Legit            56,645               6          (False Positive Rate: 0.0106%)
Actual Fraud                20              75          (Detection Rate: 78.95%)
```
- **True Positives (TP)**: 75 fraudulent transactions stopped.
- **False Negatives (FN)**: 20 fraudulent transactions missed.
- **False Positives (FP)**: Only **6** false alarms across 56,651 legitimate transactions.
- **False Positive Ratio**: Less than 1 false alarm per 9,400 legitimate purchases!

---

## 6. Threshold Sensitivity Analysis

Because fraud risk appetite varies by institution and operational capacity, we analyzed how varying the decision threshold shifts the precision-recall trade-off on the validation split:

| Operating Threshold | Precision | Recall | F1-Score | F2-Score | Recommended Deployment Scenario |
|:---:|:---:|:---:|:---:|:---:|:---|
| **0.05** | 71.93% | 87.23% | 0.7885 | 0.8367 | High-risk emergency / active cyber-attack mode |
| **0.10** | 81.00% | 86.17% | 0.8351 | 0.8508 | Aggressive fraud prevention |
| **0.20** | 87.10% | 86.17% | 0.8663 | 0.8635 | Balanced prevention |
| **0.28 (Optimal)** | **91.01%** | **86.17%** | **0.8852** | **0.8710** | **Fraud Shield Default Operating Threshold** |
| **0.40** | 93.02% | 85.11% | 0.8889 | 0.8658 | VIP cardholder / low friction mode |
| **0.50** | 93.02% | 85.11% | 0.8889 | 0.8658 | Standard uncalibrated default |
| **0.70** | 94.12% | 85.11% | 0.8939 | 0.8677 | High-confidence automated blocking only |
| **0.90** | 94.05% | 84.04% | 0.8876 | 0.8587 | Strict automated blacklisting |

---

## 7. Artifact Verification & Next Steps

All Phase 2 artifacts have been verified on disk:
- `artifacts/models/xgboost.joblib` (436 KB) & `xgboost_metadata.json`
- `artifacts/models/random_forest.joblib` & `random_forest_metadata.json`
- `artifacts/models/logistic_regression.joblib` & `logistic_regression_metadata.json`
- `artifacts/models/decision_tree.joblib` & `decision_tree_metadata.json`
- `artifacts/models/best_model.joblib` (points to selected winner: XGBoost)
- `artifacts/models/best_model_metadata.json`
- `artifacts/reports/model_comparison.json` (Structured JSON of all experiments)
- `tests/test_model_inference.py` (Passed 3/3 automated verification tests)

Phase 2 is complete and verified. Ready for Phase 3 (Explainability with SHAP, Calibration, and Risk Tier Assignment).
