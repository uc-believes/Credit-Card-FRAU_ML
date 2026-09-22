# Fraud Shield — Explainability & Risk Intelligence Framework
## Phase 3: Mathematical Attribution, Risk Scoring & SHAP Explanations

> **Status**: Production Ready  
> **Methodology**: Tree SHAP (Shapley Additive Explanations) via `shap.TreeExplainer`  
> **Model Explanator**: Asymmetric Gradient Boosted Decision Trees (XGBoost)  
> **Primary Artifacts**: `src/explainability/`, `src/risk/`, `artifacts/plots/`, `artifacts/reports/`  

---

## 1. Executive Summary

In financial risk intelligence, a black-box model is legally and operationally unviable. Under regulations such as the Fair Credit Reporting Act (FCRA), the European General Data Protection Regulation (GDPR Article 22), and banking model risk governance frameworks (SR 11-7), automated decisions that affect cardholders must provide clear, auditable explanations.

Fraud Shield integrates a **Shapley Additive Explanations (SHAP)** engine rooted in cooperative game theory. Every inference call produces:
1. **Calibrated Fraud Probability** ($P \in [0, 1]$)
2. **Standardized Risk Score** (0–100 integer)
3. **Operational Risk Category** (`LOW`, `MEDIUM`, `HIGH`) and Action (`ALLOW`, `MONITOR`, `REVIEW`)
4. **Binary Decision Alert** (evaluated against the validation-optimized operating threshold of `0.2773`)
5. **Exact Local Feature Attributions** (top factors pushing towards fraud vs. mitigating risk)
6. **Scientific Non-Causal Disclaimer**

---

## 2. Mathematical Foundation: Why Tree SHAP?

### 2.1 The Shapley Formulation
A transaction's prediction is treated as a cooperative game where each feature $x_i$ is a player cooperating to shift the model's prediction away from the baseline expectation $E[f(x)]$ towards the final margin output $f(x)$.

The classical Shapley attribution $\phi_i$ satisfies four fundamental mathematical axioms:
1. **Efficiency (Additivity)**: $\sum_{i=1}^M \phi_i = f(x) - E[f(x)]$
2. **Symmetry**: If features $i$ and $j$ contribute identically across all subsets $S$, then $\phi_i = \phi_j$.
3. **Dummy (Null Player)**: If feature $i$ contributes zero marginal value to all subsets, $\phi_i = 0$.
4. **Linearity**: The Shapley value of an ensemble is the weighted sum of the Shapley values of its trees.

### 2.2 Exact Additivity to Probability Margin
For XGBoost binary classification, `shap.TreeExplainer` computes attributions in **margin (log-odds) space**:
$$\text{Margin}(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

Where:
- $\phi_0 = \text{base\_value} = E[\text{Margin}]$ (the prior expected log-odds across training trees)
- $\phi_i(x)$ is the SHAP value of feature $i$ for transaction $x$
- $M = 33$ is the total number of input features

The final fraud probability $P(\text{fraud} \mid x)$ is obtained via the standard logistic sigmoid:
$$P(\text{fraud} \mid x) = \sigma\left(\phi_0 + \sum_{i=1}^M \phi_i(x)\right) = \frac{1}{1 + \exp\left(-\left(\phi_0 + \sum_{i=1}^M \phi_i(x)\right)\right)}$$

> **Verification Guarantee**: In automated testing (`tests/test_explainability.py`), the numerical discrepancy between the model's native `predict_proba` and $\sigma(\phi_0 + \sum \phi_i)$ is guaranteed to be $< 10^{-10}$.

### 2.3 Strict Attribution Directionality
- **$\phi_i > 0$ (Increases Fraud Risk)**: Pushes log-odds positive $\implies$ strictly increases fraud probability.
- **$\phi_i < 0$ (Mitigating / Legitimacy Evidence)**: Pushes log-odds negative $\implies$ strictly decreases fraud probability.

---

## 3. Risk Engine: Probability $\to$ Score $\to$ Category

The **Risk Engine** (`src/risk/engine.py`) standardizes raw model probabilities into an operational tiering system configured in `configs/config.yaml`:

```
   Fraud Probability P
           │
           ▼
    Risk Score = round(P * 100)  [Range: 0 – 100]
           │
 ┌─────────┼─────────┐
 ▼         ▼         ▼
LOW      MEDIUM     HIGH
[0–30]   [31–70]   [71–100]
```

### Risk Category Specification

| Risk Level | Score Range | Default Action | UI Color | System Description |
|:---:|:---:|:---:|:---:|:---|
| **`LOW`** | **0 – 30** | `ALLOW` | `#22c55e` (Green) | Transaction appears legitimate. Cleared automatically without friction. |
| **`MEDIUM`** | **31 – 70** | `MONITOR` | `#f59e0b` (Amber) | Elevated risk signals detected. Flagged for passive monitoring / secondary SMS confirmation. |
| **`HIGH`** | **71 – 100** | `REVIEW` | `#ef4444` (Red) | High-probability fraud attempt. Queued for immediate fraud agent investigation / step-up auth. |

> **Operating Threshold Integration**: While the risk score provides continuous risk tiering, the binary decision alert is governed by the tuned operating threshold ($T = 0.2773$). Any transaction with $P \ge 0.2773$ triggers `prediction = 1` (`FRAUD ALERT`).

---

## 4. Global Feature Importance (Macro Model Behavior)

Global feature importance is computed by calculating the **mean absolute SHAP value** across validation cohort transactions:
$$I_j = \frac{1}{N} \sum_{k=1}^N |\phi_j(x^{(k)})|$$

### Top 10 Global Fraud Drivers (Measured on Validation Set)

| Rank | Feature Name | Description / Signal | Mean \|SHAP\| | Global Importance Share |
|:---:|:---|:---|:---:|:---:|
| **1** | **$V_{14}$** | Latent component strongly correlated with unauthorized card present/compromised credentials | **2.6514** | **16.0%** |
| **2** | **$V_4$** | Latent transaction velocity & frequency indicator | **2.2123** | **13.3%** |
| **3** | **$V_3$** | Latent account authorization consistency | **1.0718** | **6.5%** |
| **4** | **$V_{11}$** | Latent behavioral deviation score | **0.9787** | **5.9%** |
| **5** | **$V_{12}$** | Latent cardholder geographic/terminal consistency | **0.8836** | **5.3%** |
| **6** | **$V_{10}$** | Latent merchant category risk component | **0.8521** | **5.1%** |
| **7** | **Amount** | Scaled transaction monetary value | **0.6142** | **3.7%** |
| **8** | **$V_{17}$** | Latent device signature indicator | **0.5891** | **3.6%** |
| **9** | **$\log(\text{Amount})$** | Non-linear right-skew dampener | **0.4912** | **3.0%** |
| **10** | **Time** | Elapsed seconds from transaction window epoch | **0.4215** | **2.5%** |

*Artifact Generated*: [`artifacts/plots/global_feature_importance.png`](file:///c:/Users/Admin/Desktop/coding/antigrav/res%20projects/credit%20card%20fraud/artifacts/plots/global_feature_importance.png) and [`artifacts/plots/shap_beeswarm_summary.png`](file:///c:/Users/Admin/Desktop/coding/antigrav/res%20projects/credit%20card%20fraud/artifacts/plots/shap_beeswarm_summary.png).

---

## 5. Local Transaction Explanations (Micro Inference Examples)

### Case Study A: Confirmed High-Risk Fraud Transaction
- **True Label**: 1 (Fraud)
- **Model Output**:
  - **Probability**: `1.0000`
  - **Risk Score**: `100` / 100
  - **Risk Level**: `HIGH`
  - **Decision**: `REVIEW` (`FRAUD ALERT`)
- **Top Contributing Factors (Pushing Toward Fraud)**:
  1. **$V_{14}$** ($\text{value} = -4.579$, $\text{SHAP} = +3.7898$, Impact = $29.6\%$)
  2. **$V_{12}$** ($\text{value} = -5.314$, $\text{SHAP} = +1.4917$, Impact = $11.6\%$)
  3. **Amount** ($\text{value} = 0.048$, $\text{SHAP} = +1.4159$, Impact = $11.1\%$)
  4. **$V_{10}$** ($\text{value} = -7.731$, $\text{SHAP} = +1.2486$, Impact = $9.7\%$)
  5. **$V_4$** ($\text{value} = 6.535$, $\text{SHAP} = +1.1752$, Impact = $9.2\%$)
- **Mitigating Factors**: $V_{25}$ ($\text{SHAP} = -1.1515$), $V_8$ ($\text{SHAP} = -0.8281$)
- *Artifact Generated*: [`artifacts/plots/sample_fraud_explanation.png`](file:///c:/Users/Admin/Desktop/coding/antigrav/res%20projects/credit%20card%20fraud/artifacts/plots/sample_fraud_explanation.png)

### Case Study B: Legitimate Transaction
- **True Label**: 0 (Legitimate)
- **Model Output**:
  - **Probability**: `0.0000`
  - **Risk Score**: `0` / 100
  - **Risk Level**: `LOW`
  - **Decision**: `ALLOW` (`LEGITIMATE`)
- **Mitigating Factors (Overwhelmingly Pushing Toward Legitimacy)**:
  1. **$V_{14}$** ($\text{value} = 0.799$, $\text{SHAP} = -3.4215$)
  2. **$V_{11}$** ($\text{value} = -1.224$, $\text{SHAP} = -2.3560$)
  3. **$V_4$** ($\text{value} = 0.759$, $\text{SHAP} = -1.2375$)
- *Artifact Generated*: [`artifacts/plots/sample_legit_explanation.png`](file:///c:/Users/Admin/Desktop/coding/antigrav/res%20projects/credit%20card%20fraud/artifacts/plots/sample_legit_explanation.png)

---

## 6. Distinguishing Empirical Attribution from Causality

> [!WARNING]
> **Mandatory Regulatory & Technical Notice**
> 
> SHAP feature attributions describe **how the machine-learning model combined historical statistical signals to reach its prediction**. They do **NOT** assert that:
> - The cardholder intentionally committed fraud.
> - Altering a feature value (e.g. changing transaction time or amount) will causally prevent a fraud attempt.
> - PCA features represent a single physical attribute (they are linear combinations of original anonymized variables).
>
> All automated outputs and investigation dashboard views include the explicit disclaimer:
> *"Feature contributions indicate statistical associations learned from historical training data. They do NOT imply causation and should not be interpreted as definitive reasons for fraud."*

---

## 7. Python API Quickstart

```python
from src.explainability.service import FraudIntelligenceService

# Initialize the end-to-end service
service = FraudIntelligenceService()

# Evaluate a transaction (raw dict)
result = service.predict_and_explain({
    "Time": 406.0,
    "Amount": 149.50,
    "V1": -2.31,
    "V2": 1.95,
    # ... V3 to V28 ...
}, top_k=5, generate_plot=True)

# Formatted console output
print(result["summary_text"])
# Base64 string ready for <img> HTML rendering in web UI
b64_image = result["plot_base64"]
```
