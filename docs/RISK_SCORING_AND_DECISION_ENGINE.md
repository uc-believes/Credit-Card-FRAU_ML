# Fraud Shield — Risk Scoring & Decision Engine
## Phase 4: Operational Policy, Decision Tiers & Boundary Governance

> **Status**: Production Ready  
> **Module**: `src.risk.engine.RiskEngine`  
> **Primary Artifact**: `configs/config.yaml` (`risk_engine` block)  
> **Test Suite**: `tests/test_risk_engine.py` (9 passing boundary tests)  

---

## 1. Executive Summary

In enterprise fraud risk management, a raw machine-learning model output (e.g. `0.003471` or `0.8912`) cannot be directly consumed by transaction authorization pipelines, fraud investigators, or core banking ledgers. 

The **Risk Scoring and Decision Engine** translates continuous statistical model probabilities into:
1. A **Normalized Risk Score** ($0–100$ integer) intuitive for humans and policy rule engines.
2. A **Risk Category** (`LOW RISK`, `MEDIUM RISK`, `HIGH RISK`).
3. An explicit **Business Decision** (`ACCEPT` vs. `REVIEW` / `FLAGGED FOR INVESTIGATION`) based on an independent, configurable business decision threshold.
4. An automated **Regulatory Non-Warranty Notice** ensuring that automated clearance (`ACCEPT`) is never misrepresented as an absolute certification of legitimacy.

---

## 2. Core Concepts: Probability vs. Risk Score vs. Decision Threshold

A frequent failure mode in naive machine-learning deployments is conflating model probability, normalized risk scores, and business decision thresholds. In Fraud Shield, these three entities are mathematically and operationally decoupled:

```
  Raw Transaction x
         │
         ▼
[Machine Learning Model]
         │
         ▼ (Continuous statistical estimate)
┌─────────────────────────────────┐
│ 1. Model Probability P ∈ [0, 1] │
└────────────────┬────────────────┘
                 │
        ┌────────┴────────────────────────┐
        ▼                                 ▼
┌───────────────────────────────┐ ┌──────────────────────────────────────┐
│ 2. Normalized Risk Score      │ │ 3. Business Decision Threshold       │
│    S = round(P × 100) ∈ [0,100]│ │    T ∈ [0, 1] (e.g., T = 0.2773)     │
│    Category: LOW / MED / HIGH │ │    Policy Cutoff: P ≥ T ?            │
└───────────────────────────────┘ └──────────────────┬───────────────────┘
                                                     │
                                                     ▼
                                  ┌──────────────────────────────────────┐
                                  │ Business Decision:                   │
                                  │ • P ≥ T ──> REVIEW (FLAGGED)         │
                                  │ • P < T ──> ACCEPT (LOW RISK)        │
                                  └──────────────────────────────────────┘
```

### Comparative Breakdown

| Attribute | 1. Model Probability ($P$) | 2. Normalized Risk Score ($S$) | 3. Business Decision Threshold ($T$) |
|:---|:---|:---|:---|
| **Mathematical Domain** | Continuous real number $[0.0, 1.0]$ | Discrete integer $[0, 100]$ | Continuous policy parameter $[0.0, 1.0]$ |
| **Primary Meaning** | Empirical statistical likelihood that transaction is fraudulent conditioned on input features $x$. | Standardized, model-agnostic risk index easily understood by humans, analysts, and UI widgets. | Operational policy cutoff defining an institution's risk tolerance. |
| **System Owner** | Data Science & Machine Learning Engineers | Fraud Analytics & Operations Teams | Chief Risk Officer (CRO) & Fraud Strategy Team |
| **Sensitivity to Change** | Fluctuates with model architecture, calibration, retraining, and feature drift. | Invariant standard scale; provides consistent meaning across model upgrades. | Dynamically adjusted according to operational capacity, seasonality, and threat levels. |
| **Decision Role** | Input feature to scoring engine; threshold-free. | Determines operational categorization (`LOW RISK`, `MEDIUM RISK`, `HIGH RISK`). | Dictates automated action (`ACCEPT` vs. `REVIEW` / `FLAGGED FOR INVESTIGATION`). |

---

## 3. Operational Risk Categories

The Risk Engine classifies transactions into three standard operational tiers:

```
  Score: 0                   30 31                   70 71                  100
         [───── LOW RISK ─────] [──── MEDIUM RISK ────] [──── HIGH RISK ────]
         Color: #22c55e (Green)  Color: #f59e0b (Amber)  Color: #ef4444 (Red)
```

| Tier Label | Default Score Range | Operational Meaning | Recommended Banking Workflow |
|:---:|:---:|:---|:---|
| **`LOW RISK`** | **$0 – 30$** | Transaction displays behavioral characteristics strongly aligned with historical legitimate transactions. | Automated clearance via payment rails with zero cardholder friction. |
| **`MEDIUM RISK`** | **$31 – 70$** | Transaction displays elevated anomaly or moderate fraud probability signals. | Step-up 3-D Secure authentication (SMS OTP, banking app biometric confirmation). |
| **`HIGH RISK`** | **$71 – 100$** | High-confidence fraud indicators detected (e.g. rapid velocity, compromised credentials). | Immediate authorization block; placed in Tier-1 analyst triage queue. |

*Configurability*: Category boundaries (`low_max` and `medium_max`) are configurable in `configs/config.yaml` or programmatically in `RiskEngine(low_max=..., medium_max=...)`.

---

## 4. The Business Decision Threshold Policy

The operational decision is evaluated independently against a configurable **Fraud Decision Threshold** ($T$):

$$\text{Decision}(P) = \begin{cases} \text{REVIEW (FLAGGED FOR INVESTIGATION)}, & \text{if } P \ge T \\ \text{ACCEPT (LOW RISK)}, & \text{if } P < T \end{cases}$$

### Why Score Category and Decision Threshold Are Decoupled
A transaction can have a `LOW RISK` score (e.g. $S = 25$) yet still be `FLAGGED FOR INVESTIGATION` if risk management lowers the operating threshold to $T = 0.20$ during an active card-cloning wave. 
Conversely, if an operations center is understaffed, management can raise $T$ to $0.40$ to restrict manual investigations to only the highest confidence cases without altering the objective risk score.

---

## 5. Critical Policy: Why ACCEPT is NOT a Guarantee of Legitimacy

> [!WARNING]
> **Mandatory Risk Governance Principle**
> 
> In Fraud Shield, an `ACCEPT` decision strictly denotes:
> *"The transaction satisfied current statistical risk tolerance criteria for automated payment clearance."*
> 
> An `ACCEPT` decision **NEVER** certifies, guarantees, or warrants that a transaction is genuine or free of fraud.

### Why Zero-Risk Cannot Be Guaranteed
1. **Asymmetric Base Rates**: Fraud represents $0.172\%$ of transactions. Even with $99.99\%$ specificity, zero false negatives across millions of transactions is mathematically impossible without rejecting massive legitimate volume.
2. **Novel Fraud Vectors**: Zero-day social engineering, account takeovers, or stolen cards used before reporting cannot be detected prior to behavioral deviation signals.
3. **Card Scheme Rules**: Under Visa and Mastercard core regulations, all authorized transactions remain subject to 60-to-120-day post-settlement chargeback rights.

Every `RiskAssessment` object and API response includes the mandatory compliance statement:
```
"ACCEPT indicates the transaction satisfies current risk tolerance criteria
for automated clearance. It does NOT constitute a warranty, certification,
or guarantee of transaction legitimacy. All transactions remain subject to
post-settlement dispute and chargeback rules."
```

---

## 6. Boundary Condition Governance & Verification

The engine was subjected to rigorous boundary tests in `tests/test_risk_engine.py`:

| Test Boundary | Input Value | Expected Normalized Score | Expected Category | Expected Decision ($T=0.2773$) |
|:---|:---:|:---:|:---:|:---:|
| **Zero Floor** | $P = 0.0$ | `0` | `LOW RISK` | `ACCEPT` |
| **Sub-zero Clamping** | $P = -0.05$ | `0` (clamped) | `LOW RISK` | `ACCEPT` |
| **Category 1 Boundary** | $P = 0.3049$ | `30` | `LOW RISK` | `ACCEPT` |
| **Category 1 Transition** | $P = 0.3050$ | `31` | `MEDIUM RISK` | `REVIEW` |
| **Category 2 Boundary** | $P = 0.7049$ | `70` | `MEDIUM RISK` | `REVIEW` |
| **Category 2 Transition** | $P = 0.7050$ | `71` | `HIGH RISK` | `REVIEW` |
| **Ceiling** | $P = 1.0$ | `100` | `HIGH RISK` | `REVIEW` |
| **Super-one Clamping** | $P = 1.05$ | `100` (clamped) | `HIGH RISK` | `REVIEW` |
| **Decision Boundary - Sub** | $P = 0.2772$ | `28` | `LOW RISK` | `ACCEPT` |
| **Decision Boundary - Exact** | $P = 0.2773$ | `28` | `LOW RISK` | `REVIEW` |
| **Decision Boundary - Super** | $P = 0.2774$ | `28` | `LOW RISK` | `REVIEW` |

---

## 7. Python API Quickstart

### Standard Evaluation
```python
from src.risk.engine import RiskEngine

# Initialize with default config (low_max=30, medium_max=70, threshold=0.2773)
engine = RiskEngine()

assessment = engine.evaluate(0.35)
print(assessment.format_summary())
```
**Output**:
```text
Probability:           0.3500
Normalized Risk Score: 35/100
Risk Category:         MEDIUM RISK
Business Decision:     REVIEW (FLAGGED FOR INVESTIGATION)
Decision Threshold:    0.2773

Assessment: Transaction FLAGGED FOR INVESTIGATION: probability 0.3500 exceeds decision threshold 0.2773 (risk score: 35/100, MEDIUM RISK).
```

### Automated Clearance Example
```python
assessment = engine.evaluate(0.04)
print(assessment.format_summary())
```
**Output**:
```text
Probability:           0.0400
Normalized Risk Score: 4/100
Risk Category:         LOW RISK
Business Decision:     ACCEPT (LOW RISK)
Decision Threshold:    0.2773

Assessment: Transaction cleared automatically under current operational threshold (risk score: 4/100, LOW RISK). NOTE: ACCEPT is an operational clearance, not a guarantee of transaction legitimacy.

Notice: ACCEPT indicates the transaction satisfies current risk tolerance criteria for automated clearance. It does NOT constitute a warranty, certification, or guarantee of transaction legitimacy. All transactions remain subject to post-settlement dispute and chargeback rules.
```
