# Fraud Shield — API Reference

## Flask Application Endpoints

> **Base URL**: `http://localhost:5000`
> **Version**: 1.0

---

## Page Routes (HTML)

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Overview dashboard |
| `/predict` | GET | Transaction prediction form |
| `/investigation` | GET | Investigation queue |
| `/comparison` | GET | Model comparison page |
| `/explainability` | GET | SHAP explainability page |
| `/monitoring` | GET | Model monitoring page |

---

## API Endpoints (JSON)

### POST `/api/predict`
Submit a transaction for fraud risk scoring.

**Request Body** (JSON):
```json
{
  "Time": 50000.0,
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}
```

**Response** (JSON):
```json
{
  "transaction_id": "TXN-1725851234-001",
  "fraud_probability": 0.914,
  "risk_score": 91,
  "risk_level": "HIGH",
  "decision": "REVIEW",
  "decision_color": "#ef4444",
  "threshold_used": 0.38,
  "features_used": ["V1", "V2", ..., "Amount", "log_amount", "hour_of_day", "is_night"],
  "top_contributors": [
    {"feature": "V14", "shap_value": -4.21, "direction": "fraud"},
    {"feature": "V4",  "shap_value":  3.88, "direction": "fraud"},
    {"feature": "Amount", "shap_value": 2.14, "direction": "fraud"},
    {"feature": "V17", "shap_value": -1.93, "direction": "legitimate"}
  ],
  "disclaimer": "Contributions indicate statistical associations, not causal reasons.",
  "timestamp": "2026-09-09T10:30:00+05:30",
  "model_used": "XGBoost",
  "model_version": "1.0.0"
}
```

**Error Response**:
```json
{
  "error": "Missing required field: Amount",
  "status": 400
}
```

---

### GET `/api/model-summary`
Returns current model performance summary.

**Response**:
```json
{
  "best_model": "XGBoost",
  "metrics": {
    "precision": 0.87,
    "recall": 0.83,
    "f1_score": 0.85,
    "f2_score": 0.84,
    "roc_auc": 0.97,
    "pr_auc": 0.83
  },
  "threshold": 0.38,
  "trained_on": "2026-09-09T09:00:00",
  "dataset_size": 284807
}
```

---

### GET `/api/monitoring-stats`
Returns monitoring statistics for the dashboard.

**Response**:
```json
{
  "total_predictions": 1250,
  "fraud_detections": 47,
  "fraud_rate_recent": 0.038,
  "avg_risk_score": 18.4,
  "risk_distribution": {
    "LOW": 1120,
    "MEDIUM": 83,
    "HIGH": 47
  },
  "drift_alert": false,
  "psi_score": 0.04
}
```

---

### GET `/api/investigation-queue`
Returns current investigation queue.

**Response**:
```json
{
  "count": 47,
  "items": [
    {
      "transaction_id": "TXN-001",
      "fraud_probability": 0.914,
      "risk_score": 91,
      "risk_level": "HIGH",
      "amount": 149.62,
      "timestamp": "2026-09-09T10:30:00",
      "status": "PENDING"
    }
  ]
}
```

---

### GET `/api/health`
Health check endpoint.

**Response**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "pipeline_loaded": true,
  "version": "1.0.0"
}
```

---

*API Reference v1.0 | Fraud Shield | Code Masala*
