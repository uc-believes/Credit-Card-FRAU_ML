"""
Unit and Integration Tests for Fraud Shield Dashboard API
=========================================================
Tests for Flask server routes, API endpoints, tracker integration,
and real ML inference via the API.
"""

from __future__ import annotations

import json
import pytest

from app.server import create_app
from src.monitoring.tracker import TransactionTracker


@pytest.fixture(scope="module")
def app():
    """Create Flask application fixture for testing."""
    test_app = create_app()
    test_app.config["TESTING"] = True
    return test_app


@pytest.fixture(scope="module")
def client(app):
    """Flask test client."""
    return app.test_client()


def test_health_check(client):
    """Test /health endpoint returns 200 and healthy status."""
    res = client.get("/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "healthy"
    assert "Fraud Shield" in data["service"]
    assert "XGBoost" in data["model"]


def test_index_page(client):
    """Test root dashboard UI loads HTML with all 6 required sections."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "Fraud Shield" in html
    assert 'id="tab-overview"' in html
    assert 'id="tab-predict"' in html
    assert 'id="tab-queue"' in html
    assert 'id="tab-performance"' in html
    assert 'id="tab-explainability"' in html
    assert 'id="tab-monitoring"' in html
    assert "Chart.js" in html or "chart.js" in html


def test_overview_endpoint(client):
    """Test GET /api/overview returns KPI numbers and model metadata."""
    res = client.get("/api/overview")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "kpis" in data
    kpis = data["kpis"]
    assert "total_processed" in kpis
    assert "suspected_fraud_count" in kpis
    assert "high_risk_count" in kpis
    assert "model_precision" in kpis
    assert "model_recall" in kpis
    assert "model_f2" in kpis
    assert kpis["total_processed"] >= 0

    assert "model_info" in data
    info = data["model_info"]
    assert "XGBoost" in info["selected_model"]
    assert info["operating_threshold"] > 0


def test_sample_transactions_endpoint(client):
    """Test GET /api/sample-transactions returns presets."""
    res = client.get("/api/sample-transactions")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "presets" in data
    presets = data["presets"]
    assert "fraud_high" in presets
    assert "legit_normal" in presets
    assert "suspicious_night" in presets
    assert "Amount" in presets["fraud_high"]["data"]
    assert "V14" in presets["fraud_high"]["data"]


def test_predict_endpoint_valid(client):
    """Test POST /api/predict with full feature dictionary returns real inference result."""
    payload = {
        "Amount": 149.62,
        "Time": 406.0,
        "V1": -2.312,
        "V2": 1.951,
        "V3": -1.609,
        "V4": 3.997,
        "V5": -0.522,
        "V6": -1.426,
        "V7": -2.537,
        "V8": 1.391,
        "V9": -2.770,
        "V10": -2.772,
        "V11": 3.202,
        "V12": -2.899,
        "V13": -0.595,
        "V14": -4.289,
        "V15": 0.389,
        "V16": -1.140,
        "V17": -2.830,
        "V18": -0.016,
        "V19": 0.416,
        "V20": 0.126,
        "V21": 0.517,
        "V22": -0.035,
        "V23": -0.465,
        "V24": 0.320,
        "V25": 0.044,
        "V26": 0.177,
        "V27": 0.261,
        "V28": -0.143,
    }

    res = client.post(
        "/api/predict",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "probability" in data
    assert 0.0 <= data["probability"] <= 1.0
    assert "risk_score" in data
    assert 0 <= data["risk_score"] <= 100
    assert data["risk_category"] in ["LOW RISK", "MEDIUM RISK", "HIGH RISK"]
    assert data["decision"] in ["ACCEPT", "REVIEW"]
    assert "legitimacy_disclaimer" in data
    assert "guarantee of transaction legitimacy" in data["legitimacy_disclaimer"]
    assert isinstance(data["top_contributing_factors"], list)
    assert len(data["top_contributing_factors"]) > 0


def test_predict_endpoint_empty(client):
    """Test POST /api/predict handles empty payload gracefully."""
    res = client.post(
        "/api/predict",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["status"] == "error"


def test_queue_endpoints(client):
    """Test GET /api/queue and POST /api/queue/<id>/action."""
    res = client.get("/api/queue")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "items" in data
    assert isinstance(data["items"], list)
    assert data["count"] == len(data["items"])

    if len(data["items"]) > 0:
        target_id = data["items"][0]["id"]

        # Action: CONFIRMED_FRAUD
        act_res = client.post(
            f"/api/queue/{target_id}/action",
            data=json.dumps({
                "status": "CONFIRMED_FRAUD",
                "notes": "Verified fraud in test suite.",
            }),
            content_type="application/json",
        )
        assert act_res.status_code == 200
        act_data = act_res.get_json()
        assert act_data["status"] == "success"
        assert act_data["item"]["status"] == "CONFIRMED_FRAUD"

        # Check nonexistent item returns 404
        bad_res = client.post(
            "/api/queue/TX-NONEXISTENT-9999/action",
            data=json.dumps({"status": "DISMISSED_FALSE_ALARM"}),
            content_type="application/json",
        )
        assert bad_res.status_code == 404


def test_performance_endpoint(client):
    """Test GET /api/performance returns benchmark and matrix data."""
    res = client.get("/api/performance")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "test_results" in data
    assert "validation_benchmark" in data
    assert "confusion_matrix" in data
    assert "tp" in data["confusion_matrix"]
    assert "tn" in data["confusion_matrix"]
    assert "fp" in data["confusion_matrix"]
    assert "fn" in data["confusion_matrix"]
    assert "threshold_sensitivity" in data
    assert len(data["threshold_sensitivity"]) >= 5


def test_explainability_endpoint(client):
    """Test GET /api/explainability returns global importance rankings and plots."""
    res = client.get("/api/explainability")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "global_importance" in data
    assert "top_features" in data
    assert "plots" in data
    assert "disclaimer" in data
    assert "NOT imply causation" in data["disclaimer"]


def test_monitoring_endpoint(client):
    """Test GET /api/monitoring returns PSI telemetry and distribution statistics."""
    res = client.get("/api/monitoring")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "telemetry" in data
    telem = data["telemetry"]
    assert "psi" in telem
    assert "status" in telem["psi"]
    assert "system_health" in telem
    assert "probability_histogram" in telem
    assert "high_risk_count" in telem
