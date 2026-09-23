"""
Hostile QA & Adversarial Test Suite
==================================
Tests designed to attack the system boundaries, attempt to produce 500s,
inject invalid and adversarial payloads, and verify zero-tolerance quality.
"""

import math
import pytest
from app.server import create_app
from src.features.engineering import FeatureEngineer
from src.monitoring.tracker import TransactionTracker
import pandas as pd
import numpy as np


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestAdversarialInputs:
    """Test malformed, adversarial, and edge-case inputs to prediction API."""

    def test_predict_rejects_string_amount(self, client):
        payload = {"Amount": "NOT_A_NUMBER", "Time": 100.0}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "Amount" in data["message"]

    def test_predict_rejects_negative_amount(self, client):
        payload = {"Amount": -50.0, "Time": 100.0}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "non-negative" in data["message"]

    def test_predict_rejects_nan_amount(self, client):
        payload = {"Amount": float("nan"), "Time": 100.0}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "finite" in data["message"]

    def test_predict_rejects_inf_feature(self, client):
        payload = {"Amount": 100.0, "Time": 100.0, "V1": float("inf")}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "finite" in data["message"]

    def test_predict_rejects_string_feature(self, client):
        payload = {"Amount": 100.0, "Time": 100.0, "V14": "<script>alert(1)</script>"}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"

    def test_predict_extreme_large_amount(self, client):
        payload = {"Amount": 999999999.0, "Time": 100.0}
        resp = client.post("/api/predict", json=payload)
        assert resp.status_code == 400
        data = resp.get_json()
        assert "maximum" in data["message"]


class TestQueueSecurity:
    """Test investigation queue authorization, validation, and sanitization."""

    def test_queue_action_invalid_status_rejected(self, client):
        resp = client.post(
            "/api/queue/TX-TEST01/action",
            json={"status": "SQL_INJECTION' OR '1'='1", "notes": "malicious status"},
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["status"] == "error"
        assert "Invalid status" in data["message"]

    def test_queue_action_valid_status_sanitizes_notes(self, client):
        tracker = TransactionTracker.get_instance()
        tracker.queue.append({
            "id": "TX-QA-SANITIZE",
            "amount": 250.0,
            "risk_score": 85,
            "status": "PENDING_REVIEW",
            "notes": "",
        })

        payload = {
            "status": "CONFIRMED_FRAUD",
            "notes": "<script>alert('XSS')</script> & malicious tag",
        }
        resp = client.post("/api/queue/TX-QA-SANITIZE/action", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "success"

        # Verify notes stored sanitized
        updated_item = next(i for i in tracker.queue if i["id"] == "TX-QA-SANITIZE")
        notes_val = updated_item.get("analyst_notes", "")
        assert "<script>" not in notes_val
        assert "&lt;script&gt;" in notes_val
        assert "&amp;" in notes_val


class TestPipelineRobustness:
    """Test feature engineering against silent row dropping and NaN leaks."""

    def test_feature_engineering_preserves_duplicate_rows(self):
        fe = FeatureEngineer()
        # Create DataFrame with identical rows
        identical_row = {"Time": 500.0, "Amount": 100.0}
        for i in range(1, 29):
            identical_row[f"V{i}"] = 0.5
        df = pd.DataFrame([identical_row, identical_row, identical_row])

        transformed = fe.transform(df)
        assert len(transformed) == 3, "FeatureEngineer silently dropped duplicate rows!"

    def test_feature_engineering_negative_amount_safe(self):
        fe = FeatureEngineer()
        df = pd.DataFrame([
            {"Time": 100.0, "Amount": -10.0, **{f"V{i}": 0.0 for i in range(1, 29)}}
        ])
        transformed = fe.transform(df)
        assert not transformed["log_amount"].isna().any(), "log_amount produced NaN on negative amount!"


class TestTelemetryRealism:
    """Verify that telemetry never invents numbers when empty."""

    def test_tracker_empty_state_has_zero_fabricated_counts(self):
        tracker = TransactionTracker()
        tracker.transactions = []
        tracker.queue = []

        telemetry = tracker.get_monitoring_telemetry()
        assert telemetry["total_processed"] == 0
        assert telemetry["suspected_fraud_count"] == 0
        assert telemetry["high_risk_count"] == 0
        assert telemetry["fraud_rate_pct"] == 0.0


class TestSecurityHeaders:
    """Verify HTTP security headers on all responses."""

    def test_security_headers_present(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.headers.get("X-Content-Type-Options") == "nosniff"
        assert resp.headers.get("X-Frame-Options") == "SAMEORIGIN"
        assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
