"""
Unit and Integration Tests for Phase 6: Adaptive Fraud Intelligence
===================================================================
Tests for:
1. What-If Threshold Trade-Off Analysis & Decision Optimization
2. Dynamic Runtime Threshold Switching (RiskEngine & API)
3. Multi-Feature Data Drift Telemetry (PSI, KS, Wasserstein)
4. Empirical Risk Score Distribution & Quantile Modeling
5. Multi-Factor Investigation Queue Prioritization
"""

from __future__ import annotations

import json
import pytest

from app.server import create_app
from src.monitoring.adaptive import AdaptiveIntelligenceEngine
from src.risk.engine import RiskEngine


@pytest.fixture(scope="module")
def app():
    test_app = create_app()
    test_app.config["TESTING"] = True
    return test_app


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


@pytest.fixture(scope="module")
def adaptive_engine():
    return AdaptiveIntelligenceEngine.get_instance()


# =============================================================================
# 1. RISK ENGINE & THRESHOLD SWITCHING TESTS
# =============================================================================

def test_risk_engine_set_decision_threshold():
    """Verify runtime dynamic threshold update on RiskEngine."""
    engine = RiskEngine(operating_threshold=0.2773)
    assert abs(engine.decision_threshold - 0.2773) < 1e-4

    new_t = engine.set_decision_threshold(0.40)
    assert abs(new_t - 0.40) < 1e-4
    assert abs(engine.decision_threshold - 0.40) < 1e-4

    # Test invalid boundaries
    with pytest.raises(ValueError):
        engine.set_decision_threshold(-0.01)

    with pytest.raises(ValueError):
        engine.set_decision_threshold(1.05)


def test_api_set_decision_threshold(client):
    """Test POST /api/threshold/set updates threshold dynamically."""
    res = client.post(
        "/api/threshold/set",
        data=json.dumps({"threshold": 0.35}),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert abs(data["new_threshold"] - 0.35) < 1e-4

    # Test invalid input
    res_bad = client.post(
        "/api/threshold/set",
        data=json.dumps({"threshold": 1.5}),
        content_type="application/json",
    )
    assert res_bad.status_code == 400

    # Reset back to default 0.2773
    client.post(
        "/api/threshold/set",
        data=json.dumps({"threshold": 0.2773}),
        content_type="application/json",
    )


# =============================================================================
# 2. WHAT-IF THRESHOLD TRADE-OFF ANALYSIS TESTS
# =============================================================================

def test_what_if_tradeoff_math(adaptive_engine):
    """Verify that lower threshold yields higher recall and more false alerts."""
    res = adaptive_engine.compute_threshold_tradeoffs(cost_per_false_alert=15.0)
    assert res["status"] == "success"
    assert "demonstration_table" in res
    assert "curve_grid" in res
    assert "cost_optimal_threshold" in res

    demo = res["demonstration_table"]
    assert len(demo) >= 5

    # Check key trade-off monotonic trends:
    # As threshold increases, false alarms (fp) must decrease or stay equal
    fp_vals = [row["fp"] for row in demo]
    for i in range(len(fp_vals) - 1):
        assert fp_vals[i] >= fp_vals[i + 1], f"FP at {demo[i]['threshold']} ({fp_vals[i]}) should be >= FP at {demo[i+1]['threshold']} ({fp_vals[i+1]})"


def test_api_what_if_analysis(client):
    """Test GET /api/adaptive/what-if endpoint."""
    res = client.get("/api/adaptive/what-if?cost_review=20.0&fraud_mult=1.2")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "demonstration_table" in data
    assert "curve_grid" in data
    assert "cost_assumptions" in data
    assert data["cost_assumptions"]["cost_per_false_alert"] == 20.0


def test_api_simulate_custom_tradeoffs(client):
    """Test POST /api/adaptive/simulate endpoint."""
    payload = {
        "cost_per_false_alert": 25.0,
        "fraud_multiplier": 1.5,
        "threshold_grid": [0.10, 0.25, 0.50, 0.75],
    }
    res = client.post(
        "/api/adaptive/simulate",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert len(data["curve_grid"]) == 4


# =============================================================================
# 3. MULTI-FEATURE DATA DRIFT TESTS
# =============================================================================

def test_multi_feature_drift(adaptive_engine):
    """Test multi-feature drift computation (PSI, KS, Wasserstein)."""
    dummy_recent = [
        {"amount": 100.0, "time": 50000.0, "raw_features": {"V14": -0.5, "V12": 0.2, "V10": -0.1, "V4": 0.8, "V11": 0.5}},
        {"amount": 45.0, "time": 50010.0, "raw_features": {"V14": 0.1, "V12": -0.2, "V10": 0.0, "V4": 0.1, "V11": -0.3}},
        {"amount": 250.0, "time": 50020.0, "raw_features": {"V14": -1.2, "V12": -0.8, "V10": -0.9, "V4": 1.5, "V11": 1.2}},
    ]
    drift = adaptive_engine.compute_feature_drift(dummy_recent)
    assert drift["status"] == "success"
    assert drift["monitored_feature_count"] >= 5
    assert drift["overall_drift_status"] in ["STABLE", "MODERATE", "CRITICAL"]

    feature_names = [f["feature"] for f in drift["features"]]
    assert "V14" in feature_names
    assert "Amount" in feature_names


def test_api_adaptive_drift(client):
    """Test GET /api/adaptive/drift endpoint."""
    res = client.get("/api/adaptive/drift")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "overall_drift_status" in data
    assert "features" in data
    assert len(data["features"]) > 0


# =============================================================================
# 4. RISK SCORE DISTRIBUTION & QUANTILES TESTS
# =============================================================================

def test_risk_distribution(adaptive_engine):
    """Test empirical risk distribution, histogram deciles, and quantiles."""
    res = adaptive_engine.compute_risk_distribution()
    assert res["status"] == "success"
    assert "quantiles" in res
    assert "histogram" in res
    assert "category_summary" in res

    q = res["quantiles"]
    assert q["p10"] <= q["p50"] <= q["p90"] <= q["p99_9"]
    assert len(res["histogram"]["counts"]) == 10
    assert sum(res["histogram"]["counts"]) == res["total_transactions"]


def test_api_adaptive_risk_distribution(client):
    """Test GET /api/adaptive/risk-distribution endpoint."""
    res = client.get("/api/adaptive/risk-distribution")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    assert "quantiles" in data
    assert "histogram" in data


# =============================================================================
# 5. INVESTIGATION QUEUE PRIORITIZATION TESTS
# =============================================================================

def test_calculate_priority_logic():
    """Verify multi-factor priority score combines risk score and dollar exposure."""
    # High score + High dollar amount -> P1 - CRITICAL
    crit = AdaptiveIntelligenceEngine.calculate_priority(
        risk_score=95, amount=1200.0, is_flagged=True
    )
    assert crit["priority_tier"] == "P1 - CRITICAL"
    assert crit["badge_class"] == "priority-p1"
    assert crit["priority_score"] >= 350

    # Low score + Small dollar amount -> P4 - LOW
    low = AdaptiveIntelligenceEngine.calculate_priority(
        risk_score=10, amount=15.0, is_flagged=False
    )
    assert low["priority_tier"] == "P4 - LOW"
    assert low["badge_class"] == "priority-p4"
    assert low["priority_score"] < 100


def test_queue_priority_sorting(client):
    """Test GET /api/queue returns priority fields and sorts by priority."""
    res = client.get("/api/queue?sort=priority_score&order=desc")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "success"
    items = data["items"]
    assert len(items) > 0

    first = items[0]
    assert "priority_score" in first
    assert "priority_tier" in first
    assert "badge_class" in first

    # Verify descending priority sort
    if len(items) >= 2:
        assert items[0]["priority_score"] >= items[1]["priority_score"]
