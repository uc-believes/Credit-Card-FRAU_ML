"""
Tests for Phase 4: Risk Scoring and Decision Engine
===================================================
Comprehensive unit tests covering:
1. Normalization and clamping of probability inputs (including negatives and > 1.0).
2. Category boundary conditions (0-30 LOW, 31-70 MEDIUM, 71-100 HIGH).
3. Configurable category boundaries.
4. Configurable business decision threshold (ACCEPT vs. REVIEW / FLAGGED FOR INVESTIGATION).
5. Decoupling between score category and decision threshold.
6. Non-guarantee legitimacy disclaimer verification on ACCEPT.
7. Batch evaluation and serialization.
"""

from __future__ import annotations

import pathlib
import sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk.engine import LEGITIMACY_DISCLAIMER, Decision, RiskAssessment, RiskCategory, RiskEngine


# ---------------------------------------------------------------------------
# 1. Normalization & Probability Boundary Tests
# ---------------------------------------------------------------------------

def test_probability_clamping_and_normalization():
    engine = RiskEngine()

    # Exact zero
    assert engine.normalize_score(0.0) == 0
    # Exact one
    assert engine.normalize_score(1.0) == 100

    # Negative inputs clamped to 0
    assert engine.normalize_score(-0.0001) == 0
    assert engine.normalize_score(-10.5) == 0

    # Inputs > 1.0 clamped to 100
    assert engine.normalize_score(1.0001) == 100
    assert engine.normalize_score(5.0) == 100

    # Standard rounding
    assert engine.normalize_score(0.004) == 0
    assert engine.normalize_score(0.005) == 1
    assert engine.normalize_score(0.914) == 91
    assert engine.normalize_score(0.916) == 92

    # Invalid types
    with pytest.raises(TypeError):
        engine.normalize_score("high_prob")  # type: ignore

    with pytest.raises(TypeError):
        engine.normalize_score(None)  # type: ignore


# ---------------------------------------------------------------------------
# 2. Default Category Boundaries (0-30, 31-70, 71-100)
# ---------------------------------------------------------------------------

def test_default_category_boundaries():
    engine = RiskEngine(low_max=30, medium_max=70)

    # Boundary: Score 0 -> LOW
    cat_0, color_0, _ = engine.get_risk_category(0)
    assert cat_0 == "LOW RISK"
    assert cat_0 == "LOW"
    assert color_0 == "#22c55e"

    # Boundary: Score 30 -> LOW (inclusive)
    cat_30, _, _ = engine.get_risk_category(30)
    assert cat_30 == "LOW RISK"
    assert cat_30 == "LOW"

    # Boundary: Score 31 -> MEDIUM
    cat_31, color_31, _ = engine.get_risk_category(31)
    assert cat_31 == "MEDIUM RISK"
    assert cat_31 == "MEDIUM"
    assert color_31 == "#f59e0b"

    # Boundary: Score 70 -> MEDIUM (inclusive)
    cat_70, _, _ = engine.get_risk_category(70)
    assert cat_70 == "MEDIUM RISK"
    assert cat_70 == "MEDIUM"

    # Boundary: Score 71 -> HIGH
    cat_71, color_71, _ = engine.get_risk_category(71)
    assert cat_71 == "HIGH RISK"
    assert cat_71 == "HIGH"
    assert color_71 == "#ef4444"

    # Boundary: Score 100 -> HIGH
    cat_100, _, _ = engine.get_risk_category(100)
    assert cat_100 == "HIGH RISK"
    assert cat_100 == "HIGH"


def test_floating_point_category_transitions():
    engine = RiskEngine(low_max=30, medium_max=70)

    # 0.3049 -> 30 -> LOW RISK
    res_low = engine.evaluate(0.3049)
    assert res_low.risk_score == 30
    assert res_low.risk_category == "LOW RISK"
    assert res_low.risk_level == "LOW"

    # 0.3050 -> 31 -> MEDIUM RISK
    res_med = engine.evaluate(0.3050)
    assert res_med.risk_score == 31
    assert res_med.risk_category == "MEDIUM RISK"
    assert res_med.risk_level == "MEDIUM"

    # 0.7049 -> 70 -> MEDIUM RISK
    res_med_top = engine.evaluate(0.7049)
    assert res_med_top.risk_score == 70
    assert res_med_top.risk_category == "MEDIUM RISK"

    # 0.7050 -> 71 -> HIGH RISK
    res_high = engine.evaluate(0.7050)
    assert res_high.risk_score == 71
    assert res_high.risk_category == "HIGH RISK"
    assert res_high.risk_level == "HIGH"


# ---------------------------------------------------------------------------
# 3. Configurable Category Boundaries
# ---------------------------------------------------------------------------

def test_configurable_category_boundaries():
    # Custom boundaries: 0-15 LOW, 16-50 MEDIUM, 51-100 HIGH
    custom_engine = RiskEngine(low_max=15, medium_max=50)

    res_15 = custom_engine.evaluate(0.15)
    assert res_15.risk_score == 15
    assert res_15.risk_category == "LOW RISK"

    res_16 = custom_engine.evaluate(0.16)
    assert res_16.risk_score == 16
    assert res_16.risk_category == "MEDIUM RISK"

    res_50 = custom_engine.evaluate(0.50)
    assert res_50.risk_score == 50
    assert res_50.risk_category == "MEDIUM RISK"

    res_51 = custom_engine.evaluate(0.51)
    assert res_51.risk_score == 51
    assert res_51.risk_category == "HIGH RISK"

    # Invalid boundary configurations raise ValueError
    with pytest.raises(ValueError):
        RiskEngine(low_max=70, medium_max=30)  # low_max > medium_max

    with pytest.raises(ValueError):
        RiskEngine(low_max=50, medium_max=50)  # low_max == medium_max

    with pytest.raises(ValueError):
        RiskEngine(low_max=-5, medium_max=50)  # negative

    with pytest.raises(ValueError):
        RiskEngine(low_max=50, medium_max=105)  # > 100


# ---------------------------------------------------------------------------
# 4. Decision Threshold Boundary Tests (ACCEPT vs. REVIEW)
# ---------------------------------------------------------------------------

def test_decision_threshold_boundaries():
    # Operating threshold = 0.2773
    engine = RiskEngine(decision_threshold=0.2773)

    # Just below threshold -> ACCEPT
    res_below = engine.evaluate(0.2772)
    assert res_below.decision == "ACCEPT"
    assert res_below.status_label == "LOW RISK"
    assert not res_below.is_flagged
    assert res_below.prediction == 0

    # Exactly at threshold -> REVIEW / FLAGGED FOR INVESTIGATION
    res_exact = engine.evaluate(0.2773)
    assert res_exact.decision == "REVIEW"
    assert res_exact.status_label == "FLAGGED FOR INVESTIGATION"
    assert res_exact.is_flagged
    assert res_exact.prediction == 1

    # Above threshold -> REVIEW / FLAGGED FOR INVESTIGATION
    res_above = engine.evaluate(0.2774)
    assert res_above.decision == "REVIEW"
    assert res_above.status_label == "FLAGGED FOR INVESTIGATION"
    assert res_above.is_flagged
    assert res_above.prediction == 1


def test_custom_decision_threshold_override():
    engine = RiskEngine(decision_threshold=0.2773)

    # Override with conservative threshold 0.10 per call
    res_override = engine.evaluate(0.12, threshold=0.10)
    assert res_override.decision == "REVIEW"
    assert res_override.is_flagged
    assert res_override.decision_threshold == 0.10

    # Override with permissive threshold 0.80
    res_permissive = engine.evaluate(0.75, threshold=0.80)
    assert res_permissive.decision == "ACCEPT"
    assert not res_permissive.is_flagged
    assert res_permissive.decision_threshold == 0.80


# ---------------------------------------------------------------------------
# 5. Decoupling: Probability vs. Risk Score vs. Decision Threshold
# ---------------------------------------------------------------------------

def test_decoupling_score_and_decision():
    """
    Demonstrates that category (score) and business decision (threshold)
    are independent dimensions:
    A transaction can be LOW RISK (score 25) but FLAGGED FOR REVIEW if threshold=0.20.
    """
    engine = RiskEngine(low_max=30, medium_max=70, decision_threshold=0.20)

    # Probability = 0.25 -> Risk Score = 25 (LOW RISK)
    # But 0.25 >= threshold 0.20 -> REVIEW (FLAGGED FOR INVESTIGATION)
    res = engine.evaluate(0.25)
    assert res.risk_score == 25
    assert res.risk_category == "LOW RISK"
    assert res.decision == "REVIEW"
    assert res.status_label == "FLAGGED FOR INVESTIGATION"
    assert res.is_flagged


# ---------------------------------------------------------------------------
# 6. Non-Guarantee Legitimacy Disclaimer Verification
# ---------------------------------------------------------------------------

def test_legitimacy_disclaimer_on_accept():
    engine = RiskEngine()

    res_accept = engine.evaluate(0.05)
    assert res_accept.decision == "ACCEPT"

    # Must contain clear disclaimer that ACCEPT is not a guarantee of legitimacy
    assert "not a guarantee of transaction legitimacy" in res_accept.description.lower()
    assert res_accept.legitimacy_disclaimer == LEGITIMACY_DISCLAIMER
    assert "not constitute a warranty" in res_accept.legitimacy_disclaimer.lower()

    # Formatted summary text check
    summary = res_accept.format_summary()
    assert "Business Decision:     ACCEPT" in summary
    assert "Notice: ACCEPT indicates the transaction satisfies current risk tolerance" in summary


# ---------------------------------------------------------------------------
# 7. Batch Evaluation & Serialization
# ---------------------------------------------------------------------------

def test_batch_evaluation_and_dict_serialization():
    engine = RiskEngine(decision_threshold=0.2773)

    probabilities = [0.02, 0.35, 0.92]
    batch = engine.batch_evaluate(probabilities)

    assert len(batch) == 3
    assert batch[0].decision == "ACCEPT" and batch[0].risk_category == "LOW RISK"
    assert batch[1].decision == "REVIEW" and batch[1].risk_category == "MEDIUM RISK"
    assert batch[2].decision == "REVIEW" and batch[2].risk_category == "HIGH RISK"

    # Dict serialization
    d = batch[0].to_dict()
    assert isinstance(d, dict)
    assert d["risk_score"] == 2
    assert d["decision"] == "ACCEPT"
    assert "legitimacy_disclaimer" in d
