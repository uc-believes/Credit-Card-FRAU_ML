"""
Tests for Phase 3: Explainable Fraud Prediction
================================================
Verifies:
1. RiskEngine categorization, scoring, and thresholds.
2. FraudExplainer exact mathematical additivity and feature preservation.
3. Non-causal disclaimer presence.
4. ExplainerVisualizer chart and base64 generation.
5. FraudIntelligenceService end-to-end execution.
"""

from __future__ import annotations

import pathlib
import sys
import numpy as np
import pandas as pd
import pytest
import scipy.special

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.risk.engine import RiskEngine
from src.explainability.explainer import FraudExplainer, FeatureContribution
from src.explainability.visualizer import ExplainerVisualizer
from src.explainability.service import FraudIntelligenceService
from src.utils.serialization import load_model


def test_risk_engine_scoring_and_categories():
    engine = RiskEngine(operating_threshold=0.2773)

    # Low Risk
    res_low = engine.evaluate(0.08)
    assert res_low.risk_score == 8
    assert res_low.risk_category == "LOW"
    assert res_low.decision == "ALLOW"
    assert res_low.prediction == 0

    # Medium Risk
    res_med = engine.evaluate(0.45)
    assert res_med.risk_score == 45
    assert res_med.risk_category == "MEDIUM"
    assert res_med.decision == "MONITOR"
    assert res_med.prediction == 1  # 0.45 >= 0.2773 threshold

    # High Risk
    res_high = engine.evaluate(0.914)
    assert res_high.risk_score == 91
    assert res_high.risk_category == "HIGH"
    assert res_high.decision == "REVIEW"
    assert res_high.prediction == 1

    # Clamping boundaries
    assert engine.compute_risk_score(-0.2) == 0
    assert engine.compute_risk_score(1.5) == 100


def test_shap_exact_additivity_and_directionality():
    model = load_model("best_model")
    feature_names = (
        ["Amount", "Time", "log_amount", "hour_of_day"]
        + [f"V{i}" for i in range(1, 29)]
        + ["is_night"]
    )
    explainer = FraudExplainer(model=model, feature_names=feature_names)

    # Load 5 sample rows from validation parquet
    val_path = ROOT / "data" / "processed" / "validation.parquet"
    assert val_path.exists(), "validation.parquet missing!"
    df_val = pd.read_parquet(val_path)
    X_val = df_val.drop(columns=["Class"]).values[:5]

    for i in range(len(X_val)):
        sample_x = X_val[i]
        prob = float(model.predict_proba(sample_x.reshape(1, -1))[:, 1][0])

        explanation = explainer.explain_transaction(
            x_features=sample_x,
            probability=prob,
            risk_score=int(round(prob * 100)),
            risk_level="HIGH" if prob > 0.7 else "LOW",
            prediction=1 if prob >= 0.2773 else 0,
        )

        # 1. Exact mathematical additivity:
        # sigmoid(base_value + sum(shap_values)) == model_probability
        sum_shaps = float(np.sum(explanation.raw_shap_values))
        computed_prob = float(scipy.special.expit(explanation.base_value + sum_shaps))
        diff = abs(prob - computed_prob)
        assert diff < 1e-6, f"SHAP additivity violated! diff={diff}"

        # 2. Check directionality of top contributors
        for factor in explanation.top_contributing_factors:
            assert factor.shap_value > 0
            assert factor.direction == "increases_risk"

        for factor in explanation.mitigating_factors:
            assert factor.shap_value < 0
            assert factor.direction == "decreases_risk"

        # 3. Check disclaimer
        assert "not imply" in explanation.disclaimer.lower()


def test_explainer_visualizer():
    model = load_model("best_model")
    feature_names = (
        ["Amount", "Time", "log_amount", "hour_of_day"]
        + [f"V{i}" for i in range(1, 29)]
        + ["is_night"]
    )
    explainer = FraudExplainer(model=model, feature_names=feature_names)
    visualizer = ExplainerVisualizer()

    val_path = ROOT / "data" / "processed" / "validation.parquet"
    df_val = pd.read_parquet(val_path)
    sample_x = df_val.drop(columns=["Class"]).values[0]

    explanation = explainer.explain_transaction(
        x_features=sample_x,
        probability=0.05,
        risk_score=5,
        risk_level="LOW",
        prediction=0,
    )

    b64_chart = visualizer.plot_local_attributions(explanation, top_n=8)
    assert isinstance(b64_chart, str)
    assert len(b64_chart) > 500  # Valid base64 payload


def test_end_to_end_fraud_intelligence_service():
    service = FraudIntelligenceService()

    # Raw transaction dictionary
    raw_tx = {
        "Time": 406.0,
        "Amount": 199.99,
    }
    for i in range(1, 29):
        raw_tx[f"V{i}"] = -1.5 if i in [14, 10, 12, 17] else 0.1

    result = service.predict_and_explain(raw_tx, top_k=5, generate_plot=True)

    assert "probability" in result
    assert 0.0 <= result["probability"] <= 1.0
    assert "risk_score" in result
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    assert result["prediction"] in [0, 1]
    assert "top_contributing_factors" in result
    assert len(result["top_contributing_factors"]) <= 5
    assert "summary_text" in result
    assert "Probability:" in result["summary_text"]
    assert "Risk Score:" in result["summary_text"]
    assert "Risk Level:" in result["summary_text"]
    assert result["plot_base64"] is not None
    assert "disclaimer" in result
