"""
Tests for Model Inference & Serialization
=========================================
Verifies that:
1. All saved models can be loaded.
2. The best model can run inference on preprocessed data.
3. Raw data can pass through the preprocessing pipeline and into best_model.
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.serialization import load_model, model_exists, load_metrics, load_json
from src.features.pipeline import PreprocessingPipeline


def test_saved_models_exist():
    expected_models = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost",
        "best_model",
    ]
    for m in expected_models:
        assert model_exists(m), f"Model {m} was not saved!"


def test_best_model_inference():
    best_model = load_model("best_model")
    # 33 features
    dummy_input = np.random.randn(5, 33)
    preds = best_model.predict(dummy_input)
    probas = best_model.predict_proba(dummy_input)[:, 1]

    assert len(preds) == 5
    assert len(probas) == 5
    assert (probas >= 0.0).all() and (probas <= 1.0).all()


def test_end_to_end_raw_transaction_inference():
    from src.features.engineering import FeatureEngineer

    # Load pipeline and best model
    pipe = PreprocessingPipeline.load()
    best_model = load_model("best_model")
    meta = load_json(ROOT / "artifacts" / "models" / "best_model_metadata.json")

    threshold = meta.get("optimal_threshold", 0.5)

    # Construct single sample raw transaction
    sample = {
        "Time": [100.0],
        "Amount": [149.99],
    }
    for i in range(1, 29):
        sample[f"V{i}"] = [0.0]

    raw_df = pd.DataFrame(sample)
    fe = FeatureEngineer()
    engineered_df = fe.transform(raw_df)

    processed_features = pipe.transform(engineered_df)

    assert processed_features.shape == (1, 33)

    prob = float(best_model.predict_proba(processed_features)[:, 1][0])
    is_fraud = prob >= threshold

    assert 0.0 <= prob <= 1.0
    assert isinstance(is_fraud, (bool, np.bool_))
