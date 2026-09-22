"""
Fraud Shield — Fraud Risk Intelligence Service
===============================================
Unified service combining preprocessing, model inference, calibrated risk
scoring, SHAP explanations, and dashboard visualization rendering.

Usage:
    from src.explainability.service import FraudIntelligenceService

    service = FraudIntelligenceService()
    result = service.predict_and_explain({
        "Time": 406.0,
        "Amount": 150.0,
        "V1": -2.31, ...
    })
    print(result["summary_text"])
"""

from __future__ import annotations

import logging
import pathlib
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from src.explainability.explainer import FraudExplainer, TransactionExplanation
from src.explainability.visualizer import ExplainerVisualizer
from src.features.engineering import FeatureEngineer
from src.features.pipeline import PreprocessingPipeline
from src.risk.engine import RiskAssessment, RiskEngine
from src.utils.config import get_config, get_project_root
from src.utils.logger import get_logger
from src.utils.serialization import load_json, load_model

logger = get_logger(__name__)


class FraudIntelligenceService:
    """
    End-to-end interface for explainable credit card fraud detection.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        pipeline: Optional[PreprocessingPipeline] = None,
        risk_engine: Optional[RiskEngine] = None,
        explainer: Optional[FraudExplainer] = None,
        visualizer: Optional[ExplainerVisualizer] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.root = get_project_root()
        self.cfg = config if config is not None else get_config()

        # 1. Load pipeline & feature engineer
        self.fe = FeatureEngineer()
        self.pipeline = (
            pipeline if pipeline is not None else PreprocessingPipeline.load()
        )

        # 2. Load best model & operating threshold
        if model is not None:
            self.model = model
            thresh = float(self.cfg.get("threshold", {}).get("default", 0.5))
        else:
            self.model = load_model("best_model")
            meta_path = self.root / "artifacts" / "models" / "best_model_metadata.json"
            if meta_path.exists():
                meta = load_json(meta_path)
                thresh = float(meta.get("optimal_threshold", 0.5))
            else:
                thresh = float(self.cfg.get("threshold", {}).get("default", 0.5))

        self.operating_threshold = thresh

        # 3. Risk Engine
        self.risk_engine = (
            risk_engine
            if risk_engine is not None
            else RiskEngine(
                config=self.cfg, operating_threshold=self.operating_threshold
            )
        )

        # 4. Feature Names
        # Scaled pipeline output feature order
        self.feature_names = self._resolve_feature_names()

        # 5. Explainer
        self.explainer = (
            explainer
            if explainer is not None
            else FraudExplainer(
                model=self.model,
                feature_names=self.feature_names,
                config=self.cfg,
            )
        )

        # 6. Visualizer
        self.visualizer = (
            visualizer if visualizer is not None else ExplainerVisualizer()
        )

        logger.info(
            "FraudIntelligenceService ready (Model: %s, Threshold: %.4f, Features: %d)",
            type(self.model).__name__,
            self.operating_threshold,
            len(self.feature_names),
        )

    def _resolve_feature_names(self) -> list[str]:
        """Extract output feature names from metadata or pipeline."""
        pipe_meta_path = (
            self.root / "artifacts" / "pipelines" / "preprocessing_metadata.json"
        )
        if pipe_meta_path.exists():
            try:
                meta = load_json(pipe_meta_path)
                if "feature_names_out" in meta:
                    return list(meta["feature_names_out"])
            except Exception as e:
                logger.warning("Could not read pipeline metadata feature names: %s", e)

        # Fallback to standard 33 features
        return (
            ["Amount", "Time", "log_amount", "hour_of_day"]
            + [f"V{i}" for i in range(1, 29)]
            + ["is_night"]
        )

    def preprocess_raw(
        self, raw_input: Union[dict[str, Any], pd.DataFrame, pd.Series]
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Run raw transaction through FeatureEngineer and PreprocessingPipeline.

        Returns:
            (scaled_features_array, engineered_dataframe)
        """
        if isinstance(raw_input, dict):
            df_raw = pd.DataFrame([raw_input])
        elif isinstance(raw_input, pd.Series):
            df_raw = pd.DataFrame([raw_input.to_dict()])
        elif isinstance(raw_input, pd.DataFrame):
            df_raw = raw_input.copy()
        else:
            raise TypeError(f"Unsupported input type: {type(raw_input)}")

        # Drop target if present
        target_col = self.cfg.get("dataset", {}).get("target_column", "Class")
        if target_col in df_raw.columns:
            df_raw = df_raw.drop(columns=[target_col])

        # Feature Engineering (creates log_amount, hour_of_day, is_night)
        df_eng = self.fe.transform(df_raw)

        # Scaling through fitted pipeline
        X_scaled = self.pipeline.transform(df_eng)
        return X_scaled, df_eng

    def predict_and_explain(
        self,
        raw_input: Union[dict[str, Any], pd.DataFrame, pd.Series],
        top_k: int = 5,
        generate_plot: bool = False,
        plot_output_path: Optional[pathlib.Path] = None,
    ) -> dict[str, Any]:
        """
        Predict fraud risk and produce mathematical SHAP attribution for a transaction.

        Args:
            raw_input: Raw transaction features (dict or single-row DataFrame).
            top_k: Number of top contributing factors to highlight.
            generate_plot: Whether to generate local attribution chart.
            plot_output_path: Optional path to save PNG file.

        Returns:
            Dictionary containing prediction, risk assessment, SHAP explanation,
            narrative summary, and optional visualization.
        """
        X_scaled, df_eng = self.preprocess_raw(raw_input)
        feature_vec = X_scaled[0]

        # 1. Model Inference
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(X_scaled)[:, 1][0])
        else:
            probability = float(self.model.predict(X_scaled)[0])

        # 2. Risk Engine Assessment
        assessment = self.risk_engine.evaluate(probability)

        # 3. SHAP Explanation
        explanation = self.explainer.explain_transaction(
            x_features=feature_vec,
            probability=assessment.probability,
            risk_score=assessment.risk_score,
            risk_level=assessment.risk_category,
            prediction=assessment.prediction,
            top_k=top_k,
        )

        # 4. Optional Plot Generation
        plot_b64 = None
        if generate_plot:
            plot_b64 = self.visualizer.plot_local_attributions(
                explanation=explanation,
                output_path=plot_output_path,
                top_n=top_k * 2,
            )

        # Format output payload
        return {
            "probability": assessment.probability,
            "risk_score": assessment.risk_score,
            "risk_level": assessment.risk_category,
            "prediction": assessment.prediction,
            "decision": assessment.decision,
            "color": assessment.color,
            "threshold": assessment.threshold,
            "description": assessment.description,
            "base_value": explanation.base_value,
            "margin_value": explanation.margin_value,
            "top_contributing_factors": [
                f.to_dict() for f in explanation.top_contributing_factors
            ],
            "mitigating_factors": [
                f.to_dict() for f in explanation.mitigating_factors
            ],
            "disclaimer": explanation.disclaimer,
            "summary_text": explanation.format_summary(top_n=top_k),
            "plot_base64": plot_b64,
            "explanation_object": explanation,
        }
