"""
Fraud Shield — Explainability Engine (SHAP)
=============================================
Provides mathematically grounded explanations for individual predictions and
global model dynamics using Shapley values (SHAP TreeExplainer).

CRITICAL GUARANTEES:
1. Exact Additivity:
   base_value + sum(shap_values) == log_odds_margin
   sigmoid(base_value + sum(shap_values)) == model_probability (diff < 1e-10)
2. Directionality:
   Positive SHAP (phi > 0) strictly increases predicted fraud risk.
   Negative SHAP (phi < 0) strictly mitigates predicted fraud risk.
3. Non-Causal Distinction:
   All explanations carry an explicit scientific disclaimer distinguishing
   empirical statistical correlation from real-world causality.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import scipy.special
import shap

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CAUSAL_DISCLAIMER = (
    "Feature contributions indicate statistical associations learned by the model "
    "from historical training data. They represent empirical risk signals and do "
    "NOT imply real-world causation, motive, or legal culpability."
)


@dataclass
class FeatureContribution:
    """Detailed attribution of a single feature to a transaction's prediction."""

    feature_name: str
    feature_value: float
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk" | "neutral"
    rank: int
    relative_impact: float  # Percentage of total positive/negative attribution

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransactionExplanation:
    """Complete mathematical and narrative explanation of a transaction prediction."""

    probability: float
    risk_score: int
    risk_level: str
    prediction: int
    base_value: float
    margin_value: float
    top_contributing_factors: list[FeatureContribution]
    mitigating_factors: list[FeatureContribution]
    all_features: list[FeatureContribution] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    raw_shap_values: list[float] = field(default_factory=list)
    raw_feature_values: list[float] = field(default_factory=list)
    disclaimer: str = DEFAULT_CAUSAL_DISCLAIMER

    def to_dict(self, include_all_features: bool = False) -> dict[str, Any]:
        d = {
            "probability": round(self.probability, 6),
            "risk_score": self.risk_score,
            "risk_level": self.risk_level,
            "prediction": self.prediction,
            "base_value": round(self.base_value, 6),
            "margin_value": round(self.margin_value, 6),
            "top_contributing_factors": [
                f.to_dict() for f in self.top_contributing_factors
            ],
            "mitigating_factors": [f.to_dict() for f in self.mitigating_factors],
            "disclaimer": self.disclaimer,
        }
        if include_all_features:
            d["all_features"] = [f.to_dict() for f in self.all_features]
            d["feature_names"] = self.feature_names
            d["raw_shap_values"] = self.raw_shap_values
            d["raw_feature_values"] = self.raw_feature_values
        return d

    def format_summary(self, top_n: int = 5) -> str:
        """User-friendly text summary matching project reporting guidelines."""
        lines = [
            f"Probability: {self.probability:.4f}",
            f"Risk Score:  {self.risk_score}",
            f"Risk Level:  {self.risk_level}",
            f"Prediction:  {'FRAUD ALERT' if self.prediction == 1 else 'LEGITIMATE'}",
            "",
            "Top contributing factors (increasing fraud risk):",
        ]
        if not self.top_contributing_factors:
            lines.append("  (None — no features pushed prediction towards fraud)")
        else:
            for i, factor in enumerate(self.top_contributing_factors[:top_n], 1):
                sign = "+" if factor.shap_value > 0 else ""
                lines.append(
                    f"  {i}. {factor.feature_name:<16} (value={factor.feature_value:.3f}, "
                    f"SHAP={sign}{factor.shap_value:.4f}, impact={factor.relative_impact*100:.1f}%)"
                )

        if self.mitigating_factors:
            lines.append("")
            lines.append("Primary mitigating factors (reducing fraud risk):")
            for i, factor in enumerate(self.mitigating_factors[:3], 1):
                lines.append(
                    f"  {i}. {factor.feature_name:<16} (value={factor.feature_value:.3f}, "
                    f"SHAP={factor.shap_value:.4f})"
                )

        lines.append("")
        lines.append(f"Note: {self.disclaimer}")
        return "\n".join(lines)


class FraudExplainer:
    """
    Computes local and global Shapley feature attributions for fraud models.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Sequence[str],
        config: Optional[dict] = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.cfg = config if config is not None else get_config()
        self.top_k_default = int(
            self.cfg.get("explainability", {}).get("top_k_features", 10)
        )
        self.disclaimer = self.cfg.get("explainability", {}).get(
            "disclaimer", DEFAULT_CAUSAL_DISCLAIMER
        ).strip()

        # Initialize TreeExplainer
        logger.info("Initializing shap.TreeExplainer on %s...", type(model).__name__)
        self.explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )
        # Expected base value (margin/log-odds)
        if hasattr(self.explainer, "expected_value"):
            exp_val = self.explainer.expected_value
            if isinstance(exp_val, (list, np.ndarray)):
                self.base_value = float(exp_val[1] if len(exp_val) > 1 else exp_val[0])
            else:
                self.base_value = float(exp_val)
        else:
            self.base_value = 0.0

    def explain_transaction(
        self,
        x_features: np.ndarray,
        probability: float,
        risk_score: int,
        risk_level: str,
        prediction: int,
        top_k: Optional[int] = None,
    ) -> TransactionExplanation:
        """
        Generate local SHAP explanation for a single preprocessed transaction vector.

        Args:
            x_features: 1D or 2D array of shape (n_features,) or (1, n_features).
            probability: Predicted fraud probability in [0, 1].
            risk_score: 0-100 risk score.
            risk_level: 'LOW', 'MEDIUM', 'HIGH'.
            prediction: Binary alert (0 or 1).
            top_k: Number of top features to isolate.

        Returns:
            Populated TransactionExplanation.
        """
        k = top_k if top_k is not None else self.top_k_default
        x_vec = np.asarray(x_features, dtype=float).ravel()
        if len(x_vec) != len(self.feature_names):
            raise ValueError(
                f"Feature vector length ({len(x_vec)}) does not match expected "
                f"number of features ({len(self.feature_names)})."
            )

        # Compute SHAP values for single sample (1, n_features)
        X_2d = x_vec.reshape(1, -1)
        shap_res = self.explainer(X_2d)

        # Extract values
        raw_shaps = shap_res.values[0]
        base_val = float(shap_res.base_values[0]) if hasattr(shap_res, "base_values") else self.base_value
        margin = float(base_val + np.sum(raw_shaps))

        # Separate positive contributors (pushing towards fraud)
        # and negative contributors (mitigating fraud)
        pos_indices = np.where(raw_shaps > 0)[0]
        neg_indices = np.where(raw_shaps < 0)[0]

        # Sort positive by magnitude descending
        pos_sorted = pos_indices[np.argsort(-raw_shaps[pos_indices])]
        # Sort negative by magnitude descending (most negative first)
        neg_sorted = neg_indices[np.argsort(raw_shaps[neg_indices])]

        total_pos = float(np.sum(raw_shaps[pos_indices])) if len(pos_indices) > 0 else 1.0
        total_neg = float(np.sum(np.abs(raw_shaps[neg_indices]))) if len(neg_indices) > 0 else 1.0

        top_contributors: list[FeatureContribution] = []
        for rank, idx in enumerate(pos_sorted[:k], 1):
            s_val = float(raw_shaps[idx])
            top_contributors.append(
                FeatureContribution(
                    feature_name=self.feature_names[idx],
                    feature_value=round(float(x_vec[idx]), 4),
                    shap_value=round(s_val, 4),
                    direction="increases_risk",
                    rank=rank,
                    relative_impact=round(s_val / total_pos, 4) if total_pos > 0 else 0.0,
                )
            )

        mitigating: list[FeatureContribution] = []
        for rank, idx in enumerate(neg_sorted[:k], 1):
            s_val = float(raw_shaps[idx])
            mitigating.append(
                FeatureContribution(
                    feature_name=self.feature_names[idx],
                    feature_value=round(float(x_vec[idx]), 4),
                    shap_value=round(s_val, 4),
                    direction="decreases_risk",
                    rank=rank,
                    relative_impact=round(abs(s_val) / total_neg, 4) if total_neg > 0 else 0.0,
                )
            )

        # All features ordered by absolute impact
        all_sorted = np.argsort(-np.abs(raw_shaps))
        all_features: list[FeatureContribution] = []
        for rank, idx in enumerate(all_sorted, 1):
            s_val = float(raw_shaps[idx])
            direction = (
                "increases_risk"
                if s_val > 1e-6
                else "decreases_risk"
                if s_val < -1e-6
                else "neutral"
            )
            all_features.append(
                FeatureContribution(
                    feature_name=self.feature_names[idx],
                    feature_value=round(float(x_vec[idx]), 4),
                    shap_value=round(s_val, 4),
                    direction=direction,
                    rank=rank,
                    relative_impact=round(abs(s_val) / (np.sum(np.abs(raw_shaps)) + 1e-12), 4),
                )
            )

        return TransactionExplanation(
            probability=probability,
            risk_score=risk_score,
            risk_level=risk_level,
            prediction=prediction,
            base_value=base_val,
            margin_value=margin,
            top_contributing_factors=top_contributors,
            mitigating_factors=mitigating,
            all_features=all_features,
            feature_names=self.feature_names,
            raw_shap_values=[round(float(v), 6) for v in raw_shaps],
            raw_feature_values=[round(float(v), 4) for v in x_vec],
            disclaimer=self.disclaimer,
        )

    def explain_global(
        self,
        X_sample: np.ndarray,
        max_samples: int = 500,
    ) -> list[dict[str, Any]]:
        """
        Compute global feature importance by averaging absolute SHAP values.

        Args:
            X_sample: Preprocessed feature matrix.
            max_samples: Subsample size to ensure fast execution.

        Returns:
            List of dicts sorted by mean absolute SHAP value descending.
        """
        X_arr = np.asarray(X_sample, dtype=float)
        if len(X_arr) > max_samples:
            indices = np.random.RandomState(42).choice(
                len(X_arr), size=max_samples, replace=False
            )
            X_arr = X_arr[indices]

        logger.info("Computing global SHAP values over %d samples...", len(X_arr))
        shap_res = self.explainer(X_arr)
        mean_abs_shaps = np.mean(np.abs(shap_res.values), axis=0)

        total_importance = float(np.sum(mean_abs_shaps)) + 1e-12
        sorted_indices = np.argsort(-mean_abs_shaps)

        global_importance: list[dict[str, Any]] = []
        for rank, idx in enumerate(sorted_indices, 1):
            val = float(mean_abs_shaps[idx])
            global_importance.append({
                "rank": rank,
                "feature": self.feature_names[idx],
                "mean_abs_shap": round(val, 6),
                "importance_share": round(val / total_importance, 4),
            })

        return global_importance
