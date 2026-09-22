"""
Fraud Shield — Risk Engine
===========================
Translates continuous fraud probabilities into calibrated risk scores (0–100),
actionable risk categories (LOW, MEDIUM, HIGH), and operational recommendations.

Configurable via configs/config.yaml under `risk_engine`.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RiskAssessment:
    """Structured assessment of transaction risk."""

    probability: float
    risk_score: int
    risk_category: str
    decision: str
    color: str
    prediction: int
    threshold: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Convert assessment to serializable dictionary."""
        return asdict(self)


class RiskEngine:
    """
    Evaluates risk score and operational category from model probability.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        operating_threshold: Optional[float] = None,
    ) -> None:
        self.cfg = config if config is not None else get_config()
        self.risk_cfg = self.cfg.get("risk_engine", {})
        self.thresholds_cfg = self.risk_cfg.get("thresholds", {})
        self.categories_cfg = self.risk_cfg.get("categories", {})

        self.low_max = int(self.thresholds_cfg.get("low_max", 30))
        self.medium_max = int(self.thresholds_cfg.get("medium_max", 70))

        # Default operating threshold if not supplied
        if operating_threshold is not None:
            self.operating_threshold = float(operating_threshold)
        else:
            self.operating_threshold = float(
                self.cfg.get("threshold", {}).get("default", 0.5)
            )

    def compute_risk_score(self, probability: float) -> int:
        """
        Convert continuous fraud probability in [0, 1] to integer risk score [0, 100].

        Args:
            probability: Fraud probability in [0, 1].

        Returns:
            Integer score between 0 and 100.
        """
        clamped = max(0.0, min(1.0, float(probability)))
        return int(round(clamped * 100))

    def evaluate(
        self,
        probability: float,
        threshold: Optional[float] = None,
    ) -> RiskAssessment:
        """
        Evaluate full risk profile for a transaction.

        Args:
            probability: Raw fraud probability in [0, 1].
            threshold: Optional threshold override.

        Returns:
            Populated RiskAssessment dataclass.
        """
        operating_thresh = (
            float(threshold) if threshold is not None else self.operating_threshold
        )
        score = self.compute_risk_score(probability)
        prediction = 1 if probability >= operating_thresh else 0

        if score <= self.low_max:
            cat_key = "low"
        elif score <= self.medium_max:
            cat_key = "medium"
        else:
            cat_key = "high"

        meta = self.categories_cfg.get(cat_key, {})
        label = meta.get("label", cat_key.upper())
        color = meta.get("color", "#6b7280")
        decision = meta.get("decision", "ALLOW" if cat_key == "low" else "REVIEW")
        description = meta.get(
            "description", f"Transaction classified as {label} risk."
        )

        return RiskAssessment(
            probability=round(float(probability), 6),
            risk_score=score,
            risk_category=label,
            decision=decision,
            color=color,
            prediction=prediction,
            threshold=round(operating_thresh, 4),
            description=description,
        )
