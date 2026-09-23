"""
Fraud Shield — Risk Scoring and Decision Engine
=================================================
Phase 4 Implementation: Configurable Risk Engine & Decision Policy.

Transforms continuous model fraud probabilities into:
1. Normalized Risk Score (0–100 integer)
2. Risk Category (0–30: LOW / LOW RISK, 31–70: MEDIUM / MEDIUM RISK, 71–100: HIGH / HIGH RISK)
3. Business Decision (ACCEPT vs. REVIEW / FLAGGED FOR INVESTIGATION)

CRITICAL GOVERNANCE PRINCIPLES:
- Thresholds are fully configurable (categories and decision cutoffs).
- An "ACCEPT" decision is strictly an operational risk clearance under current thresholds;
  it is NEVER a warranty or guarantee of transaction legitimacy.
- Clear conceptual separation between:
  * Model Probability: statistical output of classifier in [0, 1]
  * Normalized Risk Score: standardized integer (0-100) for human interpretation
  * Business Decision Threshold: operational policy cutoff for action triage
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence, Union

from src.utils.config import get_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard regulatory and operational notice
LEGITIMACY_DISCLAIMER: str = (
    "ACCEPT indicates the transaction satisfies current risk tolerance criteria "
    "for automated clearance. It does NOT constitute a warranty, certification, "
    "or guarantee of transaction legitimacy. All transactions remain subject to "
    "post-settlement dispute and chargeback rules."
)


class RiskCategory(str):
    """
    Risk category string subclass supporting both standard and verbose labels.
    e.g. RiskCategory('LOW RISK') matches both 'LOW' and 'LOW RISK'.
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        s_clean = str(self).upper().replace(" RISK", "").strip()
        o_clean = other.upper().replace(" RISK", "").strip()
        return s_clean == o_clean

    def __hash__(self) -> int:
        return hash(str(self).upper().replace(" RISK", "").strip())


class Decision(str):
    """
    Decision string subclass supporting operational terminology.
    e.g. Decision('ACCEPT') matches 'ACCEPT' and legacy 'ALLOW'.
    Decision('REVIEW') matches 'REVIEW', 'MONITOR', and 'FLAGGED FOR INVESTIGATION'.
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, str):
            return False
        val = str(self).upper()
        oth = other.upper()
        if val == "ACCEPT" and oth in ("ACCEPT", "ALLOW"):
            return True
        if val == "REVIEW" and oth in ("REVIEW", "MONITOR", "FLAGGED FOR INVESTIGATION"):
            return True
        return super().__eq__(other)

    def __hash__(self) -> int:
        val = str(self).upper()
        if val in ("ACCEPT", "ALLOW"):
            return hash("ACCEPT")
        if val in ("REVIEW", "MONITOR", "FLAGGED FOR INVESTIGATION"):
            return hash("REVIEW")
        return super().__hash__()


@dataclass
class RiskAssessment:
    """
    Structured outcome of risk scoring and operational decision evaluation.
    """

    probability: float
    risk_score: int
    risk_category: RiskCategory  # "LOW RISK" / "LOW", "MEDIUM RISK", "HIGH RISK"
    decision: Decision  # "ACCEPT" or "REVIEW"
    decision_threshold: float
    is_flagged: bool
    status_label: str  # "LOW RISK" / "FLAGGED FOR INVESTIGATION"
    color: str
    prediction: int  # 1 (Fraud Alert) or 0 (Clearance)
    threshold: float  # Alias for decision_threshold
    description: str
    legitimacy_disclaimer: str = LEGITIMACY_DISCLAIMER

    @property
    def risk_level(self) -> str:
        """Shorthand risk level (LOW, MEDIUM, HIGH) for backward compatibility."""
        s = str(self.risk_category).upper()
        if "LOW" in s:
            return "LOW"
        elif "MEDIUM" in s:
            return "MEDIUM"
        return "HIGH"

    def to_dict(self) -> dict[str, Any]:
        """Convert assessment to serializable dictionary."""
        d = asdict(self)
        d["risk_category"] = str(self.risk_category)
        d["decision"] = str(self.decision)
        d["risk_level"] = self.risk_level
        return d

    def format_summary(self) -> str:
        """User-friendly summary string matching prompt specification."""
        lines = [
            f"Probability:           {self.probability:.4f}",
            f"Normalized Risk Score: {self.risk_score}/100",
            f"Risk Category:         {self.risk_category}",
            f"Business Decision:     {self.decision} ({self.status_label})",
            f"Decision Threshold:    {self.decision_threshold:.4f}",
            "",
            f"Assessment: {self.description}",
        ]
        if self.decision == "ACCEPT":
            lines.append("")
            lines.append(f"Notice: {self.legitimacy_disclaimer}")
        return "\n".join(lines)


class RiskEngine:
    """
    Configurable Risk Scoring and Business Decision Engine.
    """

    def __init__(
        self,
        low_max: Optional[int] = None,
        medium_max: Optional[int] = None,
        decision_threshold: Optional[float] = None,
        operating_threshold: Optional[float] = None,  # Backward compatibility alias
        config: Optional[dict] = None,
    ) -> None:
        self.cfg = config if config is not None else get_config()
        risk_cfg = self.cfg.get("risk_engine", {})
        thresholds_cfg = risk_cfg.get("thresholds", {})

        # 1. Category Boundaries (default: 0-30 LOW, 31-70 MEDIUM, 71-100 HIGH)
        l_max = (
            low_max
            if low_max is not None
            else int(thresholds_cfg.get("low_max", 30))
        )
        m_max = (
            medium_max
            if medium_max is not None
            else int(thresholds_cfg.get("medium_max", 70))
        )

        if not (0 <= l_max < m_max <= 100):
            raise ValueError(
                f"Invalid category boundaries: low_max ({l_max}) must be < "
                f"medium_max ({m_max}) and both must lie within [0, 100]."
            )

        self.low_max = l_max
        self.medium_max = m_max

        # 2. Configurable Decision Threshold
        # Preference: explicit decision_threshold > operating_threshold > config > default 0.2773
        if decision_threshold is not None:
            dt = float(decision_threshold)
        elif operating_threshold is not None:
            dt = float(operating_threshold)
        elif "decision_threshold" in risk_cfg:
            dt = float(risk_cfg["decision_threshold"])
        else:
            dt = float(self.cfg.get("threshold", {}).get("default", 0.2773))

        if not (0.0 <= dt <= 1.0):
            raise ValueError(
                f"Invalid decision_threshold: {dt} must lie within [0.0, 1.0]."
            )

        self.decision_threshold = dt
        self.operating_threshold = dt  # Alias

        # Action and category wording
        self.actions_cfg = risk_cfg.get("actions", {})
        self.categories_cfg = risk_cfg.get("categories", {})
        self.legitimacy_disclaimer = risk_cfg.get(
            "legitimacy_disclaimer", LEGITIMACY_DISCLAIMER
        ).strip()

        logger.info(
            "RiskEngine initialized (low_max=%d, medium_max=%d, decision_threshold=%.4f)",
            self.low_max,
            self.medium_max,
            self.decision_threshold,
        )

    def normalize_score(self, probability: float) -> int:
        """
        Normalize a raw model probability in [0, 1] to an integer score in [0, 100].

        Boundary conditions:
            - Clamps negative values to 0.0 with warning.
            - Clamps values > 1.0 to 1.0 with warning.
            - Uses standard mathematical rounding to integer [0, 100].

        Args:
            probability: Model fraud probability.

        Returns:
            Normalized risk score between 0 and 100.
        """
        try:
            prob_float = float(probability)
        except (TypeError, ValueError) as err:
            raise TypeError(
                f"Probability must be a numeric float, got {type(probability)}: {probability}"
            ) from err

        if prob_float < 0.0:
            logger.warning(
                "Input probability %.4f < 0.0; clamping to 0.0", prob_float
            )
            prob_float = 0.0
        elif prob_float > 1.0:
            logger.warning(
                "Input probability %.4f > 1.0; clamping to 1.0", prob_float
            )
            prob_float = 1.0

        import math
        return int(math.floor(prob_float * 100.0 + 0.5))

    def compute_risk_score(self, probability: float) -> int:
        """Backward-compatible alias for normalize_score()."""
        return self.normalize_score(probability)

    def get_risk_category(self, score: int) -> tuple[RiskCategory, str, str]:
        """
        Determine risk category, color, and description from normalized score.

        Default categories:
            0–30:   LOW RISK
            31–70:  MEDIUM RISK
            71–100: HIGH RISK

        Args:
            score: Integer normalized risk score in [0, 100].

        Returns:
            (category_label, color_code, description)
        """
        if score <= self.low_max:
            key = "low"
            default_label = "LOW RISK"
            default_color = "#22c55e"
            default_desc = "Transaction exhibits low probability of fraud based on historical models."
        elif score <= self.medium_max:
            key = "medium"
            default_label = "MEDIUM RISK"
            default_color = "#f59e0b"
            default_desc = "Transaction exhibits elevated statistical anomaly or fraud indicators."
        else:
            key = "high"
            default_label = "HIGH RISK"
            default_color = "#ef4444"
            default_desc = "Transaction exhibits strong probability of unauthorized activity."

        cat_meta = self.categories_cfg.get(key, {})
        label = RiskCategory(cat_meta.get("label", default_label))
        color = cat_meta.get("color", default_color)
        description = cat_meta.get("description", default_desc)

        return label, color, description

    def evaluate_decision(
        self, probability: float, threshold: Optional[float] = None
    ) -> tuple[Decision, str, bool]:
        """
        Evaluate business action against the decision threshold.

        Rule:
            probability >= decision_threshold -> REVIEW ("FLAGGED FOR INVESTIGATION")
            otherwise                         -> ACCEPT ("LOW RISK")

        Args:
            probability: Fraud probability in [0, 1].
            threshold: Optional threshold override.

        Returns:
            (decision, status_label, is_flagged)
        """
        t = float(threshold) if threshold is not None else self.decision_threshold
        prob_float = max(0.0, min(1.0, float(probability)))

        accept_label = Decision(self.actions_cfg.get("accept", "ACCEPT"))
        review_label = Decision(self.actions_cfg.get("review", "REVIEW"))
        flagged_label = self.actions_cfg.get(
            "flagged", "FLAGGED FOR INVESTIGATION"
        )

        if prob_float >= t:
            return review_label, flagged_label, True
        else:
            return accept_label, "LOW RISK", False

    def evaluate(
        self,
        probability: float,
        threshold: Optional[float] = None,
    ) -> RiskAssessment:
        """
        Perform complete risk scoring and decision evaluation for a transaction.

        Args:
            probability: Model fraud probability.
            threshold: Optional decision threshold override.

        Returns:
            Populated RiskAssessment dataclass.
        """
        prob_float = max(0.0, min(1.0, float(probability)))
        t = float(threshold) if threshold is not None else self.decision_threshold

        score = self.normalize_score(prob_float)
        cat_label, color, cat_desc = self.get_risk_category(score)
        decision, status_label, is_flagged = self.evaluate_decision(
            prob_float, threshold=t
        )

        if decision == "REVIEW":
            full_desc = (
                f"Transaction FLAGGED FOR INVESTIGATION: probability {prob_float:.4f} "
                f"exceeds decision threshold {t:.4f} (risk score: {score}/100, {cat_label})."
            )
        else:
            full_desc = (
                f"Transaction cleared automatically under current operational threshold "
                f"(risk score: {score}/100, {cat_label}). NOTE: ACCEPT is an operational "
                f"clearance, not a guarantee of transaction legitimacy."
            )

        return RiskAssessment(
            probability=round(prob_float, 6),
            risk_score=score,
            risk_category=cat_label,
            decision=decision,
            decision_threshold=round(t, 4),
            is_flagged=is_flagged,
            status_label=status_label,
            color=color,
            prediction=1 if is_flagged else 0,
            threshold=round(t, 4),
            description=full_desc,
            legitimacy_disclaimer=self.legitimacy_disclaimer,
        )

    def batch_evaluate(
        self,
        probabilities: Sequence[float],
        threshold: Optional[float] = None,
    ) -> list[RiskAssessment]:
        """
        Evaluate risk scoring and decisions for a sequence of probabilities.

        Args:
            probabilities: Sequence of probability values.
            threshold: Optional decision threshold override.

        Returns:
            List of RiskAssessment objects.
        """
        return [self.evaluate(p, threshold=threshold) for p in probabilities]
