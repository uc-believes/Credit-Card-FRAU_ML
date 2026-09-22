"""
Fraud Shield — Evaluation Metrics
===================================
All metric computations for fraud detection.

Design principles:
    - Every function takes (y_true, y_pred_proba) — threshold-independent inputs
    - Threshold-dependent metrics are computed after threshold application
    - F2-score is the primary metric (recall-weighted, beta=2)
    - PR-AUC is preferred over ROC-AUC for imbalanced data

Verified class distribution (Phase 1):
    Train: 170,235 rows — 284 fraud (0.167%)
    Val:    56,745 rows —  94 fraud (0.166%)
    Test:   56,746 rows —  95 fraud (0.167%)

Usage:
    from src.evaluation.metrics import (
        compute_all_metrics, find_optimal_threshold, MetricsReport
    )
    report = compute_all_metrics(y_true, y_pred_proba)
    threshold = find_optimal_threshold(y_true, y_pred_proba)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Structured report
# ---------------------------------------------------------------------------

@dataclass
class MetricsReport:
    """All evaluation metrics for one model at one threshold."""

    # Threshold-free (area-based)
    roc_auc: float = 0.0
    pr_auc: float = 0.0

    # Threshold-dependent
    threshold: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    f2_score: float = 0.0
    accuracy: float = 0.0
    specificity: float = 0.0

    # Composite selection score
    composite_score: float = 0.0

    # Confusion matrix elements
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    # Metadata
    n_samples: int = 0
    n_fraud: int = 0
    n_legitimate: int = 0
    split: str = "validation"
    model_name: str = ""

    # Curve data (for plotting)
    precision_curve: list = field(default_factory=list)
    recall_curve: list = field(default_factory=list)
    fpr_curve: list = field(default_factory=list)
    tpr_curve: list = field(default_factory=list)
    threshold_curve: list = field(default_factory=list)

    def to_dict(self, include_curves: bool = False) -> dict:
        """Serializable dict, optionally excluding large curve arrays."""
        d = {
            "model_name": self.model_name,
            "split": self.split,
            "threshold": round(self.threshold, 4),
            "roc_auc": round(self.roc_auc, 6),
            "pr_auc": round(self.pr_auc, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1_score": round(self.f1_score, 6),
            "f2_score": round(self.f2_score, 6),
            "accuracy": round(self.accuracy, 6),
            "specificity": round(self.specificity, 6),
            "composite_score": round(self.composite_score, 6),
            "confusion_matrix": {
                "tp": self.tp, "tn": self.tn,
                "fp": self.fp, "fn": self.fn,
            },
            "n_samples": self.n_samples,
            "n_fraud": self.n_fraud,
            "n_legitimate": self.n_legitimate,
        }
        if include_curves:
            d["curves"] = {
                "precision": [round(v, 6) for v in self.precision_curve],
                "recall": [round(v, 6) for v in self.recall_curve],
                "fpr": [round(v, 6) for v in self.fpr_curve],
                "tpr": [round(v, 6) for v in self.tpr_curve],
                "thresholds": [round(v, 6) for v in self.threshold_curve],
            }
        return d

    def summary_line(self) -> str:
        return (
            f"{self.model_name:<28} "
            f"P={self.precision:.4f}  R={self.recall:.4f}  "
            f"F1={self.f1_score:.4f}  F2={self.f2_score:.4f}  "
            f"ROC={self.roc_auc:.4f}  PR={self.pr_auc:.4f}  "
            f"Comp={self.composite_score:.4f}  @t={self.threshold:.2f}"
        )


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: Optional[float] = None,
    model_name: str = "",
    split: str = "validation",
    optimize_threshold: bool = True,
    optimization_metric: str = "f2_score",
) -> MetricsReport:
    """
    Compute all fraud-detection metrics for one model.

    Args:
        y_true: Binary ground-truth labels (0=legit, 1=fraud).
        y_pred_proba: Predicted fraud probabilities in [0, 1].
        threshold: Classification threshold. If None and optimize_threshold=True,
                   the threshold is found by maximizing optimization_metric on y_true.
        model_name: Label for logging.
        split: 'train', 'validation', or 'test'.
        optimize_threshold: If True and threshold is None, find optimal threshold.
        optimization_metric: 'f2_score' or 'f1_score'.

    Returns:
        MetricsReport with all computed values.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    if len(y_true) != len(y_pred_proba):
        raise ValueError(
            f"y_true ({len(y_true)}) and y_pred_proba ({len(y_pred_proba)}) "
            f"must have the same length."
        )

    report = MetricsReport(
        model_name=model_name,
        split=split,
        n_samples=len(y_true),
        n_fraud=int(y_true.sum()),
        n_legitimate=int((y_true == 0).sum()),
    )

    # --- Threshold-free metrics ---
    try:
        report.roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    except Exception as e:
        logger.warning("ROC-AUC failed: %s", e)
        report.roc_auc = 0.0

    try:
        report.pr_auc = float(average_precision_score(y_true, y_pred_proba))
    except Exception as e:
        logger.warning("PR-AUC failed: %s", e)
        report.pr_auc = 0.0

    # --- Curve data ---
    try:
        prec_c, rec_c, thresh_c = precision_recall_curve(y_true, y_pred_proba)
        report.precision_curve = prec_c.tolist()
        report.recall_curve = rec_c.tolist()
        report.threshold_curve = thresh_c.tolist()
    except Exception:
        pass

    try:
        fpr_c, tpr_c, _ = roc_curve(y_true, y_pred_proba)
        report.fpr_curve = fpr_c.tolist()
        report.tpr_curve = tpr_c.tolist()
    except Exception:
        pass

    # --- Threshold selection ---
    if threshold is not None:
        report.threshold = threshold
    elif optimize_threshold:
        report.threshold = find_optimal_threshold(
            y_true, y_pred_proba, metric=optimization_metric
        )
    else:
        report.threshold = 0.5

    # --- Threshold-dependent metrics ---
    y_pred = (y_pred_proba >= report.threshold).astype(int)

    try:
        report.precision = float(precision_score(y_true, y_pred, zero_division=0))
        report.recall = float(recall_score(y_true, y_pred, zero_division=0))
        report.f1_score = float(f1_score(y_true, y_pred, zero_division=0))
        report.f2_score = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        report.accuracy = float(accuracy_score(y_true, y_pred))
    except Exception as e:
        logger.warning("Threshold metrics failed: %s", e)

    # Confusion matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        report.tn = int(cm[0, 0])
        report.fp = int(cm[0, 1])
        report.fn = int(cm[1, 0])
        report.tp = int(cm[1, 1])
        total_neg = report.tn + report.fp
        report.specificity = report.tn / total_neg if total_neg > 0 else 0.0
    except Exception as e:
        logger.warning("Confusion matrix failed: %s", e)

    # Composite selection score
    report.composite_score = compute_composite_score(report)

    logger.info(
        "[%s | %s] F2=%.4f  PR-AUC=%.4f  ROC-AUC=%.4f  "
        "P=%.4f  R=%.4f  t=%.2f  Composite=%.4f",
        model_name, split,
        report.f2_score, report.pr_auc, report.roc_auc,
        report.precision, report.recall, report.threshold,
        report.composite_score,
    )

    return report


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f2_score",
    search_range: tuple[float, float] = (0.05, 0.95),
    n_steps: int = 100,
) -> float:
    """
    Find the classification threshold that maximizes the given metric.

    IMPORTANT: This must be called ONLY on validation data.
    Never optimize the threshold on test data.

    Args:
        y_true: Ground-truth labels.
        y_pred_proba: Predicted probabilities.
        metric: 'f2_score' or 'f1_score'.
        search_range: (min_threshold, max_threshold).
        n_steps: Number of thresholds to evaluate.

    Returns:
        Optimal threshold as float.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_proba = np.asarray(y_pred_proba, dtype=float)

    thresholds = np.linspace(search_range[0], search_range[1], n_steps)
    best_score = -1.0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        if metric == "f2_score":
            score = float(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        else:
            score = float(f1_score(y_true, y_pred, zero_division=0))

        if score > best_score:
            best_score = score
            best_threshold = float(t)

    logger.debug(
        "Optimal threshold for %s: %.4f (score=%.4f)",
        metric, best_threshold, best_score
    )
    return best_threshold


def compute_composite_score(report: MetricsReport) -> float:
    """
    Compute the weighted model selection score.

    Formula (from config.yaml model_selection.metric_weights):
        Score = 0.35 × F2 + 0.30 × PR-AUC + 0.20 × ROC-AUC + 0.15 × F1

    This is the single number used to rank models.

    Args:
        report: Populated MetricsReport.

    Returns:
        Composite score in [0, 1].
    """
    return (
        0.35 * report.f2_score
        + 0.30 * report.pr_auc
        + 0.20 * report.roc_auc
        + 0.15 * report.f1_score
    )


def check_minimum_thresholds(report: MetricsReport) -> tuple[bool, list[str]]:
    """
    Check whether a model meets minimum acceptable thresholds.

    Thresholds (from config.yaml model_selection.minimum_thresholds):
        recall   >= 0.70
        precision >= 0.10
        roc_auc  >= 0.90

    Returns:
        (passes: bool, failures: list of failure descriptions)
    """
    failures = []
    if report.recall < 0.70:
        failures.append(f"recall={report.recall:.4f} < 0.70")
    if report.precision < 0.10:
        failures.append(f"precision={report.precision:.4f} < 0.10")
    if report.roc_auc < 0.90:
        failures.append(f"roc_auc={report.roc_auc:.4f} < 0.90")
    return len(failures) == 0, failures


def build_comparison_table(reports: list[MetricsReport]) -> str:
    """
    Build an ASCII comparison table of all models ranked by composite score.

    Args:
        reports: List of MetricsReport — one per model, on validation set.

    Returns:
        Formatted string table.
    """
    sorted_reports = sorted(reports, key=lambda r: r.composite_score, reverse=True)

    header = (
        f"{'Rank':<5} {'Model':<28} {'F2':>7} {'Recall':>8} {'Prec':>7} "
        f"{'F1':>7} {'ROC':>7} {'PR-AUC':>8} {'Comp':>7} {'@Thresh':>8} {'Min?':>5}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for rank, r in enumerate(sorted_reports, 1):
        passes, _ = check_minimum_thresholds(r)
        flag = "OK" if passes else "FAIL"
        winner = " <-- SELECTED" if rank == 1 and passes else ""
        line = (
            f"{rank:<5} {r.model_name:<28} {r.f2_score:>7.4f} {r.recall:>8.4f} "
            f"{r.precision:>7.4f} {r.f1_score:>7.4f} {r.roc_auc:>7.4f} "
            f"{r.pr_auc:>8.4f} {r.composite_score:>7.4f} {r.threshold:>8.3f} "
            f"{flag:>5}{winner}"
        )
        lines.append(line)

    lines.append(sep)
    return "\n".join(lines)
