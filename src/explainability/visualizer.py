"""
Fraud Shield — Explainability Visualizer
=========================================
Generates dashboard-ready visualizations for local and global SHAP explanations.

Uses headless Matplotlib (Agg backend) to render:
- Local Waterfall / Diverging attribution charts
- Global Mean |SHAP| Importance charts
- Beeswarm summary distributions
- Base64 encoding for direct web UI embedding
"""

from __future__ import annotations

import base64
import io
import pathlib
from typing import Any, Optional, Sequence

import matplotlib

# Set non-GUI backend before importing pyplot
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.explainability.explainer import TransactionExplanation
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Standard sleek color palette
COLOR_FRAUD_PUSH = "#ef4444"      # Red / Crimson for increasing fraud risk
COLOR_LEGIT_PUSH = "#10b981"      # Emerald green for mitigating fraud risk
COLOR_NEUTRAL = "#64748b"         # Slate neutral
BG_COLOR = "#ffffff"
TEXT_COLOR = "#0f172a"


class ExplainerVisualizer:
    """
    Renders publication-grade charts from SHAP explanation objects.
    """

    def __init__(self, style_theme: str = "whitegrid") -> None:
        self.style_theme = style_theme

    @staticmethod
    def _to_base64(fig: plt.Figure) -> str:
        """Convert a matplotlib figure to base64 PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return encoded

    def plot_local_attributions(
        self,
        explanation: TransactionExplanation,
        output_path: Optional[pathlib.Path] = None,
        top_n: int = 10,
    ) -> str:
        """
        Plot horizontal diverging bar chart of top positive & negative contributors.

        Args:
            explanation: TransactionExplanation object.
            output_path: Optional filepath to save the PNG image.
            top_n: Maximum features to display in chart.

        Returns:
            Base64-encoded PNG string.
        """
        # Collect top contributors by absolute SHAP value
        sorted_feats = sorted(
            explanation.all_features, key=lambda f: abs(f.shap_value), reverse=True
        )[:top_n]
        # Invert order for horizontal bar chart (top rank at top of plot)
        sorted_feats = sorted_feats[::-1]

        names = [f.feature_name for f in sorted_feats]
        values = [f.shap_value for f in sorted_feats]
        feat_vals = [f.feature_value for f in sorted_feats]
        colors = [
            COLOR_FRAUD_PUSH if v > 0 else COLOR_LEGIT_PUSH for v in values
        ]

        fig, ax = plt.subplots(figsize=(9, max(4.5, len(names) * 0.45)))
        bars = ax.barh(names, values, color=colors, height=0.65, edgecolor="none")

        # Zero reference line
        ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.8)

        # Annotate bars with feature value and SHAP value
        x_min, x_max = ax.get_xlim()
        padding = (x_max - x_min) * 0.02 if (x_max - x_min) > 0 else 0.1

        for bar, s_val, f_val in zip(bars, values, feat_vals):
            sign = "+" if s_val > 0 else ""
            label_text = f"val={f_val:.2f} ({sign}{s_val:.2f})"
            if s_val >= 0:
                ax.text(
                    s_val + padding,
                    bar.get_y() + bar.get_height() / 2,
                    label_text,
                    va="center",
                    ha="left",
                    fontsize=9,
                    fontweight="bold",
                    color=COLOR_FRAUD_PUSH,
                )
            else:
                ax.text(
                    s_val - padding,
                    bar.get_y() + bar.get_height() / 2,
                    label_text,
                    va="center",
                    ha="right",
                    fontsize=9,
                    fontweight="bold",
                    color=COLOR_LEGIT_PUSH,
                )

        # Title & Labels
        pred_label = (
            "FRAUD ALERT" if explanation.prediction == 1 else "LEGITIMATE"
        )
        title = (
            f"Transaction Explanation — Risk Score: {explanation.risk_score}/100 "
            f"({explanation.risk_level})\n"
            f"Prediction: {pred_label} | Prob: {explanation.probability:.4f}"
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14, color=TEXT_COLOR)
        ax.set_xlabel(
            "SHAP Attribution (Push toward Fraud > 0 | Mitigating < 0)",
            fontsize=10,
            labelpad=8,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.grid(axis="x", linestyle=":", alpha=0.5, color="#cbd5e1")

        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            logger.info("Saved local attribution chart: %s", output_path)

        return self._to_base64(fig)

    def plot_global_importance(
        self,
        global_importance: list[dict[str, Any]],
        output_path: Optional[pathlib.Path] = None,
        top_n: int = 15,
    ) -> str:
        """
        Plot global mean absolute SHAP value bar chart.

        Args:
            global_importance: Output list from FraudExplainer.explain_global().
            output_path: Optional filepath for saving PNG.
            top_n: Number of top features to plot.

        Returns:
            Base64-encoded PNG string.
        """
        top_items = global_importance[:top_n][::-1]
        names = [item["feature"] for item in top_items]
        values = [item["mean_abs_shap"] for item in top_items]
        shares = [item["importance_share"] * 100 for item in top_items]

        fig, ax = plt.subplots(figsize=(9, max(5.0, len(names) * 0.4)))
        bars = ax.barh(names, values, color="#3b82f6", height=0.65, edgecolor="none")

        # Value annotations
        x_max = max(values) if values else 1.0
        padding = x_max * 0.02

        for bar, val, share in zip(bars, values, shares):
            ax.text(
                val + padding,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f} ({share:.1f}%)",
                va="center",
                ha="left",
                fontsize=9,
                color="#1e293b",
                fontweight="medium",
            )

        ax.set_title(
            f"Global Feature Importance — Top {len(names)} Fraud Indicators\n"
            "(Mean Absolute SHAP Value across validation cohort)",
            fontsize=12,
            fontweight="bold",
            pad=14,
            color=TEXT_COLOR,
        )
        ax.set_xlabel("Mean |SHAP Value| (Impact on Model Log-Odds)", fontsize=10, labelpad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.grid(axis="x", linestyle=":", alpha=0.5, color="#cbd5e1")

        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=180, bbox_inches="tight")
            logger.info("Saved global importance chart: %s", output_path)

        return self._to_base64(fig)

    def plot_summary_beeswarm(
        self,
        explainer_obj: Any,
        X_sample: np.ndarray,
        feature_names: Sequence[str],
        output_path: Optional[pathlib.Path] = None,
        max_display: int = 15,
    ) -> str:
        """
        Generate and save a standard SHAP beeswarm summary plot.
        """
        fig = plt.figure(figsize=(9, 6))
        shap_res = explainer_obj(X_sample)
        shap_res.feature_names = list(feature_names)

        shap.plots.beeswarm(shap_res, max_display=max_display, show=False)
        plt.title(
            "SHAP Summary — Feature Value Distribution & Fraud Impact",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        plt.tight_layout()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=180, bbox_inches="tight")
            logger.info("Saved beeswarm summary chart: %s", output_path)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("utf-8")
        plt.close("all")
        return encoded
