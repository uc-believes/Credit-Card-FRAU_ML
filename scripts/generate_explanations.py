"""
Fraud Shield — Generate SHAP Explanations & Plots
===================================================
Generates:
1. Global feature importance bar chart & beeswarm distribution.
2. Local transaction explanations (Fraud vs. Legitimate examples).
3. Saves plots to artifacts/plots/ and reports to artifacts/reports/.

Usage:
    python scripts/generate_explanations.py
"""

from __future__ import annotations

import pathlib
import sys
import time

# Ensure project root is in sys.path
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.explainability.service import FraudIntelligenceService
from src.utils.config import get_config, get_project_root
from src.utils.logger import get_logger
from src.utils.serialization import save_json

logger = get_logger("scripts.generate_explanations")


def main() -> None:
    start_time = time.time()
    root = get_project_root()
    cfg = get_config()

    print("=" * 80)
    print(" FRAUD SHIELD -- PHASE 3: EXPLAINABILITY & RISK INTELLIGENCE")
    print("=" * 80)

    # 1. Initialize FraudIntelligenceService
    print("[*] Initializing FraudIntelligenceService with best model and pipeline...")
    service = FraudIntelligenceService()

    # 2. Load validation data to extract real samples & compute global importance
    val_path = root / cfg["paths"]["data"]["validation"]
    print(f"[*] Loading validation cohort from {val_path}...")
    df_val = pd.read_parquet(val_path)

    X_val_scaled = df_val[service.feature_names].values
    y_val = df_val[cfg["dataset"]["target_column"]].values

    # 3. Global Feature Importance
    print("\n[*] Computing Global SHAP Feature Importance over validation sample...")
    plots_dir = root / cfg["paths"]["artifacts"]["plots"]
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = root / cfg["paths"]["artifacts"]["reports"]
    reports_dir.mkdir(parents=True, exist_ok=True)

    global_importance = service.explainer.explain_global(X_val_scaled, max_samples=500)
    save_json(global_importance, reports_dir / "global_importance.json")

    # Plot Global Bar Chart
    global_plot_path = plots_dir / "global_feature_importance.png"
    service.visualizer.plot_global_importance(
        global_importance=global_importance,
        output_path=global_plot_path,
        top_n=15,
    )
    print(f"    [OK] Saved global importance chart to {global_plot_path}")

    # Plot Beeswarm Summary
    beeswarm_path = plots_dir / "shap_beeswarm_summary.png"
    sample_sub = X_val_scaled[:300]
    service.visualizer.plot_summary_beeswarm(
        explainer_obj=service.explainer.explainer,
        X_sample=sample_sub,
        feature_names=service.feature_names,
        output_path=beeswarm_path,
        max_display=15,
    )
    print(f"    [OK] Saved beeswarm distribution to {beeswarm_path}")

    # Print Top 5 Global Features
    print("\n  Top 5 Global Fraud Drivers (Mean |SHAP|):")
    for item in global_importance[:5]:
        print(f"    {item['rank']}. {item['feature']:<14} mean|SHAP|={item['mean_abs_shap']:.4f} ({item['importance_share']*100:.1f}%)")

    # 4. Local Explanation for an Actual Fraud Transaction (High Risk)
    print("\n" + "=" * 80)
    print(" SAMPLE 1: ACTUAL FRAUDULENT TRANSACTION (HIGH RISK)")
    print("=" * 80)
    # Find a confirmed high-risk fraud case (TP with probability > 0.90)
    fraud_indices = [
        i for i, y in enumerate(y_val)
        if y == 1 and float(service.model.predict_proba(X_val_scaled[i].reshape(1, -1))[:, 1][0]) >= 0.90
    ]
    sample_fraud_idx = fraud_indices[0] if fraud_indices else 0

    # Extract raw features from validation set
    raw_csv = root / cfg["paths"]["data"]["raw"]
    # We can reconstruct or pass the preprocessed features directly through explainer
    sample_fraud_x = X_val_scaled[sample_fraud_idx]
    fraud_prob = float(service.model.predict_proba(sample_fraud_x.reshape(1, -1))[:, 1][0])
    fraud_assessment = service.risk_engine.evaluate(fraud_prob)
    fraud_explanation = service.explainer.explain_transaction(
        x_features=sample_fraud_x,
        probability=fraud_assessment.probability,
        risk_score=fraud_assessment.risk_score,
        risk_level=fraud_assessment.risk_category,
        prediction=fraud_assessment.prediction,
        top_k=5,
    )
    fraud_plot_path = plots_dir / "sample_fraud_explanation.png"
    service.visualizer.plot_local_attributions(
        explanation=fraud_explanation,
        output_path=fraud_plot_path,
        top_n=10,
    )
    print(fraud_explanation.format_summary(top_n=5))
    print(f"\n    [OK] Saved fraud attribution chart to {fraud_plot_path}")

    # 5. Local Explanation for an Actual Legitimate Transaction
    print("\n" + "=" * 80)
    print(" SAMPLE 2: ACTUAL LEGITIMATE TRANSACTION (LOW RISK)")
    print("=" * 80)
    legit_indices = [i for i, y in enumerate(y_val) if y == 0]
    sample_legit_idx = legit_indices[0] if legit_indices else 0

    sample_legit_x = X_val_scaled[sample_legit_idx]
    legit_prob = float(service.model.predict_proba(sample_legit_x.reshape(1, -1))[:, 1][0])
    legit_assessment = service.risk_engine.evaluate(legit_prob)
    legit_explanation = service.explainer.explain_transaction(
        x_features=sample_legit_x,
        probability=legit_assessment.probability,
        risk_score=legit_assessment.risk_score,
        risk_level=legit_assessment.risk_category,
        prediction=legit_assessment.prediction,
        top_k=5,
    )
    legit_plot_path = plots_dir / "sample_legit_explanation.png"
    service.visualizer.plot_local_attributions(
        explanation=legit_explanation,
        output_path=legit_plot_path,
        top_n=10,
    )
    print(legit_explanation.format_summary(top_n=5))
    print(f"\n    [OK] Saved legit attribution chart to {legit_plot_path}")

    print("\n" + "=" * 80)
    print(f" [OK] Phase 3 explanation artifacts generated in {time.time() - start_time:.2f}s")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
