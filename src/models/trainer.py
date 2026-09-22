"""
Fraud Shield — Model Trainer & Evaluator
=========================================
Orchestrates model training, threshold tuning, and multi-metric comparison.

Models supported:
    - Logistic Regression (class_weight='balanced')
    - Decision Tree (class_weight='balanced')
    - Random Forest (class_weight='balanced', n_estimators=200)
    - XGBoost (scale_pos_weight = n_neg / n_pos)
    - Dummy Baseline (stratified random)

All models are evaluated using fraud-appropriate metrics:
    F2-score (primary), PR-AUC, ROC-AUC, F1-score, Precision, Recall.
Threshold is optimized on the VALIDATION split only.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.evaluation.metrics import (
    MetricsReport,
    check_minimum_thresholds,
    compute_all_metrics,
    compute_composite_score,
)
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.serialization import save_metrics, save_model, save_report

logger = get_logger(__name__)


class ModelTrainer:
    """
    Trains and evaluates all enabled fraud classification models.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self.cfg = config if config is not None else get_config()
        self.models_cfg = self.cfg.get("models", {})
        self.threshold_cfg = self.cfg.get("threshold", {})
        self.imbalance_cfg = self.cfg.get("imbalance", {})
        self.selection_cfg = self.cfg.get("model_selection", {})

    def build_model(self, model_key: str, y_train: Optional[np.ndarray] = None) -> Any:
        """
        Instantiate a model with config hyperparameters.

        Args:
            model_key: Key in configs/config.yaml under 'models'.
            y_train: Optional labels used to dynamically compute class ratio for XGBoost.

        Returns:
            Instantiated classifier object.
        """
        model_entry = self.models_cfg.get(model_key)
        if not model_entry:
            raise ValueError(f"Unknown model key: {model_key}")

        params = dict(model_entry.get("params", {}))

        if model_key == "logistic_regression":
            return LogisticRegression(**params)

        elif model_key == "decision_tree":
            return DecisionTreeClassifier(**params)

        elif model_key == "random_forest":
            return RandomForestClassifier(**params)

        elif model_key == "xgboost":
            if y_train is not None:
                n_neg = int((y_train == 0).sum())
                n_pos = int((y_train == 1).sum())
                scale_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
                params["scale_pos_weight"] = round(scale_weight, 2)
                logger.info(
                    "XGBoost scale_pos_weight computed: %.2f (neg=%d, pos=%d)",
                    scale_weight,
                    n_neg,
                    n_pos,
                )
            return XGBClassifier(**params)

        elif model_key == "dummy_baseline":
            return DummyClassifier(**params)

        else:
            raise ValueError(f"Unsupported model: {model_key}")

    def apply_smote(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE oversampling to training split only, if enabled in config.
        """
        if not self.imbalance_cfg.get("use_smote", False):
            return X_train, y_train

        from imblearn.over_sampling import SMOTE

        k_neighbors = self.imbalance_cfg.get("smote_k_neighbors", 5)
        seed = self.imbalance_cfg.get("smote_random_state", 42)
        logger.info("Applying SMOTE on training split (k_neighbors=%d)...", k_neighbors)
        smote = SMOTE(k_neighbors=k_neighbors, random_state=seed)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        logger.info(
            "SMOTE complete: %d samples (pos=%d, neg=%d)",
            len(y_res),
            int((y_res == 1).sum()),
            int((y_res == 0).sum()),
        )
        return X_res, y_res

    def train_one(
        self,
        model_key: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> tuple[Any, MetricsReport]:
        """
        Train a single model and evaluate on the validation set.

        Args:
            model_key: Key in config (e.g. 'random_forest').
            X_train: Preprocessed training features.
            y_train: Training labels.
            X_val: Preprocessed validation features.
            y_val: Validation labels.

        Returns:
            (trained_model, validation_metrics_report)
        """
        model_entry = self.models_cfg[model_key]
        display_name = model_entry.get("name", model_key)
        logger.info("Starting training: %s", display_name)

        model = self.build_model(model_key, y_train=y_train)

        start_time = time.time()
        model.fit(X_train, y_train)
        train_duration = time.time() - start_time
        logger.info("Fitted %s in %.2fs", display_name, train_duration)

        # Generate validation probabilities
        val_start = time.time()
        if hasattr(model, "predict_proba"):
            y_val_proba = model.predict_proba(X_val)[:, 1]
        else:
            y_val_proba = model.predict(X_val).astype(float)
        val_duration = time.time() - val_start

        # Optimization config
        optimize_thresh = self.threshold_cfg.get("optimize", True)
        opt_metric = self.threshold_cfg.get("optimization_metric", "f2_score")

        # Dummy baseline shouldn't optimize threshold to artificial numbers
        if model_key == "dummy_baseline":
            optimize_thresh = False

        report = compute_all_metrics(
            y_true=y_val,
            y_pred_proba=y_val_proba,
            threshold=0.5 if not optimize_thresh else None,
            model_name=display_name,
            split="validation",
            optimize_threshold=optimize_thresh,
            optimization_metric=opt_metric,
        )

        # Save model artifact and metrics
        model_meta = {
            "model_key": model_key,
            "display_name": display_name,
            "training_duration_seconds": round(train_duration, 3),
            "inference_duration_seconds": round(val_duration, 3),
            "optimal_threshold": report.threshold,
            "validation_composite_score": report.composite_score,
            "validation_f2": report.f2_score,
            "validation_pr_auc": report.pr_auc,
            "validation_roc_auc": report.roc_auc,
        }
        save_model(model, model_key, metadata=model_meta)
        save_metrics(report.to_dict(include_curves=True), model_key)

        return model, report

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> dict[str, tuple[Any, MetricsReport]]:
        """
        Train all enabled models from the config.

        Returns:
            Dictionary mapping model_key -> (trained_model, MetricsReport).
        """
        # Apply SMOTE if configured
        X_tr, y_tr = self.apply_smote(X_train, y_train)

        results: dict[str, tuple[Any, MetricsReport]] = {}
        for key, entry in self.models_cfg.items():
            if not entry.get("enabled", True):
                logger.info("Skipping disabled model: %s", key)
                continue

            model, report = self.train_one(key, X_tr, y_tr, X_val, y_val)
            results[key] = (model, report)

        return results

    def select_best_model(
        self, results: dict[str, tuple[Any, MetricsReport]]
    ) -> tuple[str, Any, MetricsReport]:
        """
        Select the winning model using composite score and minimum thresholds.

        Args:
            results: Output from train_all().

        Returns:
            (best_model_key, best_model_obj, best_report)
        """
        valid_candidates = []
        all_candidates = []

        for key, (model, report) in results.items():
            if key == "dummy_baseline":
                continue  # Never select baseline as winner

            passes, failures = check_minimum_thresholds(report)
            all_candidates.append((key, model, report))
            if passes:
                valid_candidates.append((key, model, report))
            else:
                logger.warning(
                    "Model %s failed minimum thresholds: %s",
                    report.model_name,
                    ", ".join(failures),
                )

        # Prioritize candidates passing minimum gates
        pool = valid_candidates if valid_candidates else all_candidates
        if not pool:
            raise RuntimeError("No candidate models available for selection.")

        # Sort descending by composite score
        best_key, best_model, best_report = max(
            pool, key=lambda item: item[2].composite_score
        )

        logger.info(
            "Selected WINNER: %s (Composite=%.4f, F2=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f)",
            best_report.model_name,
            best_report.composite_score,
            best_report.f2_score,
            best_report.pr_auc,
            best_report.roc_auc,
        )

        # Save as the designated production winner
        save_model(
            best_model,
            "best_model",
            metadata={
                "selected_model_key": best_key,
                "display_name": best_report.model_name,
                "optimal_threshold": best_report.threshold,
                "validation_composite_score": best_report.composite_score,
                "selection_criterion": "weighted_composite_score",
            },
        )

        return best_key, best_model, best_report

    def evaluate_test(
        self,
        model: Any,
        model_key: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float,
    ) -> MetricsReport:
        """
        Evaluate the selected model on the held-out test split.
        Uses the PRE-DETERMINED optimal threshold from the validation split.

        Args:
            model: Trained classifier.
            model_key: Identifier string.
            X_test: Held-out test features.
            y_test: Held-out test labels.
            threshold: Fixed threshold from validation tuning.

        Returns:
            MetricsReport for the test set.
        """
        model_name = self.models_cfg.get(model_key, {}).get("name", model_key)
        logger.info(
            "Evaluating %s on test set with fixed threshold %.4f...",
            model_name,
            threshold,
        )

        if hasattr(model, "predict_proba"):
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_test_proba = model.predict(X_test).astype(float)

        test_report = compute_all_metrics(
            y_true=y_test,
            y_pred_proba=y_test_proba,
            threshold=threshold,
            model_name=model_name,
            split="test",
            optimize_threshold=False,
        )

        save_metrics(
            test_report.to_dict(include_curves=True), f"{model_key}_test"
        )
        return test_report
