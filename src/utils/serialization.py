"""
Fraud Shield — Model & Pipeline Serialization Utilities
=========================================================
Centralised save/load functions for sklearn models and pipelines.

All serialization goes through joblib (efficient binary format for
sklearn objects with numpy arrays). JSON is used for human-readable
metadata and metrics.

Usage:
    from src.utils.serialization import save_model, load_model, save_json, load_json
    path = save_model(model, "random_forest")
    model = load_model("random_forest")
"""

from __future__ import annotations

import json
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any, Optional

import joblib

from src.utils.config import get_config, get_project_root
from src.utils.logger import get_logger

logger = get_logger(__name__)

_cfg = None
_root = None


def _get_artifacts_dir(subdir: str) -> pathlib.Path:
    """Return absolute path to an artifact subdirectory, creating it if needed."""
    global _cfg, _root
    if _cfg is None:
        _cfg = get_config()
        _root = get_project_root()
    path = _root / _cfg["paths"]["artifacts"][subdir]
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Model Serialization
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    name: str,
    metadata: Optional[dict] = None,
) -> pathlib.Path:
    """
    Save a trained model to the artifacts/models/ directory.

    Args:
        model: Any sklearn-compatible model object.
        name: Identifier used as the filename stem (e.g., 'random_forest').
        metadata: Optional dict to save alongside the model as JSON.

    Returns:
        Path to the saved .joblib file.
    """
    models_dir = _get_artifacts_dir("models")
    model_path = models_dir / f"{name}.joblib"
    joblib.dump(model, model_path)
    logger.info("Model saved: %s", model_path)

    if metadata is not None:
        meta_path = models_dir / f"{name}_metadata.json"
        save_json({
            **metadata,
            "model_name": name,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }, meta_path)

    return model_path


def load_model(name: str) -> Any:
    """
    Load a trained model from the artifacts/models/ directory.

    Args:
        name: Identifier used as the filename stem (e.g., 'random_forest').

    Returns:
        Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    models_dir = _get_artifacts_dir("models")
    model_path = models_dir / f"{name}.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run training first: python scripts/train.py"
        )

    model = joblib.load(model_path)
    logger.info("Model loaded: %s", model_path)
    return model


def model_exists(name: str) -> bool:
    """Check whether a serialised model file exists."""
    models_dir = _get_artifacts_dir("models")
    return (models_dir / f"{name}.joblib").exists()


# ---------------------------------------------------------------------------
# JSON Utilities
# ---------------------------------------------------------------------------

def save_json(data: Any, path: pathlib.Path) -> pathlib.Path:
    """
    Save a dict/list to a JSON file.

    Args:
        data: JSON-serializable object.
        path: Absolute path for the output file.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.debug("JSON saved: %s", path)
    return path


def load_json(path: pathlib.Path) -> Any:
    """
    Load a JSON file.

    Args:
        path: Absolute path to the JSON file.

    Returns:
        Parsed Python object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Metrics Persistence
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, model_name: str) -> pathlib.Path:
    """
    Save model evaluation metrics to artifacts/metrics/.

    Args:
        metrics: Dict of metric name → value.
        model_name: Used as filename stem.

    Returns:
        Path to the saved JSON file.
    """
    metrics_dir = _get_artifacts_dir("metrics")
    path = metrics_dir / f"{model_name}_metrics.json"
    return save_json({
        "model_name": model_name,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        **metrics,
    }, path)


def load_metrics(model_name: str) -> dict:
    """Load saved metrics for a model."""
    metrics_dir = _get_artifacts_dir("metrics")
    path = metrics_dir / f"{model_name}_metrics.json"
    return load_json(path)


def save_report(report: dict, filename: str) -> pathlib.Path:
    """
    Save a comparison report to artifacts/reports/.

    Args:
        report: Report data dict.
        filename: Output filename (without directory).

    Returns:
        Path to saved file.
    """
    reports_dir = _get_artifacts_dir("reports")
    path = reports_dir / filename
    return save_json(report, path)
