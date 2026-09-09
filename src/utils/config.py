"""
Fraud Shield — Centralized Configuration Loader
================================================
Loads and validates the YAML configuration file.
Provides typed access to all configuration values.

Usage:
    from src.utils.config import get_config, get_project_root
    cfg = get_config()
    raw_path = get_project_root() / cfg.paths.data.raw
"""

from __future__ import annotations

import logging
import pathlib
from functools import lru_cache
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Project root is 2 levels up from this file (src/utils/config.py → project/)
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def get_project_root() -> pathlib.Path:
    """Return the absolute path to the project root directory."""
    return _PROJECT_ROOT


def get_config_path() -> pathlib.Path:
    """Return the absolute path to the config.yaml file."""
    return _PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """
    Load and cache the project configuration from configs/config.yaml.

    Returns:
        dict: Parsed YAML configuration as a nested dictionary.

    Raises:
        FileNotFoundError: If config.yaml does not exist.
        yaml.YAMLError: If config.yaml has invalid syntax.
    """
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Expected at: configs/config.yaml relative to project root."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty.")

    logger.debug("Configuration loaded from: %s", config_path)
    return config


def get_path(key_path: str) -> pathlib.Path:
    """
    Resolve a dot-separated path key from config to an absolute filesystem path.

    Args:
        key_path: Dot-separated config key, e.g. "paths.data.raw"

    Returns:
        Absolute pathlib.Path resolved relative to project root.

    Example:
        get_path("paths.data.raw")
        → PosixPath('/project/data/raw/creditcard.csv')
    """
    cfg = get_config()
    keys = key_path.split(".")
    value = cfg
    for k in keys:
        if not isinstance(value, dict) or k not in value:
            raise KeyError(f"Config key '{key_path}' not found. Missing segment: '{k}'")
        value = value[k]

    if not isinstance(value, str):
        raise TypeError(
            f"Config key '{key_path}' must be a string path, got {type(value).__name__}"
        )

    return _PROJECT_ROOT / value


def get_risk_thresholds() -> dict[str, int]:
    """
    Return the risk categorization thresholds from config.

    Returns:
        dict with keys: low_max (int), medium_max (int)
    """
    cfg = get_config()
    return cfg["risk_engine"]["thresholds"]


def get_model_configs() -> dict[str, Any]:
    """Return the models section of the config."""
    return get_config()["models"]


def get_splitting_config() -> dict[str, Any]:
    """Return train/val/test splitting configuration."""
    return get_config()["splitting"]


def get_imbalance_config() -> dict[str, Any]:
    """Return imbalance handling configuration."""
    return get_config()["imbalance"]


def get_threshold_config() -> dict[str, Any]:
    """Return threshold optimization configuration."""
    return get_config()["threshold"]


def get_explainability_config() -> dict[str, Any]:
    """Return SHAP explainability configuration."""
    return get_config()["explainability"]


def get_model_selection_config() -> dict[str, Any]:
    """Return model selection criterion configuration."""
    return get_config()["model_selection"]


def reload_config() -> dict[str, Any]:
    """
    Force reload of configuration (clears the LRU cache).
    Useful during testing or when config file is updated at runtime.
    """
    get_config.cache_clear()
    return get_config()
