"""
Fraud Shield — Logging Setup
============================
Configures application-wide logging with rotating file handler.
Call setup_logging() once at application startup.

Usage:
    from src.utils.logger import setup_logging, get_logger
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Module started")
"""

from __future__ import annotations

import logging
import logging.handlers
import pathlib
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10_485_760,  # 10 MB
    backup_count: int = 3,
) -> None:
    """
    Configure root logger with console and optional rotating file handler.

    Args:
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to log file. If None, logs to console only.
        max_bytes: Max file size before rotation.
        backup_count: Number of backup log files to keep.
    """
    # Attempt to load from config; fall back to defaults if config not yet available
    try:
        from src.utils.config import get_config, get_project_root
        cfg = get_config().get("logging", {})
        level = cfg.get("level", level)
        log_fmt = cfg.get("format", _DEFAULT_FORMAT)
        if log_file is None:
            log_file = cfg.get("file")
        max_bytes = cfg.get("max_bytes", max_bytes)
        backup_count = cfg.get("backup_count", backup_count)
        if log_file:
            log_file = str(get_project_root() / log_file)
    except Exception:
        log_fmt = _DEFAULT_FORMAT

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    formatter = logging.Formatter(log_fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicate logs on re-setup
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (rotating)
    if log_file:
        log_path = pathlib.Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging initialized | level=%s | file=%s", level, log_file or "console only"
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger.

    Args:
        name: Logger name (typically __name__ of the calling module).

    Returns:
        logging.Logger instance.
    """
    return logging.getLogger(name)


_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
