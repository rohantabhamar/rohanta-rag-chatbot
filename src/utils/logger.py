"""
src/utils/logger.py
───────────────────
Structured, levelled logger used across the entire application.
Uses Python's built-in logging with a clean format suitable for
both local development and production log aggregators.
"""

import logging
import sys
from src.config import get_settings


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Usage:
        from src.utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Vector store loaded", extra={"index": "faiss_index"})
    """
    settings = get_settings()
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    return logger
