"""Loguru sink setup. Centralized so every CLI subcommand logs identically."""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger as _logger

__all__ = ["logger", "setup_logging"]

logger = _logger

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def setup_logging(level: str = "INFO", *, sink: Any = None) -> None:
    """Replace any existing loguru sink with a single configured one.

    Idempotent — safe to call repeatedly (e.g. from `from_config`).
    """
    _logger.remove()
    _logger.add(
        sink or sys.stderr,
        level=level,
        format=_FORMAT,
        backtrace=False,
        diagnose=False,
        enqueue=False,
    )
