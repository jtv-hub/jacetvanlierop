"""System capacity checks to ensure sufficient resources before live trading."""

from __future__ import annotations

import logging
import os
import shutil
from typing import Tuple

try:  # optional dependency for detailed memory info
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - psutil optional
    psutil = None

logger = logging.getLogger(__name__)


def _available_memory_mb() -> Tuple[float | None, bool]:
    """Return (available_mem_mb, precise_flag)."""

    if psutil is not None:
        try:
            mem = psutil.virtual_memory()
            return mem.available / (1024 * 1024), True
        except Exception:  # pragma: no cover - psutil failures are rare
            logger.exception("psutil.virtual_memory() failed; falling back to sysconf")

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")  # bytes
        avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        if page_size <= 0 or avail_pages <= 0:
            return None, False
        return (page_size * avail_pages) / (1024 * 1024), False
    except (ValueError, OSError, AttributeError):
        return None, False


def ensure_system_capacity(min_disk_mb: float = 100.0, min_mem_mb: float = 256.0) -> None:
    """Validate disk and memory availability before entering live trading.

    Raises ``ConfigurationError`` when requirements are not met.
    """

    from crypto_trading_bot.config import ConfigurationError  # late import to avoid cycles

    total, used, free = shutil.disk_usage(os.getcwd())
    free_mb = free / (1024 * 1024)
    if free_mb < min_disk_mb:
        message = f"Insufficient disk space: {free_mb:.1f} MB available (required {min_disk_mb} MB)."
        raise ConfigurationError(message)

    mem_mb, precise = _available_memory_mb()
    if mem_mb is not None and mem_mb < min_mem_mb:
        detail = "precise" if precise else "estimated"
        message = f"Insufficient memory: {mem_mb:.1f} MB {detail} available " f"(required {min_mem_mb} MB)."
        raise ConfigurationError(message)

    logger.info(
        "System capacity check passed | free_disk=%.1f MB available_mem=%s",
        free_mb,
        f"{mem_mb:.1f} MB" if mem_mb is not None else "unknown",
    )
