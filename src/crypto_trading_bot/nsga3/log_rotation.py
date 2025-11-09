"""
rotate_logs.py — utility helpers to compress/rotate NSGA-III log files.
"""

from __future__ import annotations

import gzip
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from crypto_trading_bot.nsga3.regime_utils import normalize_regime_label

LOGGER = logging.getLogger(__name__)
__all__ = ["rotate_nsga3_logs", "DEFAULT_TARGETS"]

LOG_DIR = Path("logs")
ARCHIVE_DIR = LOG_DIR / "archive"
DEFAULT_TARGETS = [
    LOG_DIR / "nsga3_checkpoint.json",
    LOG_DIR / "nsga3_checkpoint_global.json",
    LOG_DIR / "nsga3_promotions.jsonl",
    LOG_DIR / "nsga3_promotions_global.jsonl",
    LOG_DIR / "nsga3_front.jsonl",
    LOG_DIR / "nsga3_front_global.jsonl",
    LOG_DIR / "nsga3_top_candidates.jsonl",
    LOG_DIR / "nsga3_top_candidates_global.jsonl",
    LOG_DIR / "nsga3_shadow_results.jsonl",
    LOG_DIR / "nsga3_shadow_results_global.jsonl",
    LOG_DIR / "nsga3_scheduler.log",
]
MAX_SIZE_MB = 10
MAX_AGE_DAYS = 30


def _regime_default_targets(regime: str) -> list[Path]:
    label = normalize_regime_label(regime)
    return [
        LOG_DIR / f"nsga3_checkpoint_{label}.json",
        LOG_DIR / f"nsga3_promotions_{label}.jsonl",
        LOG_DIR / f"nsga3_front_{label}.jsonl",
        LOG_DIR / f"nsga3_top_candidates_{label}.jsonl",
        LOG_DIR / f"nsga3_shadow_results_{label}.jsonl",
        LOG_DIR / "nsga3_scheduler.log",
    ]


def rotate_nsga3_logs(
    files: Sequence[str | Path] | None = None,
    *,
    regime: str | None = None,
) -> None:
    """Rotate NSGA-III logs based on age or size thresholds."""
    if files is not None:
        targets: Iterable[Path] = [Path(p) for p in files]
    elif regime is not None:
        targets = _regime_default_targets(regime)
    else:
        targets = DEFAULT_TARGETS
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for path in targets:
        if _should_rotate(path):
            try:
                _rotate_file(path)
                LOGGER.info("Rotated NSGA-III log %s", path)
            except OSError as exc:
                LOGGER.warning("Failed to rotate %s: %s", path, exc)


def _should_rotate(path: Path) -> bool:
    """Return True if file exceeds max size or age thresholds."""
    if not path.exists():
        return False
    try:
        size_mb = path.stat().st_size / (1024**2)
        age_seconds = datetime.now(timezone.utc).timestamp() - path.stat().st_mtime
        age_days = age_seconds / (60 * 60 * 24)
        return size_mb >= MAX_SIZE_MB or age_days >= MAX_AGE_DAYS
    except OSError:
        return False


def _rotate_file(path: Path) -> None:
    """Compress path into logs/archive and truncate original."""
    if not path.exists():
        return
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    archive_path = ARCHIVE_DIR / f"{path.name}.{timestamp}.gz"
    with path.open("rb") as src, gzip.open(archive_path, "wb") as dst:
        dst.writelines(src)
    path.write_text("", encoding="utf-8")
