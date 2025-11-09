# pylint: disable=duplicate-code
"""
integration_hook.py — Phase 2
Promotes top individuals from NSGA-3 into the Learning Machine pipeline.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from crypto_trading_bot.nsga3.regime_utils import normalize_regime_label

LOGGER = logging.getLogger(__name__)

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover
    fcntl = None


def promote_to_learning_machine(
    individuals: Iterable[dict],
    generation: int = 0,
    reason: str = "pareto_elite",
    regime: str = "global",
) -> None:
    """
    Append individuals to the NSGA-3 promotions log.

    Placeholder implementation for Phase 2 scaffolding — actual promotion
    criteria will be wired up once the evolution loop is complete.
    """
    os.makedirs("logs", exist_ok=True)
    label = normalize_regime_label(regime)
    file = Path(f"logs/nsga3_promotions_{label}.jsonl")
    ts = datetime.now(timezone.utc).isoformat()
    iterable = [individuals] if isinstance(individuals, dict) else list(individuals)
    try:
        with file.open("a", encoding="utf-8") as handle:
            if fcntl:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except OSError:
                    pass
            for individual in iterable:
                record = {
                    "timestamp": ts,
                    "generation": generation,
                    "reason": reason,
                    "regime": label,
                    **individual,
                }
                handle.write(json.dumps(record) + "\n")
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
    except Exception as exc:  # pragma: no cover - durability safeguard  # pylint: disable=broad-exception-caught
        LOGGER.error("[PromotionHook] Write failed: %s", exc)
