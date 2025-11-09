"""Continual learning bridge for adaptive NSGA-III parameters."""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from crypto_trading_bot.nsga3.jsonl_utils import load_jsonl

LOGGER = logging.getLogger(__name__)
ADAPTATION_LOG = Path("logs/nsga3_adaptation.log")
DEFAULT_FEEDBACK_PATH = Path("logs/learning_feedback.jsonl")
MAX_HISTORY = 512


def get_adjustments(
    feedback_path: str | Path = DEFAULT_FEEDBACK_PATH,
    decay: float = 0.95,
    caps: float = 0.1,
) -> dict[str, float]:
    """
    Parse recent learning feedback, compute bounded adjustments for objective weights
    and genetic operators. Returns a dict of multipliers/deltas that the NSGA-3 engine
    applies prior to each generation.
    """
    neutral = {
        "roi_w": 1.0,
        "drawdown_w": 1.0,
        "winrate_w": 1.0,
        "mutation_rate": 0.0,
        "cx_eta": 0.0,
        "cx_prob": 0.0,
        "mut_eta": 0.0,
    }

    seed = _resolve_seed()
    feedback_file = Path(feedback_path)
    decay = min(max(float(decay), 0.0), 0.999)
    caps = abs(float(caps))

    entries = _load_feedback_entries(feedback_file)
    if not entries:
        _log_adaptation(neutral, seed, "no_feedback")
        return neutral

    signals = _aggregate_signals(entries, decay)
    adjustments = {
        "roi_w": _compute_weight(signals["roi"], caps),
        "drawdown_w": _compute_weight(signals["drawdown"], caps),
        "winrate_w": _compute_weight(signals["win_rate"], caps),
        "mutation_rate": _clamp(signals["mutation_rate"], -caps, caps),
        "cx_prob": _clamp(signals["cx_prob"], -caps, caps),
        "cx_eta": _clamp(signals["cx_eta"], -caps, caps),
        "mut_eta": _clamp(signals["mut_eta"], -caps, caps),
    }
    _log_adaptation(adjustments, seed, "computed", samples=len(entries))
    return adjustments


def _resolve_seed() -> int:
    """Return the deterministic seed supplied by the NSGA-III engine."""
    raw = os.environ.get("NSGA3_ADAPT_SEED", "0")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _load_feedback_entries(path: Path) -> List[Dict[str, Any]]:
    """Load recent learning feedback entries."""
    entries = load_jsonl(path, logger=LOGGER)
    if not entries:
        return []
    return entries[-MAX_HISTORY:]


# pylint: disable=too-many-locals


def _aggregate_signals(entries: List[Dict[str, Any]], decay: float) -> dict[str, float]:
    """Compute exponentially decayed signals for each adjustment channel."""
    roi_signal = drawdown_signal = win_signal = 0.0
    mutation_signal = cx_prob_signal = mut_eta_signal = cx_eta_signal = 0.0
    total_weight = 0.0

    for idx, entry in enumerate(reversed(entries)):
        weight = decay**idx
        total_weight += weight

        metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        roi_value = _safe_float(metrics.get("roi"), fallback=_safe_float(entry.get("roi")))
        if roi_value is not None:
            roi_signal += weight * (roi_value - 0.02)

        rar_value = _safe_float(entry.get("rar"), fallback=_safe_float(metrics.get("risk_adjusted_roi")))
        if rar_value is not None:
            roi_signal += weight * rar_value * 0.1

        drawdown_value = _safe_float(metrics.get("drawdown"), fallback=_safe_float(entry.get("drawdown")))
        if drawdown_value is not None:
            drawdown_signal += weight * (0.12 - drawdown_value)

        win_value = _safe_float(metrics.get("win_rate"), fallback=_safe_float(entry.get("win_rate")))
        if win_value is None:
            win_value = _safe_float(entry.get("winrate"))
        if win_value is not None:
            win_signal += weight * (win_value - 0.55)

        mutation_hint = _safe_float(entry.get("mutation_rate_hint"), fallback=_safe_float(metrics.get("mutation_rate")))
        if mutation_hint is not None:
            mutation_signal += weight * (mutation_hint - 0.05)

        cx_prob_hint = _safe_float(entry.get("cx_prob_hint"))
        if cx_prob_hint is not None:
            cx_prob_signal += weight * cx_prob_hint

        cx_eta_hint = _safe_float(entry.get("cx_eta_hint"))
        if cx_eta_hint is not None:
            cx_eta_signal += weight * cx_eta_hint

        mut_eta_hint = _safe_float(entry.get("mut_eta_hint"))
        if mut_eta_hint is not None:
            mut_eta_signal += weight * mut_eta_hint

    normalizer = total_weight if total_weight > 0 else 1.0
    return {
        "roi": roi_signal / normalizer,
        "drawdown": drawdown_signal / normalizer,
        "win_rate": win_signal / normalizer,
        "mutation_rate": mutation_signal / normalizer,
        "cx_prob": cx_prob_signal / normalizer,
        "cx_eta": cx_eta_signal / normalizer,
        "mut_eta": mut_eta_signal / normalizer,
    }


def _compute_weight(signal: float, caps: float) -> float:
    """Convert a signal into a bounded multiplicative weight."""
    return 1.0 + _clamp(signal, -caps, caps)


def _safe_float(value: Any, *, fallback: float | None = None) -> float | None:
    """Convert value to float when possible."""
    if value is None:
        return fallback
    try:
        val = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(val) or math.isinf(val):
        return fallback
    return val


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _log_adaptation(adjustments: dict[str, float], seed: int, reason: str, samples: int = 0) -> None:
    """Write durable adaptation summaries for auditing."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "reason": reason,
        "samples": samples,
        "adjustments": adjustments,
    }
    ADAPTATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ADAPTATION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as exc:  # pragma: no cover - durability safeguard
        LOGGER.warning("Unable to log adaptation entry: %s", exc)
