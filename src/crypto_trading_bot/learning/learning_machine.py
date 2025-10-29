"""
Learning Machine for Crypto Trading Bot.

Reads closed trades from trades.log, calculates performance metrics, and can
emit learning suggestions to logs/learning_feedback.jsonl for the dashboard.

This module provides both function-level APIs (run_learning_cycle) and a small
wrapper class (LearningMachine) for compatibility with scripts that expect a
class interface with generate_report().
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
from typing import Callable, Dict, Iterable, List

try:
    import numpy as np  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore[assignment]

from crypto_trading_bot.bot.utils.log_rotation import get_rotating_handler
from crypto_trading_bot.risk.risk_manager import get_dynamic_buffer
from crypto_trading_bot.utils.file_locks import _locked_file

logger = logging.getLogger("learning_machine")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    # Review report log (historical)
    logger.addHandler(get_rotating_handler("learning_review.log"))
    # Dedicated debug/ops log for this module as requested
    logger.addHandler(get_rotating_handler("learning_machine.log"))
    logger.propagate = False

_CONFIDENCE_MIN = 0.1
_CONFIDENCE_MAX = 1.0
_MAX_CONFIDENCE_DELTA = 0.1
_SHADOW_SUCCESS_THRESHOLD = 0.7
_SHADOW_MIN_SUCCESSFUL_RUNS = 100
_DRAW_DOWN_HALT_THRESHOLD = 0.05
_TRADE_DECAY_DAYS = 30
_MIN_WEIGHT = 0.05


def _compute_trade_weight(trade: dict, *, now: datetime.datetime | None = None) -> float:
    """Return an exponentially decayed weight for ``trade`` based on age."""

    timestamp = trade.get("timestamp")
    now = now or datetime.datetime.now(datetime.UTC)
    if not isinstance(timestamp, str):
        return 1.0
    try:
        trade_ts = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return 1.0
    if trade_ts.tzinfo is None:
        trade_ts = trade_ts.replace(tzinfo=datetime.UTC)
    age = now - trade_ts
    if age <= datetime.timedelta(days=_TRADE_DECAY_DAYS):
        return 1.0
    excess_days = max(age.days - _TRADE_DECAY_DAYS, 0)
    # Exponential decay after the threshold with a 30-day half-life.
    decay = math.exp(-excess_days / float(_TRADE_DECAY_DAYS))
    return max(_MIN_WEIGHT, min(1.0, decay))


def _safe_float(value: object) -> float | None:
    """Best-effort conversion of ``value`` to float, returning None when invalid."""

    try:
        candidate = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return candidate if math.isfinite(candidate) else None


def _validate_feedback_record(record: dict) -> None:
    """Ensure ``record`` follows the expected JSONL schema before persistence."""

    required_fields = {
        "timestamp": str,
        "type": str,
        "strategy": str,
        "status": str,
    }
    for field, expected in required_fields.items():
        if field not in record:
            raise ValueError(f"Missing required feedback field: {field}")
        if not isinstance(record[field], expected):
            raise TypeError(f"Field '{field}' must be {expected}, got {type(record[field])}")

    # When confidence fields are present, validate their ranges.
    for key in ("confidence_before", "confidence_after", "suggested_confidence", "applied_confidence"):
        if key in record and record[key] is not None:
            value = _safe_float(record[key])
            if value is None:
                raise TypeError(f"Field '{key}' must be numeric when provided.")
            if not (_CONFIDENCE_MIN <= value <= _CONFIDENCE_MAX):
                raise ValueError(f"Confidence {value} for '{key}' outside [{_CONFIDENCE_MIN}, {_CONFIDENCE_MAX}]")


def _append_jsonl(
    path: str,
    records: Iterable[dict],
    *,
    validator: Callable[[dict], None] | None = None,
) -> int:
    """Append ``records`` to ``path`` with a locked write, returning count written."""

    written = 0
    with _locked_file(path, "a") as handle:
        for record in records:
            if validator is not None:
                validator(record)
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
            written += 1
    return written


def _is_valid_learning_trade(trade: dict) -> bool:
    """Return True if the trade should influence learning decisions."""

    confidence = trade.get("confidence")
    try:
        confidence_val = float(confidence)
    except (TypeError, ValueError):
        logger.warning("Skipping trade %s — confidence missing or invalid", trade.get("trade_id"))
        return False

    if confidence_val <= 0 or confidence_val > 1:
        logger.warning(
            "Skipping trade %s — confidence %.4f outside acceptable range",
            trade.get("trade_id"),
            confidence_val,
        )
        return False

    if math.isclose(confidence_val, 0.5, abs_tol=1e-6):
        logger.warning(
            "Skipping trade %s — confidence %.4f flagged as deprecated placeholder",
            trade.get("trade_id"),
            confidence_val,
        )
        return False

    roi = trade.get("roi")
    try:
        roi_val = float(roi)
    except (TypeError, ValueError):
        logger.warning("Skipping trade %s — ROI missing or invalid", trade.get("trade_id"))
        return False

    # Reject outliers that indicate synthetic or corrupted entries
    if abs(roi_val) > 1.5:
        logger.warning(
            "Skipping trade %s — ROI %.4f outside learning bounds",
            trade.get("trade_id"),
            roi_val,
        )
        return False

    return True


def load_trades(log_path: str = "logs/trades.log") -> List[dict]:
    """Load closed trades with valid ROI from the trades log.

    Skips malformed lines and never raises so the learning loop is resilient.
    """
    trades: List[dict] = []
    if not os.path.exists(log_path):
        return trades

    now = datetime.datetime.now(datetime.UTC)
    with _locked_file(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                trade = json.loads(line)
            except json.JSONDecodeError:
                continue
            status = (trade.get("status") or "").lower()
            roi = trade.get("roi")
            size = trade.get("size")
            if (
                status == "closed"
                and isinstance(roi, (int, float))
                and isinstance(size, (int, float))
                and float(size) > 0.0
                and _is_valid_learning_trade(trade)
            ):
                trade["_weight"] = _compute_trade_weight(trade, now=now)
                trades.append(trade)
    return trades


def calculate_metrics(trades: List[dict]) -> Dict[str, float | int]:
    """Calculate core ROI-based metrics from closed trades."""
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_roi": 0.0,
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            # convenience extras for suggestion heuristics
            "roi_percent": 0.0,
            "sortino_ratio": 0.0,
        }

    roi_values: List[float] = []
    weights: List[float] = []
    for trade in trades:
        roi_val = _safe_float(trade.get("roi"))
        if roi_val is None:
            continue
        weight = _safe_float(trade.get("_weight"))
        if weight is None or weight <= 0:
            weight = 1.0
        weights.append(weight)
        roi_values.append(roi_val)

    if not roi_values:
        # All trades invalid; return zeroed metrics.
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "avg_roi": 0.0,
            "cumulative_return": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "roi_percent": 0.0,
        }

    total_weight = sum(weights) or float(len(roi_values))
    wins_weight = sum(w for roi, w in zip(roi_values, weights) if roi > 0)
    losses_weight = total_weight - wins_weight
    win_rate = wins_weight / total_weight if total_weight > 0 else 0.0
    wins_count = sum(1 for roi in roi_values if roi > 0)
    losses_count = len(roi_values) - wins_count

    logger.info(
        "Learning cycle win rate computed",
        extra={
            "wins": wins_weight,
            "losses": losses_weight,
            "total_trades": total_trades,
            "win_rate": win_rate,
        },
    )

    # Apply exponential decay weights to ROI contributions to minimize drift.
    adjusted_rois = [roi * min(1.0, max(_MIN_WEIGHT, weight)) for roi, weight in zip(roi_values, weights)]

    if np is not None:
        rois = np.array(roi_values, dtype=float)
        w = np.array(weights, dtype=float)
        if np.allclose(w.sum(), 0.0):
            w = np.ones_like(rois)

        avg_roi = float(np.average(rois, weights=w))
        cumulative_return = float(np.prod(1 + np.array(adjusted_rois, dtype=float)) - 1)

        mean_roi = avg_roi
        variance = np.average((rois - mean_roi) ** 2, weights=w)
        std = float(np.sqrt(variance))
        sharpe_ratio = float(np.mean(rois) / std) if std > 0 else 0.0

        downside = rois[rois < 0]
        if downside.size > 0:
            downside_weights = w[rois < 0]
            downside_mean = float(np.average(downside, weights=downside_weights))
            downside_variance = np.average(
                (downside - downside_mean) ** 2,
                weights=downside_weights,
            )
            dd = float(np.sqrt(downside_variance))
        else:
            dd = 0.0
        sortino_ratio = float(mean_roi / dd) if dd > 0 else 0.0

        cumulative = np.cumprod(1 + np.array(adjusted_rois, dtype=float))
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdowns)) if drawdowns.size > 0 else 0.0
    else:
        avg_roi = sum(roi * weight for roi, weight in zip(roi_values, weights)) / total_weight

        cumulative_product = 1.0
        for roi in adjusted_rois:
            cumulative_product *= 1 + roi
        cumulative_return = cumulative_product - 1.0

        if len(roi_values) > 1:
            variance = sum(weight * (roi - avg_roi) ** 2 for roi, weight in zip(roi_values, weights)) / total_weight
            std = math.sqrt(variance)
        else:
            std = 0.0
        sharpe_ratio = avg_roi / std if std > 0 else 0.0

        downside_values = [(roi, weight) for roi, weight in zip(roi_values, weights) if roi < 0]
        if downside_values:
            weighted_sum = sum(weight * roi for roi, weight in downside_values)
            downside_weight = sum(weight for _, weight in downside_values) or 1.0
            downside_mean = weighted_sum / downside_weight
            variance_downside = (
                sum(weight * (roi - downside_mean) ** 2 for roi, weight in downside_values) / downside_weight
            )
            dd = math.sqrt(variance_downside)
        else:
            dd = 0.0
        sortino_ratio = avg_roi / dd if dd > 0 else 0.0

        running_balance = 1.0
        running_max = 1.0
        max_drawdown = 0.0
        for roi in adjusted_rois:
            running_balance *= 1 + roi
            running_max = max(running_max, running_balance)
            if running_max > 0:
                drawdown = (running_balance - running_max) / running_max
                if drawdown < max_drawdown:
                    max_drawdown = drawdown

    return {
        "total_trades": int(total_trades),
        "wins": int(wins_count),
        "losses": int(losses_count),
        "win_rate": round(win_rate, 6),
        "avg_roi": round(avg_roi, 6),
        "cumulative_return": round(cumulative_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 6),
        "sortino_ratio": round(sortino_ratio, 6),
        "max_drawdown": round(max_drawdown, 6),
        # convenience for heuristics/UX
        "roi_percent": round(cumulative_return * 100.0, 4),
    }


def run_learning_cycle() -> Dict[str, float | int | str]:
    """Run a single learning cycle and return metrics (with timestamp/buffer)."""
    trades = load_trades()
    metrics = calculate_metrics(trades)
    metrics["timestamp"] = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")
    metrics["capital_buffer"] = get_dynamic_buffer()
    return metrics


class LearningMachine:
    """Lightweight wrapper exposing a class interface used by some scripts."""

    def generate_report(self) -> Dict[str, float | int | str]:  # pragma: no cover - thin wrapper
        """Return the latest learning cycle metrics."""
        return run_learning_cycle()


def _load_shadow_results(path: str = "logs/shadow_test_results.jsonl") -> List[dict]:
    """Load shadow test results without raising on malformed rows."""
    if not os.path.exists(path):
        return []
    rows: List[dict] = []
    with _locked_file(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _evaluate_shadow_promotions(
    *,
    output_path: str,
    timestamp: str,
    shadow_path: str = "logs/shadow_test_results.jsonl",
) -> None:
    """Track shadow test streaks and flag promotion candidates."""
    rows = _load_shadow_results(shadow_path)
    if not rows:
        return

    stats: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}

    for record in rows:
        strategy = record.get("strategy") or record.get("strategy_name") or "Unknown"
        if not isinstance(strategy, str):
            continue
        success = record.get("success_rate")
        if success is None:
            success = record.get("win_rate")
        try:
            success_val = float(success)
        except (TypeError, ValueError):
            continue
        counts[strategy] = counts.get(strategy, 0) + 1
        stats[strategy] = stats.get(strategy, {"success_sum": 0.0})
        stats[strategy]["success_sum"] += success_val

    promotion_records: List[dict] = []
    for strategy, total_runs in counts.items():
        if total_runs == 0:
            continue
        success_sum = stats[strategy]["success_sum"]
        average_success = success_sum / total_runs
        if total_runs >= 10000 and average_success >= 0.99:
            msg = (
                f"[Promotion] Strategy '{strategy}' reached promotion threshold "
                f"(runs={total_runs}, success_rate={average_success:.2%})."
            )
            logger.info(msg)
            promotion_records.append(
                {
                    "timestamp": timestamp,
                    "type": "promotion_decision",
                    "strategy": strategy,
                    "shadow_runs": int(total_runs),
                    "shadow_success_rate": round(average_success, 6),
                    "status": "promoted",
                    "action": "manual_promotion_recorded",
                    "approval": "manual",
                }
            )
            continue
        if total_runs >= 100 and average_success > 0.7:
            msg = (
                f"[Promotion] Strategy '{strategy}' qualifies as a promotion candidate "
                f"(runs={total_runs}, success_rate={average_success:.2%})."
            )
            logger.info(msg)
            promotion_record = {
                "timestamp": timestamp,
                "type": "promotion_candidate",
                "strategy": strategy,
                "shadow_runs": int(total_runs),
                "shadow_success_rate": round(average_success, 6),
                "status": "promotion_candidate",
                "action": "pending_manual_promotion",
            }
            promotion_records.append(promotion_record)

    if not promotion_records:
        return

    _append_jsonl(output_path, promotion_records, validator=_validate_feedback_record)


def run_learning_machine(output_path: str = "logs/learning_feedback.jsonl") -> int:
    """Generate simple learning suggestions and append to JSONL file.

    Uses run_learning_cycle() metrics and the optimization.generate_suggestions
    helper to produce a compact suggestion record. Returns count written.
    """
    try:
        # Imported lazily to avoid circulars and heavy deps at import time
        # pylint: disable=import-outside-toplevel
        from .optimization import generate_suggestions  # type: ignore
    except ImportError:  # pragma: no cover - fallback path

        def generate_suggestions(_report):  # type: ignore
            return [{"suggestion": "Monitor performance", "confidence": 0.7, "reason": "fallback"}]

    # Load closed trades for diagnostics and diversity checks
    trades = load_trades()
    total_trades = len(trades)
    # Confidence stats (graceful parsing)
    valid_conf: List[float] = []
    for t in trades:
        try:
            c = float(t.get("confidence"))
            valid_conf.append(c)
        except (TypeError, ValueError):
            continue
    n_conf_not_half = sum(1 for v in valid_conf if abs(v - 0.5) > 1e-9)

    # Strategy distribution
    by_strategy: Dict[str, int] = {}
    recent_samples: List[str] = []
    for t in trades[-10:]:
        tid = t.get("trade_id")
        strat = t.get("strategy") or "Unknown"
        ts = t.get("timestamp")
        if isinstance(strat, str):
            by_strategy[strat] = by_strategy.get(strat, 0) + 1
        if isinstance(tid, str) and isinstance(ts, str):
            recent_samples.append(f"{ts} • {tid} • {strat}")

    strategies_considered = len(by_strategy)

    # Emit debug summary to both console and log
    summary_line = (
        f"[LearningMachine] Loaded {total_trades} closed trades "
        f"({len(valid_conf)} with valid confidence, {n_conf_not_half} != 0.5). "
        f"Strategies={strategies_considered}: {by_strategy}"
    )
    logger.info(summary_line)
    if recent_samples:
        for line in recent_samples:
            logger.info("[LearningMachine] Sample: %s", line)

    # Compute report metrics and generate suggestions
    report = run_learning_cycle()
    drawdown_val = _safe_float(report.get("max_drawdown"))
    if drawdown_val is not None and abs(drawdown_val) > _DRAW_DOWN_HALT_THRESHOLD:
        logger.info("Pausing learning due to drawdown.")
        return 0

    suggestions = generate_suggestions(report) or []

    # If no suggestions, write a diagnostic entry to learning_feedback.jsonl
    if not suggestions:
        if total_trades == 0:
            reason = "No closed trades available"
        elif valid_conf and n_conf_not_half == 0:
            reason = "All trades had confidence = 0.5"
        elif strategies_considered == 0:
            reason = "No strategies detected in closed trades"
        else:
            reason = "Optimizer returned no suggestions"

        # heuristically set strategy label for diagnostics
        strat_label = (
            "Multiple"
            if strategies_considered > 1
            else (next(iter(by_strategy.keys())) if strategies_considered == 1 else "Unknown")
        )

        diag = {
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
            "type": "no_suggestions_generated",
            "strategy": strat_label,
            "status": "no_suggestions_generated",
            "reason": reason,
            "total_trades": total_trades,
        }
        _append_jsonl(output_path, [diag], validator=_validate_feedback_record)
        msg = (
            f"[LearningMachine] No suggestions generated — reason='{reason}', "
            f"total_trades={total_trades}, strategies={strategies_considered}"
        )
        logger.info(msg)
        return 0

    # Otherwise, append generated suggestions
    ts = datetime.datetime.now(datetime.UTC).isoformat()
    accepted_records: List[dict] = []
    for index, suggestion in enumerate(suggestions, start=1):
        primary_label = suggestion.get("strategy") or suggestion.get("strategy_name")
        strategy_name = primary_label or suggestion.get("category") or "Unknown"

        before_raw = (
            suggestion.get("confidence_before") or suggestion.get("current_confidence") or suggestion.get("confidence")
        )
        after_raw = (
            suggestion.get("confidence_after") or suggestion.get("suggested_confidence") or suggestion.get("confidence")
        )

        confidence_after = _safe_float(after_raw)
        if confidence_after is None:
            logger.warning("Rejecting suggestion with invalid confidence_after: %s", after_raw)
            continue

        confidence_before = _safe_float(before_raw)
        if confidence_before is None:
            confidence_before = confidence_after

        # Cap confidence change
        if abs(confidence_after - confidence_before) > _MAX_CONFIDENCE_DELTA:
            logger.warning(
                "Rejecting drastic confidence change: %s → %s",
                confidence_before,
                confidence_after,
            )
            continue

        # Clamp to safe range
        if not (_CONFIDENCE_MIN <= confidence_after <= _CONFIDENCE_MAX):
            logger.warning("Invalid confidence: %s", confidence_after)
            continue

        status_val = suggestion.get("status") or "pending"
        rec = {
            "timestamp": ts,
            "type": "learning_suggestion",
            "strategy": strategy_name,
            "confidence_before": round(confidence_before, 6),
            "confidence_after": round(confidence_after, 6),
            "reason": suggestion.get("reason", ""),
            "status": status_val,
        }
        for key in ("parameter", "old_value", "new_value", "suggestion", "category"):
            if key in suggestion and key not in rec:
                rec[key] = suggestion[key]
        accepted_records.append(rec)
        logger.info(
            "[LearningMachine] Suggestion #%s: strategy=%s before=%.4f after=%.4f status=%s reason=%s",
            index,
            rec["strategy"],
            rec["confidence_before"],
            rec["confidence_after"],
            rec["status"],
            rec.get("reason"),
        )

    if suggestions and not accepted_records:
        logger.warning("All suggestions rejected by safety filters; nothing written.")

    wrote = _append_jsonl(output_path, accepted_records, validator=_validate_feedback_record) if accepted_records else 0
    # Shadow test recent performance per suggested strategy
    try:
        _shadow_log_path = "logs/shadow_test_results.jsonl"
        recent_trades = load_trades()
        recent_trades = recent_trades[-100:]
        shadow_records: List[dict] = []
        if recent_trades and accepted_records:
            by_strategy: Dict[str, List[dict]] = {}
            for trade in recent_trades:
                strat = trade.get("strategy") or "Unknown"
                by_strategy.setdefault(strat, []).append(trade)
            for record in accepted_records:
                strat = record.get("strategy") or "Unknown"
                rows = by_strategy.get(strat, [])
                if not rows:
                    continue
                rois = []
                for entry in rows:
                    roi_candidate = _safe_float(entry.get("roi"))
                    if roi_candidate is not None:
                        rois.append(roi_candidate)
                if not rois:
                    continue
                wins = sum(1 for value in rois if value > 0)
                total = len(rois)
                success_rate = wins / total if total else 0.0
                avg_roi = sum(rois) / total if total else 0.0
                shadow_records.append(
                    {
                        "timestamp": ts,
                        "strategy": strat,
                        "success_rate": round(success_rate, 4),
                        "avg_roi": round(avg_roi, 6),
                        "confidence": record.get("confidence_after"),
                    }
                )
                logger.info(
                    "[SHADOW TEST] %s -> %.2f%%, ROI=%.2f",
                    strat,
                    success_rate * 100,
                    avg_roi,
                )
        if shadow_records:
            _append_jsonl(_shadow_log_path, shadow_records)
    except (OSError, ValueError, TypeError) as _e:  # pragma: no cover - diagnostics only
        logger.info("Shadow test logging skipped: %s", _e)

    # Bayesian optimization for confidence (optional dependency)
    try:
        from skopt import gp_minimize  # type: ignore  # pylint: disable=import-outside-toplevel
        from skopt.space import Real  # type: ignore  # pylint: disable=import-outside-toplevel

        def _sharpe_like(conf: float, rois: List[float]) -> float:
            arr = np.asarray(rois, dtype=float)
            if arr.size == 0:
                return 0.0
            mu = float(arr.mean()) * conf
            sd = float(arr.std())
            return 0.0 if sd == 0 else mu / sd

        # Build objective from recent trades regardless of strategy
        rois_all: List[float] = []
        for t in recent_trades:
            try:
                rois_all.append(float(t.get("roi")))
            except (TypeError, ValueError):
                continue
        if rois_all:

            def objective(x):
                conf = x[0]
                return -_sharpe_like(conf, rois_all)

            res = gp_minimize(
                objective,
                [Real(0.1, 1.0, name="confidence")],
                n_calls=20,
                random_state=0,
            )
            best_conf = float(res.x[0])
            best_score = float(-res.fun)
            out_rec = {
                "timestamp": ts,
                "type": "learning_suggestion",
                "strategy": "SimpleRSIStrategy",
                "confidence_before": None,
                "confidence_after": round(best_conf, 4),
                "suggested_confidence": round(best_conf, 4),
                "sharpe": round(best_score, 4),
                "reason": "bayesian_optimizer",
                "status": "analysis",
                "source": "bayesian_optimization",
            }
            _append_jsonl(output_path, [out_rec], validator=_validate_feedback_record)
            wrote += 1
            opt_msg = f"[OPTIMIZER] Suggested confidence={best_conf:.4f} " f"with Sharpe={best_score:.4f}"
            logger.info(opt_msg)
    except (ImportError, ValueError, TypeError) as _e:  # pragma: no cover - optional dep fallback
        logger.info("Bayesian optimizer unavailable or failed: %s", _e)

    # Auto-apply suggestions based on shadow tests
    try:
        shadow_path = "logs/shadow_test_results.jsonl"
        if os.path.exists(shadow_path):
            by_strat: Dict[str, List[dict]] = {}
            with _locked_file(shadow_path, "r") as handle:
                for line in handle:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sname = rec.get("strategy") or "Unknown"
                    by_strat.setdefault(sname, []).append(rec)
            applied_records: List[dict] = []
            for sname, rows in by_strat.items():
                success_runs = sum(
                    1
                    for rec in rows
                    if (rate := _safe_float(rec.get("success_rate"))) is not None and rate >= _SHADOW_SUCCESS_THRESHOLD
                )
                if success_runs < _SHADOW_MIN_SUCCESSFUL_RUNS:
                    continue

                latest_conf = None
                try:
                    with _locked_file(output_path, "r") as lf:
                        for line in lf:
                            try:
                                entry = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            has_after = entry.get("confidence_after")
                            has_suggested = entry.get("suggested_confidence")
                            if entry.get("strategy") == sname and (has_after or has_suggested):
                                latest_conf = has_after or has_suggested
                except FileNotFoundError:
                    latest_conf = None

                confidence_val = _safe_float(latest_conf)
                if confidence_val is None:
                    continue
                if not (_CONFIDENCE_MIN <= confidence_val <= _CONFIDENCE_MAX):
                    logger.warning(
                        "Auto-apply confidence out of range for %s: %s",
                        sname,
                        latest_conf,
                    )
                    continue

                applied_record = {
                    "timestamp": ts,
                    "type": "learning_suggestion",
                    "strategy": sname,
                    "confidence_before": confidence_val,
                    "confidence_after": confidence_val,
                    "status": "applied",
                    "applied_confidence": confidence_val,
                    "reason": "auto_apply_from_shadow_test",
                    "source": "auto_apply_from_shadow_test",
                }
                applied_records.append(applied_record)
                logger.info(
                    "[LEARNING APPLY] %s updated with confidence=%.3f (success_runs=%s)",
                    sname,
                    confidence_val,
                    success_runs,
                )

            if applied_records:
                wrote += _append_jsonl(output_path, applied_records, validator=_validate_feedback_record)
    except (OSError, ValueError, TypeError) as _e:  # pragma: no cover - diagnostics only
        logger.info("Auto-apply skipped: %s", _e)
    _evaluate_shadow_promotions(output_path=output_path, timestamp=ts)

    logger.info("Wrote %s suggestion(s) to %s", wrote, output_path)
    return wrote


def _debug_main() -> None:  # pragma: no cover
    """Run a single learning cycle and emit suggestions for manual inspection."""

    result = run_learning_cycle()
    logger.info("Learning Machine metrics", extra={"metrics": result})
    try:
        suggestions_written = run_learning_machine()
        logger.info(
            "Suggestions written via debug run",
            extra={"count": suggestions_written},
        )
    except (OSError, RuntimeError, ValueError) as exc:  # safety for ad-hoc runs
        logger.error("Failed to write suggestions in debug mode", extra={"error": str(exc)})


if __name__ == "__main__":  # pragma: no cover
    _debug_main()
