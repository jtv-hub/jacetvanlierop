"""
Trading Logic Module

Evaluates signals using predefined strategies, executes mock trades,
manages open positions, and checks exit conditions.
"""

# pylint: disable=too-many-lines,line-too-long

import datetime
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal, getcontext
from pathlib import Path
from typing import Any, Mapping, Optional

# Optional dependency for correlation checks
try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

import crypto_trading_bot.utils.price_feed as price_feed_module
from crypto_trading_bot.bot.state.portfolio_state import (
    load_portfolio_state,
    refresh_portfolio_state,
)
from crypto_trading_bot.bot.utils.alerts import send_alert
from crypto_trading_bot.config import (
    CANARY_MAX_FRACTION,
    CONFIG,
    DEPLOY_PHASE,
    get_mode_label,
    is_live,
    set_live_mode,
)
from crypto_trading_bot.config.constants import KILL_SWITCH_FILE
from crypto_trading_bot.context.trading_context import TradingContext
from crypto_trading_bot.ledger.trade_ledger import TradeLedger
from crypto_trading_bot.safety import risk_guard
from crypto_trading_bot.utils.file_locks import _locked_file
from crypto_trading_bot.utils.kraken_api import get_ohlc_data
from crypto_trading_bot.utils.kraken_client import (
    KrakenAPIError,
    KrakenAuthError,
    _invalidate_pair_cache,
    kraken_get_asset_pair_meta,
)
from crypto_trading_bot.utils.kraken_client import (
    kraken_place_order as _kraken_place_order,
)
from crypto_trading_bot.utils.kraken_client import kraken_place_order as kraken_place_order
from crypto_trading_bot.utils.kraken_pairs import ensure_usdc_pair
from crypto_trading_bot.utils.price_feed import get_current_price
from crypto_trading_bot.utils.price_history import (
    HistoryUnavailable,
    append_live_price,
    get_history_prices,
)
from crypto_trading_bot.utils.system_logger import get_system_logger

# Optional RSI calculator (import may vary by environment)
try:
    from crypto_trading_bot.indicators.rsi import calculate_rsi  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    calculate_rsi = None  # type: ignore[assignment]

from .strategies.advanced_strategies import (
    ADXStrategy,
    BollingerBandStrategy,
    CompositeStrategy,
    KeltnerBreakoutStrategy,
    MACDStrategy,
    StochRSIStrategy,
    VWAPStrategy,
)
from .strategies.dual_threshold_strategies import DualThresholdStrategy
from .strategies.simple_rsi_strategies import SimpleRSIStrategy

context = TradingContext()
logger = get_system_logger().getChild("trading_logic")

TRADES_LOG_PATH = "logs/trades.log"
PORTFOLIO_STATE_PATH = "logs/portfolio_state.json"
PRICE_STALE_THRESHOLD = datetime.timedelta(minutes=5)
DRAWDOWN_GUARD_LIMIT = 0.10

_DISK_GUARD_THRESHOLD_MB = float(os.getenv("CRYPTO_TRADING_BOT_DISK_GUARD_MB", "500"))
_DISK_GUARD_PATH = os.getenv("CRYPTO_TRADING_BOT_DISK_GUARD_PATH", os.getcwd())

STATE_SNAPSHOT_PATH = Path("logs/state_snapshot.json")
_STATE_SNAPSHOT_INTERVAL = float(os.getenv("CRYPTO_TRADING_BOT_STATE_SNAPSHOT_SECONDS", "3600") or "3600")
_LAST_STATE_SNAPSHOT = 0.0
_STARTUP_SNAPSHOT_LOGGED = False

TRADE_INTERVAL = 300
MAX_PORTFOLIO_RISK = CONFIG.get("max_portfolio_risk", 0.10)
PAPER_STARTING_BALANCE = float(CONFIG.get("paper_mode", {}).get("starting_balance", 100_000.0))
SLIPPAGE = 0.0  # slippage handled per-asset in ledger; do not apply here
_TEST_MODE_SENTINELS = {"1", "true", "yes", "on"}
TEST_MODE = os.getenv("CRYPTO_TRADING_BOT_TEST_MODE", "0").strip().lower() in _TEST_MODE_SENTINELS

_VOLUME_CACHE: dict[str, tuple[float, float]] = {}
_VOLUME_CACHE_TTL_SECONDS = 60.0


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=16),
    retry=retry_if_exception_type((KrakenAPIError, RuntimeError, OSError)),
)
def _kraken_place_order_retry(*args, **kwargs):
    """Resilient wrapper around Kraken order placement."""

    return _kraken_place_order(*args, **kwargs)


def _resolve_trade_size_bounds(pair: str) -> tuple[float, float]:
    """Return (min_volume, max_volume) for the given trading pair."""

    trade_size_cfg = CONFIG.get("trade_size", {}) or {}
    default_min = float(trade_size_cfg.get("default_min", 0.001) or 0.001)
    default_max = float(trade_size_cfg.get("default_max", 0.005) or 0.005)
    per_pair = trade_size_cfg.get("per_pair", {}) or {}
    pair_cfg = per_pair.get(pair, {})
    min_volume = float(pair_cfg.get("min_volume", default_min) or default_min)
    max_volume = float(pair_cfg.get("max_volume", default_max) or default_max)
    return max(min_volume, 0.0), max(max_volume, min_volume)


# Real-time price feed imported above per lint ordering


# get_current_price now lives in utils.price_feed; removed local duplicate.


# Removed hardcoded ASSETS list. Pairs are now centralized in CONFIG["tradable_pairs"].


@dataclass
class _RuntimeState:
    """Mutable runtime flags tracked across trading loop iterations."""

    live_block_logged: bool = False
    last_capital_log: tuple[float | None, str | None] = (None, None)
    kraken_pause_until: float | None = None
    kraken_failure_pause_until: float | None = None
    last_mode_label: str | None = None
    auto_paused_reason: str | None = None
    dry_run_logged: bool = False
    emergency_stop_triggered: bool = False
    kill_switch_auto_cleared: bool = False
    disk_space_block_active: bool = False
    drawdown_block_active: bool = False


_STATE = _RuntimeState()
_KRAKEN_FAILURE_PAUSE_SECONDS = 60.0
_KRAKEN_FAILURE_PAUSE_UNTIL: float | None = 0.0

getcontext().prec = 18


strategy_context: dict[str, dict[str, list[Any]]] = {"live": {}, "paper": {}}


def _monotonic_now() -> float:
    """Return the current monotonic timestamp."""

    return time.monotonic()


def _set_kraken_failure_pause(pause_until: float) -> None:
    """Record the Kraken failure pause deadline for tests and runtime guards."""

    global _KRAKEN_FAILURE_PAUSE_UNTIL  # pylint: disable=global-statement
    pause_until = float(pause_until)
    _KRAKEN_FAILURE_PAUSE_UNTIL = pause_until
    _STATE.kraken_failure_pause_until = pause_until
    _STATE.kraken_pause_until = pause_until


def _clear_kraken_failure_pause(*, clear_runtime_pause: bool = True) -> None:
    """Clear any Kraken failure pause deadline."""

    global _KRAKEN_FAILURE_PAUSE_UNTIL  # pylint: disable=global-statement
    _KRAKEN_FAILURE_PAUSE_UNTIL = None
    _STATE.kraken_failure_pause_until = None
    if clear_runtime_pause:
        _STATE.kraken_pause_until = None


def _normalise_mode_label(mode: str | None) -> str:
    """Normalise free-form mode labels to canonical cache keys."""

    if not mode:
        return "paper"
    lowered = mode.strip().lower()
    if "live" in lowered:
        return "live"
    if "paper" in lowered or "demo" in lowered:
        return "paper"
    return lowered or "paper"


def _build_strategy_pipeline(
    *,
    per_asset_params: Mapping[str, Mapping[str, Any]] | None = None,
    mode: str | None = None,
) -> list[Any]:
    """Construct a fresh list of strategy instances for the requested mode."""

    del mode  # Mode-specific adjustments may be added in the future.

    per_asset_params = per_asset_params or {}

    def _per_asset(name: str) -> dict[str, Any]:
        raw = per_asset_params.get(name, {})
        if isinstance(raw, Mapping):
            return dict(raw)
        return dict(raw) if hasattr(raw, "items") else {}

    rsi_cfg = CONFIG.get("rsi", {}) or {}
    return [
        SimpleRSIStrategy(
            period=rsi_cfg.get("period", 21),
            lower=rsi_cfg.get("lower", 48),
            upper=rsi_cfg.get("upper", 75),
            per_asset=_per_asset("SimpleRSIStrategy"),
        ),
        DualThresholdStrategy(),
        MACDStrategy(per_asset=_per_asset("MACDStrategy")),
        KeltnerBreakoutStrategy(per_asset=_per_asset("KeltnerBreakoutStrategy")),
        StochRSIStrategy(per_asset=_per_asset("StochRSIStrategy")),
        BollingerBandStrategy(per_asset=_per_asset("BollingerBandStrategy")),
        VWAPStrategy(per_asset=_per_asset("VWAPStrategy")),
        ADXStrategy(per_asset=_per_asset("ADXStrategy")),
        CompositeStrategy(per_asset=_per_asset("CompositeStrategy")),
    ]


def _get_locked_strategy_pipeline(
    pair: str,
    mode: str | None,
    per_asset_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Any]:
    """Return the cached strategy pipeline for ``pair`` and ``mode``."""

    canonical_mode = _normalise_mode_label(mode)
    cache = strategy_context.setdefault(canonical_mode, {})
    if pair not in cache:
        cache[pair] = _build_strategy_pipeline(
            per_asset_params=per_asset_params,
            mode=canonical_mode,
        )
    return cache[pair]


def _ensure_startup_snapshot() -> None:
    """Log a one-time snapshot of deployment configuration at startup."""

    global _STARTUP_SNAPSHOT_LOGGED  # pylint: disable=global-statement
    if _STARTUP_SNAPSHOT_LOGGED:
        return

    try:
        strategy_instances = _build_strategy_pipeline()
        strategy_names = [s.__class__.__name__ for s in strategy_instances]
    except Exception:  # pylint: disable=broad-exception-caught
        strategy_names = []

    risk_state = risk_guard.load_state()
    snapshot = {
        "deployment_phase": DEPLOY_PHASE,
        "canary_max_fraction": CANARY_MAX_FRACTION,
        "is_live": is_live,
        "kraken_endpoint": CONFIG.get("kraken", {}).get("api_base") or "https://api.kraken.com",
        "strategies": strategy_names,
        "tradable_pairs": CONFIG.get("tradable_pairs", []),
        "initial_balance": PAPER_STARTING_BALANCE,
        "risk_paused": bool(risk_state.get("paused")),
        "risk_reason": risk_state.get("pause_reason"),
    }
    logger.info("[startup] configuration snapshot: %s", snapshot)
    send_alert("Startup snapshot emitted", level="INFO", context=snapshot)
    _STARTUP_SNAPSHOT_LOGGED = True


def _maybe_write_state_checkpoint(snapshot_context: Optional[dict[str, Any]] = None) -> None:
    """Persist a lightweight state snapshot periodically."""

    global _LAST_STATE_SNAPSHOT  # pylint: disable=global-statement
    if _STATE_SNAPSHOT_INTERVAL <= 0:
        return
    now = time.time()
    if now - _LAST_STATE_SNAPSHOT < _STATE_SNAPSHOT_INTERVAL:
        return
    checkpoint = {
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "mode": get_mode_label(),
        "is_live": is_live,
        "deployment_phase": DEPLOY_PHASE,
        "risk_state": risk_guard.load_state(),
        "context": snapshot_context or {},
    }
    try:
        STATE_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with STATE_SNAPSHOT_PATH.open("w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        _LAST_STATE_SNAPSHOT = now
    except OSError as exc:
        logger.error("Failed to write state snapshot: %s", exc)


def _apply_deploy_phase_limits(
    pair: str,
    size: float,
    price: float,
    *,
    emit_alert: bool = True,
) -> tuple[float, Optional[dict[str, Any]]]:
    """Scale trade size when running in canary deployment mode."""

    if DEPLOY_PHASE != "canary":
        return size, None
    trade_size_cfg = CONFIG.get("trade_size", {}) or {}
    base_max = float(trade_size_cfg.get("max", max(size, 0.0)) or max(size, 0.0))
    allowed_size = max(base_max * CANARY_MAX_FRACTION, 0.0)
    limit_context = {
        "phase": DEPLOY_PHASE,
        "pair": pair,
        "requested_size": size,
        "allowed_size": allowed_size,
        "price": price,
        "max_fraction": CANARY_MAX_FRACTION,
    }
    if allowed_size <= 0:
        if emit_alert:
            send_alert(
                f"[canary] Trade blocked for {pair} — effective canary limit is zero.",
                level="WARNING",
                context=limit_context,
            )
        return 0.0, limit_context
    if size <= allowed_size:
        return size, None
    scaled = allowed_size
    limit_context["scaled_size"] = scaled
    logger.warning(
        "Canary deployment limiting trade size for %s: %.8f -> %.8f (fraction %.2f%%)",
        pair,
        size,
        scaled,
        CANARY_MAX_FRACTION * 100,
    )
    if emit_alert:
        send_alert(
            f"[canary] Trade size limited for {pair}",
            level="WARNING",
            context=limit_context,
        )
    return scaled, limit_context


def _coerce_strategy_param(value: Any, seen: set[int]) -> Any:
    """Return a stable, serializable representation of strategy state values."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _coerce_strategy_param(val, seen) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_strategy_param(item, seen) for item in value]
    if isinstance(value, (set, frozenset)):
        coerced_items = [_coerce_strategy_param(item, seen) for item in value]
        try:
            return sorted(coerced_items, key=repr)
        except TypeError:
            return coerced_items
    if callable(getattr(value, "generate_signal", None)):
        return _serialize_strategy(value, _seen=seen)
    if hasattr(value, "__dict__"):
        obj_id = id(value)
        if obj_id in seen:
            return f"<recursive:{value.__class__.__name__}>"
        seen.add(obj_id)
        try:
            public_attrs = {
                key: _coerce_strategy_param(val, seen) for key, val in value.__dict__.items() if not key.startswith("_")
            }
        finally:
            seen.discard(obj_id)
        return {
            "class": value.__class__.__name__,
            "attrs": public_attrs,
        }
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return repr(value)


def _serialize_strategy(strategy: Any, *, _seen: set[int] | None = None) -> dict[str, Any]:
    """Represent ``strategy`` as a lightweight dictionary for parity tests."""

    seen = _seen if _seen is not None else set()
    obj_id = id(strategy)
    if obj_id in seen:
        return {"name": strategy.__class__.__name__, "params": {"_circular": True}}
    seen.add(obj_id)
    try:
        spec: dict[str, Any] = {"name": strategy.__class__.__name__}
        params: dict[str, Any] = {}
        state = getattr(strategy, "__dict__", None)
        if isinstance(state, dict):
            for key, value in state.items():
                if key.startswith("_"):
                    continue
                params[key] = _coerce_strategy_param(value, seen)
        config = getattr(strategy, "config", None)
        if isinstance(config, dict):
            params.setdefault("config", _coerce_strategy_param(config, seen))
        spec["params"] = params
        return spec
    finally:
        seen.discard(obj_id)


def _consecutive_losses(limit: int) -> int:
    """Return the number of trailing consecutive losing trades."""

    if limit <= 0:
        return 0
    if not os.path.exists(TRADES_LOG_PATH):
        return 0

    count = 0
    try:
        with _locked_file(TRADES_LOG_PATH, "r") as handle:
            lines = handle.readlines()
    except OSError:
        return 0

    for line in reversed(lines):
        if count >= limit:
            break
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (rec.get("status") or "").lower() != "closed":
            continue
        roi = rec.get("roi")
        try:
            roi_val = float(roi)
        except (TypeError, ValueError):
            continue
        if roi_val < 0:
            count += 1
        else:
            break

    return count


def _evaluate_auto_pause(state_snapshot: dict[str, Any] | None) -> tuple[bool, str | None]:
    """Inspect configured auto-pause thresholds and return status."""

    cfg = CONFIG.get("auto_pause", {})
    drawdown_limit = float(cfg.get("max_drawdown_pct", 0.10) or 0.0)
    roi_limit = float(cfg.get("max_total_roi_pct", -0.15) or 0.0)
    loss_limit = int(cfg.get("max_consecutive_losses", 5) or 0)

    state = state_snapshot or {}

    try:
        drawdown = float(state.get("drawdown_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        drawdown = 0.0

    try:
        state_drawdown_limit = float(state.get("drawdown_limit"))
    except (TypeError, ValueError):
        state_drawdown_limit = None
    if state_drawdown_limit is not None and state_drawdown_limit > 0:
        drawdown_limit = state_drawdown_limit

    if drawdown_limit > 0 and drawdown >= drawdown_limit:
        return True, f"Max drawdown {drawdown:.2%} ≥ limit {drawdown_limit:.2%}"

    try:
        total_roi = float(state.get("total_roi", 0.0) or 0.0)
    except (TypeError, ValueError):
        total_roi = 0.0

    if roi_limit < 0 and total_roi <= roi_limit:
        return True, f"Total ROI {total_roi:.2%} ≤ limit {roi_limit:.2%}"

    if loss_limit > 0:
        streak = _consecutive_losses(loss_limit)
        if streak >= loss_limit:
            return True, f"{streak} consecutive losses ≥ limit {loss_limit}"

    return False, None


def _submit_live_trade(
    *,
    pair: str,
    side: str,
    size: float,
    price: float,
    strategy: str,
    confidence: float,
) -> bool:
    """Submit an order to the live exchange when live trading is enabled.

    Returns ``True`` if the attempt should proceed, ``False`` when blocked.
    """

    if not is_live:
        if not _STATE.live_block_logged:
            logger.warning(
                "🚫 Live trade blocked (paper) pair=%s side=%s size=%.6f price=%.4f strat=%s conf=%.3f",
                pair,
                side,
                size,
                price,
                strategy,
                confidence,
            )
            _STATE.live_block_logged = True
        else:
            logger.debug(
                "Live trade attempt blocked (paper mode) | pair=%s side=%s size=%.6f",
                pair,
                side,
                size,
            )
        return False

    if risk_guard.is_paused():
        _, pause_reason = risk_guard.check_pause()
        message = f"Kill-switch active — skipping live trade for {pair} ({pause_reason or 'paused'})"
        logger.critical(message)
        send_alert(
            message,
            level="CRITICAL",
            context={"pair": pair, "strategy": strategy, "phase": DEPLOY_PHASE},
        )
        return False

    limited_size, _ = _apply_deploy_phase_limits(pair, size, price, emit_alert=False)
    if limited_size <= 0:
        logger.warning("Skipping live trade for %s — canary limit reduced size to zero.", pair)
        return False
    if limited_size != size:
        size = limited_size

    now = _monotonic_now()
    pause_deadline = _STATE.kraken_failure_pause_until or _STATE.kraken_pause_until
    if pause_deadline is not None and now < pause_deadline:
        pause_remaining = pause_deadline - now
        logger.error(
            "Kraken trading paused (%.1fs remaining) due to earlier error; skipping %s",
            pause_remaining,
            pair,
        )
        return False

    logger.info(
        "Submitting live trade | pair=%s side=%s size=%.6f price=%.4f strategy=%s confidence=%.3f",
        pair,
        side,
        size,
        price,
        strategy,
        confidence,
    )

    kraken_cfg = CONFIG.get("kraken", {}) or {}
    tif = kraken_cfg.get("time_in_force") or None
    validate_flag = bool(kraken_cfg.get("validate_orders", False))

    min_cost_default = float(
        CONFIG.get(
            "kraken_min_cost_threshold",
            kraken_cfg.get("min_cost_threshold", 0.5),
        )
    )
    min_cost_by_pair = kraken_cfg.get("pair_cost_minimums", {}) or {}
    config_cost_threshold = float(min_cost_by_pair.get(pair, min_cost_default))

    try:
        pair_meta = kraken_get_asset_pair_meta(pair)
    except KrakenAPIError as exc:
        pause_until = _monotonic_now() + _KRAKEN_FAILURE_PAUSE_SECONDS
        _set_kraken_failure_pause(pause_until)
        logger.error(
            "Kraken pair metadata failed; pause %.0fs for %s (%s)",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            exc,
        )
        return False

    config_min_volume, _ = _resolve_trade_size_bounds(pair)
    min_volume = max(float(pair_meta.get("ordermin", 0.0) or 0.0), config_min_volume)
    metadata_cost_threshold = float(pair_meta.get("costmin", 0.0) or 0.0)
    price_decimals = int(pair_meta.get("pair_decimals", 5) or 5)
    volume_decimals = int(pair_meta.get("lot_decimals", 8) or 8)

    price_step = Decimal("1").scaleb(-price_decimals)
    volume_step = Decimal("1").scaleb(-volume_decimals)

    price_dec = Decimal(str(price)).quantize(price_step, rounding=ROUND_DOWN)
    size_dec = Decimal(str(size)).quantize(volume_step, rounding=ROUND_DOWN)

    min_volume_dec = Decimal(str(min_volume)) if min_volume else Decimal("0")
    if min_volume and size_dec < min_volume_dec:
        logger.warning(
            "Live order volume below min | pair=%s side=%s size=%.10f min=%.10f",
            pair,
            side,
            float(size_dec),
            min_volume,
        )
        return False

    attempted_cost_dec = price_dec * size_dec
    effective_cost_threshold = max(metadata_cost_threshold, config_cost_threshold)
    if effective_cost_threshold:
        min_cost_dec = Decimal(str(effective_cost_threshold))
    else:
        min_cost_dec = Decimal("0")
    if effective_cost_threshold and attempted_cost_dec < min_cost_dec:
        logger.warning(
            "Live order cost below min | pair=%s side=%s cost=%.10f min=%.10f",
            pair,
            side,
            float(attempted_cost_dec),
            effective_cost_threshold,
        )
        return False

    price = float(price_dec)
    size = float(size_dec)
    attempted_cost = float(attempted_cost_dec)

    try:
        response = _kraken_place_order_retry(
            pair,
            side,
            size,
            price,
            time_in_force=tif,
            validate=validate_flag,
            min_cost_threshold=effective_cost_threshold,
        )
    except (KrakenAuthError, KrakenAPIError) as exc:
        pause_until = _monotonic_now() + _KRAKEN_FAILURE_PAUSE_SECONDS
        _set_kraken_failure_pause(pause_until)
        logger.error(
            "Kraken order error; pause %.0fs | pair=%s side=%s err=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
            exc,
        )
        send_alert(
            "[kraken] Order submission error (auth/api)",
            level="ERROR",
            context={"pair": pair, "side": side, "strategy": strategy, "error": str(exc)},
        )
        return False
    except Exception:  # pylint: disable=broad-exception-caught
        pause_until = _monotonic_now() + _KRAKEN_FAILURE_PAUSE_SECONDS
        _set_kraken_failure_pause(pause_until)
        logger.exception(
            "Unexpected Kraken order issue; pause %.0fs | pair=%s side=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
        )
        send_alert(
            "[kraken] Unexpected exception during order placement",
            level="CRITICAL",
            context={"pair": pair, "side": side, "strategy": strategy},
        )
        return False

    if not isinstance(response, dict):
        pause_until = _monotonic_now() + _KRAKEN_FAILURE_PAUSE_SECONDS
        _set_kraken_failure_pause(pause_until)
        logger.error(
            "Kraken order rejected; pause %.0fs | pair=%s side=%s err=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
            response,
        )
        send_alert(
            "[kraken] Non-dict response from order placement",
            level="ERROR",
            context={"pair": pair, "side": side, "strategy": strategy, "response": str(response)},
        )
        return False

    success_response = None
    if response.get("ok", False):
        success_response = response
    else:
        response_code = response.get("code")
        response_error = response.get("error")
        response_cost = response.get("attempted_cost", attempted_cost)
        response_threshold = response.get("threshold", effective_cost_threshold)
        if response_code == "cost_minimum_not_met":
            _clear_kraken_failure_pause()
            logger.warning(
                "Kraken cost min not met | pair=%s side=%s cost=%.6f min=%.6f err=%s",
                pair,
                side,
                float(response_cost or 0.0),
                float(response_threshold or effective_cost_threshold),
                response_error,
            )
            return False

        if response_code == "volume_minimum_not_met":
            _clear_kraken_failure_pause()
            logger.warning(
                "Kraken volume min not met | pair=%s side=%s size=%.6f min=%.6f err=%s",
                pair,
                side,
                size,
                float(pair_meta.get("ordermin", min_volume)),
                response_error,
            )
            send_alert(
                "[kraken] Volume minimum not met",
                level="WARNING",
                context={
                    "pair": pair,
                    "side": side,
                    "size": size,
                    "min": float(pair_meta.get("ordermin", min_volume)),
                    "error": response_error,
                },
            )
            return False

        if response_code == "rate_limit":
            pause_until = _monotonic_now() + 90.0
            _set_kraken_failure_pause(pause_until)
            _invalidate_pair_cache()
            logger.error(
                "Kraken rate limit encountered; pausing live trading for 90s | pair=%s side=%s",
                pair,
                side,
            )
            send_alert(
                "[kraken] Rate limit encountered",
                level="CRITICAL",
                context={"pair": pair, "side": side, "strategy": strategy},
            )
            return False

        pause_until = _monotonic_now() + _KRAKEN_FAILURE_PAUSE_SECONDS
        _set_kraken_failure_pause(pause_until)
        logger.error(
            "Kraken order rejected; pause %.0fs | pair=%s side=%s err=%s code=%s",
            _KRAKEN_FAILURE_PAUSE_SECONDS,
            pair,
            side,
            response_error,
            response_code,
        )
        send_alert(
            "[kraken] Order rejected",
            level="ERROR",
            context={
                "pair": pair,
                "side": side,
                "strategy": strategy,
                "code": response_code,
                "error": response_error,
            },
        )
        return False

    response = success_response

    _clear_kraken_failure_pause()

    # Kraken returns txid values as a list; unwrap singletons for convenience while retaining the list.
    txid_list = response.get("txid")
    if not isinstance(txid_list, list):
        alt_list = response.get("txid_list")
        txid_list = alt_list if isinstance(alt_list, list) else []
    if txid_list and len(txid_list) == 1:
        response["txid_list"] = txid_list
        response["txid_single"] = txid_list[0]
        response["txid"] = txid_list[0]
        txid_repr = txid_list[0]
    else:
        response["txid_list"] = txid_list
        if txid_list:
            response["txid_single"] = txid_list[0]
            response["txid"] = txid_list
        txid_repr = response.get("txid_single") if len(txid_list) <= 1 else txid_list
    descr = response.get("descr")

    logger.info(
        "Kraken live order acknowledged | pair=%s side=%s size=%.6f price=%s txid=%s descr=%s",
        pair,
        side,
        size,
        f"{price:.4f}" if isinstance(price, (int, float)) else price,
        txid_repr,
        descr,
    )
    # Downstream callers treat response["txid"] as a convenience string for single fills, while
    # response["txid_list"] always preserves the original list returned by Kraken.
    return response


def _log_capital(amount: float, source: str) -> None:
    """Log the deployable capital when it changes for observability."""

    last_amount, last_source = _STATE.last_capital_log
    if last_amount == amount and last_source == source:
        return

    logger.info(
        "Active capital (%s): %s (source=%s)",
        get_mode_label(),
        f"${amount:,.2f}",
        source,
    )
    _STATE.last_capital_log = (amount, source)


# NOTE: Clear any stale kill-switch flag once at startup so live flow is not permanently blocked.
def _ensure_kill_switch_cleared() -> None:
    """Clean up a stale kill-switch file so the engine can resume trading."""

    if _STATE.kill_switch_auto_cleared:
        return
    if not os.path.exists(KILL_SWITCH_FILE):
        _STATE.kill_switch_auto_cleared = True
        return
    try:
        os.remove(KILL_SWITCH_FILE)
        message = "[EMERGENCY STOP] Kill-switch file " f"{KILL_SWITCH_FILE} detected at startup — auto-cleared."
        logger.warning(message)
        send_alert(message, level="WARNING")
        _STATE.kill_switch_auto_cleared = True
        _STATE.emergency_stop_triggered = False
    except OSError as exc:
        logger.error("Failed to auto-clear kill-switch file %s: %s", KILL_SWITCH_FILE, exc)
        if not os.path.exists(KILL_SWITCH_FILE):
            _STATE.kill_switch_auto_cleared = True


# NOTE: Block new trades while disk space is critically low.
# Resume automatically once capacity returns.
def _guard_low_disk_space() -> bool:
    """Return True if trades should be paused due to low disk space."""

    try:
        _, _, free = shutil.disk_usage(_DISK_GUARD_PATH)
    except OSError as exc:
        logger.error("Disk usage check failed for %s: %s", _DISK_GUARD_PATH, exc)
        return False

    free_mb = free / (1024 * 1024)
    if free_mb < _DISK_GUARD_THRESHOLD_MB:
        if not _STATE.disk_space_block_active:
            message = (
                "Disk space critically low ("
                f"{free_mb:.1f} MB free < {_DISK_GUARD_THRESHOLD_MB:.1f} MB) — "
                "disabling new trade submissions."
            )
            logger.critical(message)
            send_alert(
                message,
                context={"free_mb": round(free_mb, 2), "threshold_mb": _DISK_GUARD_THRESHOLD_MB},
                level="CRITICAL",
            )
            logger.critical(message)
        _STATE.disk_space_block_active = True
        return True

    if _STATE.disk_space_block_active:
        logger.info("Disk space recovered (%.1f MB free); resuming trade evaluation.", free_mb)
        _STATE.disk_space_block_active = False
    return False


def _fetch_recent_volume(pair: str, *, candles: int = 5, fallback: float = 0.0) -> float | None:
    """Return an approximate recent volume for ``pair`` using Kraken OHLC data."""

    now = time.monotonic()
    cached = _VOLUME_CACHE.get(pair)
    if cached and now - cached[0] <= _VOLUME_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        rows = get_ohlc_data(pair, interval=1, limit=max(1, candles))
    # pylint: disable=broad-exception-caught
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Volume fetch failed for %s: %s", pair, exc)
        if TEST_MODE:
            return fallback
        return None

    volumes = [float(row.get("volume", 0.0)) for row in rows if row.get("volume")]
    if not volumes:
        if TEST_MODE:
            return fallback
        logger.warning("No volume data returned for %s", pair)
        return None

    avg_volume = sum(volumes[-candles:]) / min(len(volumes), candles)
    _VOLUME_CACHE[pair] = (now, avg_volume)
    return avg_volume


# Configurable minimum volume threshold (testing lower assets like XRP/LINK/SOL)
MIN_VOLUME = CONFIG.get("min_volume", 100)


def _load_history_series(
    pair: str,
    *,
    window: int,
    cache: dict[str, list[float]],
) -> list[float]:
    """Return cached price history for ``pair`` with a minimum window length."""

    series = cache.get(pair)
    if series is None:
        try:
            raw_history = get_history_prices(pair, min_len=window)
            series = [p for p in raw_history if p is not None]
        except HistoryUnavailable:
            series = []
        cache[pair] = series
    return series


def _compute_pair_correlation(
    pair_a: str,
    pair_b: str,
    *,
    window: int,
    cache: dict[str, list[float]],
) -> float | None:
    """Return Pearson correlation for the trailing ``window`` prices of two pairs."""

    if np is None:
        raise ImportError("numpy not available")

    series_a = _load_history_series(pair_a, window=window, cache=cache)
    series_b = _load_history_series(pair_b, window=window, cache=cache)
    if len(series_a) < window or len(series_b) < window:
        return None

    compare_len = min(len(series_a), len(series_b), window)
    if compare_len < 2:
        return None

    segment_a = series_a[-compare_len:]
    segment_b = series_b[-compare_len:]
    corr_matrix = np.corrcoef(segment_a, segment_b)
    value = float(corr_matrix[0, 1])
    if math.isnan(value):
        return None
    return value


def _evaluate_correlation_blocks(
    asset: str,
    proposed_trades: list[dict[str, Any]],
    *,
    cache: dict[str, list[float]],
    window: int,
    threshold: float,
) -> tuple[bool, list[dict[str, float]]]:
    """Return (should_skip, diagnostics) for correlation control."""

    skip_due_to_corr = False
    corr_rows: list[dict[str, float]] = []
    for trade in proposed_trades:
        other = trade.get("asset")
        if other is None or other == asset:
            continue
        pair_a = ensure_usdc_pair(f"{asset}/USDC")
        pair_b = ensure_usdc_pair(f"{other}/USDC")
        corr_value = _compute_pair_correlation(
            pair_a,
            pair_b,
            window=window,
            cache=cache,
        )
        if corr_value is None:
            continue
        corr_rows.append({"pair": f"{asset}-{other}", "other": other, "corr": corr_value})
        if corr_value >= threshold:
            skip_due_to_corr = True
            break
    return skip_due_to_corr, corr_rows


class PositionManager:
    """Manages trade positions including open, exit, and persistence."""

    def __init__(self):
        """Initializes the PositionManager with an empty positions dictionary."""
        self.positions = {}

    def open_position(
        self,
        trade_id,
        pair,
        size,
        entry_price,
        strategy,
        confidence,
        entry_adx: float | None = None,
        entry_rsi: float | None = None,
        timestamp: str | None = None,
    ):
        """Opens a new position and writes it to the positions log.
        Expects entry_price to be the final effective entry (e.g., after slippage)
        so it is persisted exactly as used for the trade.
        """
        ts = timestamp or datetime.datetime.now(datetime.UTC).isoformat()
        self.positions[trade_id] = {
            "trade_id": trade_id,
            "pair": pair,
            "size": size,
            "entry_price": entry_price,
            "timestamp": ts,
            "strategy": strategy,
            "confidence": confidence,
            "high_water_mark": entry_price,
            "entry_adx": entry_adx,
            "entry_rsi": entry_rsi,
        }
        try:
            os.makedirs("logs", exist_ok=True)
            with _locked_file("logs/positions.jsonl", "a") as f:
                f.write(json.dumps(self.positions[trade_id]) + "\n")
            logger.debug("Position persisted", extra={"trade_id": trade_id, "pair": pair, "size": size})
        except (OSError, IOError) as e:
            logger.error("Failed writing positions.jsonl", extra={"error": str(e)})

    def check_exits(
        self,
        current_prices,
        tp=0.002,
        sl=0.0015,
        trailing_stop=0.01,
        max_hold_bars=14,
    ):
        """Checks each open position for exit criteria like SL, TP, or max hold."""
        exits = []
        current_time = datetime.datetime.now(datetime.UTC)
        keys_to_delete = []
        history_cache: dict[str, list[float]] = {}
        for trade_id, pos in self.positions.items():
            # Current price (ledger applies exit slippage at write time)
            price = current_prices.get(pos["pair"], pos["entry_price"])
            if price <= 0:
                continue
            if "high_water_mark" not in pos:
                pos["high_water_mark"] = pos["entry_price"]
            ret = (price - pos["entry_price"]) / pos["entry_price"]
            pos["high_water_mark"] = max(pos["high_water_mark"], price)
            trailing_threshold = pos["high_water_mark"] * (1 - trailing_stop)
            bars_held = (
                current_time - datetime.datetime.fromisoformat(pos["timestamp"])
            ).total_seconds() // TRADE_INTERVAL

            # RSI-based exit check before other exits
            try:
                if calculate_rsi is not None:
                    pair = pos["pair"]
                    rsi_period = int(CONFIG.get("rsi", {}).get("period", 14))
                    history = history_cache.get(pair)
                    if history is None:
                        try:
                            raw_history = get_history_prices(pair, min_len=rsi_period + 1)
                            history = [p for p in raw_history if p is not None]
                            history_cache[pair] = history
                        except HistoryUnavailable:
                            history = []
                    if price and (not history or history[-1] != price):
                        history = ((history or []) + [price])[-(rsi_period + 5) :]
                        history_cache[pair] = history
                    if history and len(history) >= rsi_period + 1:
                        lookback = history[-(rsi_period + 1) :]
                        rsi_val = calculate_rsi(lookback, rsi_period)  # type: ignore[misc]
                        rsi_cfg = CONFIG.get("rsi", {})
                        exit_upper = rsi_cfg.get("exit_upper", rsi_cfg.get("upper", 70))
                        if rsi_val is not None and float(rsi_val) >= float(exit_upper):
                            exit_price = price
                            reason = "RSI_EXIT"
                            logger.info(
                                "RSI exit triggered",
                                extra={"trade_id": trade_id, "pair": pair, "rsi": float(rsi_val)},
                            )
                            exits.append((trade_id, exit_price, reason))
                            keys_to_delete.append(trade_id)
                            continue
                else:
                    logger.warning(
                        "RSI calculator unavailable; skipping RSI exit check",
                        extra={"trade_id": trade_id, "pair": pos["pair"]},
                    )
            except (HistoryUnavailable, KeyError, ValueError, TypeError, IndexError) as e:
                logger.error(
                    "RSI exit check encountered error",
                    extra={"trade_id": trade_id, "pair": pos["pair"], "error": str(e)},
                )

            if ret <= -sl:
                exit_price = price
                reason = "STOP_LOSS"
            elif price <= trailing_threshold:
                exit_price = price
                reason = "TRAILING_STOP"
            elif bars_held >= max_hold_bars:
                exit_price = price
                reason = "MAX_HOLD"
            elif ret >= tp:
                exit_price = price
                reason = "TAKE_PROFIT"
            else:
                continue

            exits.append((trade_id, exit_price, reason))
            keys_to_delete.append(trade_id)

        for trade_id in keys_to_delete:
            del self.positions[trade_id]
        return exits

    def load_positions_from_file(self, file_path="logs/positions.jsonl"):
        """Loads existing positions from the JSONL file into memory."""
        self.positions = {}
        if os.path.exists(file_path):
            try:
                with _locked_file(file_path, "r") as f:
                    seen_trade_ids = set()
                    for line in f:
                        try:
                            pos = json.loads(line.strip())
                            trade_id = pos["trade_id"]
                            if trade_id not in seen_trade_ids:
                                seen_trade_ids.add(trade_id)
                                if "high_water_mark" not in pos:
                                    pos["high_water_mark"] = pos["entry_price"]
                                pos["timestamp"] = pos.get("timestamp")
                                self.positions[trade_id] = pos
                                logger.debug(
                                    "Loaded position from disk",
                                    extra={"trade_id": trade_id, "pair": pos.get("pair")},
                                )
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse position entry from positions.jsonl")
            except (OSError, IOError) as e:
                logger.error("Error reading positions.jsonl", extra={"error": str(e)})
        else:
            logger.info("Positions file not found; starting with empty state")


position_manager = PositionManager()
ledger = TradeLedger(position_manager)


def calculate_total_risk(trades):
    """Calculates the sum of the risk from all proposed trades."""
    return sum(trade.get("risk", 0.0) for trade in trades)


def _committed_notional(trades) -> float:
    """Total notional already allocated across pending trades."""
    total = 0.0
    for trade in trades or []:
        try:
            total += max(float(trade.get("notional", 0.0)), 0.0)
        except (TypeError, ValueError):
            continue
    return total


def _compute_position_sizing(
    *,
    total_capital: float,
    remaining_capital: float,
    current_price: float,
    confidence: float,
    base_risk_pct: float,
    buffer: float,
    reinvestment_rate: float | None,
    liquidity_factor: float,
    min_size: float,
    max_size: float,
) -> tuple[float, float, float]:
    """Return (size, notional, risk_fraction) for the proposed trade."""
    invalid_inputs = any(
        value <= 0
        for value in (
            total_capital,
            remaining_capital,
            current_price,
            confidence,
            base_risk_pct,
        )
    )
    if invalid_inputs:
        return 0.0, 0.0, 0.0

    reinvestment_factor = float(reinvestment_rate) if reinvestment_rate is not None else 1.0
    reinvestment_factor = min(max(reinvestment_factor, 0.0), 1.0)
    liquidity_factor = min(max(liquidity_factor, 0.0), 1.0)
    buffer = max(buffer, 0.0)

    effective_pct = base_risk_pct * buffer * confidence * liquidity_factor * reinvestment_factor
    effective_pct = min(effective_pct, base_risk_pct)

    notional_target = remaining_capital * effective_pct
    if notional_target <= 0:
        return 0.0, 0.0, 0.0

    raw_units = notional_target / current_price
    if raw_units <= 0:
        return 0.0, 0.0, 0.0

    # Respect maximum size; if size falls below minimum, skip this trade
    max_size = max(max_size, min_size)
    units = min(raw_units, max_size)
    min_units = float(min_size)
    if units < min_units:
        min_notional = min_units * current_price
        if min_notional <= remaining_capital:
            logger.debug("Adjusting size to meet minimum | target=%.6f min=%.6f", units, min_units)
            units = min_units
        else:
            logger.warning(
                "Insufficient capital for min size | required=%.4f available=%.4f",
                min_notional,
                remaining_capital,
            )
            return 0.0, 0.0, 0.0

    units = round(units, 6)
    notional = units * current_price
    risk_fraction = notional / total_capital if total_capital > 0 else 0.0
    risk_fraction = min(risk_fraction, base_risk_pct)

    return units, notional, risk_fraction


def save_portfolio_state(ctx):
    """Saves the current trading context to portfolio_state.json."""
    os.makedirs("logs", exist_ok=True)
    with open(PORTFOLIO_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(ctx.get_snapshot(), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    logger.info("Portfolio state saved", extra={"path": str(PORTFOLIO_STATE_PATH)})


def evaluate_signals_and_trade(
    check_exits_only: bool = False,
    tradable_pairs: list[str] | None = None,
    *,
    available_capital: float | None = None,
    risk_per_trade: float | None = None,
    reinvestment_rate: float | None = None,
):
    """Evaluates trade signals and manages trade execution and exits."""
    # REFACTOR-HOOKS: harmless calls while we peel logic out
    try:
        _signals = gather_signals(prices=None, volumes=None, ctx=None)  # type: ignore[name-defined]
        _ok = risk_screen(_signals, ctx=None)  # type: ignore[name-defined]
        _ = (
            _signals,
            _ok,
            execute_trade(_signals, ctx=None),  # type: ignore[name-defined]
            check_and_close_exits(ctx=None),  # type: ignore[name-defined]
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("evaluate_signals_and_trade helper error", extra={"error": str(e)})

    _ensure_startup_snapshot()

    executed_trades = 0  # ensure initialized for check_exits_only
    position_manager.load_positions_from_file()
    _ensure_kill_switch_cleared()
    disk_block = _guard_low_disk_space()
    if disk_block and not check_exits_only:
        check_exits_only = True
    pause_new_trades, pause_reason = ledger.consume_pause_request()
    if pause_new_trades:
        pause_msg = pause_reason or "[PAUSE] skipping new trades this cycle (ledger check)."
        logger.warning(pause_msg)
        if not check_exits_only:
            check_exits_only = True
    if CONFIG.get("live_mode", {}).get("dry_run") and not _STATE.dry_run_logged:
        msg = "[DRY-RUN] Validate-only trades active; live orders use validate=True."
        logger.warning(msg)
        _STATE.dry_run_logged = True

    if os.path.exists(KILL_SWITCH_FILE):
        if not _STATE.emergency_stop_triggered:
            alert_message = f"EMERGENCY STOP enabled via kill-switch file ({KILL_SWITCH_FILE})."
            send_alert(alert_message, level="CRITICAL")
            logger.critical(alert_message)
            _STATE.emergency_stop_triggered = True
        if is_live:
            set_live_mode(False)
        check_exits_only = True

    paused, pause_reason = risk_guard.check_pause()
    if paused:
        if _STATE.auto_paused_reason != pause_reason:
            message = f"Kill-switch: risk guard paused. {pause_reason or 'No reason provided'}"
            logger.critical(message)
            send_alert(
                message,
                level="CRITICAL",
                context={"phase": DEPLOY_PHASE, "reason": pause_reason},
            )
        _STATE.auto_paused_reason = pause_reason
        check_exits_only = True
    else:
        if _STATE.auto_paused_reason:
            _STATE.auto_paused_reason = None

    # Resolve the list of tradable pairs centrally (config-driven)
    # Added to ensure the engine scans all requested assets with no hardcoding.
    pairs: list[str] = tradable_pairs or CONFIG.get("tradable_pairs", [])
    if not pairs:
        logger.warning("No tradable_pairs configured; skipping evaluation.")
        return

    mode_label = get_mode_label()
    if mode_label != _STATE.last_mode_label:
        logger.info("Trading mode: %s (is_live=%s)", mode_label, is_live)
        _STATE.last_mode_label = mode_label

    state_snapshot: dict | None = None
    resolved_capital: float | None = None
    capital_source = "manual_override"
    if available_capital is not None:
        try:
            resolved_capital = float(available_capital)
        except (TypeError, ValueError):
            resolved_capital = None

    if resolved_capital is None or resolved_capital <= 0:
        state_snapshot = load_portfolio_state(
            refresh=is_live,
            starting_balance=PAPER_STARTING_BALANCE,
        )
        resolved_capital = float(state_snapshot.get("available_capital", PAPER_STARTING_BALANCE))
        capital_source = state_snapshot.get("capital_source", "portfolio_state")

    if state_snapshot is None:
        state_snapshot = load_portfolio_state(
            refresh=is_live,
            starting_balance=PAPER_STARTING_BALANCE,
        )
        capital_source = state_snapshot.get("capital_source", capital_source)
        if resolved_capital is None or resolved_capital <= 0:
            resolved_capital = float(state_snapshot.get("available_capital", PAPER_STARTING_BALANCE))

    resolved_capital = max(resolved_capital or 0.0, 0.0)
    _log_capital(resolved_capital, capital_source)

    if not check_exits_only:
        paused, reason = _evaluate_auto_pause(state_snapshot)
        if paused:
            if _STATE.auto_paused_reason != reason:
                logger.error("[AUTO-PAUSE] %s", reason)
            else:
                logger.debug("Auto-pause active: %s", reason)
            _STATE.auto_paused_reason = reason
            return
        if _STATE.auto_paused_reason is not None:
            logger.info("Auto-pause cleared; resuming trade evaluation.")
            _STATE.auto_paused_reason = None

    drawdown = ledger.get_current_drawdown()
    if drawdown >= DRAWDOWN_GUARD_LIMIT:
        if not _STATE.drawdown_block_active:
            logger.error(
                "Drawdown guard active %.2f%% ≥ limit %.2f%%; halting trade evaluation.",
                drawdown * 100,
                DRAWDOWN_GUARD_LIMIT * 100,
            )
            send_alert(
                "[risk] Drawdown guard triggered",
                level="CRITICAL",
                context={"drawdown_pct": round(drawdown, 6), "limit_pct": round(DRAWDOWN_GUARD_LIMIT, 6)},
            )
        _STATE.drawdown_block_active = True
        ledger.request_pause_new_trades("drawdown_guard_active")
        return
    if _STATE.drawdown_block_active:
        logger.info("Drawdown guard cleared; resuming trade evaluation.")
        _STATE.drawdown_block_active = False

    if reinvestment_rate is None:
        reinvestment_rate = float(state_snapshot.get("reinvestment_rate", 0.0))

    trade_risk_pct = 0.02 if risk_per_trade is None else max(float(risk_per_trade), 0.0)
    # Build current price map using live feed for each pair
    current_prices: dict[str, float] = {}
    current_volumes: dict[str, float] = {}
    historical_price_cache: dict[str, list[float]] = {}
    price_times: dict[str, datetime.datetime] = {}
    missing_pairs: set[str] = set()
    cache_map = getattr(price_feed_module, "_cache", {})
    for pair in pairs:
        price_now = get_current_price(pair)
        asset = pair.split("/")[0]
        if price_now is not None and price_now > 0:
            current_prices[pair] = price_now
            logger.debug("Price feed update", extra={"asset": asset, "pair": pair, "price": price_now})
            # Access the internal cache timestamp to detect stale prices safely.
            cache_entry = None
            if isinstance(cache_map, dict):
                cache_entry = cache_map.get(pair.upper())
            if cache_entry and isinstance(cache_entry, tuple) and cache_entry:
                cache_epoch = cache_entry[0]
                try:
                    cache_dt = datetime.datetime.fromtimestamp(float(cache_epoch), datetime.UTC)
                except (TypeError, ValueError, OSError):
                    cache_dt = datetime.datetime.now(datetime.UTC)
            else:
                cache_dt = datetime.datetime.now(datetime.UTC)
            price_times[pair] = cache_dt
            volume_estimate = _fetch_recent_volume(pair, fallback=float(MIN_VOLUME))
            if volume_estimate is not None and volume_estimate > 0:
                current_volumes[pair] = volume_estimate
            else:
                logger.warning("No volume data for %s; skipping volume map entry", pair)
        else:
            missing_pairs.add(pair)
            logger.warning("No current price available; skipping price map entry", extra={"pair": pair})

    now_ts = datetime.datetime.now(datetime.UTC)
    stale_pairs = [pair for pair, ts in price_times.items() if now_ts - ts > PRICE_STALE_THRESHOLD]
    if stale_pairs or missing_pairs:
        affected = sorted(set(stale_pairs).union(missing_pairs))
        logger.warning("Price data stale or missing for %s; halting trade evaluation.", affected)
        ledger.request_pause_new_trades("stale_price_data")
        return

    # Preload seeded history for all pairs (startup fallback)
    try:
        preload_period = CONFIG.get("rsi", {}).get("period", 21)
        preload_min = max(int(preload_period) + 1, 14)
        for p in pairs:
            _ = get_history_prices(p, min_len=preload_min)
    except Exception:  # pylint: disable=broad-exception-caught
        # Non-fatal: trading loop can still proceed using live prices only
        pass

    # Refresh current market regime and capital buffer before signal evaluation
    context.update_context()
    base_buffer = context.get_buffer()
    composite_buffer = context.get_buffer_for_strategy("CompositeStrategy")
    logger.info(
        "Context refreshed",
        extra={
            "regime": context.get_regime(),
            "buffer": base_buffer,
            "composite_buffer": composite_buffer,
        },
    )
    save_portfolio_state(context)

    # Daily trade limit (configurable; ENV override allowed)
    max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "0") or 0)

    def _today_trade_count() -> int:
        try:
            if not os.path.exists(TRADES_LOG_PATH):
                return 0
            today = datetime.datetime.now(datetime.UTC).date().isoformat()
            count = 0
            with _locked_file(TRADES_LOG_PATH, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        ts = rec.get("timestamp")
                        if ts and ts[:10] == today:
                            if (rec.get("status") or "").lower() in {"executed", "closed"}:
                                count += 1
                    except json.JSONDecodeError:
                        continue
            return count
        except OSError:
            return 0

    def _last_closed_trades(n: int) -> list[dict]:
        """Return the most recent n closed trades with valid ROI.

        Parses `logs/trades.log` and sorts by timestamp (UTC).
        """
        items: list[dict] = []
        try:
            if not os.path.exists(TRADES_LOG_PATH):
                return items
            with _locked_file(TRADES_LOG_PATH, "r") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (rec.get("status") or "").lower() != "closed":
                        continue
                    if rec.get("roi") is None:
                        continue
                    ts = rec.get("timestamp") or "1970-01-01T00:00:00+00:00"
                    rec["_ts"] = ts
                    items.append(rec)
            # Sort by timestamp string (ISO 8601 lexicographic works for UTC)
            items.sort(key=lambda r: r.get("_ts", ""))
            return items[-n:]
        except OSError:
            return items[:n]

    # ---- Daily limits with streak rules ----
    default_daily_cap = int(CONFIG.get("MAX_TRADES_PER_DAY", 5))
    bonus_cap = int(CONFIG.get("BONUS_LIMIT_IF_WINNING_STREAK", 7))
    loss_streak_stop = int(CONFIG.get("STOP_IF_LOSS_STREAK", 3))

    # Start from ENV override if provided, else config default
    current_daily_cap = max_trades_per_day if max_trades_per_day > 0 else default_daily_cap

    today_count = _today_trade_count()
    recent = _last_closed_trades(5)

    # Loss streak: last loss_streak_stop closed trades all losers
    if len(recent) >= loss_streak_stop:
        last_n = recent[-loss_streak_stop:]
        if all((t.get("roi") or 0) < 0 for t in last_n):
            logger.warning("Loss streak detected — trading paused for the remainder of the day")
            return

    # Winning streak: last 5 closed trades all winners
    if len(recent) >= 5 and all((t.get("roi") or 0) > 0 for t in recent[-5:]):
        if current_daily_cap < bonus_cap:
            current_daily_cap = bonus_cap
            logger.info("Winning streak detected — daily trade cap increased", extra={"cap": current_daily_cap})

    # Enforce daily cap before scanning assets
    if today_count >= current_daily_cap:
        logger.info("Daily trade limit reached", extra={"count": today_count, "limit": current_daily_cap})
        return

    total_capital = resolved_capital
    if not check_exits_only and total_capital <= 0:
        logger.warning("Available capital is zero — skipping new trade evaluation")
        check_exits_only = True

    if not check_exits_only:
        proposed_trades = []
        executed_trades = 0
        logger.info("Scanning assets", extra={"pairs": pairs, "count": len(pairs)})
        for pair in pairs:
            try:
                # Derive asset symbol from pair (e.g., "BTC" from "BTC/USDC")
                asset = pair.split("/")[0]
                # Fetch current price
                current_price = current_prices.get(pair)
                if current_price is None:
                    current_price = get_current_price(pair)
                if current_price is None or current_price <= 0:
                    logger.warning("Skipping asset due to invalid price", extra={"pair": pair})
                    continue

                # Ensure seeded history is available, then append live price
                rsi_period = CONFIG.get("rsi", {}).get("period", 21)
                # Ensure we always provide at least 30 points to RSI-based strategies
                min_needed = max(int(rsi_period) + 1, 30)
                try:
                    _ = get_history_prices(pair, min_len=min_needed)
                except HistoryUnavailable as exc:
                    logger.error("Skipping %s: %s", pair, exc)
                    continue

                append_live_price(pair, float(current_price))

                try:
                    safe_prices = get_history_prices(pair, min_len=min_needed)
                except HistoryUnavailable as exc:
                    logger.error("Skipping %s after live price append: %s", pair, exc)
                    continue
                # Filter out None values before strategy evaluation
                safe_prices = [p for p in safe_prices if p is not None]
                historical_price_cache[pair] = safe_prices
                # Pre-pair debug trace
                logger.info(
                    "Generating signal",
                    extra={
                        "pair": pair,
                        "asset": asset,
                        "valid_candles": len(safe_prices),
                        "latest_close": safe_prices[-1] if safe_prices else None,
                    },
                )
                last5 = safe_prices[-5:]
                logger.debug("Recent prices", extra={"pair": pair, "last5": last5})

                # Pre-compute RSI for diagnostics/logging
                rsi_val = None
                try:
                    if calculate_rsi is not None and len(safe_prices) >= int(rsi_period) + 1:
                        rsi_val = float(calculate_rsi(safe_prices, int(rsi_period)))
                except (TypeError, ValueError):
                    rsi_val = None
                # Skip if insufficient history
                if len(safe_prices) < min_needed:
                    logger.warning(
                        "Skipping due to insufficient price history",
                        extra={"pair": pair, "available": len(safe_prices), "required": min_needed},
                    )
                    continue
                # Compute ADX gate using recent prices
                adx_val = context.get_adx(pair, safe_prices)
                if adx_val is not None:
                    logger.debug("ADX computed", extra={"pair": pair, "adx": adx_val})
                volume = current_volumes.get(pair)
                if volume is None:
                    volume = _fetch_recent_volume(pair, fallback=float(MIN_VOLUME))
                    if volume is not None and volume > 0:
                        current_volumes[pair] = volume
                if volume is None or volume <= 0:
                    if TEST_MODE:
                        volume = float(MIN_VOLUME)
                        logger.debug(
                            "Test mode volume fallback applied",
                            extra={"pair": pair, "volume": volume},
                        )
                    else:
                        logger.warning(
                            "Skipping asset due to missing volume",
                            extra={"asset": asset, "pair": pair, "rsi": rsi_val},
                        )
                        continue
                if volume < MIN_VOLUME:
                    logger.warning(
                        "Skipping asset due to low volume",
                        extra={"asset": asset, "pair": pair, "volume": volume, "min_volume": MIN_VOLUME},
                    )
                    continue
                logger.info(
                    "Signal context inputs",
                    extra={
                        "pair": pair,
                        "asset": asset,
                        "price": current_price,
                        "rsi": rsi_val,
                        "volume": volume,
                        "adx": adx_val,
                    },
                )
                # Reinitialize strategies per iteration to reset state
                per_asset_params = CONFIG.get("strategy_params", {})
                strategies = _get_locked_strategy_pipeline(
                    pair,
                    mode_label,
                    per_asset_params=per_asset_params,
                )
                regime = context.get_regime()
                strategy_candidates: list[dict[str, Any]] = []
                signals_count = 0
                buy_count = 0
                sell_count = 0
                skipped_low_conf = 0
                skipped_none = 0

                for strategy in strategies:
                    try:
                        try:
                            signal_result = strategy.generate_signal(
                                safe_prices,
                                volume=volume,
                                asset=asset,
                                adx=adx_val,
                            )
                        except TypeError:
                            signal_result = strategy.generate_signal(
                                safe_prices,
                                volume=volume,
                            )
                        scan_msg = {
                            "asset": asset,
                            "strategy": strategy.__class__.__name__,
                            "price": round(float(current_price), 4),
                            "volume": volume,
                            "signal": signal_result,
                        }
                        logger.debug("Strategy scan result", extra=scan_msg)
                    except (ValueError, RuntimeError) as e:
                        log_path = "logs/anomalies.log"
                        timestamp = datetime.datetime.now(datetime.UTC).isoformat()
                        with open(log_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "timestamp": timestamp,
                                        "type": "Signal Error",
                                        "error": str(e),
                                        "strategy": strategy.__class__.__name__,
                                    }
                                )
                                + "\n"
                            )
                            try:
                                f.flush()
                                os.fsync(f.fileno())
                            except (OSError, IOError):
                                pass
                        continue

                    logger.debug(
                        "Strategy output",
                        extra={"strategy": strategy.__class__.__name__, "signal": signal_result},
                    )
                    candidate_signal = signal_result.get("signal") or signal_result.get("side")
                    candidate_confidence = float(signal_result.get("confidence", 0.0) or 0.0)
                    strategy_name = strategy.__class__.__name__

                    if candidate_signal in {"buy", "sell"}:
                        logger.info(
                            "Candidate signal selected",
                            extra={"pair": pair, "signal": candidate_signal, "confidence": candidate_confidence},
                        )
                        strategy_candidates.append(
                            {
                                "strategy": strategy_name,
                                "signal": candidate_signal,
                                "confidence": candidate_confidence,
                                "raw": signal_result,
                                "buffer": context.get_buffer_for_strategy(strategy_name),
                                "strategy_obj": strategy,
                            }
                        )
                        signals_count += 1
                        if candidate_signal == "buy":
                            buy_count += 1
                        else:
                            sell_count += 1
                    else:
                        skipped_none += 1
                        logger.debug("No actionable signal produced", extra={"pair": pair})

                actionable_candidates = [
                    candidate for candidate in strategy_candidates if candidate["signal"] in {"buy", "sell"}
                ]
                if not actionable_candidates:
                    logger.info(
                        "No actionable signals",
                        extra={
                            "pair": pair,
                            "asset": asset,
                            "generated_signals": signals_count,
                            "skipped_none": skipped_none,
                            "skipped_low_conf": skipped_low_conf,
                        },
                    )
                    continue

                selected = max(actionable_candidates, key=lambda c: c["confidence"])
                signal = selected["signal"]
                confidence = selected["confidence"]
                strategy_name = selected["strategy"]
                selected_strategy = selected["strategy_obj"]
                buffer = selected["buffer"]

                # ADX gating (regime filter)
                if adx_val is not None:
                    if adx_val < 20.0:
                        skipped_none += 1
                        logger.info(
                            "ADX below threshold; skipping candidate",
                            extra={"pair": pair, "adx": adx_val},
                        )
                        continue
                    if adx_val > 40.0:
                        before = confidence
                        confidence = min(1.0, confidence * 1.2)
                        logger.info(
                            "ADX strong trend adjustment",
                            extra={"pair": pair, "confidence_before": before, "confidence_after": confidence},
                        )
                    logger.info(
                        "ADX summary",
                        extra={"pair": pair, "adx": adx_val, "confidence": confidence},
                    )

                if confidence < 0.4:
                    skipped_low_conf += 1
                    logger.info(
                        "Skipping candidate due to low confidence",
                        extra={"pair": pair, "confidence": confidence},
                    )
                    continue
                if volume is None or volume < MIN_VOLUME:
                    logger.info("Skipping candidate due to low volume", extra={"pair": pair, "volume": volume})
                    continue
                logger.info(
                    "Candidate volume snapshot",
                    extra={"pair": pair, "volume": volume, "confidence": confidence},
                )
                signals_count += 1
                if signal == "buy":
                    buy_count += 1
                elif signal == "sell":
                    sell_count += 1

                committed_capital = _committed_notional(proposed_trades)
                remaining_capital = max(total_capital - committed_capital, 0.0)
                if remaining_capital <= 0:
                    logger.warning("Capital exhausted for new positions; stopping scans")
                    break

                min_sz, max_sz = _resolve_trade_size_bounds(pair)
                dynamic_buffer = context.get_buffer_for_strategy(strategy_name)
                liquidity_factor = min(volume / 1000, 1.0)

                adjusted_size, position_notional, trade_risk = _compute_position_sizing(
                    total_capital=total_capital,
                    remaining_capital=remaining_capital,
                    current_price=current_price,
                    confidence=confidence,
                    base_risk_pct=trade_risk_pct,
                    buffer=dynamic_buffer,
                    reinvestment_rate=reinvestment_rate,
                    liquidity_factor=liquidity_factor,
                    min_size=min_sz,
                    max_size=max_sz,
                )

                if adjusted_size <= 0:
                    logger.warning(
                        "Skipping due to insufficient capital for minimum position size",
                        extra={"pair": pair},
                    )
                    continue

                limited_size, limit_context = _apply_deploy_phase_limits(
                    pair,
                    adjusted_size,
                    float(current_price),
                )
                if limited_size <= 0:
                    logger.info(
                        "Trade size reduced to zero by canary limits; skipping",
                        extra={"pair": pair},
                    )
                    continue
                if limited_size != adjusted_size:
                    adjusted_size = limited_size
                    position_notional = adjusted_size * float(current_price)

                trade_data = {
                    "asset": asset,
                    "size": adjusted_size,
                    "risk": trade_risk,
                    "strategy": strategy_name,
                    "confidence": confidence,
                    "signal_score": confidence,
                    "regime": regime,
                    "notional": round(position_notional, 2),
                }
                if limit_context:
                    trade_data["deploy_limit"] = limit_context

                proposed_trades.append(trade_data)
                total_risk = calculate_total_risk(proposed_trades)
                if total_risk > MAX_PORTFOLIO_RISK:
                    msg = f"Skipping trade for {asset} — portfolio risk ({total_risk:.2%}) exceeds cap"
                    logger.warning(msg)
                    proposed_trades.pop()
                    continue

                # Correlation control against already proposed trades
                # Optional numpy correlation check; skip if unavailable
                try:
                    if np is None:
                        raise ImportError("numpy not available")

                    corr_cfg = CONFIG.get("correlation", {})
                    corr_threshold = float(corr_cfg.get("threshold", 0.7))
                    corr_window = max(int(corr_cfg.get("window", 30)), 5)
                    skip_due_to_corr, corr_rows = _evaluate_correlation_blocks(
                        asset,
                        proposed_trades,
                        cache=historical_price_cache,
                        window=corr_window,
                        threshold=corr_threshold,
                    )
                    if corr_rows:
                        os.makedirs("logs", exist_ok=True)
                        with open(
                            "logs/portfolio_metrics.log",
                            "a",
                            encoding="utf-8",
                        ) as f:
                            for row in corr_rows:
                                corr_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
                                f.write(
                                    json.dumps(
                                        {
                                            "timestamp": corr_timestamp,
                                            "type": "correlation",
                                            **row,
                                        }
                                    )
                                    + "\n"
                                )
                            f.flush()
                            os.fsync(f.fileno())
                    if skip_due_to_corr and corr_rows:
                        last_row = corr_rows[-1]
                        os.makedirs("logs", exist_ok=True)
                        with open(
                            "logs/portfolio_metrics.log",
                            "a",
                            encoding="utf-8",
                        ) as f:
                            corr_timestamp = datetime.datetime.now(datetime.UTC).isoformat()
                            f.write(
                                json.dumps(
                                    {
                                        "timestamp": corr_timestamp,
                                        "type": "correlation_block",
                                        "asset": asset,
                                        "other": last_row.get("other"),
                                        "corr": last_row["corr"],
                                    }
                                )
                                + "\n"
                            )
                            f.flush()
                            os.fsync(f.fileno())
                    if skip_due_to_corr:
                        logger.info(
                            "Skipping asset due to correlation threshold",
                            extra={"asset": asset, "last_corr": last_row.get("corr") if corr_rows else None},
                        )
                        proposed_trades.pop()
                        continue
                except (ImportError, ValueError, TypeError, KeyError, IndexError) as e:
                    logger.error("Correlation check error", extra={"asset": asset, "error": str(e)})

                logger.info("Proposed trade details", extra=trade_data)

                trade_side = signal
                start_latency = time.perf_counter()
                order_result = _submit_live_trade(
                    pair=pair,
                    side=trade_side,
                    size=adjusted_size,
                    price=float(current_price),
                    strategy=strategy_name,
                    confidence=confidence,
                )
                submitted_live = bool(order_result)
                live_latency = time.perf_counter() - start_latency if is_live else None

                if is_live and not submitted_live:
                    logger.warning("Live trade suppressed; running in paper mode", extra={"pair": pair})
                    continue

                if not is_live:
                    logger.debug(
                        "Paper trade | pair=%s side=%s size=%.6f price=%.4f strategy=%s conf=%.3f",
                        pair,
                        trade_side,
                        adjusted_size,
                        float(current_price),
                        strategy_name,
                        confidence,
                    )

                # Daily trade limit check (re-evaluate per attempt)
                if current_daily_cap > 0:
                    today_count = _today_trade_count()
                    logger.debug(
                        "Daily trade count check",
                        extra={"count": today_count, "limit": current_daily_cap},
                    )
                    if today_count >= current_daily_cap:
                        logger.info("Daily trade limit reached during loop; skipping new trade")
                        continue

                trade_id = str(uuid.uuid4())
                logger.debug("Generated trade identifier", extra={"trade_id": trade_id})

                entry_raw = safe_prices[-1]
                # Let ledger apply entry slippage consistently; pass raw price
                fill_context = order_result if isinstance(order_result, dict) else {}
                txid_list_for_ledger = fill_context.get("txid_list")
                if not isinstance(txid_list_for_ledger, list):
                    txid_list_candidate = fill_context.get("txid")
                    if isinstance(txid_list_candidate, list):
                        txid_list_for_ledger = txid_list_candidate
                    elif isinstance(txid_list_candidate, str):
                        txid_list_for_ledger = [txid_list_candidate]
                ledger_kwargs = {
                    "txid": txid_list_for_ledger,
                    "fills": fill_context.get("fills"),
                    "gross_amount": fill_context.get("gross_amount"),
                    "fee": fill_context.get("fee"),
                    "net_amount": fill_context.get("net_amount"),
                    "balance_delta": fill_context.get("balance_delta"),
                    "fill_price": fill_context.get("average_price"),
                    "filled_volume": fill_context.get("filled_volume"),
                }

                ledger.log_trade(
                    trading_pair=pair,
                    trade_size=adjusted_size,
                    strategy_name=strategy_name,
                    trade_id=trade_id,
                    strategy_instance=selected_strategy,
                    confidence=confidence,
                    entry_price=entry_raw,
                    regime=regime,
                    capital_buffer=buffer,
                    rsi=rsi_val,
                    adx=adx_val,
                    **{k: v for k, v in ledger_kwargs.items() if v is not None},
                )
                # Use the exact entry_price and timestamp
                # as written by the ledger (after slippage & rounding)
                logged_trade = ledger.trade_index.get(trade_id)
                if logged_trade is None:
                    logged_trade = next(
                        (t for t in ledger.trades if t.get("trade_id") == trade_id),
                        None,
                    )
                logged_price = logged_trade.get("entry_price") if logged_trade else None
                logged_ts = logged_trade.get("timestamp") if logged_trade else None
                if logged_price is None:
                    # Fallback (should not happen): approximate using same computation
                    logged_price = round(entry_raw * (1 + 0.002), 4)
                position_manager.open_position(
                    trade_id=trade_id,
                    pair=pair,
                    size=adjusted_size,
                    entry_price=logged_price,
                    strategy=strategy_name,
                    confidence=confidence,
                    entry_adx=adx_val,
                    entry_rsi=rsi_val,
                    timestamp=logged_ts,
                )
                executed_trades += 1
                logger.debug("Trade confidence score", extra={"trade_id": trade_id, "confidence": confidence})
                # Print confirmation only when a valid trade record exists and size > 0
                if adjusted_size > 0 and logged_trade:
                    logger.info(
                        "Trade logged",
                        extra={
                            "trade_id": trade_id,
                            "pair": pair,
                            "size": adjusted_size,
                            "strategy": strategy_name,
                        },
                    )

                txid_for_summary = None
                if submitted_live:
                    ctx_txid = fill_context.get("txid")
                    if isinstance(ctx_txid, list):
                        txid_for_summary = ctx_txid[0]
                    else:
                        txid_for_summary = ctx_txid

                trade_summary_context = {
                    "trade_id": trade_id,
                    "pair": pair,
                    "strategy": strategy_name,
                    "size": adjusted_size,
                    "confidence": confidence,
                    "phase": DEPLOY_PHASE,
                    "live_latency_seconds": live_latency,
                    "submitted_live": bool(submitted_live),
                    "kraken_txid": txid_for_summary,
                }
                if limit_context:
                    trade_summary_context["deploy_limit"] = limit_context
                send_alert(
                    f"[trade] Position opened for {pair}",
                    level="INFO",
                    context=trade_summary_context,
                )
                logger.info(
                    "Trade opened | id=%s pair=%s size=%.6f strategy=%s live_latency=%.3fs phase=%s",
                    trade_id,
                    pair,
                    adjusted_size,
                    strategy_name,
                    0.0 if live_latency is None else live_latency,
                    DEPLOY_PHASE,
                )

                # Summary per asset
                logger.debug(
                    "Asset evaluation summary",
                    extra={
                        "asset": asset,
                        "signals": signals_count,
                        "buy_signals": buy_count,
                        "sell_signals": sell_count,
                        "skipped_none": skipped_none,
                        "skipped_low_conf": skipped_low_conf,
                    },
                )
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.exception("Exception while evaluating pair", extra={"pair": pair, "error": str(e)})

    if executed_trades == 0:
        logger.info("No executable signals produced this cycle; no trades submitted.")
    # Exit evaluation now relies solely on observed market prices; no synthetic injections.
    exits = position_manager.check_exits(current_prices)
    for trade_id, exit_price, reason in exits:
        trade_position = position_manager.positions.get(trade_id)
        if reason == "TAKE_PROFIT" and trade_position:
            current_prices[trade_position["pair"]] = exit_price  # Apply immediately
        logger.info(
            "Closing trade",
            extra={
                "trade_id": trade_id,
                "pair": (trade_position or {}).get("pair"),
                "exit_price": exit_price,
                "reason": reason,
            },
        )
        ledger.update_trade(
            trade_id=trade_id,
            exit_price=exit_price,
            reason=reason,
        )
        trade_record = ledger.trade_index.get(trade_id)
        roi_value = None
        slippage_amount = None
        canonical_reason = reason
        reason_display = reason
        if trade_record:
            roi_value = trade_record.get("roi")
            slippage_amount = trade_record.get("exit_slippage_amount")
            canonical_reason = trade_record.get("exit_reason") or canonical_reason
            reason_display = trade_record.get("reason_display") or reason_display
        level = "INFO"
        try:
            if roi_value is not None and float(roi_value) < 0:
                level = "WARNING"
        except (TypeError, ValueError):
            level = "INFO"
        exit_summary = {
            "trade_id": trade_id,
            "pair": (trade_position or {}).get("pair"),
            "strategy": (trade_position or {}).get("strategy"),
            "exit_price": exit_price,
            "reason": canonical_reason,
            "reason_display": reason_display,
            "roi": roi_value,
            "slippage_amount": slippage_amount,
            "phase": DEPLOY_PHASE,
        }
        send_alert(
            f"[trade] Position closed for {(trade_position or {}).get('pair', trade_id)}",
            level=level,
            context=exit_summary,
        )
        logger.info(
            "Trade closed | id=%s pair=%s roi=%s reason=%s display_reason=%s phase=%s",
            trade_id,
            (trade_position or {}).get("pair"),
            roi_value,
            canonical_reason,
            reason_display,
            DEPLOY_PHASE,
        )
        with _locked_file(TRADES_LOG_PATH, "a"):
            pass
        try:
            refresh_portfolio_state()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.exception("Failed to refresh portfolio state after closing trade %s: %s", trade_id, exc)
    if not exits:
        logger.info("No exit conditions triggered.")

    # Shadow test result logging
    shadow_path = "logs/shadow_test_results.jsonl"
    win_count = sum(1 for e in exits if e[2] == "TAKE_PROFIT")
    with _locked_file(shadow_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
                    "win_rate": win_count / len(exits) if exits else 0.0,
                    "num_exits": len(exits),
                }
            )
            + "\n"
        )

    _maybe_write_state_checkpoint(
        {
            "executed_trades": executed_trades,
            "check_exits_only": check_exits_only,
            "phase": DEPLOY_PHASE,
        }
    )


def gather_signals(prices, volumes, ctx=None, **kwargs):
    """Collect and compute minimal indicators safely.

    Computes RSI and returns a dict with simple fields. Accepts optional
    ``ctx`` or ``context`` for config lookup.
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    out = {"rsi": None, "trend": None, "raw": {"prices": prices, "volumes": volumes}}

    def _to_scalar(x):
        """Coerce various container types to a single representative value.

        Takes the last item for sequences, preferred keys for mappings, or
        returns the value unchanged for numerics/others.
        """
        # Accept list/tuple/dict/number; take a reasonable "last value".
        if isinstance(x, (list, tuple)) and x:
            return x[-1]
        if isinstance(x, dict):
            for k in ("rsi", "current", "value", "last"):
                if k in x:
                    return x[k]
        return x

    try:
        if prices and hasattr(prices, "__len__"):
            n = len(prices)
            if n >= 3:
                # default = 14; take from context if available
                period = 14
                try:
                    if ctx is not None:
                        cfg = getattr(ctx, "config", None) or getattr(ctx, "CONFIG", None) or {}
                        if hasattr(cfg, "get"):
                            period = cfg.get("rsi", {}).get("period", 14)
                        elif isinstance(cfg, dict):
                            period = cfg.get("rsi", {}).get("period", 14)
                except (AttributeError, TypeError, ValueError):
                    pass
                # clamp to valid range (some impls need <= n-1)
                period = max(2, min(int(period), max(2, n - 1)))
                if calculate_rsi is None:
                    logger.warning("RSI calculator unavailable — skipping RSI computation.")
                else:
                    try:
                        val = calculate_rsi(prices, period)  # type: ignore[misc]
                        out["rsi"] = _to_scalar(val)
                    except (ValueError, TypeError, ZeroDivisionError, IndexError):
                        out["rsi"] = None
    except Exception:  # pylint: disable=broad-exception-caught
        # fail-open: never break the loop due to indicator calc
        pass
    return out


def risk_screen(signals, ctx=None, **kwargs) -> bool:
    """Lightweight risk gate. Returns True to allow, False to block.

    Rules:
    - Block new *long* entries if RSI > 70 (overbought).
    - Enforce max open positions (default 3).
    - Enforce cash buffer ratio (cash/equity) >= capital_buffer (default 0.25).
    - Fail-open on unexpected errors (never halt the loop).
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    try:
        sb = signals or {}

        # ---- config (defensive) ----
        cfg = getattr(ctx, "config", None) or getattr(ctx, "CONFIG", None) or {}

        def cfg_get(path, default):
            """Safely traverse nested config dicts with defaults.

            Accepts a tuple path like ("risk", "max_open_positions") and
            returns the value if present; otherwise returns ``default``.
            """
            cur = cfg
            for k in path:
                if hasattr(cur, "get"):
                    cur = cur.get(k, default if k is path[-1] else {})
                elif isinstance(cur, dict):
                    cur = cur.get(k, default if k is path[-1] else {})
                else:
                    return default
            return cur

        max_positions = int(cfg_get(("risk", "max_open_positions"), 3))
        capital_buffer = float(cfg_get(("risk", "capital_buffer"), 0.25))

        # ---- portfolio guards ----
        portfolio = getattr(ctx, "portfolio", None)
        if portfolio is not None:
            opens = len(getattr(portfolio, "open_positions", []) or [])
            if opens >= max_positions:
                return False

            cash = float(getattr(portfolio, "cash", 0.0) or 0.0)
            equity = getattr(portfolio, "equity", None)

            if equity is None or equity == 0:
                logger.warning("Portfolio equity missing or zero — skipping trade.")
                return False

            equity = float(equity)
            if equity > 0 and (cash / equity) < capital_buffer:
                return False

        # ---- RSI guard for long entries ----
        intent = sb.get("signal")
        rsi = sb.get("rsi")

        # scalarize rsi
        if isinstance(rsi, (list, tuple)):
            rsi = rsi[-1] if rsi else None
        elif isinstance(rsi, dict):
            rsi = rsi.get("rsi") or rsi.get("current") or rsi.get("value") or rsi.get("last")

        # numeric cast (best effort)
        try:
            rsi_num = float(rsi) if rsi is not None else None
        except (ValueError, TypeError):
            rsi_num = None

        if intent == "buy" and rsi_num is not None and rsi_num > 70.0:
            return False

        return True
    except Exception:  # pylint: disable=broad-exception-caught
        # fail-open so the engine doesn't halt on a guard bug
        return True


def execute_trade(signals, ctx=None, **kwargs):
    """Execute trade(s) based on signals.

    Returns a list of trade-like dicts for testing only; production flow
    still uses existing paths until we finish the split.
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    _ = (signals, ctx)
    return []


def check_and_close_exits(ctx=None, **kwargs) -> int:
    """Check exit rules and close positions if needed.

    Returns the number of closed positions (0 in stub).
    """
    # Back-compat: accept callers passing 'context='
    if ctx is None:
        ctx = kwargs.get("context")
    _ = ctx
    return 0


if __name__ == "__main__":
    evaluate_signals_and_trade()
