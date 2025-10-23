"""
Trade Ledger Module (Class-Based)

Encapsulates trade lifecycle logging, ROI updates, schema validation,
and trade syncing using a class-based approach.
"""

# pylint: disable=too-many-lines

import fcntl
import json
import logging
import math
import os
import random
import time
import uuid
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional

from crypto_trading_bot import config as bot_config
from crypto_trading_bot.bot.utils.alerts import send_alert
from crypto_trading_bot.bot.utils.log_rotation import get_anomalies_logger
from crypto_trading_bot.bot.utils.schema_validator import validate_trade_schema
from crypto_trading_bot.config import CONFIG, IS_LIVE
from crypto_trading_bot.utils.kraken_client import get_usdc_balance, kraken_get_balance
from crypto_trading_bot.utils.system_logger import (
    SYSTEM_LOG_PATH as SHARED_SYSTEM_LOG_PATH,
)
from crypto_trading_bot.utils.system_logger import (
    get_system_logger,
)

# Enable extra debug output when explicitly requested
DEBUG_MODE = os.getenv("DEBUG_MODE", "0") == "1"

TRADES_LOG_PATH = "logs/trades.log"
SYSTEM_LOG_PATH = str(SHARED_SYSTEM_LOG_PATH)
POSITIONS_PATH = "logs/positions.jsonl"
SLIPPAGE = 0.002  # default fallback; per-asset slippage from CONFIG overrides this

# System logger (warnings/errors/debug) to logs/system.log
system_logger = get_system_logger()

# Trade logger (message-only JSON lines) to logs/trades.log
trade_logger = logging.getLogger("trade_ledger.trades")
if not trade_logger.hasHandlers():
    os.makedirs("logs", exist_ok=True)
    trade_handler = RotatingFileHandler(
        TRADES_LOG_PATH,
        maxBytes=10 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    trade_handler.setFormatter(logging.Formatter("%(message)s"))
    trade_logger.addHandler(trade_handler)
    trade_logger.setLevel(logging.INFO)
    trade_logger.propagate = False

# Shared anomalies logger (compact JSONL)
anomalies_logger = get_anomalies_logger()

_MISSING_TRADE_ALERT_LIMIT = int(os.getenv("MISSING_TRADE_ALERT_LIMIT", "2"))
_MISSING_TRADE_WINDOW_SECONDS = float(os.getenv("MISSING_TRADE_ALERT_WINDOW", "300"))
_DUPLICATE_WINDOW_SECONDS = float(os.getenv("TRADE_DUPLICATE_WINDOW_SECONDS", "90"))
_DUPLICATE_SIZE_REL_TOL = float(os.getenv("TRADE_DUPLICATE_SIZE_REL_TOL", "0.001"))
_DUPLICATE_SIZE_ABS_TOL = float(os.getenv("TRADE_DUPLICATE_SIZE_ABS_TOL", "1e-6"))
_DECIMAL_TWO_PLACES = Decimal("0.01")

_EXIT_REASON_CANONICAL_MAP = {
    "STOP_LOSS": "sl_triggered",
    "STOPLOSS": "sl_triggered",
    "SL_TRIGGERED": "sl_triggered",
    "TRAILING_STOP": "trailing_exit",
    "TRAIL_STOP": "trailing_exit",
    "TRAILING_EXIT": "trailing_exit",
    "TAKE_PROFIT": "tp_hit",
    "TAKEPROFIT": "tp_hit",
    "TP_HIT": "tp_hit",
    "MAX_HOLD": "max_hold_expired",
    "MAX_HOLD_EXPIRED": "max_hold_expired",
    "RSI_EXIT": "indicator_exit",
    "RSI": "indicator_exit",
    "INDICATOR_EXIT": "indicator_exit",
    "MANUAL_EXIT": "manual_exit",
    "MANUAL_CLOSE": "manual_exit",
    "AUTO_PAUSE": "auto_pause_exit",
    "KILL_SWITCH": "emergency_stop",
    "EMERGENCY_STOP": "emergency_stop",
}


def _to_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return Decimal(str(value))
    try:
        candidate = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return candidate if candidate.is_finite() else None


def _round_balance(value: Any) -> Optional[float]:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    try:
        rounded = decimal_value.quantize(_DECIMAL_TWO_PLACES, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    return float(rounded)


def _safe_float(value: Any) -> float | None:
    decimal_value = _to_decimal(value)
    if decimal_value is None:
        return None
    return float(decimal_value)


def _calculate_balance_delta(entry_balance: Any, exit_balance: Any) -> float | None:
    entry_decimal = _to_decimal(entry_balance)
    exit_decimal = _to_decimal(exit_balance)
    if entry_decimal is None or exit_decimal is None:
        return None
    delta = exit_decimal - entry_decimal
    try:
        quantized = delta.quantize(_DECIMAL_TWO_PLACES, rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    return float(quantized)


def _canonical_exit_reason(*candidates: Any) -> tuple[str, str]:
    """Return (canonical_reason, display_reason) for supplied candidates."""

    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        upper = text.upper()
        mapped = _EXIT_REASON_CANONICAL_MAP.get(upper)
        if mapped:
            return mapped, text
        lowered = text.lower()
        if lowered in _EXIT_REASON_CANONICAL_MAP.values():
            return lowered, text
        sanitized = lowered.replace(" ", "_")
        if sanitized:
            return sanitized, text
    return "unknown", "unknown"


def _parse_iso_datetime(raw: Any) -> datetime | None:
    if not isinstance(raw, str):
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _slippage_rate_for_pair(pair: str) -> float:
    base = (pair or "BTC/USDC").split("/")[0]
    majors = {"BTC", "ETH"}
    if base in majors:
        return float(CONFIG.get("slippage", {}).get("majors", 0.001))
    cfg = CONFIG.get("slippage", {})
    use_random = bool(cfg.get("use_random", False))
    lo = float(cfg.get("alts_min", 0.005))
    hi = float(cfg.get("alts_max", 0.01))
    if use_random:
        return random.uniform(lo, hi)
    return (lo + hi) / 2.0


def _apply_slippage(pair: str, price: float, side: str) -> tuple[float, float, float]:
    """Apply side-aware slippage using configured rates.

    Returns (adjusted_price, slippage_amount, slippage_rate).
    """
    rate = _slippage_rate_for_pair(pair)
    raw = float(price)
    if side == "buy":
        adjusted = raw * (1 + rate)
        amount = adjusted - raw
    else:  # sell
        adjusted = raw * (1 - rate)
        amount = raw - adjusted
    adjusted = round(adjusted, 4)
    amount = round(amount, 4)
    system_logger.debug(
        "Applied slippage pair=%s side=%s raw=%.6f adj=%.6f rate=%.6f amount=%.6f",
        pair,
        side,
        raw,
        adjusted,
        rate,
        amount,
    )
    return adjusted, amount, rate


def _apply_exit_slippage(pair: str, exit_price: float) -> float:
    """Backward-compatible no-op wrapper retained for tests.

    Exit slippage is applied via _apply_slippage in update paths. This function
    preserves previous external expectations that it returns the input value.
    """
    _ = pair
    return float(exit_price)


def _infer_side(strategy: str | None, roi: float | None) -> str:
    """Infer trade side ('long' or 'short') from strategy hints and ROI.

    - If strategy mentions 'RSI' or 'Threshold' and ROI >= 0 -> long
    - If ROI < 0 -> short
    - Fallback default -> long
    """
    s = strategy.lower() if isinstance(strategy, str) else ""
    try:
        r = float(roi) if roi is not None else None
    except (TypeError, ValueError):
        r = None
    if ("rsi" in s or "threshold" in s) and (r is None or r >= 0):
        return "long"
    if r is not None and r < 0:
        return "short"
    return "long"


def _normalize_side(side: str | None, strategy: str | None = None, roi: float | None = None) -> str:
    """Normalize side to 'long'/'short', mapping common aliases.

    Accepts 'buy' -> 'long', 'sell' -> 'short'; falls back to inference.
    """
    if isinstance(side, str):
        s = side.strip().lower()
        if s in {"long", "short"}:
            return s
        if s in {"buy", "sell"}:
            return "long" if s == "buy" else "short"
    return _infer_side(strategy, roi)


def validate_trade_entry(trade: dict) -> dict:
    """Best-effort validation and normalization for a trade entry.

    - Ensures 'strategy' is non-empty string; defaults to 'Unknown'
    - Normalizes 'side' to 'long'/'short' (infers if missing)
    - If 'exit_price' present, ensures 'roi' is numeric; if not, warns and sets 0.0
    - Logs warnings to system_logger; returns mutated trade dict.
    """
    # Strategy
    if not isinstance(trade.get("strategy"), str) or not trade.get("strategy").strip():
        system_logger.warning("Missing/invalid strategy; defaulting to 'Unknown'")
        trade["strategy"] = "Unknown"

    # Side normalization
    norm_side = _normalize_side(
        trade.get("side"),
        strategy=trade.get("strategy"),
        roi=trade.get("roi"),
    )
    if trade.get("side") != norm_side:
        system_logger.warning("Normalized side %s -> %s", trade.get("side"), norm_side)
        trade["side"] = norm_side

    # Exit/ROI consistency (do not raise)
    if trade.get("exit_price") is not None:
        try:
            _ = float(trade.get("roi"))  # ensure numeric
        except (TypeError, ValueError):
            system_logger.warning("Exit present but ROI invalid; setting ROI=0.0")
            trade["roi"] = 0.0

    return trade


def _ensure_closed_trade_metadata(trade: dict) -> bool:
    """Ensure legacy closed trades have required metadata populated."""

    if trade.get("status") != "closed":
        return False

    mutated = False

    norm_side = _normalize_side(
        trade.get("side"),
        strategy=trade.get("strategy"),
        roi=trade.get("roi"),
    )
    side_value = trade.get("side")
    if not isinstance(side_value, str) or side_value.strip().lower() not in {"long", "short"}:
        trade["side"] = norm_side
        mutated = True

    buffer_value = trade.get("capital_buffer")
    if not isinstance(buffer_value, (int, float)):
        trade["capital_buffer"] = 0.25
        mutated = True

    return mutated


class TradeLedger:
    """
    A class to manage trade lifecycle logging, updating, and validation.
    Handles logging trade entries, updating trade exits, syncing positions,
    reloading historical trades, and verifying lifecycle status.
    """

    def __init__(self, position_manager):
        """
        Initialize the TradeLedger with a reference to the PositionManager.
        Loads existing trades from the trades log.
        """
        self.trades = []
        self.trade_index: dict[str, dict] = {}
        self.position_manager = position_manager
        self._missing_trade_alerted: set[str] = set()
        self._missing_trade_window: Dict[str, Dict[str, Any]] = {}
        self._pause_new_trades_once = False
        self._pause_reason: str | None = None
        self.account_balance: Optional[float] = None
        self.txid_index: dict[str, str] = {}
        self._balance_source: str = "unknown"
        self.reload_trades()

    def _fetch_runtime_balance(self) -> Optional[float]:
        """Return the current USDC balance from portfolio/market data helpers."""

        try:
            balance = get_usdc_balance()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            system_logger.warning("USDC balance fetch failed: %s", exc)
            return None

        value = _safe_float(balance)
        if value is None:
            system_logger.warning("USDC balance fetch returned non-numeric value: %r", balance)
            return None

        if math.isclose(value, 0.0, abs_tol=1e-9):
            system_logger.warning("USDC balance fetch returned zero. Verify live balance configuration.")
        system_logger.info("Fetched runtime USDC balance %.2f", value)
        return value

    def _apply_runtime_balance(
        self,
        trade: Dict[str, Any],
        *,
        entry_balance: Optional[float] = None,
    ) -> None:
        """Inject the latest account balance into ``trade`` if available."""

        existing_balance = trade.get("account_balance")
        existing_delta = trade.get("balance_delta")

        balance_value = self._fetch_runtime_balance()
        if balance_value is None:
            if existing_balance is None:
                trade["account_balance"] = None
            if entry_balance is not None:
                if existing_delta is None:
                    trade["balance_delta"] = None
                system_logger.warning(
                    "Missing live balance when closing trade %s; balance_delta unavailable.",
                    trade.get("trade_id"),
                )
            else:
                if existing_balance is None:
                    system_logger.warning(
                        "Missing live balance when logging trade %s; account_balance unavailable.",
                        trade.get("trade_id"),
                    )
            return

        rounded_balance = _round_balance(balance_value)
        if rounded_balance is None:
            if existing_balance is None:
                trade["account_balance"] = None
            if entry_balance is not None:
                if existing_delta is None:
                    trade["balance_delta"] = None
                system_logger.warning(
                    "Unable to round live balance for trade %s; balance_delta unavailable.",
                    trade.get("trade_id"),
                )
            else:
                if existing_balance is None:
                    system_logger.warning(
                        "Unable to round live balance for trade %s; account_balance unavailable.",
                        trade.get("trade_id"),
                    )
            return

        self.account_balance = rounded_balance
        trade["account_balance"] = rounded_balance
        system_logger.info(
            "Trade %s balance snapshot=%s",
            trade.get("trade_id"),
            rounded_balance,
        )

        if entry_balance is not None:
            delta_value = _calculate_balance_delta(entry_balance, rounded_balance)
            trade["balance_delta"] = delta_value
            if delta_value is None:
                system_logger.warning(
                    "Unable to compute balance_delta for trade %s (entry=%s exit=%s).",
                    trade.get("trade_id"),
                    entry_balance,
                    rounded_balance,
                )
            else:
                system_logger.info(
                    "Trade %s balance lifecycle entry=%.2f exit=%.2f delta=%.2f",
                    trade.get("trade_id"),
                    entry_balance,
                    rounded_balance,
                    delta_value,
                )
        else:
            trade["balance_delta"] = existing_delta

    def _find_recent_duplicate(self, trade: dict) -> dict | None:
        """Return an existing trade that matches the provided signature."""

        timestamp = _parse_iso_datetime(trade.get("timestamp"))
        if timestamp is None:
            return None

        window = max(1.0, _DUPLICATE_WINDOW_SECONDS)
        pair = str(trade.get("pair") or "").upper()
        side = str(trade.get("side") or "").lower()
        strategy = str(trade.get("strategy") or "")
        try:
            size = float(trade.get("size", 0.0))
        except (TypeError, ValueError):
            return None

        tolerance = max(_DUPLICATE_SIZE_ABS_TOL, abs(size) * _DUPLICATE_SIZE_REL_TOL)

        for existing in reversed(self.trades[-100:]):
            if existing.get("trade_id") == trade.get("trade_id"):
                continue
            if str(existing.get("pair") or "").upper() != pair:
                continue
            if str(existing.get("side") or "").lower() != side:
                continue
            if str(existing.get("strategy") or "") != strategy:
                continue

            existing_time = _parse_iso_datetime(existing.get("timestamp"))
            if existing_time is None:
                continue
            delta = abs((timestamp - existing_time).total_seconds())
            if delta > window:
                if timestamp >= existing_time:
                    break
                continue

            try:
                existing_size = float(existing.get("size"))
            except (TypeError, ValueError):
                continue
            if abs(existing_size - size) <= tolerance:
                return existing
        return None

    def request_pause_new_trades(self, reason: str | None = None) -> None:
        """Pause new trade execution for a single evaluation cycle."""

        self._pause_new_trades_once = True
        self._pause_reason = reason

    def consume_pause_request(self) -> tuple[bool, str | None]:
        """Return (should_pause, reason) and reset the one-shot pause flag."""

        if self._pause_new_trades_once:
            self._pause_new_trades_once = False
            reason = self._pause_reason
            self._pause_reason = None
            return True, reason
        return False, None

    def log_trade(self, trading_pair, trade_size, strategy_name, **kwargs):
        """
        Log a new trade to the trades log.
        Validates schema and writes the trade to disk.
        Returns the trade ID.
        """
        if not isinstance(strategy_name, str):
            raise ValueError(f"[Ledger] Invalid strategy_name: {strategy_name}")
        pair_token = str(trading_pair or "").strip().upper()
        if not pair_token:
            raise ValueError("[Ledger] Trading pair is required.")
        live_mode = bool(getattr(bot_config, "is_live", False))
        if live_mode:
            if pair_token.endswith("/USD"):
                message = (
                    f"[Ledger] Trading pair {pair_token} uses USD quote. "
                    "Switch configuration to USDC before logging trades."
                )
                system_logger.error(message)
                raise ValueError(message)
            if not pair_token.endswith("/USDC"):
                message = f"[Ledger] Trading pair {pair_token} is not a USDC market."
                system_logger.error(message)
                raise ValueError(message)
        else:
            if pair_token.endswith("/USD"):
                system_logger.warning(
                    "[Ledger] USD-quoted trade %s logged outside live mode. "
                    "Ensure conversion to USDC before enabling live trading.",
                    pair_token,
                )
        trading_pair = pair_token
        confidence = float(kwargs.get("confidence", 0.0) or 0.0)
        if math.isclose(confidence, 0.5, abs_tol=1e-9):
            message = " ".join(
                [
                    "[Ledger] Confidence of 0.5 is no longer permitted — ensure strategies emit",
                    "calibrated values.",
                ]
            )
            raise ValueError(message)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"[Ledger] Invalid confidence value: {confidence}")
        if isinstance(strategy_name, str) and strategy_name.strip().upper() == "S":
            warning = " ".join(
                [
                    "[Ledger] Strategy identifier 'S' is reserved for testing and cannot be",
                    "logged in production.",
                ]
            )
            raise ValueError(warning)
        if not isinstance(trade_size, (int, float)) or trade_size <= 0:
            raise ValueError(f"[Ledger] Invalid trade size: {trade_size}")
        # Bound-check size based on CONFIG
        trade_size = float(trade_size)
        min_sz = float(CONFIG.get("trade_size", {}).get("min", 0.0001))
        max_sz = float(CONFIG.get("trade_size", {}).get("max", max(trade_size, 1.0)))
        trade_size = max(min_sz, min(trade_size, max_sz))
        if "entry_price" in kwargs:
            entry_price_val = kwargs["entry_price"]
            if not isinstance(entry_price_val, (int, float)) or entry_price_val <= 0:
                raise ValueError(f"[Ledger] Invalid entry_price: {entry_price_val}")

        fills = kwargs.get("fills")
        txid = kwargs.get("txid")
        gross_amount = kwargs.get("gross_amount")
        fee_amount = kwargs.get("fee")
        net_amount = kwargs.get("net_amount")
        fill_price_override = kwargs.get("fill_price")
        filled_volume = kwargs.get("filled_volume")
        reconciled_flag = bool(kwargs.get("reconciled", False))
        pending_flag = bool(kwargs.get("pending_reconciliation", False))
        source_flag = kwargs.get("source") or ("kraken" if reconciled_flag else "local")
        pending_at = kwargs.get("pending_reconciliation_at")
        if isinstance(pending_at, datetime):
            pending_at = pending_at.isoformat()

        trade_id = kwargs.get("trade_id") or str(uuid.uuid4())
        entry_price = kwargs.get("entry_price", 18000 + 250 * (0.5 - random.random()))
        # Normalize side to long/short; map to order side for slippage
        side_norm = _normalize_side(
            kwargs.get("side"),
            strategy=strategy_name,
            roi=kwargs.get("roi"),
        )
        order_side = "buy" if side_norm == "long" else "sell"
        entry_price_adj, slippage_amount_entry, slip_rate = _apply_slippage(
            trading_pair,
            float(entry_price),
            order_side,
        )

        if filled_volume is not None:
            try:
                filled_volume = float(filled_volume)
                if filled_volume > 0:
                    trade_size = float(filled_volume)
            except (TypeError, ValueError):
                filled_volume = None

        if fill_price_override:
            try:
                entry_price_adj = float(fill_price_override)
            except (TypeError, ValueError):
                pass

        capital_buffer = kwargs.get("capital_buffer", 0.25)
        tax_method = kwargs.get("tax_method", "FIFO")
        regime = kwargs.get("regime", "unknown")

        normalized_txid = None
        if txid:
            if isinstance(txid, (list, tuple)):
                normalized_txid = [str(item) for item in txid if item]
            else:
                normalized_txid = [str(txid)]

        # If Kraken provided a precise cost basis, prefer it.
        gross_amount_float = _safe_float(gross_amount)
        fee_amount_float = _safe_float(fee_amount)
        net_amount_float = _safe_float(net_amount)

        if gross_amount_float is not None:
            cost_basis = abs(gross_amount_float)
        else:
            cost_basis = round(entry_price_adj * trade_size, 4)

        try:
            if fill_price_override is not None:
                fill_price_value = float(fill_price_override)
            else:
                fill_price_value = None
        except (TypeError, ValueError):
            fill_price_value = None

        trade = {
            "trade_id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pair": trading_pair,
            "size": trade_size,
            "strategy": strategy_name,
            "confidence": confidence,
            "status": "executed",
            "capital_buffer": capital_buffer,
            "tax_method": tax_method,
            "cost_basis": round(cost_basis, 4),
            "entry_price": entry_price_adj,
            # Persist side so downstream logic can apply correct exit behavior
            "side": side_norm,
            "exit_price": None,
            "realized_gain": None,
            "holding_period_days": None,
            "roi": None,
            "reason": None,
            "regime": regime,
            # Optional diagnostics
            "rsi": kwargs.get("rsi"),
            "adx": kwargs.get("adx"),
            # Slippage metadata
            "entry_slippage_rate": round(slip_rate, 6),
            "entry_slippage_amount": slippage_amount_entry,
            "fills": fills if isinstance(fills, list) else None,
            "gross_amount": (round(gross_amount_float, 8) if gross_amount_float is not None else None),
            "fee": round(fee_amount_float, 8) if fee_amount_float is not None else None,
            "net_amount": (round(net_amount_float, 8) if net_amount_float is not None else None),
            "balance_delta": None,
            "account_balance": None,
            "fill_price": fill_price_value,
            "filled_volume": (float(filled_volume) if isinstance(filled_volume, (int, float)) else None),
            "txid": normalized_txid,
            "reconciled": reconciled_flag,
            "pending_reconciliation": pending_flag,
            "source": source_flag,
            "pending_reconciliation_at": pending_at,
        }

        # Debug: emit the trade being logged for diagnostics (opt-in)
        if DEBUG_MODE:
            try:
                system_logger.debug("Logging trade: %s", json.dumps(trade, indent=2))
            except (TypeError, OSError):
                pass

        # Best-effort validate/normalize fields to prevent schema issues
        trade = validate_trade_entry(trade)

        # Normalize side for audit compatibility
        # Convert canonical internal values to order sides expected by audits
        try:
            s = str(trade.get("side", "")).strip().lower()
            if s == "long":
                trade["side"] = "buy"
            elif s == "short":
                trade["side"] = "sell"
        except (AttributeError, TypeError, ValueError):
            system_logger.exception("Failed to normalize trade side for trade_id=%s", trade_id)

        # Validate schema; on failure, log anomaly for audit and re-raise.
        try:
            validate_trade_schema(trade)
        except (ValueError, TypeError) as e:
            try:
                anomalies_logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "Trade Schema Error",
                            "error": str(e),
                            "trade": trade,
                        },
                        separators=(",", ":"),
                    )
                )
            except (TypeError, ValueError):
                pass
            raise

        duplicate = self._find_recent_duplicate(trade)
        if duplicate is not None:
            existing_id = duplicate.get("trade_id") or "<unknown>"
            system_logger.warning(
                "Duplicate trade detected within %.0fs window; existing trade_id=%s pair=%s "
                "size=%.8f strategy=%s. Skipping new entry %s.",
                _DUPLICATE_WINDOW_SECONDS,
                existing_id,
                trade.get("pair"),
                trade.get("size"),
                trade.get("strategy"),
                trade_id,
            )
            try:
                anomalies_logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "DuplicateTrade",
                            "existing_trade_id": existing_id,
                            "duplicate_trade_id": trade_id,
                            "pair": trade.get("pair"),
                            "side": trade.get("side"),
                            "size": trade.get("size"),
                            "strategy": trade.get("strategy"),
                        },
                        separators=(",", ":"),
                    )
                )
            except (TypeError, ValueError):
                pass
            send_alert(
                "[Ledger] Duplicate trade signature detected",
                level="WARNING",
                context={
                    "existing_trade_id": existing_id,
                    "duplicate_trade_id": trade_id,
                    "pair": trade.get("pair"),
                    "strategy": trade.get("strategy"),
                    "size": trade.get("size"),
                },
            )
            return existing_id

        balance_delta_value = None
        balance_delta_provided = "balance_delta" in kwargs and kwargs.get("balance_delta") is not None
        if balance_delta_provided:
            balance_delta_value = _safe_float(kwargs.get("balance_delta"))
        if balance_delta_value is None:
            balance_delta_value = net_amount_float

        existing_balance = _safe_float(self.account_balance)
        if balance_delta_value is not None:
            updated_balance = (existing_balance or 0.0) + balance_delta_value
            self.account_balance = updated_balance
            trade["balance_delta"] = balance_delta_value
            trade["account_balance"] = updated_balance
        else:
            trade["balance_delta"] = None
            trade["account_balance"] = _round_balance(existing_balance)

        if not balance_delta_provided:
            previous_balance = trade.get("account_balance")
            previous_delta = trade.get("balance_delta")

            # Attach live/paper balance snapshot for diagnostics.
            self._apply_runtime_balance(trade)

            if trade.get("account_balance") is None and previous_balance is not None:
                trade["account_balance"] = previous_balance
            if trade.get("balance_delta") is None and previous_delta is not None:
                trade["balance_delta"] = previous_delta
            if trade.get("account_balance") is None:
                system_logger.warning(
                    "Trade %s logged without account_balance; review live balance configuration.",
                    trade_id,
                )

        # Write compact, one-line JSON to trades.log via trade_logger
        trade_logger.info(json.dumps(trade, separators=(",", ":")))
        self.trades.append(trade)
        self.trade_index[trade_id] = trade
        if normalized_txid:
            for tx in normalized_txid:
                self.txid_index[tx] = trade_id
        return trade_id

    def _register_missing_trade(
        self,
        trade_id: str,
        pair: str | None,
    ) -> tuple[bool, Dict[str, Any]]:
        """Track missing-trade alerts and suppress repeated notifications per symbol."""
        bucket = (pair or "unknown").upper()
        now = time.monotonic()
        state = self._missing_trade_window.get(bucket, {})
        window_start = float(state.get("start", 0.0))
        if not state or now - window_start > _MISSING_TRADE_WINDOW_SECONDS:
            state = {"start": now, "count": 0, "suppressed": 0}

        state["count"] = int(state.get("count", 0)) + 1
        state["last_trade_id"] = trade_id
        state["last_seen_iso"] = datetime.now(timezone.utc).isoformat()
        self._missing_trade_window[bucket] = state

        alert_limit = max(1, _MISSING_TRADE_ALERT_LIMIT)
        should_alert = state["count"] <= alert_limit
        if not should_alert:
            state["suppressed"] = int(state.get("suppressed", 0)) + 1
        return should_alert, state

    def get_missing_trade_metrics(self, reset: bool = False) -> Dict[str, Dict[str, Any]]:
        """Return snapshot of missing-trade counters, optionally clearing the state."""
        snapshot = {bucket: dict(state) for bucket, state in self._missing_trade_window.items()}
        if reset:
            self._missing_trade_window.clear()
        return snapshot

    def open_position(self, trade_id, trade):
        """
        Record a newly opened trade into the positions file (positions.jsonl).
        Uses the trade ID as key reference.
        """
        # Validate essential fields with safe fallbacks
        trade = dict(trade)
        trade["trade_id"] = trade_id
        # Enforce basic required fields
        if not isinstance(trade.get("pair"), str) or not trade.get("pair").strip():
            system_logger.warning("Position missing pair; defaulting to BTC/USDC")
            trade["pair"] = "BTC/USDC"
        if not isinstance(trade.get("strategy"), str) or not trade.get("strategy").strip():
            system_logger.warning("Position missing strategy; defaulting to Unknown")
            trade["strategy"] = "Unknown"
        # Ensure entry_price
        try:
            ep = float(trade.get("entry_price"))
            if ep <= 0:
                raise ValueError
        except (TypeError, ValueError):
            # Fallback: reuse cost_basis/size or set to plausible default
            cb = trade.get("cost_basis")
            sz = trade.get("size") or 0.001
            try:
                trade["entry_price"] = round(float(cb) / float(sz), 4) if cb else 18000.0
            except (TypeError, ZeroDivisionError):
                trade["entry_price"] = 18000.0
            system_logger.warning("Position missing/invalid entry_price; default set")
        # Normalize side
        trade["side"] = _normalize_side(
            trade.get("side"),
            strategy=trade.get("strategy"),
            roi=trade.get("roi"),
        )

        with open(POSITIONS_PATH, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            json.dump(trade, f)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
        system_logger.debug("Position %s written to positions.jsonl", trade_id)

    def update_trade(self, trade_id, exit_price, reason, exit_reason=None):
        """
        Update a trade with exit information such as exit_price, reason, and ROI.
        If trade is missing, attempts to reload from file and/or reconstruct from positions.
        Retries up to 3 times on failure and safely resyncs positions.jsonl.

        Exit reasons are normalized to snake_case and stored in both ``reason`` and
        ``exit_reason``; the original label is preserved in ``reason_display``.
        """
        # If exit_price is missing/invalid, log anomaly and return without raising.
        if exit_price is None or not isinstance(exit_price, (int, float)) or exit_price <= 0:
            # Retrieve trade object safely without broad exception catching
            t_obj = None
            trade_idx = getattr(self, "trade_index", None)
            if isinstance(trade_idx, dict):
                t_obj = trade_idx.get(trade_id)
            system_logger.error(
                "Cannot update trade %s — invalid exit_price %s (reason=%s)",
                trade_id,
                exit_price,
                reason or exit_reason or "<unspecified>",
            )
            try:
                anomalies_logger.info(
                    json.dumps(
                        {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "type": "Missing Exit Price",
                            "trade_id": trade_id,
                            "pair": (t_obj or {}).get("pair"),
                            "strategy": (t_obj or {}).get("strategy"),
                            "reason": "exit_price is None or not logged",
                            "confidence": (t_obj or {}).get("confidence"),
                            "roi": (t_obj or {}).get("roi"),
                            "regime": (t_obj or {}).get("regime"),
                            "side": (t_obj or {}).get("side"),
                        },
                        separators=(",", ":"),
                    )
                )
            except (TypeError, ValueError):
                pass
            return

        if not os.path.exists(TRADES_LOG_PATH):
            system_logger.error("Trade file not found: %s", TRADES_LOG_PATH)
            return

        positions_missing = not hasattr(self.position_manager, "positions")
        positions_none = getattr(self.position_manager, "positions", None) is None
        if positions_missing or positions_none:
            system_logger.error("position_manager.positions unavailable for trade %s", trade_id)
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                updated = False

                # Ensure we have latest trades in memory
                trades = self.trades or []
                if trade_id not in self.trade_index:
                    _ = self.reload_trades()

                # If still not present, try reconstruct from positions
                if trade_id not in self.trade_index:
                    pos = self.position_manager.positions.get(trade_id)
                    if pos and pos.get("exit_price") is None:
                        trade = {
                            "trade_id": trade_id,
                            "timestamp": pos.get(
                                "timestamp",
                                datetime.now(timezone.utc).isoformat(),
                            ),
                            "pair": pos.get("pair", "BTC/USDC"),
                            "size": pos.get("size", 0.001),
                            "strategy": pos.get("strategy", "Unknown"),
                            "confidence": float(pos.get("confidence", 0.0) or 0.0),
                            "status": "executed",
                            "capital_buffer": 0.25,
                            "tax_method": "FIFO",
                            "cost_basis": round(
                                pos.get("entry_price", 18000) * pos.get("size", 0.001),
                                4,
                            ),
                            "entry_price": round(pos.get("entry_price", 18000), 4),
                            "exit_price": None,
                            "realized_gain": None,
                            "holding_period_days": None,
                            "roi": None,
                            "reason": None,
                            "regime": "unknown",
                        }
                        trades.append(trade)
                        self.trade_index[trade_id] = trade
                        system_logger.warning("Synced missing trade %s from position", trade_id)

                # If still not found, log a clear error and stop
                if trade_id not in self.trade_index:
                    trades = self.reload_trades()
                    if trade_id in self.trade_index:
                        system_logger.info(
                            "Trade %s located after full reload; continuing update",
                            trade_id,
                        )
                    else:
                        reason_text = reason or exit_reason or "n/a"
                        message = (
                            f"Trade ID {trade_id} not found in memory or trades.log — "
                            f"skipping update (exit_price={exit_price}, reason={reason_text})"
                        )
                        pos = self.position_manager.positions.get(trade_id)
                        pair = (pos or {}).get("pair")
                        should_alert, state = self._register_missing_trade(trade_id, pair)
                        system_logger.warning(message)
                        context = {
                            "trade_id": trade_id,
                            "attempt": attempt + 1,
                            "pair": pair,
                            "count_in_window": state.get("count"),
                            "suppressed": state.get("suppressed"),
                            "position_keys": list(self.position_manager.positions.keys()),
                        }
                        if should_alert and trade_id not in self._missing_trade_alerted:
                            self._missing_trade_alerted.add(trade_id)
                            send_alert(message, context=context, level="WARNING")
                            pause_reason = f"Ledger consistency check — missing trade {trade_id}"
                            self.request_pause_new_trades(reason=pause_reason)
                        else:
                            payload = {
                                "timestamp": datetime.now(timezone.utc).isoformat(),
                                "type": "Missing Trade Suppressed",
                                "message": message,
                                "context": context,
                            }
                            anomalies_logger.info(json.dumps(payload, separators=(",", ":")))
                        return

                # Apply update
                updated = False
                t_obj = self.trade_index.get(trade_id)
                if t_obj and t_obj.get("exit_price") is None:
                    size = t_obj.get("size")
                    entry_price = t_obj.get("entry_price")
                    # Derive entry_price if missing and cost_basis/size available
                    if entry_price is None and t_obj.get("cost_basis") is not None and size:
                        try:
                            entry_price = float(t_obj["cost_basis"]) / float(size)
                        except (TypeError, ZeroDivisionError):
                            entry_price = None
                    if entry_price is None:
                        system_logger.error(
                            "Missing entry_price for trade %s; cannot compute PnL/ROI (reason=%s)",
                            trade_id,
                            reason or exit_reason or "<unspecified>",
                        )
                        return
                    try:
                        timestamp = datetime.fromisoformat(t_obj["timestamp"])
                        if timestamp.tzinfo is None:
                            timestamp = timestamp.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError) as e:
                        system_logger.error(
                            "Invalid timestamp for trade %s: %s (error: %s)",
                            trade_id,
                            t_obj.get("timestamp"),
                            e,
                        )
                        return

                    # Use opposite order side for exit to account for shorts/longs
                    cur_side = t_obj.get("side")
                    norm_side = _normalize_side(
                        cur_side,
                        strategy=t_obj.get("strategy"),
                        roi=t_obj.get("roi"),
                    )
                    exit_side = "buy" if norm_side == "short" else "sell"

                    # Ensure critical metadata is populated before we persist updates.
                    if not isinstance(cur_side, str) or not cur_side.strip():
                        t_obj["side"] = norm_side
                    else:
                        t_obj["side"] = cur_side.strip().lower()
                    capital_buffer_value = t_obj.get("capital_buffer")
                    if not isinstance(capital_buffer_value, (int, float)):
                        capital_buffer_value = 0.25
                    t_obj["capital_buffer"] = round(float(capital_buffer_value), 6)

                    exit_adj, exit_slippage_amount, slip_rate = _apply_slippage(
                        t_obj.get("pair"),
                        float(exit_price),
                        exit_side,
                    )
                    canonical_reason, display_reason = _canonical_exit_reason(
                        exit_reason,
                        reason,
                    )
                    # Compute ROI and clamp/flag extreme values (> 500%)
                    if entry_price:
                        roi_val = round((exit_adj - entry_price) / entry_price, 6)
                    else:
                        roi_val = 0.0
                    roi_final = roi_val
                    roi_clamped = False
                    if abs(roi_val) > 5.0:
                        roi_clamped = True
                        roi_final = 5.0 if roi_val > 0 else -5.0
                        try:
                            anomalies_logger.info(
                                json.dumps(
                                    {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "type": "Extreme ROI",
                                        "trade_id": trade_id,
                                        "pair": t_obj.get("pair"),
                                        "roi_raw": roi_val,
                                        "roi_clamped": roi_final,
                                    },
                                    separators=(",", ":"),
                                )
                            )
                        except (TypeError, ValueError):
                            pass

                    t_obj.update(
                        {
                            "exit_price": exit_adj,
                            "status": "closed",
                            "reason": canonical_reason,
                            "exit_reason": canonical_reason,
                            "reason_display": display_reason,
                            "holding_period_days": round(
                                (datetime.now(timezone.utc) - timestamp).total_seconds() / 86400,
                                2,
                            ),
                            "realized_gain": round((exit_adj - entry_price) * size, 4),
                            "roi": roi_final,
                            "roi_was_clamped": roi_clamped,
                            "exit_slippage_rate": round(slip_rate, 6),
                            "exit_slippage_amount": exit_slippage_amount,
                        }
                    )
                    pre_update_balance = t_obj.get("account_balance")
                    pre_update_delta = t_obj.get("balance_delta")
                    entry_balance_snapshot = _safe_float(pre_update_balance)
                    self._apply_runtime_balance(
                        t_obj,
                        entry_balance=entry_balance_snapshot,
                    )
                    if t_obj.get("account_balance") is None and pre_update_balance is not None:
                        t_obj["account_balance"] = pre_update_balance
                    if (
                        entry_balance_snapshot is not None
                        and t_obj.get("balance_delta") is None
                        and t_obj.get("account_balance") is not None
                    ):
                        fallback_delta = _calculate_balance_delta(
                            entry_balance_snapshot,
                            t_obj.get("account_balance"),
                        )
                        if fallback_delta is not None:
                            t_obj["balance_delta"] = fallback_delta
                    if t_obj.get("balance_delta") is None and pre_update_delta is not None:
                        t_obj["balance_delta"] = pre_update_delta
                    if t_obj.get("account_balance") is None:
                        system_logger.warning(
                            "Trade %s closed without account_balance; review live balance configuration.",
                            trade_id,
                        )
                    elif entry_balance_snapshot is not None and t_obj.get("balance_delta") is None:
                        system_logger.warning(
                            "Trade %s closed without balance_delta (entry=%s exit=%s).",
                            trade_id,
                            entry_balance_snapshot,
                            t_obj.get("account_balance"),
                        )
                    # Revalidate updated trade; log anomaly but do not abort update flow
                    try:
                        validate_trade_schema(t_obj)
                    except (ValueError, TypeError) as e:
                        try:
                            anomalies_logger.info(
                                json.dumps(
                                    {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "type": "Trade Schema Error (post-update)",
                                        "error": str(e),
                                        "trade": t_obj,
                                    },
                                    separators=(",", ":"),
                                )
                            )
                        except (TypeError, ValueError):
                            pass
                    updated = True

                    # After update, if exit_price still missing/None, log anomaly but continue
                    if t_obj.get("exit_price") is None:
                        try:
                            anomalies_logger.info(
                                json.dumps(
                                    {
                                        "timestamp": datetime.now(timezone.utc).isoformat(),
                                        "type": "Missing Exit Price",
                                        "trade_id": trade_id,
                                        "pair": t_obj.get("pair"),
                                        "strategy": t_obj.get("strategy"),
                                        "reason": "exit_price is None or not logged",
                                        "confidence": t_obj.get("confidence"),
                                        "roi": t_obj.get("roi"),
                                        "regime": t_obj.get("regime"),
                                        "side": t_obj.get("side"),
                                    },
                                    separators=(",", ":"),
                                )
                            )
                        except (TypeError, ValueError):
                            pass

                if updated:
                    # Safely rewrite trades.log with exclusive lock held on source
                    temp_path = TRADES_LOG_PATH + ".tmp"
                    with open(TRADES_LOG_PATH, "r+", encoding="utf-8") as src:
                        fcntl.flock(src, fcntl.LOCK_EX)
                        try:
                            with open(temp_path, "w", encoding="utf-8") as f:
                                for t in trades:
                                    if t is not None:
                                        f.write(json.dumps(t) + "\n")
                                f.flush()
                                os.fsync(f.fileno())
                            os.replace(temp_path, TRADES_LOG_PATH)
                        finally:
                            fcntl.flock(src, fcntl.LOCK_UN)
                    self.trades = trades
                    self.trade_index = {}
                    for trade in trades:
                        if not trade:
                            continue
                        tid = trade.get("trade_id")
                        if tid:
                            self.trade_index[tid] = trade
                    system_logger.debug("Successfully updated trade %s in trades.log", trade_id)

                    # Safely resync positions.jsonl by removing the closed position
                    try:
                        if os.path.exists(POSITIONS_PATH):
                            tmp_pos = POSITIONS_PATH + ".tmp"
                            with open(POSITIONS_PATH, "r", encoding="utf-8") as src:
                                with open(tmp_pos, "w", encoding="utf-8") as dst:
                                    fcntl.flock(dst, fcntl.LOCK_EX)
                                    try:
                                        for line in src:
                                            try:
                                                obj = json.loads(line)
                                                if obj.get("trade_id") == trade_id:
                                                    # skip removed position
                                                    continue
                                            except json.JSONDecodeError:
                                                # preserve unparseable lines as-is
                                                pass
                                            dst.write(line)
                                        dst.flush()
                                        os.fsync(dst.fileno())
                                    finally:
                                        fcntl.flock(dst, fcntl.LOCK_UN)
                            os.replace(tmp_pos, POSITIONS_PATH)
                            system_logger.debug(
                                "Synchronized positions.jsonl — removed closed position %s",
                                trade_id,
                            )
                    except (OSError, IOError) as e:
                        system_logger.warning(
                            "Failed to sync positions.jsonl for %s: %s",
                            trade_id,
                            e,
                        )
                else:
                    # If trade is already closed, treat this as idempotent
                    existing = next(
                        (t for t in trades if t.get("trade_id") == trade_id),
                        None,
                    )
                    already_closed = existing and existing.get("exit_price") is not None
                    status_closed = existing and existing.get("status") == "closed"
                    if already_closed and status_closed:
                        system_logger.info(
                            "Idempotent update — trade %s already closed",
                            trade_id,
                        )
                    else:
                        system_logger.info(
                            "No update applied — trade_id %s not in open state",
                            trade_id,
                        )
                return

            except (OSError, IOError, json.JSONDecodeError) as e:
                system_logger.error("Error updating trade (attempt %s): %s", attempt + 1, e)
                time.sleep(1)

        system_logger.error("Failed to update trade %s after %s attempts", trade_id, max_retries)

    def reload_trades(self):
        """
        Reload all trades from the trades log into memory.
        Skips malformed or incomplete entries.
        Returns the list of trades.
        """
        self.trades = []
        self.txid_index = {}
        legacy_mutations = False
        if os.path.exists(TRADES_LOG_PATH):
            with open(TRADES_LOG_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        trade = json.loads(line)
                        if trade.get("trade_id"):
                            # Populate defaults for backward compatibility
                            trade.setdefault("txid", None)
                            trade.setdefault("gross_amount", None)
                            trade.setdefault("fee", None)
                            trade.setdefault("net_amount", None)
                            trade.setdefault("balance_delta", None)
                            trade.setdefault("account_balance", None)
                            trade.setdefault("reconciled", False)
                            trade.setdefault("pending_reconciliation", False)
                            trade.setdefault("source", "local")
                            trade.setdefault("pending_reconciliation_at", None)
                            tx_field = trade.get("txid")
                            if isinstance(tx_field, str):
                                trade["txid"] = [tx_field]
                            elif not isinstance(tx_field, list):
                                trade["txid"] = []
                            if _ensure_closed_trade_metadata(trade):
                                legacy_mutations = True
                            self.trades.append(trade)
                    except json.JSONDecodeError as e:
                        raw = line.strip()
                        redacted = (raw[:200] + "...") if len(raw) > 200 else raw
                        system_logger.warning(
                            "Skipping malformed line in trades.log: %s — Error: %s",
                            redacted,
                            e,
                        )
            self.trade_index = {}
            for trade in self.trades:
                if not trade:
                    continue
                trade_id = trade.get("trade_id")
                if trade_id:
                    self.trade_index[trade_id] = trade
                tx_values = trade.get("txid")
                if isinstance(tx_values, list):
                    for tx in tx_values:
                        if isinstance(tx, str):
                            self.txid_index[tx] = trade_id
            system_logger.info("Reloaded %s trades from trades.log", len(self.trades))
        else:
            system_logger.info("No trades.log file found.")
        if legacy_mutations:
            system_logger.info("Detected legacy trade entries missing metadata; rewriting trades.log.")
            self._rewrite_trades_file()
        latest_balance: Optional[float] = None
        for trade in self.trades:
            acct_value = _safe_float(trade.get("account_balance"))
            if acct_value is not None:
                latest_balance = acct_value
        self.account_balance = latest_balance
        return self.trades

    def _rewrite_trades_file(self) -> None:
        """Rewrite trades.log with current in-memory trades."""

        directory = os.path.dirname(TRADES_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(TRADES_LOG_PATH, "w", encoding="utf-8") as handle:
            for trade in self.trades:
                handle.write(json.dumps(trade, separators=(",", ":")) + "\n")

    def mark_pending_reconciliation(
        self,
        trade_id: str,
        *,
        pending: bool = True,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Toggle the pending reconciliation flag for ``trade_id``."""

        trade = self.trade_index.get(trade_id)
        if not trade:
            return
        trade["pending_reconciliation"] = pending
        if pending:
            trade["reconciled"] = False
            trade["pending_reconciliation_at"] = (timestamp or datetime.now(timezone.utc)).isoformat()
        if "account_balance" not in trade or trade["account_balance"] is None:
            trade["account_balance"] = _round_balance(self.account_balance)
        if not pending:
            trade["pending_reconciliation_at"] = None
        self._rewrite_trades_file()

    def mark_reconciled(self, trade_id: str) -> None:
        """Mark ``trade_id`` as reconciled and persist the ledger update."""

        trade = self.trade_index.get(trade_id)
        if not trade:
            return
        trade["reconciled"] = True
        trade["pending_reconciliation"] = False
        trade["pending_reconciliation_at"] = None
        if "account_balance" not in trade or trade["account_balance"] is None:
            trade["account_balance"] = _round_balance(self.account_balance)
        self._rewrite_trades_file()

    def find_trade_by_txid(self, txid: str) -> Optional[dict]:
        """Return the trade entry associated with a Kraken transaction id."""

        trade_id = self.txid_index.get(txid)
        if not trade_id:
            return None
        return self.trade_index.get(trade_id)

    def get_account_balance(self) -> float:
        """Return the latest known account balance tracked by the ledger."""

        if IS_LIVE:
            asset = (CONFIG.get("kraken", {}) or {}).get("balance_asset", "USDC")
            try:
                response = kraken_get_balance(asset)
            except Exception as exc:  # pylint: disable=broad-except
                system_logger.error("[BALANCE] Kraken balance fetch failed: %s", exc)
                raise RuntimeError("Failed to fetch live balance — aborting to prevent synthetic fallback.") from exc

            if not isinstance(response, dict):
                system_logger.error("[BALANCE] Kraken balance response malformed: %r", response)
                raise RuntimeError("Failed to fetch live balance — aborting to prevent synthetic fallback.")

            balance_value = response.get("balance")
            if balance_value is None:
                payload = response.get("result")
                if isinstance(payload, dict):
                    candidates = [
                        asset,
                        asset.upper(),
                        "USDC",
                        "USD",
                        "ZUSD",
                    ]
                    for key in candidates:
                        if key in payload and payload[key] is not None:
                            balance_value = payload[key]
                            break

            try:
                balance_float = float(balance_value)
            except (TypeError, ValueError):
                balance_float = None

            if balance_float is None or not math.isfinite(balance_float) or balance_float < 0:
                detail = response.get("error") or response.get("code") or "missing-balance"
                system_logger.error(
                    "[BALANCE] Kraken balance invalid asset=%s detail=%s payload=%s",
                    asset,
                    detail,
                    response,
                )
                raise RuntimeError("Failed to fetch live balance — aborting to prevent synthetic fallback.")

            system_logger.info("[BALANCE] Live Kraken balance fetched: $%.2f", balance_float)
            self.account_balance = balance_float
            self._balance_source = "kraken_live"
            return balance_float

        fallback = None
        paper_cfg = CONFIG.get("paper_mode", {}) or {}
        if "starting_balance" in paper_cfg:
            fallback = paper_cfg.get("starting_balance")
        if fallback is None:
            fallback = CONFIG.get("initial_capital")
        if fallback is None:
            fallback = 100_000.0

        try:
            tracked_balance = float(self.account_balance)
        except (TypeError, ValueError):
            tracked_balance = None
        if tracked_balance is not None and (tracked_balance != 0.0 or self.trades):
            system_logger.info("[BALANCE] Using tracked ledger balance: $%.2f", tracked_balance)
            self._balance_source = "paper_tracked"
            return tracked_balance

        try:
            balance_float = float(fallback)
        except (TypeError, ValueError):
            balance_float = 100_000.0

        system_logger.info("[BALANCE] Using simulated paper balance: $%.2f", balance_float)
        self.account_balance = balance_float
        self._balance_source = "paper_simulated"
        return balance_float

    def get_balance_source(self) -> str:
        """Return the most recent balance source identifier."""

        return getattr(self, "_balance_source", "unknown")

    def verify_trade_update(self, target_trade_id):
        """
        Verify if a trade with the given trade_id has been updated properly.
        Checks for required fields and status.
        Returns True if updated, False otherwise.
        """
        trades = self.trades
        matching = [t for t in trades if t.get("trade_id") == target_trade_id]
        if not matching:
            system_logger.info("No entries found for trade_id: %s", target_trade_id)
            return False
        latest = max(matching, key=lambda x: x.get("timestamp", "1970-01-01T00:00:00+00:00"))
        required_fields = ["status", "exit_price", "reason", "roi"]
        status_ok = latest.get("status") == "closed"
        fields_ok = all(latest.get(f) is not None for f in required_fields)
        if status_ok and fields_ok:
            system_logger.info("Verified update for trade %s", target_trade_id)
            return True
        system_logger.info("Trade found but not fully updated: %s", latest)
        return False
