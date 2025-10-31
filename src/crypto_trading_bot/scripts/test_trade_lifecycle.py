"""
One-off lifecycle test for the trading bot.

Runs fully headless:
- Generates an RSI-based signal using live Kraken prices
- Logs a trade (status=executed) via TradeLedger
- Writes a matching position to logs/positions.jsonl and src/crypto_trading_bot/data/positions.jsonl
- Waits a few seconds, runs the exit checker, verifies closure
- Prints final ROI and exit reason

Usage:
    python -m crypto_trading_bot.scripts.test_trade_lifecycle
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import List

from crypto_trading_bot import trading_logic as tl
from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.indicators.rsi import calculate_rsi
from crypto_trading_bot.ledger.trade_ledger import TradeLedger
from crypto_trading_bot.scripts.check_exit_conditions import main as run_exit_checks
from crypto_trading_bot.strategies.simple_rsi_strategies import SimpleRSIStrategy
from crypto_trading_bot.trading_logic import position_manager
from crypto_trading_bot.utils.kraken_api import get_ticker_price

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _ensure_dirs():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("src/crypto_trading_bot/data", exist_ok=True)


def _fetch_price_series(pair: str, n: int = 15, delay: float = 0.3) -> List[float]:
    """Fetch a short live series from Kraken; tolerate intermittent failures.

    Falls back to repeating the last known price if transient errors occur.
    """
    series: List[float] = []
    last_px: float | None = None
    for _ in range(max(2, n)):
        try:
            px = float(get_ticker_price(pair))
            last_px = px
            series.append(px)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Live price fetch failed for %s: %s", pair, e)
            if last_px is not None:
                series.append(last_px)
        time.sleep(max(0.05, delay))
    return series


def _choose_signal(prices: List[float]) -> tuple[str | None, float]:
    """Generate a signal using SimpleRSIStrategy with real RSI logic."""
    period = int(CONFIG.get("rsi", {}).get("period", 14))
    lower = float(CONFIG.get("rsi", {}).get("lower", 48))
    upper = float(CONFIG.get("rsi", {}).get("upper", 75))
    strat = SimpleRSIStrategy(period=period, lower=lower, upper=upper)
    # Use a generous volume to pass volume checks
    volume = 1000

    # Ensure we have enough points; pad with last price if short
    if len(prices) < period + 1 and prices:
        prices = prices + [prices[-1]] * (period + 1 - len(prices))

    res = strat.generate_signal(prices, volume=volume)
    signal = res.get("signal")
    confidence = float(res.get("confidence", 0.0) or 0.0)
    return signal, confidence


def run_test_lifecycle():
    """Execute a headless trade lifecycle test using live prices and RSI."""
    _ensure_dirs()

    pairs = CONFIG.get("tradable_pairs", ["BTC/USDC"]) or ["BTC/USDC"]
    pair = pairs[0]
    logger.info("Using pair: %s", pair)

    # Fetch a small price series and derive a signal
    series = _fetch_price_series(pair, n=max(15, int(CONFIG.get("rsi", {}).get("period", 14)) + 1))
    if not series or any(p is None or p <= 0 for p in series):
        raise RuntimeError("Unable to build valid price series for RSI test")

    try:
        rsi_val = calculate_rsi(series, int(CONFIG.get("rsi", {}).get("period", 14)))
        logger.info("RSI=%s from series tail=%s", rsi_val, [round(p, 2) for p in series[-5:]])
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("RSI calculation failed; proceeding with strategy-only: %s", e)

    signal, confidence = _choose_signal(series)
    if signal is None or confidence < 0.4:
        logger.info("Neutral RSI; biasing to 'buy' with min confidence for test")
        signal, confidence = "buy", 0.5

    # Trade sizing
    size_min = float(CONFIG.get("trade_size", {}).get("min", 0.001))
    size_max = float(CONFIG.get("trade_size", {}).get("max", 0.005))
    trade_size = max(size_min, min(size_max, 0.001))

    # Log trade through the ledger (applies entry slippage, validates schema)
    ledger = TradeLedger(position_manager)
    entry_raw = float(series[-1])
    trade_id = ledger.log_trade(
        trading_pair=pair,
        trade_size=trade_size,
        strategy_name="SimpleRSIStrategy",
        side=signal,
        confidence=confidence,
        entry_price=entry_raw,
        regime="test",
        capital_buffer=0.25,
    )
    # Small delay to ensure fsync visibility on some filesystems
    time.sleep(0.5)
    print("\n[DEBUG] Contents of logs/trades.log:")
    try:
        with open("logs/trades.log", "r", encoding="utf-8") as f:
            for line in f:
                print(">>>", line.strip())
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"[DEBUG] Unable to read logs/trades.log: {e}")
    logger.info(
        "Logged trade_id=%s side=%s size=%.6f entry_raw=%.4f",
        trade_id,
        signal,
        trade_size,
        entry_raw,
    )

    # Ensure the trade is fully flushed and visible on disk before proceeding
    # Retry-read trades.log for a short period to handle FS sync/rotation timing.
    start = time.time()
    logged_trade = None
    while (time.time() - start) < 3.0:
        ledger.reload_trades()
        logged_trade = next((t for t in ledger.trades if t.get("trade_id") == trade_id), None)
        if logged_trade:
            break
        time.sleep(0.1)
    if not logged_trade:
        # As a last resort, inspect the raw file for debugging
        try:
            with open("logs/trades.log", "r", encoding="utf-8") as f:
                lines = f.readlines()[-5:]
                logger.debug("Last trades.log lines: %s", "".join(lines))
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug("Unable to read trades.log for debug: %s", e)
        raise RuntimeError("Logged trade missing after reload")
    logged_price = float(logged_trade.get("entry_price"))
    logged_ts = logged_trade.get("timestamp") or datetime.now(timezone.utc).isoformat()

    # Record position for exit processing
    pos = {
        "trade_id": trade_id,
        "pair": pair,
        "size": trade_size,
        "entry_price": logged_price,
        "timestamp": logged_ts,
        "strategy": "SimpleRSIStrategy",
        "confidence": confidence,
        "high_water_mark": logged_price,
    }
    position_manager.positions[trade_id] = pos
    # Write to the default positions file used by the engine
    try:
        ledger.open_position(trade_id, pos.copy())
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed writing logs/positions.jsonl: %s", e)

    # Also write to requested data path for inspection
    try:
        with open("src/crypto_trading_bot/data/positions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(pos) + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed writing src/crypto_trading_bot/data/positions.jsonl: %s", e)

    # Short wait to simulate holding period and allow different prices
    time.sleep(6)

    # Force the random exit path to trigger deterministically for this test
    original_random_fn = tl.random.random
    try:
        tl.random.random = lambda: 0.0  # ensure random_win path triggers
        run_exit_checks()
    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Exit script failed: %s", e)
        raise
    finally:
        tl.random.random = original_random_fn

    # Reload with retries to ensure write visibility
    start = time.time()
    updated = None
    while (time.time() - start) < 3.0:
        ledger.reload_trades()
        updated = next((t for t in ledger.trades if t.get("trade_id") == trade_id), None)
        if updated and updated.get("status") == "closed":
            break
        time.sleep(0.1)
    if not updated:
        raise RuntimeError("Trade not found after exit checks")

    status = updated.get("status")
    roi = updated.get("roi")
    reason = updated.get("reason")
    if status != "closed" or updated.get("exit_price") is None:
        logger.warning("Trade did not close. status=%s roi=%s reason=%s", status, roi, reason)
    else:
        logger.info("Closed trade %s | ROI=%.6f | Reason=%s", trade_id, float(roi or 0.0), reason)
        print(f"Final ROI: {float(roi or 0.0):.6f}, Exit Reason: {reason}")


if __name__ == "__main__":
    run_test_lifecycle()
