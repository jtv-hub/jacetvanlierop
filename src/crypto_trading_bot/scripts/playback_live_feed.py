#!/usr/bin/env python3
"""
playback_live_feed.py

Replay a recorded JSONL live feed (from scripts/record_live_feed.py), compute RSI from the 'last'
price column, and simulate a simple RSI-style strategy with TP/SL/max-hold constraints.

Outputs trades to logs/live_playback_trades.jsonl by default, using the same open/close JSONL
schema as your paper trader.

Example
-------
python scripts/playback_live_feed.py \
  --file data/live/kraken_BTCUSD_20250812.jsonl \
  --size 100 --rsi-threshold 40 --tp 0.003 --sl 0.003 --max-hold 5 --debug
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# ---------- utils ----------


def utc_now_str() -> str:
    """Return current time in UTC as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def pct_change(a: float, b: float) -> float:
    """Return signed percentage change from a (entry) to b (exit). Example: +0.0123 for +1.23%."""
    if a == 0:
        return 0.0
    return (b - a) / a


def ensure_file_parent(path: Path) -> None:
    """Ensure the parent directory of a path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def log_event(path: Path, obj: Dict[str, Any]) -> None:
    """Append a JSON object as one line to the given file path."""
    ensure_file_parent(path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, separators=(",", ":")) + "\n")


# ---------- RSI calc ----------


def calc_rsi(values: List[float], period: int = 14) -> Optional[float]:
    """Classic Wilder RSI from a trailing window. Returns None until enough history is present."""
    if len(values) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(-period, 0):
        diff = values[i] - values[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff  # diff is negative, accumulate as positive loss
    if losses == 0:
        return 100.0
    rs = gains / losses
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


# ---------- config & state ----------


@dataclass
class PlaybackConfig:
    """Configuration for a playback simulation run."""

    file: Path
    size: float
    rsi_threshold: float
    tp: float
    sl: float
    max_hold: int
    rsi_period: int
    log_file: Path
    debug: bool


@dataclass
class Position:
    """Represents an open long position in the simulator."""

    entry_price: float
    bars_held: int = 0


@dataclass
class SimState:
    """Simulation state that accrues during playback."""

    pos: Optional[Position] = None
    prices: List[float] = field(default_factory=list)
    trades: int = 0
    wins: int = 0
    losses: int = 0


# ---------- core sim ----------


def open_long(cfg: PlaybackConfig, state: SimState, price: float) -> None:
    """Open a long position and write an 'open' event to the log."""
    state.pos = Position(entry_price=price)
    state.trades += 1
    if cfg.debug:
        print(f"Opened BUY at {price:.2f}")

    log_event(
        cfg.log_file,
        {
            "type": "open",
            "timestamp": utc_now_str(),
            "pair": infer_pair_from_path(cfg.file) or "UNKNOWN",
            "side": "BUY",
            "size": cfg.size,
            "price": price,
            "strategy": "PlaybackRSI",
            "note": f"rsi_th={cfg.rsi_threshold}",
        },
    )


def close_long(cfg: PlaybackConfig, state: SimState, price: float, reason: str) -> None:
    """Close the current long position (if any) and write a 'close' event to the log."""
    assert state.pos is not None
    entry = state.pos.entry_price
    chg = pct_change(entry, price)

    # win/loss bookkeeping for the summary print
    if chg >= 0:
        state.wins += 1
    else:
        state.losses += 1

    if cfg.debug:
        pct = chg * 100.0
        print(f"Closed ({reason}) at {price:.2f} | entry={entry:.2f} change={pct:+.2f}%")

    log_event(
        cfg.log_file,
        {
            "type": "close",
            "timestamp": utc_now_str(),
            "pair": infer_pair_from_path(cfg.file) or "UNKNOWN",
            "side": "SELL",
            "size": cfg.size,
            "price": price,
            "strategy": "PlaybackRSI",
            "reason": reason,
            "note": "",
        },
    )

    state.pos = None


def tick(cfg: PlaybackConfig, state: SimState, price: float, rsi: Optional[float]) -> None:
    """
    Process one bar:
    - If flat and RSI < threshold -> BUY
    - If long -> check TP/SL/max-hold; otherwise hold
    """
    # entry
    if state.pos is None and rsi is not None and rsi < cfg.rsi_threshold:
        open_long(cfg, state, price)
        return

    # management / exit
    if state.pos is not None:
        pos = state.pos
        pos.bars_held += 1

        # TP
        if cfg.tp > 0 and (price >= pos.entry_price * (1.0 + cfg.tp)):
            close_long(cfg, state, price, "TP")
            return

        # SL
        if cfg.sl > 0 and (price <= pos.entry_price * (1.0 - cfg.sl)):
            close_long(cfg, state, price, "SL")
            return

        # max_hold
        if cfg.max_hold > 0 and pos.bars_held >= cfg.max_hold:
            close_long(cfg, state, price, "MAX_HOLD")
            return


def infer_pair_from_path(p: Path) -> Optional[str]:
    """
    Infer pair from filename like 'kraken_BTCUSD_20250812.jsonl' -> 'BTC/USD'.

    Returns:
        The inferred pair (e.g., 'BTC/USD') or None if it cannot be inferred.
    """
    name = p.name
    try:
        middle = name.split("_")[1]  # may raise IndexError
        # Insert slash before last 3 chars (simple heuristic for USD/USDT-like suffixes).
        if len(middle) > 3:
            return f"{middle[:-3]}/{middle[-3:]}"
        return middle
    except (IndexError, AttributeError):
        # Be explicit: only catch errors related to filename structure, not all exceptions.
        return None


def playback(cfg: PlaybackConfig) -> None:
    """Run the full playback loop and print a brief summary."""
    print(f"▶️  Playback file: {cfg.file}")
    print(f"    Params: rsi_th={cfg.rsi_threshold} " f"tp={cfg.tp} sl={cfg.sl} max_hold={cfg.max_hold} size={cfg.size}")
    print(f"    Logging trades → {cfg.log_file}")

    ensure_file_parent(cfg.log_file)

    # read prices
    state = SimState()
    with cfg.file.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            price = rec.get("last")
            if price is None or not isinstance(price, (int, float)) or math.isnan(price):
                continue

            state.prices.append(float(price))
            rsi = calc_rsi(state.prices, period=cfg.rsi_period)

            if cfg.debug:
                if rsi is None:
                    print(f"price={price:.2f} rsi=NA")
                else:
                    print(f"price={price:.2f} rsi={rsi:.1f}")

            tick(cfg, state, float(price), rsi)

    # End-of-run: flatten any open position
    if state.pos is not None:
        close_long(cfg, state, state.prices[-1], "EOR")

    # summary
    total = state.wins + state.losses
    win_rate = (state.wins / total * 100.0) if total > 0 else 0.0
    print(
        "\n✅ Playback done."
        f"\nTrades: {state.trades} | Wins: {state.wins} | Losses: {state.losses} "
        f"| Win rate: {win_rate:.1f}%"
    )


# ---------- CLI ----------


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for playback."""
    p = argparse.ArgumentParser(description="Playback JSONL live feed and simulate an RSI strategy.")
    p.add_argument(
        "--file",
        required=True,
        help="Path to JSONL captured by record_live_feed.py",
    )
    p.add_argument(
        "--size",
        type=float,
        default=100.0,
        help="Trade size (paper) (default: %(default)s)",
    )
    p.add_argument(
        "--rsi-threshold",
        type=float,
        default=40.0,
        help="RSI buy threshold (default: %(default)s)",
    )
    p.add_argument(
        "--tp",
        type=float,
        default=0.003,
        help="Take profit as fraction (e.g., 0.003 = 0.3%%)",
    )
    p.add_argument(
        "--sl",
        type=float,
        default=0.003,
        help="Stop loss as fraction (e.g., 0.003 = 0.3%%)",
    )
    p.add_argument(
        "--max-hold",
        type=int,
        default=20,
        help="Max bars to hold a position (0=disable)",
    )
    p.add_argument(
        "--rsi-period",
        type=int,
        default=14,
        help="RSI lookback (default: %(default)s)",
    )
    p.add_argument(
        "--log-file",
        default="logs/live_playback_trades.jsonl",
        help="Output trade log (default: %(default)s)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Verbose per-bar prints",
    )
    return p.parse_args(argv)


def main() -> None:
    """Entry point for CLI usage."""
    args = parse_args()
    cfg = PlaybackConfig(
        file=Path(args.file),
        size=args.size,
        rsi_threshold=args.rsi_threshold,
        tp=args.tp,
        sl=args.sl,
        max_hold=args.max_hold,
        rsi_period=args.rsi_period,
        log_file=Path(args.log_file),
        debug=bool(args.debug),
    )
    playback(cfg)


if __name__ == "__main__":
    main()
