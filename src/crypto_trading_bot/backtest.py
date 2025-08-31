#!/usr/bin/env python3
"""
Backtest runner (baseline).

- Auto-detects price from JSONL (common fields: price/last/close/bid/ask/c)
- Computes RSI (Wilder) on closing price
- Simple long-only strategy:
    • Enter long when RSI < (100 - rsi_th)
    • Exit on take-profit, stop-loss, or max_hold bars
- Outputs:
    • logs/backtests/<stamp>_trades.jsonl  (each fill/close)
    • logs/backtests/<stamp>_summary.json  (metrics)
    • logs/backtests/backtest_summary.csv  (append one-line summary)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List, Optional, Tuple


def _ensure_dirs() -> str:
    """Ensure output directories exist and return logs/backtests path."""
    out_dir = os.path.join("logs", "backtests")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _parse_price(obj: dict) -> Optional[float]:
    """
    Extract a price from a JSON object using a best-effort field order.
    Supports string numbers as well.
    """
    candidates = [
        "price",
        "last",
        "close",
        "bid",
        "ask",
        "c",  # some feeds use 'c' for close
    ]
    for key in candidates:
        if key in obj:
            val = obj[key]
            try:
                return float(val)
            except (TypeError, ValueError):
                return None
    return None


def _iter_prices(path: str) -> Iterable[Tuple[str, float]]:
    """
    Yield (timestamp_iso, price) from a JSONL file.
    If no timestamp is present, synthesize a monotonic UTC timestamp.
    """
    synth_idx = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            price = _parse_price(obj)
            if price is None or math.isnan(price) or price <= 0:
                continue

            # try common time keys; fall back to synthetic, stable order
            ts = (
                obj.get("ts") or obj.get("timestamp") or obj.get("time") or obj.get("t")
            )
            if not isinstance(ts, str):
                ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                # make deterministic order if called twice
                ts = f"{ts[:-6]}Z+{synth_idx:06d}"
                synth_idx += 1

            yield ts, float(price)


def _rsi_wilder(closes: List[float], period: int) -> List[Optional[float]]:
    """
    Compute RSI (Wilder) for a series; returns list aligned with closes.
    First (period) entries are None until seed is available.
    """
    if period <= 0:
        raise ValueError("RSI period must be > 0")

    rsi: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return rsi

    gains: List[float] = []
    losses: List[float] = []

    # seed
    for i in range(1, period + 1):
        change = closes[i] - closes[i - 1]
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def rs_to_rsi(gn: float, ls: float) -> float:
        if ls == 0:
            return 100.0
        rs = gn / ls
        return 100.0 - (100.0 / (1.0 + rs))

    rsi[period] = rs_to_rsi(avg_gain, avg_loss)

    # Wilder smoothing
    for i in range(period + 1, len(closes)):
        change = closes[i] - closes[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        rsi[i] = rs_to_rsi(avg_gain, avg_loss)

    return rsi


@dataclass
class Params:
    """Tunable parameters for the baseline RSI backtest."""

    rsi_th: float
    tp: float
    sl: float
    max_hold: int
    rsi_period: int
    size: float


def _backtest(
    prices: List[float],
    times: List[str],
    p: Params,
) -> Tuple[List[dict], dict]:
    """
    Run a simple long-only RSI strategy; return (trades, summary).
    Entry: RSI < (100 - rsi_th)
    Exit:  take-profit OR stop-loss OR max_hold
    """
    rsi = _rsi_wilder(prices, p.rsi_period)
    trades: List[dict] = []

    in_pos = False
    entry_idx = -1
    entry_price = 0.0

    # Use enumerate to avoid range(len(...)) and index lookups
    for i, price in enumerate(prices):
        if rsi[i] is None:
            continue

        if not in_pos and rsi[i] < (100.0 - p.rsi_th):
            in_pos = True
            entry_idx = i
            entry_price = price
            trades.append(
                {
                    "ts": times[i],
                    "side": "BUY",
                    "price": entry_price,
                    "size": p.size,
                    "note": "entry",
                }
            )
            continue

        if in_pos:
            cur = price
            change = (cur - entry_price) / entry_price
            held = i - entry_idx

            exit_reason = None
            if change >= p.tp:
                exit_reason = "TP"
            elif change <= -p.sl:
                exit_reason = "SL"
            elif held >= p.max_hold:
                exit_reason = "MAX_HOLD"

            if exit_reason:
                trades.append(
                    {
                        "ts": times[i],
                        "side": "SELL",
                        "price": cur,
                        "size": p.size,
                        "note": f"exit:{exit_reason}",
                        "pnl_pct": round(change * 100.0, 4),
                    }
                )
                in_pos = False
                entry_idx = -1
                entry_price = 0.0

    wins = 0
    losses = 0
    pnls: List[float] = []
    for t in trades:
        if t.get("side") == "SELL":
            pnl = float(t.get("pnl_pct", 0.0))
            pnls.append(pnl)
            if pnl >= 0:
                wins += 1
            else:
                losses += 1

    win_rate = (wins / max(1, wins + losses)) if (wins + losses) else 0.0
    expectancy = sum(pnls) / max(1, len(pnls))

    summary = {
        "count_trades": len([t for t in trades if t["side"] == "SELL"]),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "expectancy_pct": round(expectancy, 4),
        "rsi_period": p.rsi_period,
        "rsi_th": p.rsi_th,
        "tp": p.tp,
        "sl": p.sl,
        "max_hold": p.max_hold,
        "size": p.size,
    }
    return trades, summary


def main() -> int:
    """CLI entry-point."""
    ap = argparse.ArgumentParser(description="Baseline RSI backtester")
    ap.add_argument("--file", required=True, help="Input JSONL price file")
    ap.add_argument("--rsi-period", type=int, default=14)
    ap.add_argument("--rsi-th", type=float, default=60.0)
    ap.add_argument("--tp", type=float, default=0.002)
    ap.add_argument("--sl", type=float, default=0.003)
    ap.add_argument("--max-hold", type=int, default=10)
    ap.add_argument("--size", type=float, default=100.0)
    args = ap.parse_args()

    # Load prices
    times: List[str] = []
    prices: List[float] = []
    for ts, px in _iter_prices(args.file):
        times.append(ts)
        prices.append(px)

    if len(prices) < args.rsi_period + 2:
        print(
            f"[WARN] Not enough data ({len(prices)}) for RSI period {args.rsi_period}.",
        )
        return 0

    params = Params(
        rsi_th=args.rsi_th,
        tp=args.tp,
        sl=args.sl,
        max_hold=args.max_hold,
        rsi_period=args.rsi_period,
        size=args.size,
    )

    trades, summary = _backtest(prices, times, params)

    # Outputs
    out_dir = _ensure_dirs()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    base = os.path.join(out_dir, f"bt_{stamp}")

    trades_path = f"{base}_trades.jsonl"
    with open(trades_path, "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    summary_path = f"{base}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Append to rolling CSV
    csv_path = os.path.join(out_dir, "backtest_summary.csv")
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(
                [
                    "ts_utc",
                    "file",
                    "rsi_period",
                    "rsi_th",
                    "tp",
                    "sl",
                    "max_hold",
                    "size",
                    "trades",
                    "wins",
                    "losses",
                    "win_rate",
                    "expectancy_pct",
                ]
            )
        w.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                os.path.relpath(args.file, start=os.getcwd()),
                summary["rsi_period"],
                summary["rsi_th"],
                summary["tp"],
                summary["sl"],
                summary["max_hold"],
                summary["size"],
                summary["count_trades"],
                summary["wins"],
                summary["losses"],
                summary["win_rate"],
                summary["expectancy_pct"],
            ]
        )

    print(
        f"✅ Backtest done. Trades: {summary['count_trades']} | "
        f"Win rate: {summary['win_rate']*100:.1f}% | "
        f"Expectancy: {summary['expectancy_pct']:.3f}%"
    )
    print(f"   → {trades_path}")
    print(f"   → {summary_path}")
    print(f"   ↪ appended: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
