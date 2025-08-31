#!/usr/bin/env python3
"""
Plot an equity curve and print quick stats from a backtest trades JSONL.

Inputs:
  --trades: path to a trades JSONL produced by backtest.py
  --start-equity: starting equity for compounding (default: 1000.0)
  --outdir: where to save PNG/CSV (default: logs/backtests)

The script looks for per‑trade returns in this order:
  - 'pnl_pct' (e.g., 0.01 for +1%),
  - 'return_pct' / 'ret' / 'r' (percent),
  - or absolute 'pnl' (added to equity directly).

Timestamps are taken from 'ts' / 'timestamp' / 'time' if available
(ISO8601, accepts trailing 'Z'); otherwise an index label is used.

If matplotlib is not available, it still writes the CSV and prints stats.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import List, Tuple

# Matplotlib is optional; we degrade gracefully if unavailable.
try:
    import matplotlib.pyplot as plt  # type: ignore[import]  # mypy / pylance

    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False
    plt = None  # type: ignore


def _parse_time(raw: str | None, index: int) -> str:
    """Return a readable UTC timestamp string or an index label if parsing fails."""
    if raw is None or raw == "":
        return f"idx_{index:05d}"

    text = raw
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        dt = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return f"idx_{index:05d}"

    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _extract_return(tr: dict) -> Tuple[float, bool]:
    """Return (value, is_percent). If is_percent is False, value is absolute PnL."""
    for key in ("pnl_pct", "return_pct", "ret", "r"):
        if key in tr:
            try:
                return float(tr[key]), True
            except (ValueError, TypeError):
                # fall through to try other fields
                pass

    if "pnl" in tr:
        try:
            return float(tr["pnl"]), False
        except (ValueError, TypeError):
            pass

    # default to 0% return if nothing usable
    return 0.0, True


def load_trades(path: str) -> List[dict]:
    """Load trades from a JSONL file, skipping malformed lines."""
    trades: List[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            try:
                trades.append(json.loads(text))
            except json.JSONDecodeError:
                # skip malformed lines quietly
                continue
    return trades


def build_equity(
    trades: List[dict],
    start_equity: float,
) -> Tuple[List[str], List[float], int, int]:
    """Build equity curve; return (times, equity_values, wins, losses)."""
    equity = start_equity
    times: List[str] = []
    curve: List[float] = []
    wins = 0
    losses = 0

    for idx, tr in enumerate(trades):
        ts = tr.get("ts") or tr.get("timestamp") or tr.get("time")
        label = _parse_time(str(ts) if ts is not None else None, idx)

        rv, is_pct = _extract_return(tr)
        if is_pct:
            equity *= 1.0 + rv
        else:
            equity += rv

        if rv > 0:
            wins += 1
        elif rv < 0:
            losses += 1

        times.append(label)
        curve.append(equity)

    return times, curve, wins, losses


def _save_csv(csv_path: str, times: List[str], equity: List[float]) -> None:
    """Save equity series to CSV (idx,time,equity)."""
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("idx,time,equity\n")
        for i, (t, e) in enumerate(zip(times, equity, strict=True)):
            fh.write(f"{i},{t},{e:.6f}\n")


def _save_plot(png_path: str, equity: List[float]) -> None:
    """Save a simple equity curve plot if matplotlib is available."""
    if not HAVE_MPL:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(equity)
    plt.title("Equity Curve")
    plt.xlabel("Trade #")
    plt.ylabel("Equity")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close()


def main() -> int:
    """Entry point: parse args, load trades, compute curve, write outputs."""
    parser = argparse.ArgumentParser(description="Plot & analyze backtest trades JSONL")
    parser.add_argument("--trades", required=True, help="Path to *_trades.jsonl")
    parser.add_argument(
        "--start-equity",
        type=float,
        default=1000.0,
        help="Starting equity for compounding (default: 1000.0)",
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join("logs", "backtests"),
        help="Directory to write CSV/PNG outputs (default: logs/backtests)",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    trades = load_trades(args.trades)
    if not trades:
        print(f"[WARN] No trades read from {args.trades}")
        return 1

    times, equity, wins, losses = build_equity(trades, args.start_equity)
    total = wins + losses
    final_eq = equity[-1]
    ret_pct = (
        0.0 if args.start_equity == 0 else (final_eq / args.start_equity - 1.0) * 100.0
    )
    win_rate = 0.0 if total == 0 else wins / total

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    png_path = os.path.join(args.outdir, f"equity_{stamp}.png")
    csv_path = os.path.join(args.outdir, f"equity_{stamp}.csv")

    _save_csv(csv_path, times, equity)
    _save_plot(png_path, equity)

    print("✅ Plot & analysis")
    print(
        f"Trades: {total} | Wins: {wins} | Losses: {losses} | Win rate: {win_rate:.2%}"
    )
    print(
        f"Start equity: {args.start_equity:.2f} → Final equity: {final_eq:.2f}  "
        f"(Return: {ret_pct:.2f}%)"
    )
    if HAVE_MPL:
        print(f"PNG: {png_path}")
    else:
        print("[INFO] matplotlib not installed; no PNG written.")
    print(f"CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
