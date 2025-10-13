"""Show live trading performance summary in the terminal.

Reads compact JSONL trades from logs/trades.log and prints summary metrics.

Usage:
    python scripts/show_stats.py [--json]

Notes:
    - Only closed trades with a numeric ROI are used for performance metrics
    - Positions count is derived from logs/positions.jsonl (if present)
    - Uses tabulate if installed; otherwise falls back to simple prints
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

LOG_DIR = "logs"
TRADES_LOG = os.path.join(LOG_DIR, "trades.log")
POSITIONS_LOG = os.path.join(LOG_DIR, "positions.jsonl")
LEGACY_EXIT_REASON_MAP = {
    "STOP_LOSS": "sl_triggered",
    "STOPLOSS": "sl_triggered",
    "TRAILING_STOP": "trailing_exit",
    "TRAIL_STOP": "trailing_exit",
    "TAKE_PROFIT": "tp_hit",
    "TAKEPROFIT": "tp_hit",
    "RSI_EXIT": "indicator_exit",
    "RSI": "indicator_exit",
    "MAX_HOLD": "max_hold_expired",
    "MAX HOLD": "max_hold_expired",
}
EXIT_REASON_LABELS = {
    "sl_triggered": "Stop Loss",
    "tp_hit": "Take Profit",
    "indicator_exit": "Indicator Exit",
    "trailing_exit": "Trailing Stop",
    "max_hold_expired": "Max Hold",
    "unknown": "Unknown",
}

# Optional pretty table support
try:  # pragma: no cover - optional dependency
    from tabulate import tabulate  # type: ignore
except ImportError:  # pragma: no cover
    tabulate = None  # type: ignore


def read_jsonl(path: str) -> List[str]:
    """Return file lines for a JSONL file or an empty list if missing."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def parse_closed_trades(lines: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Return (closed_trades, warnings). Skips malformed or invalid rows."""
    closed: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, ln in enumerate(lines, start=1):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError as e:
            warnings.append(f"Malformed JSON (L{i}): {e}")
            continue
        if obj.get("status") != "closed":
            continue
        roi = obj.get("roi")
        try:
            roi_val = float(roi)
        except (TypeError, ValueError):
            warnings.append(f"Missing/invalid ROI (L{i}); skipping trade_id={obj.get('trade_id')}")
            continue
        obj["roi"] = roi_val
        closed.append(obj)
    return closed, warnings


def to_dt_iso(s: str | None) -> datetime | None:
    """Parse ISO timestamp to datetime, supporting "Z" suffix. Returns None on error."""
    if not s or not isinstance(s, str):
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def compute_metrics(closed: List[Dict[str, Any]], positions_count: int) -> Dict[str, Any]:
    """Compute summary stats from closed trades and count of active positions."""
    if not closed:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_roi_pct": 0.0,
            "total_pl_usd": 0.0,
            "final_equity": 1000.0,
            "max_drawdown_pct": 0.0,
            "exit_reason_counts": {},
            "active_positions": positions_count,
        }

    # Sort by timestamp
    closed_sorted = sorted(
        closed,
        key=lambda t: (to_dt_iso(t.get("timestamp")) or datetime.fromtimestamp(0, tz=timezone.utc)),
    )

    rois = [float(t.get("roi", 0.0)) for t in closed_sorted]
    wins = sum(1 for r in rois if r > 0)
    total = len(rois)
    win_rate = (wins / total) if total else 0.0
    avg_roi = sum(rois) / total if total else 0.0

    # Equity curve starting from $1000
    equity = 1000.0
    curve: List[float] = []
    for r in rois:
        equity *= 1.0 + r
        curve.append(equity)
    final_equity = equity
    total_pl = final_equity - 1000.0

    # Max drawdown (peak-to-trough)
    peak = -1.0
    max_dd = 0.0
    for val in curve:
        if val > peak:
            peak = val
        drawdown = (val - peak) / peak if peak > 0 else 0.0
        if drawdown < max_dd:
            max_dd = drawdown
    max_drawdown_pct = abs(max_dd)

    # Exit reason counts (canonical snake_case, legacy-friendly)
    reasons: List[str] = []
    for t in closed_sorted:
        candidates = [
            t.get("exit_reason"),
            t.get("reason"),
            t.get("reason_display"),
        ]
        canonical: str | None = None
        for candidate in candidates:
            text = str(candidate or "").strip()
            if not text:
                continue
            lowered = text.lower().replace(" ", "_")
            if lowered in EXIT_REASON_LABELS:
                canonical = lowered
                break
            mapped = LEGACY_EXIT_REASON_MAP.get(text.upper())
            if mapped:
                canonical = mapped
                break
        reasons.append(canonical or "unknown")
    reason_counts = dict(Counter(reasons).most_common())

    return {
        "total_trades": total,
        "win_rate": round(win_rate * 100, 2),  # percent
        "avg_roi_pct": round(avg_roi * 100, 2),
        "total_pl_usd": round(total_pl, 2),
        "final_equity": round(final_equity, 2),
        "max_drawdown_pct": round(max_drawdown_pct * 100, 2),
        "exit_reason_counts": reason_counts,
        "active_positions": positions_count,
    }


def count_positions(path: str) -> int:
    """Return number of non-empty lines in positions file (0 if missing)."""
    if not os.path.exists(path):
        return 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for ln in f if ln.strip())
    except OSError:
        return 0


def print_table(stats: Dict[str, Any], warnings: List[str]):
    """Pretty-print the summary table and details, with optional tabulate."""
    rows = [
        ["✅ Total Trades", stats["total_trades"]],
        ["🟢 Win Rate (%)", stats["win_rate"]],
        ["📈 Average ROI (%)", stats["avg_roi_pct"]],
        ["💰 Total P/L ($)", stats["total_pl_usd"]],
        ["🏦 Final Equity ($)", stats["final_equity"]],
        ["📉 Max Drawdown (%)", stats["max_drawdown_pct"]],
        ["🔄 Active Positions", stats["active_positions"]],
    ]
    if tabulate:  # type: ignore[truthy-function]
        print(
            tabulate(  # type: ignore[misc]
                rows,
                headers=["Metric", "Value"],
                tablefmt="github",
                colalign=("left", "right"),
            )
        )
    else:
        print("\n=== Trading Performance Summary ===")
        for k, v in rows:
            print(f"{k:<24} {v:>10}")

    # Exit reasons
    rc = stats.get("exit_reason_counts") or {}
    if rc:
        if tabulate:  # type: ignore[truthy-function]
            print("\n📌 Exit Reasons")
            print(
                tabulate(  # type: ignore[misc]
                    [[EXIT_REASON_LABELS.get(k, k), v] for k, v in rc.items()],
                    headers=["Reason", "Count"],
                    tablefmt="github",
                )
            )
        else:
            print("\n📌 Exit Reasons")
            for k, v in rc.items():
                label = EXIT_REASON_LABELS.get(k, k)
                print(f"- {label}: {v}")

    # Warnings
    for w in warnings:
        print(f"⚠️  {w}")


def main():
    """Entry point: load logs, compute stats, and print (or JSON)."""
    parser = argparse.ArgumentParser(
        description="Show live trading performance summary",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw stats as JSON",
    )
    args = parser.parse_args()

    trade_lines = read_jsonl(TRADES_LOG)
    if not trade_lines:
        print("⚠️  No trades.log found or file is empty. Nothing to summarize.")
        print("Tip: ensure logs/trades.log exists and contains JSONL entries.")
        return

    closed, warnings = parse_closed_trades(trade_lines)
    positions_count = count_positions(POSITIONS_LOG)

    if not closed:
        print("ℹ️  No closed trades yet.")
        print(f"🔄 Active Positions: {positions_count}")
        return

    stats = compute_metrics(closed, positions_count)

    if args.json:
        print(json.dumps(stats, indent=2))
        return

    print_table(stats, warnings)


if __name__ == "__main__":
    main()
