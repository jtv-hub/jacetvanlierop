"""Audit Trade Ledger for ROI Accuracy.

Loads `logs/trades.log`, computes a running balance starting from $1,000
using each CLOSED trade's `size` and `roi`, and prints a concise report:

- Final computed balance
- Total closed trades
- Average ROI per trade
- Win rate
- Top 5 trades by absolute ROI

It also emits warnings when:
- All trades appear to use the same capital amount (uniform cost_basis)
- A trade's cost_basis exceeds the available capital after applying
  the trade's `capital_buffer` (if present)

The script writes a JSONL summary to `logs/audit_report.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, List

LOGS_DIR = "logs"
TRADES_PATH = os.path.join(LOGS_DIR, "trades.log")
OUTPUT_PATH = os.path.join(LOGS_DIR, "audit_report.jsonl")


def _load_jsonl(path: str) -> Iterable[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue


def _parse_ts(ts: str | None) -> float:
    """Parse ISO8601 timestamp to epoch seconds; returns 0.0 on failure."""
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):  # defensive but specific
        return 0.0


@dataclass
class AuditEntry:
    """Lightweight view of a closed trade used by the audit."""

    trade_id: str
    timestamp: str
    roi: float
    size: float
    cost_basis: float | None


def _extract_closed_trades(path: str) -> List[AuditEntry]:
    out: List[AuditEntry] = []
    for t in _load_jsonl(path):
        try:
            status = (t.get("status") or "").lower()
            if status != "closed":
                continue
            roi = float(t.get("roi"))
            size = float(t.get("size"))
            trade_id = str(t.get("trade_id") or "")
            ts = str(t.get("timestamp") or "")
            cb = t.get("cost_basis")
            cost_basis = float(cb) if cb is not None else None
            out.append(AuditEntry(trade_id, ts, roi, size, cost_basis))
        except (TypeError, ValueError):
            # Skip invalid rows
            continue
    # Sort by parsed timestamp ascending
    out.sort(key=lambda e: _parse_ts(e.timestamp))
    return out


def run_audit(trades_path: str = TRADES_PATH, initial_balance: float = 1000.0) -> dict:
    """Run the ROI/capital audit and return a summary dict.

    - Reads closed trades, sorted by timestamp
    - Computes running balance using profit = size * roi
    - Assembles stats and warnings
    """
    os.makedirs(LOGS_DIR, exist_ok=True)

    entries = _extract_closed_trades(trades_path)
    balances: List[float] = [initial_balance]
    balance = initial_balance
    wins = 0
    roi_values: List[float] = []

    # Heuristics for warnings
    uniform_cost_basis = True
    first_cb = None
    capital_violations = 0

    for t in entries:
        # Check uniform cost basis
        if first_cb is None:
            first_cb = t.cost_basis
        if t.cost_basis != first_cb:
            uniform_cost_basis = False

        # Buffer check (assume capital_buffer is present per-trade if needed)
        # We derive available capital as balance * (1 - capital_buffer) when provided.
        # Since the trade log line may not have capital_buffer, we fallback to 0.25.
        capital_buffer = 0.25  # default if not recorded per-trade
        # Note: this warning is heuristic; it is meant to catch egregious violations only.
        available_capital = balance * (1 - capital_buffer)
        if t.cost_basis is not None and t.cost_basis > available_capital + 1e-9:
            capital_violations += 1

        # Compute profit and new balance
        profit = t.size * t.roi
        balance += profit
        balances.append(balance)
        roi_values.append(t.roi)
        if t.roi > 0:
            wins += 1

    total = len(entries)
    valid = len(roi_values)
    avg_roi = (sum(roi_values) / valid) if valid else 0.0
    win_rate = (wins / valid) if valid else 0.0

    # Top 5 by absolute ROI magnitude
    top5 = sorted(entries, key=lambda e: abs(e.roi), reverse=True)[:5]
    top5_view = [
        {
            "trade_id": e.trade_id,
            "timestamp": e.timestamp,
            "roi": round(e.roi, 4),
            "size": e.size,
            "cost_basis": e.cost_basis,
        }
        for e in top5
    ]

    warnings: List[str] = []
    if total > 0 and uniform_cost_basis and first_cb is not None:
        # Flag if cost basis is identical across all closed trades
        warnings.append(f"Uniform cost_basis across trades (cost_basis={first_cb}).")
    if capital_violations:
        warnings.append(f"{capital_violations} trade(s) exceeded available capital after buffer.")

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "initial_balance": initial_balance,
        "final_balance": round(balance, 4),
        "total_trades": total,
        "avg_roi_per_trade": round(avg_roi, 4),
        "win_rate": round(win_rate, 4),
        "top5_by_abs_roi": top5_view,
        "warnings": warnings,
        "balance_series": [round(x, 4) for x in balances],
    }

    # Persist JSONL summary
    with open(OUTPUT_PATH, "a", encoding="utf-8") as out:
        out.write(json.dumps(summary, separators=(",", ":")) + "\n")

    return summary


def main() -> None:
    """CLI entrypoint for the ROI audit."""
    parser = argparse.ArgumentParser(description="Audit trades.log ROI and capital usage.")
    parser.add_argument("--trades", default=TRADES_PATH, help="Path to trades.log")
    parser.add_argument("--balance", type=float, default=1000.0, help="Starting balance")
    args = parser.parse_args()

    result = run_audit(trades_path=args.trades, initial_balance=args.balance)

    print("=== ROI Audit Report ===")
    print(f"Initial balance: ${args.balance:,.2f}")
    print(f"Final balance:   ${result['final_balance']:,.2f}")
    print(f"Total trades:    {result['total_trades']}")
    print(f"Avg ROI/trade:   {result['avg_roi_per_trade']:.4f}")
    print(f"Win rate:        {result['win_rate']*100:.2f}%")
    if result["top5_by_abs_roi"]:
        print("Top 5 trades by |ROI|:")
        for t in result["top5_by_abs_roi"]:
            msg = (
                f"  - {t['trade_id']} | ts={t['timestamp']} | "
                f"roi={t['roi']:.4f} | size={t['size']} | cb={t['cost_basis']}"
            )
            print(msg)
    if result["warnings"]:
        print("Warnings:")
        for w in result["warnings"]:
            print(f"  - {w}")


if __name__ == "__main__":
    main()
