#!/usr/bin/env python3
"""Learn from paper trades and suggest the next parameters.

Reads:  logs/paper/paper_trades_*.jsonl (one JSON object per line)
Writes: config/next_params.json

Fields expected per trade (extra fields ignored):
  symbol (str), side (str), pnl_pct (float), ts_end (str)

The rules are intentionally simple and transparent so we can iterate fast.
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import glob
import json
import statistics as stats

# Where to read / write
PAPER_GLOB = "logs/paper/paper_trades_*.jsonl"
OUT_JSON = Path("config/next_params.json")

# Baseline parameters we adjust from
ESSENTIAL_DEFAULTS = {
    "rsi_th": 60,
    "tp": 0.0020,
    "sl": 0.0030,
    "trail": {"mode": "pct", "pct": 0.0020, "activate": 0.000},
}


def load_trades() -> list[dict]:
    """Load paper trades from all matching files (best‑effort).

    Each non-empty line must be a JSON object. Unknown fields are ignored.
    Returns a list of canonicalized trade dicts with keys: symbol, side, pnl_pct.
    """
    trades: list[dict] = []
    for fname in sorted(glob.glob(PAPER_GLOB)):
        try:
            with open(fname, "r", encoding="utf-8") as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    sym = (d.get("symbol") or d.get("pair") or "UNKNOWN").upper()
                    side = str(d.get("side") or "long").lower()
                    try:
                        pnl = float(d.get("pnl_pct", 0.0))
                    except (TypeError, ValueError):
                        pnl = 0.0
                    trades.append({"symbol": sym, "pnl_pct": pnl, "side": side})
        except FileNotFoundError:
            # No paper logs yet; that's fine.
            continue
    return trades


def summarize(trades: list[dict]) -> dict:
    """Aggregate paper trades by symbol and compute suggested params.

    The heuristics are intentionally small:
      - Adjust RSI threshold by win rate
      - Tighten SL if losses dominate and are large
      - Reduce TP slightly if wins are consistently small
      - Adjust trailing stop tightness by win rate
    """
    by_sym: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        by_sym[t["symbol"]].append(t)

    out: dict[str, dict] = {}
    for sym, rows in by_sym.items():
        pnls = [float(r["pnl_pct"]) for r in rows]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        losses = n - wins

        win_rate = wins / n if n else 0.0
        avg_pnl = stats.mean(pnls) if n else 0.0
        avg_win = stats.mean([p for p in pnls if p > 0]) if wins else 0.0
        avg_loss = stats.mean([p for p in pnls if p <= 0]) if losses else 0.0

        # Start from defaults and adjust with simple rules
        rsi_th = ESSENTIAL_DEFAULTS["rsi_th"]
        tp = ESSENTIAL_DEFAULTS["tp"]
        sl = ESSENTIAL_DEFAULTS["sl"]
        trail_mode = ESSENTIAL_DEFAULTS["trail"]["mode"]
        trail_pct = ESSENTIAL_DEFAULTS["trail"]["pct"]
        trail_activate = ESSENTIAL_DEFAULTS["trail"]["activate"]

        # Rule 1: adjust RSI threshold based on win rate
        if win_rate >= 0.60:
            rsi_th = max(45, rsi_th - 5)  # more permissive
        elif win_rate <= 0.40:
            rsi_th = min(70, rsi_th + 5)  # stricter

        # Rule 2: adjust TP/SL by relative magnitudes
        if abs(avg_loss) > abs(avg_win) * 1.2 and losses >= wins:
            sl = max(0.0015, sl * 0.7)  # tighten
        if 0 < avg_win < 0.0015:
            tp = max(0.0015, tp * 0.8)  # realize sooner

        # Rule 3: trailing stop tightness by win rate
        if win_rate < 0.45:
            trail_pct = max(0.0010, trail_pct * 0.8)
        elif win_rate > 0.65:
            trail_pct = min(0.0030, trail_pct * 1.1)

        out[sym] = {
            "observations": {
                "trades": n,
                "win_rate": round(win_rate, 3),
                "avg_pnl": round(avg_pnl, 5),
                "avg_win": round(avg_win, 5),
                "avg_loss": round(avg_loss, 5),
            },
            "suggested": {
                "rsi_th": int(rsi_th),
                "tp": round(tp, 6),
                "sl": round(sl, 6),
                "trail": {
                    "mode": trail_mode,
                    "pct": round(trail_pct, 6),
                    "activate": round(trail_activate, 6),
                },
            },
        }
    return out


def main() -> None:
    """Entry point: read paper trades and write next_params.json."""
    Path("config").mkdir(parents=True, exist_ok=True)
    trades = load_trades()
    report = summarize(trades)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[learn] wrote {OUT_JSON} for {len(report)} symbols")


if __name__ == "__main__":
    main()
