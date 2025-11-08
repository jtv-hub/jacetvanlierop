"""Show live summary statistics from logs/trades.log (compact JSONL).

Usage:
    python scripts/show_live_stats.py
    python scripts/show_live_stats.py --json

The script reads UTF-8 JSONL, skips malformed lines, and only includes
trades whose status == "closed" with numeric ROI.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from crypto_trading_bot.config import CONFIG

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from crypto_trading_bot.bot.trading_logic import ppo_agent as _PPO_AGENT
except Exception:  # pragma: no cover - fallback for lightweight environments
    _PPO_AGENT = None

TRADES_PATH = os.path.join("logs", "trades.log")
LEDGER_STATE_PATH = os.path.join("logs", "ledger_state.json")
PORTFOLIO_STATE_PATH = os.path.join("logs", "portfolio_state.json")


def _read_lines(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _read_json_file(path: str) -> Dict[str, Any]:
    """Return parsed JSON dict from ``path`` or {} on error."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _load_closed_trades(path: str) -> List[Dict[str, Any]]:
    """Load closed trades with numeric ROI; skip malformed or incomplete."""
    rows: List[Dict[str, Any]] = []
    for ln in _read_lines(path):
        if not ln.strip():
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if obj.get("status") != "closed":
            continue
        roi = obj.get("roi")
        try:
            obj["roi"] = float(roi)
        except (TypeError, ValueError):
            continue
        rows.append(obj)
    return rows


def _fmt_pct(x: float, places: int = 2, sign: bool = True) -> str:
    s = f"{x*100:.{places}f}%"
    if sign and not s.startswith("-"):
        s = "+" + s
    return s


def _fmt_currency(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _stats(subset: List[Dict[str, Any]]) -> tuple[int, float, float]:
    if not subset:
        return (0, 0.0, 0.0)
    wins = sum(1 for t in subset if float(t.get("roi", 0.0)) > 0)
    total = len(subset)
    win_rate = wins / total if total else 0.0
    avg_roi = sum(float(t.get("roi", 0.0)) for t in subset) / total
    return total, win_rate, avg_roi


def _compute_summary(trades: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(trades)
    wins = sum(1 for t in trades if t["roi"] > 0)
    losses = total - wins
    win_rate = (wins / total) if total else 0.0
    cum_roi = sum(t["roi"] for t in trades) if total else 0.0
    avg_roi = (cum_roi / total) if total else 0.0
    ppo_trades = [t for t in trades if t.get("ppo_used") is True]
    hybrid_trades = [t for t in trades if t.get("hybrid") is True]
    ppo_stats = _stats(ppo_trades)
    hybrid_stats = _stats(hybrid_trades)

    # Strategy leaderboards
    wins_by_strategy: Counter[str] = Counter()
    roi_by_strategy: defaultdict[str, float] = defaultdict(float)
    for t in trades:
        strat = str(t.get("strategy") or "Unknown")
        wins_by_strategy[strat] += 1 if t["roi"] > 0 else 0
        roi_by_strategy[strat] += float(t["roi"])

    # Build combined leaderboard entries (wins, roi)
    combined: List[Dict[str, Any]] = []
    for strat in set(wins_by_strategy) | set(roi_by_strategy):
        combined.append(
            {
                "strategy": strat,
                "wins": int(wins_by_strategy.get(strat, 0)),
                "roi": float(roi_by_strategy.get(strat, 0.0)),
            }
        )
    top = sorted(combined, key=lambda x: (x["wins"], x["roi"]), reverse=True)[:5]

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "cumulative_roi": round(cum_roi, 6),
        "average_roi": round(avg_roi, 6),
        "top_strategies": top,
        "ppo_stats": {
            "count": ppo_stats[0],
            "win_rate": round(ppo_stats[1], 4),
            "average_roi": round(ppo_stats[2], 6),
        },
        "hybrid_stats": {
            "count": hybrid_stats[0],
            "win_rate": round(hybrid_stats[1], 4),
            "average_roi": round(hybrid_stats[2], 6),
        },
    }


def _recent_ppo_usage(trades: List[Dict[str, Any]], window: int = 50) -> tuple[bool, int, int]:
    """Return (active, ppo_trades, total_sample) for the last ``window`` trades."""

    if not trades:
        return False, 0, 0
    ordered = sorted(trades, key=lambda t: str(t.get("timestamp") or ""))
    recent = ordered[-window:]
    count = sum(1 for t in recent if t.get("ppo_used") is True)
    return count > 0, count, len(recent)


def _print_human(summary: Dict[str, Any]) -> None:
    print("\n📊 Trade Summary\n")
    print(f"Total Trades: {summary['total_trades']}")
    print(f"Wins: {summary['wins']} | Losses: {summary['losses']}")
    wr = _fmt_pct(summary["win_rate"], places=1, sign=False)
    print(f"Win Rate: {wr}")
    print(f"Cumulative ROI: {_fmt_pct(summary['cumulative_roi'])}")
    print(f"Average ROI/Trade: {_fmt_pct(summary['average_roi'])}")
    ppo_stats = summary.get("ppo_stats", {})
    hybrid_stats = summary.get("hybrid_stats", {})
    ppo_count = ppo_stats.get("count", 0)
    ppo_wr = ppo_stats.get("win_rate", 0.0)
    ppo_avg = ppo_stats.get("average_roi", 0.0)
    hybrid_count = hybrid_stats.get("count", 0)
    hybrid_wr = hybrid_stats.get("win_rate", 0.0)
    hybrid_avg = hybrid_stats.get("average_roi", 0.0)
    print(f"\n🤖 PPO Trades: {ppo_count} | Win Rate: {ppo_wr:.2%} | Avg ROI: {ppo_avg:.4f}")
    print(f"🔀 Hybrid Trades: {hybrid_count} | Win Rate: {hybrid_wr:.2%} | Avg ROI: {hybrid_avg:.4f}")

    if summary.get("top_strategies"):
        print("\n🏆 Top Strategies")
        for item in summary["top_strategies"]:
            s_name = item["strategy"]
            s_wins = item["wins"]
            s_roi = _fmt_pct(item["roi"]) if isinstance(item["roi"], (int, float)) else str(item["roi"])
            print(f"\t•\t{s_name}: {s_wins} wins, {s_roi}")


def _extract_capital(snapshot: Dict[str, Any]) -> float | None:
    """Return best-effort capital figure from the portfolio snapshot."""

    for key in ("available_capital", "balance", "capital", "final_equity"):
        value = snapshot.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _extract_trend_strength(snapshot: Dict[str, Any]) -> float | None:
    """Return numeric trend strength/regime confidence when present."""

    for key in (
        "regime_trend_strength",
        "trend_strength",
        "regime_strength",
        "regime_confidence",
        "buffer",
    ):
        value = snapshot.get(key)
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _print_live_overview(trades: List[Dict[str, Any]]) -> None:
    """Print capital, drawdown, and PPO context sourced from logs."""

    ledger_state = _read_json_file(LEDGER_STATE_PATH)
    portfolio_state = _read_json_file(PORTFOLIO_STATE_PATH)

    drawdown_value = portfolio_state.get("drawdown_pct")
    if not isinstance(drawdown_value, (int, float)):
        drawdown_value = ledger_state.get("drawdown_pct", ledger_state.get("last_drawdown"))
    drawdown_pct = _coerce_float(drawdown_value, 0.0)
    drawdown_text = f"{drawdown_pct * 100:.2f}%"

    capital_value = _extract_capital(portfolio_state)
    capital_text = _fmt_currency(capital_value)

    regime_label = str(portfolio_state.get("regime") or "unknown")
    raw_trend_strength = portfolio_state.get("trend_strength")
    if not isinstance(raw_trend_strength, (int, float)):
        raw_trend_strength = _extract_trend_strength(portfolio_state)
    has_trend_strength = raw_trend_strength is not None
    trend_strength = _coerce_float(raw_trend_strength, 0.0)
    if has_trend_strength:
        trend_text = f"{regime_label} ({trend_strength * 100:.1f}%)"
    else:
        trend_text = regime_label

    ppo_active, ppo_count, sample_size = _recent_ppo_usage(trades)
    if sample_size:
        ppo_text = f"{'Active' if ppo_active else 'Inactive'} ({ppo_count}/{sample_size} trades)"
    else:
        ppo_text = "Inactive (no trades)"

    risk_cfg = CONFIG.get("risk", {}) or {}
    drawdown_limit = float(risk_cfg.get("max_drawdown_pct", 0.0) or 0.0)
    loss_limit = int(risk_cfg.get("max_consecutive_losses", 0) or 0)
    threshold_text = f"{drawdown_limit * 100:.1f}% | {loss_limit} losses"

    print("\n🩺 Live Risk Snapshot")
    print(f"💼 Live Capital: {capital_text}")
    print(f"📉 Drawdown %: {drawdown_text}")
    print(f"📈 Regime trend strength: {trend_text}")
    print(f"🤖 PPO Mode: {ppo_text}")
    print(f"🛡 Max drawdown + loss streak thresholds: {threshold_text}")

    capital_numeric = capital_value if isinstance(capital_value, (int, float)) else 0.0
    ppo_mode = CONFIG.get("ppo", {}).get("mode", "disabled")
    ppo_status = _resolve_ppo_status()
    print(
        f"Drawdown: {drawdown_pct:.2%} | Live Capital: ${capital_numeric:,.2f} | "
        f"Trend Strength: {trend_strength:.2f} | PPO Mode: {ppo_mode} | Approved: {ppo_status}"
    )


def _resolve_ppo_status() -> str:
    """Return textual PPO approval state."""

    agent = _PPO_AGENT
    if agent is None:
        return "N/A"
    approver = getattr(agent, "is_approved", None)
    try:
        if callable(approver):
            return str(bool(approver()))
        if approver is not None:
            return str(bool(approver))
    except Exception:  # pragma: no cover - defensive formatting
        return "error"
    return "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(description="Show live stats from trades.log")
    parser.add_argument("--json", action="store_true", help="Emit compact JSON summary")
    args = parser.parse_args()

    if not os.path.exists(TRADES_PATH):
        print("ℹ️  logs/trades.log not found. Nothing to summarize.")
        return

    trades = _load_closed_trades(TRADES_PATH)
    if not trades:
        print("ℹ️  No closed trades with numeric ROI found.")
        return

    summary = _compute_summary(trades)
    if args.json:
        print(json.dumps(summary, separators=(",", ":")))
    else:
        _print_human(summary)
        _print_live_overview(trades)


if __name__ == "__main__":
    main()
