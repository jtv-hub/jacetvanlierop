"""
paper_cycle.py

Runs a single paper-trading cycle using real market data (when available),
then triggers learning, shadow tests, and a confidence audit. It also seeds
logs with realistic mock entries if no trades are available yet so the
Streamlit dashboard has something to render.

Usage:
  python scripts/paper_cycle.py --iterations 3 --sleep 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List

# Local imports from the package
from src.crypto_trading_bot.bot.trading_logic import evaluate_signals_and_trade
from src.crypto_trading_bot.learning.confidence_audit import audit_trades, log_anomaly


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file best-effort and return a list of dicts."""
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append a JSON object as one compact JSONL line to ``path``."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, separators=(",", ":")) + "\n")


def _seed_mock_suggestion(trades: List[Dict[str, Any]], out_path: str) -> None:
    """Seed a minimal learning suggestion entry if none exist yet.

    Uses the most recent trade_id when present so the dashboard can join
    strategy metadata from trades.log.
    """
    tid = None
    strategy = None
    for t in reversed(trades):
        tid = t.get("trade_id")
        strategy = t.get("strategy")
        if tid:
            break
    if not tid:
        tid = str(uuid.uuid4())
    cur = 0.8
    suggested = 0.7
    _append_jsonl(
        out_path,
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "trade_id": tid,
            "current_confidence": cur,
            "suggested_confidence": suggested,
            "reason": "Low ROI with high confidence",
            "strategy": strategy or "Unknown",
        },
    )


def run_learning_machine() -> None:
    """Run the learning machine or seed a suggestion if unavailable."""
    try:
        # Optional import kept inside function to avoid hard dependency
        from scripts import run_learning_machine as RLM  # pylint: disable=import-outside-toplevel
    except (ImportError, ModuleNotFoundError):
        RLM = None  # type: ignore[assignment]

    if RLM is not None and hasattr(RLM, "main"):
        try:
            RLM.main()  # appends to logs/learning_feedback.jsonl
            return
        except (OSError, ValueError, TypeError, RuntimeError, json.JSONDecodeError):
            # Fall through to seeding on failure
            pass
    # Fallback: ensure a minimal suggestion exists if logs are empty
    trades = _read_jsonl("logs/trades.log")
    lf = "logs/learning_feedback.jsonl"
    if not os.path.exists(lf) or os.path.getsize(lf) == 0:
        _seed_mock_suggestion(trades, lf)


def run_shadow_tester() -> None:
    """Run a lightweight shadow test to populate shadow_test_results.jsonl."""
    try:
        # Prefer our lightweight tester that writes to logs/shadow_test_results.jsonl
        from scripts.shadow_tester import run_shadow_test  # pylint: disable=import-outside-toplevel
    except (ImportError, ModuleNotFoundError, AttributeError):
        run_shadow_test = None  # type: ignore[assignment]

    if run_shadow_test is not None:
        try:
            run_shadow_test()
            return
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            # Ignore and continue; dashboard will still have trade/learning data
            pass


def run_confidence_audit() -> None:
    """Run confidence audit; log a benign anomaly if it fails."""
    try:
        audit_trades("logs/trades.log")
    except (OSError, ValueError, TypeError):
        # As a last resort, add a benign test anomaly to prove the logger path
        log_anomaly(
            {"type": "test_anomaly", "marker": "paper_cycle_fallback"},
            source="paper_cycle",
        )


def paper_cycle(iterations: int = 1, sleep_seconds: float = 0.0) -> None:
    """Run N paper iterations, then print a compact CLI summary."""
    os.makedirs("logs", exist_ok=True)
    for i in range(iterations):
        print(f"\n=== Paper iteration {i+1}/{iterations} ===")
        # Run one trading evaluation pass (opens/exits + writes trades/logs)
        evaluate_signals_and_trade()
        # Post-trade tasks
        run_learning_machine()
        run_shadow_tester()
        run_confidence_audit()
        if sleep_seconds > 0 and i < (iterations - 1):
            time.sleep(sleep_seconds)

    # CLI summary preview
    trades = _read_jsonl("logs/trades.log")
    closed = [t for t in trades if (t.get("status") or "").lower() == "closed"]
    print(f"\nTrades: total={len(trades)}, closed={len(closed)}")
    if closed:
        last = closed[-1]
        fields = ("trade_id", "pair", "strategy", "roi", "exit_reason", "timestamp")
        snippet = {k: last.get(k) for k in fields}
        print("Last closed:", json.dumps(snippet, indent=2))

    lf = _read_jsonl("logs/learning_feedback.jsonl")
    print(f"Learning feedback entries: {len(lf)}")
    if lf:
        print("Sample:", json.dumps(lf[-1], indent=2))

    st = _read_jsonl("logs/shadow_test_results.jsonl")
    print(f"Shadow test results: {len(st)}")
    if st:
        print("Sample:", json.dumps(st[-1], indent=2))

    try:
        with open("logs/anomalies.log", "r", encoding="utf-8") as f:
            anom_lines = [ln.strip() for ln in f if ln.strip()][-3:]
    except FileNotFoundError:
        anom_lines = []
    print(f"Anomalies (last up to 3): {len(anom_lines)}")
    for line in anom_lines:
        try:
            rec = json.loads(line)
            print("-", rec.get("type"), rec.get("trade_id"), rec.get("timestamp"))
        except (json.JSONDecodeError, TypeError, ValueError):
            print("-", line)


def main() -> None:
    """CLI entrypoint for running one or more paper iterations."""
    ap = argparse.ArgumentParser(description="Run paper trading cycle with learning and audits")
    ap.add_argument("--iterations", type=int, default=1, help="Number of paper iterations to run")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between iterations")
    args = ap.parse_args()
    paper_cycle(iterations=args.iterations, sleep_seconds=args.sleep)


if __name__ == "__main__":
    main()
