"""
Fix Trade Log Integrity

Repairs historical issues in logs/trades.log and logs/positions.jsonl.

Actions:
- Normalize/infer side to 'long'/'short' on trades
- Create synthetic positions for closed trades missing from positions
- Skip malformed lines but count them
- Validate output against schema (best-effort)

Usage:
  python scripts/fix_trade_log_integrity.py [--trades logs/trades.log] \
      [--positions logs/positions.jsonl] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

try:
    from colorama import Fore, Style  # type: ignore
    from colorama import init as colorama_init
except ImportError:  # pragma: no cover - optional

    class _Dummy:
        RESET_ALL = ""

    class _Fore(_Dummy):
        RED = GREEN = YELLOW = CYAN = ""

    class _Style(_Dummy):
        BRIGHT = NORMAL = ""

    Fore, Style = _Fore(), _Style()  # type: ignore

    def colorama_init(*_args, **_kwargs):  # type: ignore
        return None


# Project imports with src fallback
try:
    from crypto_trading_bot.bot.utils.schema_validator import validate_trade_schema
except ImportError:  # pragma: no cover
    import sys

    REPO_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
    sys.path.insert(0, os.path.abspath(REPO_SRC))
    from crypto_trading_bot.bot.utils.schema_validator import validate_trade_schema  # type: ignore


def _normalize_side(side: Any, strategy: Any, roi: Any) -> str:
    s = str(side).strip().lower() if isinstance(side, str) else ""
    if s in {"long", "short"}:
        return s
    if s in {"buy", "sell"}:
        return "long" if s == "buy" else "short"
    # Infer by simple heuristic
    strat = str(strategy or "").lower()
    try:
        r = float(roi) if roi is not None else None
    except (TypeError, ValueError):
        r = None
    if ("rsi" in strat or "threshold" in strat) and (r is None or r >= 0):
        return "long"
    if r is not None and r < 0:
        return "short"
    return "long"


def _sanitize_confidence(value: Any) -> float:
    """Return a safe confidence value in the approved range [0.6, 1.0]."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = 0.7

    if parsed <= 0 or parsed > 1:
        parsed = 0.7

    if abs(parsed - 0.5) < 1e-6:
        parsed = 0.7

    return max(0.6, min(parsed, 1.0))


def _sanitize_strategy(name: Any) -> str:
    if not isinstance(name, str) or not name.strip():
        return "UnknownStrategy"
    alias = name.strip()
    if alias.upper() == "S":
        return "UnknownStrategy"
    return alias


def _load_jsonl(path: str) -> Tuple[List[Dict], int]:
    items: List[Dict] = []
    malformed = 0
    if not os.path.exists(path):
        return items, malformed
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                malformed += 1
    return items, malformed


def _write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")


def _norm_id(x: Any) -> str | None:
    """Normalize IDs for matching: strip and lowercase string form."""
    if x is None:
        return None
    try:
        s = str(x).strip()
    except Exception:
        return None
    return s.lower() if s else None


def repair(trades_path: str, positions_path: str, dry_run: bool = False) -> Dict[str, Any]:
    trades, malformed_trades = _load_jsonl(trades_path)
    positions, malformed_positions = _load_jsonl(positions_path)
    pos_ids = {_norm_id(p.get("trade_id")) for p in positions if p.get("trade_id")}

    fixed_side = 0
    created_positions = 0
    schema_errors = 0

    # Normalize sides
    for t in trades:
        orig = t.get("side")
        norm = _normalize_side(orig, t.get("strategy"), t.get("roi"))
        if norm != orig:
            fixed_side += 1
            t["side"] = norm

    # Create synthetic positions for closed trades missing in positions
    for t in trades:
        tid = _norm_id(t.get("trade_id"))
        if not tid:
            continue
        if t.get("exit_price") is not None and tid not in pos_ids:
            side_norm = _normalize_side(t.get("side"), t.get("strategy"), t.get("roi"))
            entry_price = t.get("entry_price")
            try:
                entry_price = float(entry_price) if entry_price is not None else 1.0
            except (TypeError, ValueError):
                entry_price = 1.0
            entry_time = t.get("timestamp") or datetime.now(timezone.utc).isoformat()
            synthetic = {
                "trade_id": tid,
                "pair": t.get("pair") or "BTC/USD",
                "side": side_norm,
                "entry_price": entry_price,
                "entry_time": entry_time,
                "strategy": _sanitize_strategy(t.get("strategy")),
                "confidence": _sanitize_confidence(t.get("confidence")),
                "size": float(t.get("size", 100) or 100),
                "capital_buffer": float(t.get("capital_buffer", 0.25) or 0.25),
                "status": "closed",
            }
            positions.append(synthetic)
            pos_ids.add(tid)
            created_positions += 1
            # warn per synthetic addition
            print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} added synthetic position for trade_id={tid}")

    # Second pass: ensure open trades have a position entry
    for t in trades:
        status = str(t.get("status") or "").lower()
        if status == "closed":
            continue
        tid = _norm_id(t.get("trade_id"))
        if not tid or tid in pos_ids:
            continue
        # create synthetic open position
        side_norm = _normalize_side(t.get("side"), t.get("strategy"), t.get("roi"))
        entry_price = t.get("entry_price")
        try:
            entry_price = float(entry_price) if entry_price is not None else 1.0
        except (TypeError, ValueError):
            entry_price = 1.0
        entry_time = t.get("timestamp") or datetime.now(timezone.utc).isoformat()
        synthetic_open = {
            "trade_id": tid,
            "pair": t.get("pair") or "BTC/USD",
            "side": side_norm,
            "entry_price": entry_price,
            "entry_time": entry_time,
            "strategy": _sanitize_strategy(t.get("strategy")),
            "confidence": _sanitize_confidence(t.get("confidence")),
            "size": float(t.get("size", 100) or 100),
            "capital_buffer": float(t.get("capital_buffer", 0.25) or 0.25),
            "status": "open",
        }
        positions.append(synthetic_open)
        pos_ids.add(tid)
        created_positions += 1
        print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} Missing synthetic position for open trade_id={tid}")

    # Validate trades against schema (best-effort)
    for t in trades:
        try:
            validate_trade_schema(t)
        except Exception:
            schema_errors += 1

    if not dry_run:
        trades_out = os.path.join(os.path.dirname(trades_path), "trades_fixed.log")
        positions_out = os.path.join(os.path.dirname(positions_path), "positions_fixed.jsonl")
        _write_jsonl(trades_out, trades)
        _write_jsonl(positions_out, positions)

    return {
        "malformed_trades": malformed_trades,
        "malformed_positions": malformed_positions,
        "fixed_side": fixed_side,
        "created_positions": created_positions,
        "schema_errors": schema_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair historical trade/position logs")
    parser.add_argument("--trades", default="logs/trades.log", help="Path to trades.log")
    parser.add_argument("--positions", default="logs/positions.jsonl", help="Path to positions.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only; do not write outputs")
    args = parser.parse_args()

    colorama_init(autoreset=True)

    summary = repair(args.trades, args.positions, dry_run=args.dry_run)

    # Human summary
    print(f"{Style.BRIGHT}=== Trade Log Integrity Report ==={Style.RESET_ALL}")
    print(f"Malformed trades.jsonl lines: {summary['malformed_trades']}")
    print(f"Malformed positions.jsonl lines: {summary['malformed_positions']}")
    print(f"Sides normalized: {summary['fixed_side']}")
    print(f"Synthetic positions created: {summary['created_positions']}")
    status_ok = summary["schema_errors"] == 0
    emoji = "✅" if status_ok else "❌"
    color = Fore.GREEN if status_ok else Fore.RED
    print(f"Schema check: {color}{emoji} {summary['schema_errors']} errors{Style.RESET_ALL}")
    if args.dry_run:
        print("(dry-run) No files written.")
    else:
        print("Wrote logs/trades_fixed.log and logs/positions_fixed.jsonl")


if __name__ == "__main__":
    main()
