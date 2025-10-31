#!/usr/bin/env python3
"""Compare paper-mode vs live dry-run signals for sanity checking."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy

from crypto_trading_bot.config import CONFIG, is_live, set_live_mode
from crypto_trading_bot.simulation import collect_signal_snapshot

logging.basicConfig(level=logging.INFO)


def _first_actionable(entry):
    for signal in entry.get("signals", []):
        if signal.get("signal") in {"buy", "sell"}:
            return signal
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare paper vs live-dry signals")
    parser.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="Specific trading pairs to compare (default: CONFIG tradable pairs)",
    )
    args = parser.parse_args(argv)

    pairs = args.pairs or list(CONFIG.get("tradable_pairs", []))
    if not pairs:
        logging.error("No tradable pairs configured; nothing to compare")
        return 1

    original_mode = is_live
    original_validate = deepcopy(CONFIG.get("kraken", {}).get("validate_orders"))
    original_dry = deepcopy(CONFIG.get("live_mode", {}).get("dry_run"))

    # Paper snapshot
    set_live_mode(False)
    CONFIG.setdefault("live_mode", {})["dry_run"] = False
    CONFIG.setdefault("kraken", {})["validate_orders"] = False
    paper_snapshot = collect_signal_snapshot(pairs)

    # Live dry-run snapshot
    set_live_mode(True)
    CONFIG.setdefault("live_mode", {})["dry_run"] = True
    CONFIG.setdefault("kraken", {})["validate_orders"] = True
    live_snapshot = collect_signal_snapshot(pairs)

    # Restore original state
    set_live_mode(original_mode)
    CONFIG.setdefault("kraken", {})["validate_orders"] = original_validate
    CONFIG.setdefault("live_mode", {})["dry_run"] = original_dry

    mismatches = []
    paper_map = {entry["pair"]: entry for entry in paper_snapshot}
    live_map = {entry["pair"]: entry for entry in live_snapshot}

    for pair in pairs:
        paper_entry = paper_map.get(pair, {})
        live_entry = live_map.get(pair, {})
        paper_sig = _first_actionable(paper_entry) or {"signal": None, "confidence": 0.0}
        live_sig = _first_actionable(live_entry) or {"signal": None, "confidence": 0.0}
        if paper_sig != live_sig:
            mismatches.append({"pair": pair, "paper": paper_sig, "live": live_sig})

    summary = {
        "pairs_evaluated": len(pairs),
        "mismatches": mismatches,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not mismatches else 2


if __name__ == "__main__":
    raise SystemExit(main())
