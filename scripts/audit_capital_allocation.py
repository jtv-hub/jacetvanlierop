#!/usr/bin/env python3
"""
Audit the current capital allocation snapshot.

Reads portfolio_state.json and prints key fields needed for the pre-launch checklist.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

STATE_CANDIDATES = (
    Path("data/portfolio_state.json"),
    Path("logs/portfolio_state.json"),
)


def _load_state() -> Tuple[Dict[str, Any], Path]:
    """Return the most recent portfolio state and the file path used."""

    for candidate in STATE_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                return json.load(handle), candidate
        except (OSError, json.JSONDecodeError):
            continue
    raise FileNotFoundError("No portfolio_state.json found in expected locations.")


def _format_currency(amount: Any) -> str:
    try:
        value = float(amount)
    except (TypeError, ValueError):
        return "n/a"
    return f"${value:,.2f}"


def _format_percent(value: Any) -> str:
    try:
        pct = float(value) * 100.0
    except (TypeError, ValueError):
        return "n/a"
    return f"{pct:.2f}%"


def _print_strategy_allocation(strategy_buffers: Dict[str, Dict[str, Any]]) -> None:
    if not strategy_buffers:
        print("Strategy allocation: n/a")
        return

    print("Strategy allocation:")
    for strategy in sorted(strategy_buffers):
        allocations = strategy_buffers[strategy]
        if not isinstance(allocations, dict):
            print(f"  - {strategy}: n/a")
            continue
        print(f"  - {strategy}:")
        for regime in sorted(allocations):
            allocation = allocations[regime]
            if isinstance(allocation, (int, float)):
                print(f"      {regime}: {allocation:.4f}")
            else:
                print(f"      {regime}: n/a")


def main() -> None:
    try:
        state, path = _load_state()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Loaded portfolio state from: {path}")

    available_capital = state.get("available_capital")
    starting_balance = state.get("starting_balance")
    total_roi = state.get("total_roi")
    drawdown_pct = state.get("drawdown_pct")
    market_regime = state.get("market_regime", "unknown")
    capital_buffer = state.get("capital_buffer")
    strategy_buffers = state.get("strategy_buffers") or {}

    reinvestment_rate = state.get("reinvestment_rate")
    total_reinvested = state.get("total_reinvested")
    retained_profit = state.get("retained_profit")

    print(f"Available capital: {_format_currency(available_capital)}")
    print(f"Starting balance: {_format_currency(starting_balance)}")
    print(f"Total ROI: {_format_percent(total_roi)}")
    print(f"Drawdown %: {_format_percent(drawdown_pct)}")
    print(f"Market regime: {market_regime}")
    print(f"Capital buffer: {capital_buffer if isinstance(capital_buffer, (int, float)) else 'n/a'}")
    print(f"Reinvestment rate: {_format_percent(reinvestment_rate)}")

    if isinstance(total_reinvested, (int, float)) or isinstance(retained_profit, (int, float)):
        print(f"Total reinvested: {_format_currency(total_reinvested)}")
        print(f"Retained profit: {_format_currency(retained_profit)}")

    _print_strategy_allocation(strategy_buffers if isinstance(strategy_buffers, dict) else {})


if __name__ == "__main__":
    main()
