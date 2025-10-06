#!/usr/bin/env python3
"""Helper to clear the persistent risk guard state."""

from __future__ import annotations

import sys
from pathlib import Path

from crypto_trading_bot.safety import clear_risk_state
from crypto_trading_bot.safety.risk_guard import state_path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    state = clear_risk_state()
    path = state_path()
    print(f"Risk guard state cleared at {path}")
    print(
        "Current state: "
        f"consecutive_failures={state.get('consecutive_failures', 0)} "
        f"paused={state.get('paused', False)}"
    )


if __name__ == "__main__":
    main()
