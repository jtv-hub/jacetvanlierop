#!/usr/bin/env python3
"""Helper to clear the persistent risk guard state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

risk_guard_path = SRC / "crypto_trading_bot" / "safety" / "risk_guard.py"
spec = importlib.util.spec_from_file_location("_risk_guard", risk_guard_path)
assert spec and spec.loader, "Unable to load risk_guard module"
risk_guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk_guard)


def main() -> None:
    """Clear persisted risk guard state and print the resulting snapshot."""
    risk_guard.invalidate_cache()
    risk_guard.clear_state()
    risk_guard.resume_trading(context={"command": "reset_risk_state"})
    final_state = risk_guard.load_state(force_reload=True)
    path = risk_guard.state_path()
    paused_flag = bool(final_state.get("paused"))
    print(f"Risk guard state cleared at {path}")
    print(
        "Current state: " f"consecutive_failures={final_state.get('consecutive_failures', 0)} " f"paused={paused_flag}"
    )
    print(
        "Drawdown snapshot: "
        f"last_drawdown={final_state.get('last_drawdown', 0.0)} "
        f"max_drawdown={final_state.get('max_drawdown', 0.0)}"
    )
    if paused_flag:
        print("⚠️ Risk guard pause flag still active — manual intervention required.")
    else:
        print("✅ Risk guard pause cleared and drawdown counters reset.")


if __name__ == "__main__":
    main()
