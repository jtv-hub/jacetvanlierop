"""Quick TWAP smoke test."""

from __future__ import annotations

import json

from crypto_trading_bot.execution.twap_engine import execute


def main() -> int:
    result = execute("BTC/USDC", 0.1, "buy")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
