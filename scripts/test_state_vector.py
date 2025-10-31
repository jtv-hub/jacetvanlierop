"""
Quick sanity check for the multi-timeframe state vector builder.

Usage:
    PYTHONPATH=src python3 scripts/test_state_vector.py
"""

from __future__ import annotations

import numpy as np

from crypto_trading_bot.learning.state_builder import STATE_DIM, build_state_vector


def main() -> None:
    pair = "BTC/USDC"
    vector = build_state_vector(pair)
    if vector.shape != (STATE_DIM,):
        raise AssertionError(f"Unexpected shape {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise AssertionError("State vector contains non-finite values")
    print(f"State vector for {pair}: shape={vector.shape}, mean={vector.mean():.6f}, std={vector.std():.6f}")


if __name__ == "__main__":
    main()
