"""
test_ppo_inference.py

Smoke test that loads the PPO agent, builds a fresh state vector, and runs a
single inference pass printing action and confidence.
"""

from __future__ import annotations

import sys
from pathlib import Path

from crypto_trading_bot.learning.ppo_agent import PPOAgent
from crypto_trading_bot.learning.state_builder import build_state_vector

MODEL_PATH = Path("models/ppo/ppo_agent.zip")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"[PPO] Model checkpoint missing: {MODEL_PATH}", file=sys.stderr)
        return 1
    try:
        agent = PPOAgent(model_path=MODEL_PATH, load_existing=True)
        state = build_state_vector("BTC/USDC")
        action, confidence = agent.predict(state)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[PPO] Inference failure: {exc}", file=sys.stderr)
        return 1
    print(f"[PPO] action={action} confidence={confidence:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
