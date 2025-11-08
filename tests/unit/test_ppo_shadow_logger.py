"""Test the PPO shadow logger functionality."""

import json
import tempfile
from pathlib import Path

from crypto_trading_bot.rl import ppo_shadow_logger


def test_log_divergence_writes_valid_jsonl_entry():
    """Verify that log_divergence writes a valid JSONL entry to a file."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_path = Path(tmp_dir) / "test_divergence.jsonl"

        strategy_view = {
            "strategy_id": "RSI_Baseline",
            "confidence": 0.5,
            "size": 120,
            "allow_entry": True,
        }

        ppo_output = {
            "confidence": 0.65,
            "size": 150,
            "allow_entry": False,
        }

        original_log_path = ppo_shadow_logger.LOG_PATH
        ppo_shadow_logger.LOG_PATH = test_path
        decision = {"use_ppo": False, "reason": "unit_test"}

        try:
            ppo_shadow_logger.log_divergence(strategy_view, ppo_output, decision)
        finally:
            ppo_shadow_logger.LOG_PATH = original_log_path

        assert test_path.exists(), "Log file was not created"

        with open(test_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 1, "Expected one log entry"
            entry = json.loads(lines[0])
            assert entry["strategy_action"]["strategy_id"] == "RSI_Baseline"
            assert entry["ppo_action"]["size"] == 150
