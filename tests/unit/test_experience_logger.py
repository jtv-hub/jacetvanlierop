"""Unit tests for experience_logger.py to validate JSONL logging behavior."""

import json
import tempfile
from pathlib import Path

from crypto_trading_bot.rl.experience_logger import log_experience


def test_log_experience_writes_jsonl_line():
    """Test that log_experience writes a valid JSON object to a temp file."""
    temp_dir = tempfile.TemporaryDirectory()
    test_log_path = Path(temp_dir.name) / "test_ppo_experience.jsonl"

    # Patch the global LOG_PATH temporarily
    original_log_path = log_experience.__globals__["LOG_PATH"]
    log_experience.__globals__["LOG_PATH"] = test_log_path

    try:
        mock_row = {
            "strategy": "ppo_test",
            "confidence": 0.72,
            "indicators": {"rsi": 48, "adx": 20},
            "regime": {"trend_strength": "weak", "volatility_regime": "low"},
            "portfolio": {"capital": 9500, "positions": {"BTC": 0.5}},
            "action": [0.9, 1.0, 0.7],
            "timestamp": "2025-11-07T15:00:00Z",
        }

        log_experience(mock_row)

        assert test_log_path.exists()
        with open(test_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["strategy"] == "ppo_test"
        assert isinstance(parsed["confidence"], float)
        assert "rsi" in parsed["indicators"]
        assert "trend_strength" in parsed["regime"]

    finally:
        log_experience.__globals__["LOG_PATH"] = original_log_path
        temp_dir.cleanup()
