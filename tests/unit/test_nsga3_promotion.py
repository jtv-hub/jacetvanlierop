"""Tests for NSGA-III promotion manager persistence."""

from __future__ import annotations

import json
import os

from crypto_trading_bot.nsga3 import promotion_manager


def test_run_promotion_cycle_appends_jsonl():
    """Promotion cycle should append JSONL entries with params/objectives."""
    os.makedirs("logs", exist_ok=True)
    population = [
        {
            "params": {"rsi_low": 30, "rsi_high": 70},
            "objectives": {"roi": 0.05, "drawdown": 0.02, "win_rate": 0.6},
        }
    ]
    promotion_manager.run_promotion_cycle(
        population,
        shadow_results=None,
        generation=0,
        regime="global",
    )
    promo_file = "logs/nsga3_promotions_global.jsonl"
    assert os.path.exists(promo_file)
    with open(promo_file, "r", encoding="utf-8") as handle:
        line = json.loads(handle.readline())
        assert "params" in line
        assert "objectives" in line
