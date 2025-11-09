"""Smoke tests for the NSGA-III engine."""

from __future__ import annotations

import os

from crypto_trading_bot.nsga3 import nsga3_engine


def test_run_nsga3_cycle_creates_logs():
    """Running a dry-run generation should emit front log entries."""
    os.makedirs("logs", exist_ok=True)
    population, generation = nsga3_engine.run_nsga3_cycle(dry_run=True, regime="global")
    assert isinstance(population, list)
    assert isinstance(generation, int)
    assert os.path.exists("logs/nsga3_front_global.jsonl")
