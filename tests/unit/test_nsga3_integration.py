"""Integration smoke test covering healthcheck, engine, and promotion."""

from __future__ import annotations

import os

from crypto_trading_bot.nsga3 import nsga3_engine, nsga3_healthcheck, promotion_manager


def test_integration_flow():
    """Running the pipeline should produce promotions output."""
    os.makedirs("logs", exist_ok=True)
    assert isinstance(nsga3_healthcheck.run_healthcheck(), bool)

    population, generation = nsga3_engine.run_nsga3_cycle(dry_run=True, regime="global")
    assert isinstance(population, list)
    assert isinstance(generation, int)

    promotion_manager.run_promotion_cycle(
        population,
        shadow_results=None,
        generation=generation,
        regime="global",
    )
    assert os.path.exists("logs/nsga3_promotions_global.jsonl")
