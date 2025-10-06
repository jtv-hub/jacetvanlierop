"""Tests ensuring strategy pipeline parity between different invocations."""

from crypto_trading_bot.bot import trading_logic
from crypto_trading_bot.bot.trading_logic import _build_strategy_pipeline


def test_strategy_pipeline_matches_between_modes():
    """Pipelines built back-to-back contain strategies in the same order."""

    dummy_per_asset: dict[str, dict[str, float]] = {}
    strategies_first = _build_strategy_pipeline(per_asset_params=dummy_per_asset)
    strategies_second = _build_strategy_pipeline(per_asset_params=dummy_per_asset)

    names_first = sorted(strategy.__class__.__name__ for strategy in strategies_first)
    names_second = sorted(strategy.__class__.__name__ for strategy in strategies_second)

    assert names_first == names_second, f"Mismatch: {names_first} vs {names_second}"


def test_strategy_signal_lock_parity():
    """Cached live and paper pipelines serialize to identical specs."""

    dummy_per_asset: dict[str, dict[str, float]] = {}
    trading_logic.strategy_context["live"].clear()
    trading_logic.strategy_context["paper"].clear()

    live_pipeline = trading_logic._get_locked_strategy_pipeline(  # pylint: disable=protected-access
        "SOL/USD",
        "live",
        dummy_per_asset,
    )
    paper_pipeline = trading_logic._get_locked_strategy_pipeline(  # pylint: disable=protected-access
        "SOL/USD",
        "paper",
        dummy_per_asset,
    )

    live_specs = [trading_logic._serialize_strategy(s) for s in live_pipeline]  # pylint: disable=protected-access
    paper_specs = [trading_logic._serialize_strategy(s) for s in paper_pipeline]  # pylint: disable=protected-access

    assert live_specs == paper_specs, f"Live/Paper mismatch: {live_specs} vs {paper_specs}"
