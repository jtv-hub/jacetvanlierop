from __future__ import annotations

import pytest

from crypto_trading_bot import config


def _reset_live_mode() -> None:
    config.set_live_mode(False)


@pytest.fixture(autouse=True)
def _ensure_reset():
    _reset_live_mode()
    yield
    _reset_live_mode()


def test_set_live_mode_blocks_withdraw_enabled_key(monkeypatch):
    monkeypatch.setattr(config, "_validate_credentials", lambda: None)
    monkeypatch.setattr(config, "_kraken_client", object())
    monkeypatch.setattr(
        config,
        "query_api_key_permissions",
        lambda: {"rights": {"can_withdraw": True}},
    )

    with pytest.raises(config.ConfigurationError, match="withdraw permissions"):
        config.set_live_mode(True)

    assert config.is_live is False


def test_set_live_mode_allows_non_withdraw_key(monkeypatch):
    monkeypatch.setattr(config, "_validate_credentials", lambda: None)
    monkeypatch.setattr(config, "_kraken_client", object())
    monkeypatch.setattr(
        config,
        "query_api_key_permissions",
        lambda: {"rights": {"can_withdraw": False}},
    )

    config.set_live_mode(True)
    assert config.is_live is True

    config.set_live_mode(False)


def test_load_tradable_pairs_accepts_zusd(monkeypatch):
    monkeypatch.setenv("CRYPTO_TRADING_BOT_PAIRS", "BTC/USDC")
    monkeypatch.setattr(
        config,
        "_kraken_pair_meta",
        lambda _pair: {"quote": "", "altname": "XXBTZUSD"},
    )

    pairs = config._load_tradable_pairs()
    assert "BTC/USDC" in pairs
