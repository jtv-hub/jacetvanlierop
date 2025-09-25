"""Unit tests for Kraken integration safety checks."""

import base64
import logging

import pytest

from crypto_trading_bot import config as config_module
from crypto_trading_bot.bot import market_data
from crypto_trading_bot.config import ConfigurationError, set_live_mode
from crypto_trading_bot.utils import kraken_client


def test_get_account_balance_paper_fallback(monkeypatch):
    """Paper mode should always return the configured paper balance."""

    monkeypatch.setattr(market_data, "is_live", False, raising=False)
    paper_balance = float(market_data.CONFIG.get("paper_mode", {}).get("starting_balance", 0.0))

    result = market_data.get_account_balance(use_mock_for_paper=False)

    assert result == paper_balance


def test_decode_secret_padding():
    """Secrets missing padding characters should still decode correctly."""

    raw_secret = b"padding-test"
    no_padding_secret = base64.b64encode(raw_secret).decode().rstrip("=")

    decoded = kraken_client._decode_secret(no_padding_secret)  # type: ignore[attr-defined]

    assert decoded == raw_secret


def test_place_order_invalid_secret(monkeypatch):
    """Invalid base64 secrets should surface a structured error without HTTP calls."""

    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", "!!!invalid!!!")

    called = False

    def fake_http_post(*_, **__):
        nonlocal called
        called = True
        raise AssertionError("HTTP should not be reached when secret is invalid")

    monkeypatch.setattr(kraken_client, "_http_post", fake_http_post)

    response = kraken_client.kraken_place_order("BTC/USD", "buy", 0.01, price=20_000.0)

    assert response.get("ok") is False
    assert response.get("code") == "auth"
    assert response.get("endpoint") == "AddOrder"
    assert "Invalid Kraken API secret" in (response.get("error") or "")
    assert called is False


def test_set_live_mode_missing_credentials(monkeypatch, caplog):
    """Live mode activation fails when credentials are absent."""

    monkeypatch.setattr(config_module, "dotenv_values", lambda: {})
    monkeypatch.setenv("KRAKEN_API_KEY", "")
    monkeypatch.setenv("KRAKEN_API_SECRET", "")
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_key", "")
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_secret", "")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ConfigurationError):
            set_live_mode(True)

    assert any("Kraken API key/secret" in rec.message for rec in caplog.records)
    set_live_mode(False)


def test_set_live_mode_bad_secret(monkeypatch, caplog):
    """Live mode activation fails when the secret is malformed base64."""

    monkeypatch.setattr(config_module, "dotenv_values", lambda: {"KRAKEN_API_SECRET": "!!!invalid!!!", "KRAKEN_API_KEY": "unit-test-key"})
    monkeypatch.setenv("KRAKEN_API_KEY", "")
    monkeypatch.setenv("KRAKEN_API_SECRET", "")
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_key", "")
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_secret", "")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ConfigurationError):
            set_live_mode(True)

    assert any("failed base64" in rec.message for rec in caplog.records)
    set_live_mode(False)


def test_get_account_balance_live_bad_credentials(monkeypatch):
    """Live mode with auth errors raises BalanceFetchError."""

    monkeypatch.setattr(market_data, "is_live", True, raising=False)
    monkeypatch.setitem(market_data.CONFIG, "kraken_api_key", "live-key")
    monkeypatch.setitem(market_data.CONFIG, "kraken_api_secret", "live-secret")
    monkeypatch.setitem(market_data.CONFIG["live_mode"], "fallback_balance", 0.0)

    env_var = market_data.CONFIG["live_mode"].get("balance_env_var") or "CRYPTO_TRADING_BOT_LIVE_BALANCE"
    monkeypatch.delenv(env_var, raising=False)

    monkeypatch.setattr(
        market_data,
        "kraken_get_balance",
        lambda asset: {
            "ok": False,
            "error": "credentials missing",
            "code": "auth",
            "endpoint": "Balance",
            "balance": None,
        },
    )

    with pytest.raises(market_data.BalanceFetchError):
        market_data.get_account_balance()

    monkeypatch.setattr(market_data, "is_live", False, raising=False)


def test_sanitize_base64_secret_helper():
    """Sanitization should strip quotes, whitespace, and invalid characters."""

    dirty = "  'Zm9vYmFy?%$#'  "
    sanitized = config_module._sanitize_base64_secret(dirty)
    assert sanitized == "Zm9vYmFy=="


def test_place_order_success_with_valid_secret(monkeypatch):
    """Valid secret should allow order flow when private request succeeds."""

    valid_secret = base64.b64encode(b"unit-test").decode()
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", valid_secret)

    def fake_private_request(endpoint, *args, **kwargs):  # pylint: disable=unused-argument
        return {
            "ok": True,
            "error": None,
            "code": "ok",
            "endpoint": endpoint,
            "result": {"txid": ["abc123"], "descr": {"order": "buy"}},
            "raw": {"result": {"txid": ["abc123"], "descr": {"order": "buy"}}},
        }

    monkeypatch.setattr(kraken_client, "_private_request", fake_private_request)

    response = kraken_client.kraken_place_order("BTC/USD", "buy", 0.01, price=20_000.0)

    assert response["ok"] is True
    assert response["txid"] == ["abc123"]
    assert response["descr"] == "buy"


def test_place_order_cost_minimum_not_met(monkeypatch):
    """Cost minimum error should return structured response without raising."""

    valid_secret = base64.b64encode(b"unit-test").decode()
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", valid_secret)

    def fake_private_request(endpoint, payload=None, **_):  # pylint: disable=unused-argument
        return {
            "ok": False,
            "error": "EOrder:Cost minimum not met",
            "code": "cost_minimum_not_met",
            "endpoint": endpoint,
            "result": None,
            "raw": {"error": ["EOrder:Cost minimum not met"]},
            "errors": ["EOrder:Cost minimum not met"],
        }

    monkeypatch.setattr(kraken_client, "_private_request", fake_private_request)

    response = kraken_client.kraken_place_order(
        "BTC/USD",
        "buy",
        0.0001,
        price=10.0,
        min_cost_threshold=5.0,
    )

    assert response["ok"] is False
    assert response["code"] == "cost_minimum_not_met"
    assert "Cost minimum" in response["error"]
    assert response.get("threshold") == 5.0
    assert response.get("attempted_cost") == pytest.approx(0.001)
*** End Patch
