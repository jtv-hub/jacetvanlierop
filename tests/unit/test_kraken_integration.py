"""Unit tests for Kraken integration safety checks."""

# pylint: disable=protected-access,missing-function-docstring,missing-class-docstring

import base64
import hashlib
import hmac
import logging
from types import SimpleNamespace

import pytest
import requests

from crypto_trading_bot import config as config_module
from crypto_trading_bot.bot import market_data, trading_logic
from crypto_trading_bot.config import ConfigurationError, set_live_mode
from crypto_trading_bot.safety import risk_guard
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

    monkeypatch.setenv("KRAKEN_API_KEY", "")
    monkeypatch.setenv("KRAKEN_API_SECRET", "")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", "!!!invalid!!!")

    called = False

    def fake_http_post(*_, **__):
        """Ensure HTTP transport is skipped when secrets are invalid."""
        nonlocal called
        called = True
        raise AssertionError("HTTP should not be reached when secret is invalid")

    monkeypatch.setattr(kraken_client, "_http_post", fake_http_post)

    response = kraken_client.kraken_place_order("BTC/USDC", "buy", 0.01, price=20_000.0)

    assert response.get("ok") is False
    assert response.get("code") == "auth"
    assert response.get("endpoint") == "AddOrder"
    assert "EAuth:Invalid secret" in (response.get("error") or "")
    assert called is False


def test_set_live_mode_missing_credentials(monkeypatch, caplog):
    """Live mode activation fails when credentials are absent."""

    monkeypatch.setattr(
        "crypto_trading_bot.config.dotenv_values",
        lambda *_, **__: {},
        raising=False,
    )
    monkeypatch.setenv("KRAKEN_API_KEY", "")
    monkeypatch.setenv("KRAKEN_API_SECRET", "")
    monkeypatch.setattr(config_module, "_kraken_client", None, raising=False)
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_key", "")
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_secret", "")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ConfigurationError):
            set_live_mode(True)

    assert any("Kraken API key/secret" in rec.message for rec in caplog.records)
    set_live_mode(False)


def test_set_live_mode_bad_secret(monkeypatch, caplog):
    """Live mode activation fails when the secret is malformed base64."""

    monkeypatch.setattr(
        "crypto_trading_bot.config.dotenv_values",
        lambda *_, **__: {
            "KRAKEN_API_SECRET": "!!!invalid!!!",
            "KRAKEN_API_KEY": "unit-test-key",
        },
        raising=False,
    )
    monkeypatch.setenv("KRAKEN_API_KEY", "")
    monkeypatch.setenv("KRAKEN_API_SECRET", "")
    monkeypatch.setattr(config_module, "_kraken_client", None, raising=False)
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_key", "")
    monkeypatch.setitem(config_module.CONFIG, "kraken_api_secret", "")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(ConfigurationError):
            set_live_mode(True)

    assert any("missing for live mode" in rec.message or "validation failed" in rec.message for rec in caplog.records)
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


def test_sign_request_consistency():
    """_sign_request matches the documented Kraken HMAC signature."""

    secret_raw = b"unit-test-secret"
    secret = base64.b64encode(secret_raw).decode()
    nonce = "1616492376594"
    postdata = "nonce=1616492376594&pair=XXBTZUSD"
    uri_path = "/0/private/Balance"

    expected = base64.b64encode(
        hmac.new(
            base64.b64decode(secret),
            uri_path.encode() + hashlib.sha256((nonce + postdata).encode()).digest(),
            hashlib.sha512,
        ).digest()
    ).decode()

    actual = kraken_client._sign_request(  # type: ignore[attr-defined]
        uri_path,
        nonce,
        postdata,
        secret,
    )

    assert actual == expected


def test_http_post_success(monkeypatch):
    """_http_post returns parsed JSON when the transport succeeds."""

    payload = {"result": {"foo": "bar"}}

    class DummyResponse:
        """Stub HTTP response returning a static payload."""

        def raise_for_status(self):
            """Do nothing for the dummy response."""
            return None

        def json(self):
            """Return the mocked payload."""
            return payload

    monkeypatch.setattr(kraken_client.requests, "post", lambda *args, **kwargs: DummyResponse())

    result = kraken_client._http_post("https://example.com", "nonce=1", {}, 5.0)

    assert result == payload


def test_http_post_handles_http_error(monkeypatch):
    """HTTP errors surface as KrakenAPIError."""

    def _raise_http_error(*_, **__):
        """Raise a simulated HTTP error."""

        raise requests.exceptions.HTTPError("boom")

    monkeypatch.setattr(kraken_client.requests, "post", _raise_http_error)

    with pytest.raises(kraken_client.KrakenAPIError):
        kraken_client._http_post("https://example.com", "nonce=1", {}, 5.0)


def test_http_post_invalid_json(monkeypatch):
    """Invalid JSON responses raise KrakenAPIError."""

    class DummyResponse:
        """Stub response object raising a JSON decode error."""

        def raise_for_status(self):
            """Do nothing for the dummy response."""
            return None

        def json(self):
            """Simulate a JSON parsing error."""
            raise ValueError("not json")

    monkeypatch.setattr(kraken_client.requests, "post", lambda *args, **kwargs: DummyResponse())

    with pytest.raises(kraken_client.KrakenAPIError):
        kraken_client._http_post("https://example.com", "nonce=1", {}, 5.0)


def test_private_request_missing_credentials(monkeypatch):
    """Missing credentials should short-circuit before HTTP transport."""

    monkeypatch.setenv("KRAKEN_API_KEY", "")
    monkeypatch.setenv("KRAKEN_API_SECRET", "")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", "")

    called = False

    def fake_http_post(*_, **__):
        """Raise an assertion to ensure HTTP transport is not reached."""
        nonlocal called
        called = True
        raise AssertionError("transport should not be invoked when credentials missing")

    monkeypatch.setattr(kraken_client, "_http_post", fake_http_post)

    response = kraken_client.kraken_place_order("BTC/USDC", "buy", 0.01, price=20_000.0)

    assert response.get("ok") is False
    assert response.get("code") == "auth"
    assert called is False


def test_place_order_success_with_valid_secret(monkeypatch):
    """Valid secret should allow order flow when private request succeeds."""

    valid_secret = base64.b64encode(b"unit-test").decode()
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", valid_secret)

    def fake_private_request(endpoint, *args, **kwargs):  # pylint: disable=unused-argument
        """Return a mocked private-response payload for successful order tests."""
        return {
            "ok": True,
            "error": None,
            "code": "ok",
            "endpoint": endpoint,
            "result": {"txid": ["abc123"], "descr": {"order": "buy"}},
            "raw": {"result": {"txid": ["abc123"], "descr": {"order": "buy"}}},
        }

    monkeypatch.setattr(kraken_client, "_private_request", fake_private_request)

    response = kraken_client.kraken_place_order("BTC/USDC", "buy", 0.01, price=20_000.0)

    assert response["ok"] is True
    assert response["txid"] == ["abc123"]


def test_place_order_validate_only_success(monkeypatch):
    """Validate-only orders should surface a successful ok response."""

    valid_secret = base64.b64encode(b"unit-test").decode()
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", valid_secret)

    captured_payload = {}

    def fake_private_request(endpoint, payload=None, **_):  # pylint: disable=unused-argument
        """Capture payloads for validate-only order submissions."""
        captured_payload.update(payload or {})
        return {
            "ok": True,
            "code": "ok",
            "endpoint": endpoint,
            "result": {"txid": ["validate"], "descr": {"order": "buy"}},
            "raw": {},
        }

    monkeypatch.setattr(kraken_client, "_private_request", fake_private_request)

    response = kraken_client.kraken_place_order(
        "BTC/USDC",
        "buy",
        0.05,
        price=25_000.0,
        validate=True,
    )

    assert response["ok"] is True
    assert response["code"] == "ok"
    assert captured_payload.get("validate") is True
    assert response["descr"] == "buy"


def test_place_order_cost_minimum_not_met(monkeypatch):
    """Cost minimum error should return structured response without raising."""

    valid_secret = base64.b64encode(b"unit-test").decode()
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", valid_secret)

    def fake_private_request(
        endpoint,
        payload=None,
        **_,
    ):  # pylint: disable=unused-argument
        """Return a structured cost-minimum error payload."""
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
        "BTC/USDC",
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


def test_asset_pair_meta_missing_pair(monkeypatch):
    """Metadata fetch should fall back when Kraken omits the requested pair."""

    monkeypatch.setattr(
        kraken_client,
        "_http_public",
        lambda endpoint, params=None, timeout=10.0: {"error": [], "result": {}},
    )

    meta = kraken_client.kraken_get_asset_pair_meta("FOO/BAR")
    assert meta["ordermin"] >= 0.0
    assert meta["costmin"] >= 0.0
    assert meta.get("source") == "fallback"


def test_kraken_place_order_rate_limit_code(monkeypatch):
    """Rate limit errors should be normalized to rate_limit code."""

    valid_secret = base64.b64encode(b"unit-test").decode()
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_key", "unit-test-key")
    monkeypatch.setitem(kraken_client.CONFIG, "kraken_api_secret", valid_secret)

    def fake_private_request(endpoint, payload=None, **_):  # pylint: disable=unused-argument
        """Return a simulated rate-limit API error payload."""
        return {
            "ok": False,
            "error": "EOrder:Rate limit exceeded",
            "code": "rate_limit",
            "endpoint": endpoint,
            "result": None,
            "raw": {"error": ["EOrder:Rate limit exceeded"]},
            "errors": ["EOrder:Rate limit exceeded"],
        }

    monkeypatch.setattr(kraken_client, "_private_request", fake_private_request)

    response = kraken_client.kraken_place_order("BTC/USDC", "buy", 0.01, price=20_000.0)

    assert response["code"] == "rate_limit"


def _reset_trading_state():
    """Reset trading logic state between Kraken integration tests."""

    trading_logic.is_live = True
    trading_logic._live_block_logged = False  # type: ignore[attr-defined]
    trading_logic._KRAKEN_FAILURE_PAUSE_UNTIL = None  # type: ignore[attr-defined]
    trading_logic.DEPLOY_PHASE = "full"
    risk_guard.invalidate_cache()
    risk_guard.resume_trading(context={"test": "reset"})


def test_submit_live_trade_skips_volume_min(monkeypatch):
    """Orders below volume minimum should be skipped before hitting Kraken."""

    _reset_trading_state()
    monkeypatch.setitem(trading_logic.CONFIG, "kraken_min_cost_threshold", 0.0)
    monkeypatch.setitem(
        trading_logic.CONFIG,
        "kraken",
        {"min_cost_threshold": 0.0, "pair_cost_minimums": {}},
    )

    monkeypatch.setattr(
        trading_logic,
        "kraken_get_asset_pair_meta",
        lambda pair: {
            "ordermin": 0.01,
            "costmin": 0.05,
            "pair_decimals": 4,
            "lot_decimals": 8,
        },
    )

    called = False

    def fake_place_order(*_args, **_kwargs):
        """Simulate a cost-minimum rejection."""
        nonlocal called
        called = True
        return {"ok": True, "code": "ok"}

    monkeypatch.setattr(trading_logic, "_kraken_place_order_retry", fake_place_order)

    result = trading_logic._submit_live_trade(  # pylint: disable=protected-access
        pair="USDC/USD",
        side="buy",
        size=0.001,
        price=1.00,
        strategy="unit",
        confidence=0.9,
    )

    assert result is False
    assert called is False


def test_submit_live_trade_skips_cost_min(monkeypatch):
    """Orders below cost minimum should be rejected locally."""

    _reset_trading_state()
    monkeypatch.setitem(trading_logic.CONFIG, "kraken_min_cost_threshold", 0.0)
    monkeypatch.setitem(
        trading_logic.CONFIG,
        "kraken",
        {"min_cost_threshold": 0.0, "pair_cost_minimums": {}},
    )

    monkeypatch.setattr(
        trading_logic,
        "kraken_get_asset_pair_meta",
        lambda pair: {
            "ordermin": 0.001,
            "costmin": 5.0,
            "pair_decimals": 4,
            "lot_decimals": 8,
        },
    )

    called = False

    def fake_place_order(*_args, **_kwargs):  # pragma: no cover - should not run
        """Should not execute when credentials invalid."""
        nonlocal called
        called = True
        return {"ok": True, "code": "ok"}

    monkeypatch.setattr(trading_logic, "_kraken_place_order_retry", fake_place_order)

    result = trading_logic._submit_live_trade(  # pylint: disable=protected-access
        pair="USDC/USD",
        side="buy",
        size=0.001,
        price=2.0,
        strategy="unit",
        confidence=0.9,
    )

    assert result is False
    assert called is False


def test_submit_live_trade_boundary_executes(monkeypatch):
    """Exact minimum volume and cost should proceed to Kraken."""

    _reset_trading_state()
    monkeypatch.setitem(trading_logic.CONFIG, "kraken_min_cost_threshold", 0.0)
    monkeypatch.setitem(
        trading_logic.CONFIG,
        "kraken",
        {"min_cost_threshold": 0.0, "pair_cost_minimums": {}},
    )

    meta_payload = {
        "ordermin": 0.01000000,
        "costmin": 0.0500,
        "pair_decimals": 4,
        "lot_decimals": 8,
    }

    monkeypatch.setattr(trading_logic, "kraken_get_asset_pair_meta", lambda pair: meta_payload)

    captured_payload = {}

    def fake_place_order(pair, side, size, price, **kwargs):  # pylint: disable=unused-argument
        """Capture order payload for boundary validation tests."""
        payload = {"pair": pair, "side": side, "size": size, "price": price}
        payload.update(kwargs)
        captured_payload.update(payload)
        return {"ok": True, "code": "ok", "txid": ["test"], "descr": "buy"}

    monkeypatch.setattr(trading_logic, "kraken_place_order", fake_place_order)

    result = trading_logic._submit_live_trade(  # pylint: disable=protected-access
        pair="USDC/USD",
        side="buy",
        size=0.0100009123,
        price=5.123456,
        strategy="unit",
        confidence=1.0,
    )

    assert isinstance(result, dict)
    # _submit_live_trade unwraps the single-element txid list for convenience
    assert result.get("txid") == "test"
    # ...while retaining the original list so downstream components can persist it verbatim
    assert result.get("txid_list") == ["test"]
    assert captured_payload["size"] == pytest.approx(0.01000091)
    assert captured_payload["price"] == pytest.approx(5.1234)


def test_submit_live_trade_rate_limit_pause(monkeypatch):
    """Rate limit responses should trigger a cooldown."""

    _reset_trading_state()
    monkeypatch.setitem(trading_logic.CONFIG, "kraken_min_cost_threshold", 0.0)
    monkeypatch.setitem(
        trading_logic.CONFIG,
        "kraken",
        {"min_cost_threshold": 0.0, "pair_cost_minimums": {}},
    )

    monkeypatch.setattr(
        trading_logic,
        "kraken_get_asset_pair_meta",
        lambda pair: {
            "ordermin": 0.001,
            "costmin": 0.001,
            "pair_decimals": 4,
            "lot_decimals": 8,
        },
    )

    fake_time = {"now": 1000.0}
    monkeypatch.setattr(
        trading_logic,
        "time",
        SimpleNamespace(monotonic=lambda: fake_time["now"]),
    )

    def fake_place_order(*_args, **_kwargs):  # pylint: disable=unused-argument
        """Simulate a rate-limit error from Kraken."""
        return {
            "ok": False,
            "code": "rate_limit",
            "error": "EOrder:Rate limit exceeded",
        }

    monkeypatch.setattr(trading_logic, "kraken_place_order", fake_place_order)

    result = trading_logic._submit_live_trade(  # pylint: disable=protected-access
        pair="USDC/USD",
        side="buy",
        size=0.01,
        price=1.0,
        strategy="unit",
        confidence=0.8,
    )

    assert result is False
    pause_until = getattr(trading_logic, "_KRAKEN_FAILURE_PAUSE_UNTIL")
    assert pause_until == pytest.approx(1090.0)
