from __future__ import annotations

import logging

from crypto_trading_bot.safety import prelaunch_guard as guard


def test_signals_match_within_tolerance(caplog):
    paper = {"signal": "buy", "strategy": "rsi", "confidence": 0.50}
    live = {"signal": "buy", "strategy": "rsi", "confidence": 0.30}

    with caplog.at_level(logging.DEBUG):
        assert guard._signals_match(paper, live) is True
        assert any("Confidence deviation tolerated" in rec.message for rec in caplog.records)


def test_signals_match_beyond_tolerance():
    paper = {"signal": "buy", "strategy": "rsi", "confidence": 0.90}
    live = {"signal": "buy", "strategy": "rsi", "confidence": 0.50}
    assert guard._signals_match(paper, live) is False
