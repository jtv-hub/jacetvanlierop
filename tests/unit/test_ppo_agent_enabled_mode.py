"""Unit test for PPOAgent in enabled mode with mock model."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from crypto_trading_bot.rl.ppo_agent import PPOAgent


def mock_context():
    """Builds a representative PPO context with valid indicators."""
    return {
        "strategy_id": "test_strategy",
        "confidence": 0.6,
        "indicators": {
            "rsi": 55.0,
            "adx": 35.0,
            "volume": 1200,
            "price": 30500.0,
        },
        "regime": {
            "trend_strength": 1.0,
            "volatility_regime": 0.5,
        },
        "portfolio": {
            "capital": 15000,
            "positions": {},
        },
        "timestamp": datetime.now(UTC).isoformat(),
    }


def test_ppo_agent_enabled_mode():
    """Test PPOAgent returns live predictions when enabled with mock model."""
    agent = PPOAgent()
    agent.enabled = True

    # Mock the PPO model and its predict method
    mock_model = MagicMock()
    mock_model.predict.return_value = ([1.2, 1.0, 0.8], None)
    agent.model = mock_model

    action = agent.get_action(state=None, context=mock_context())

    assert isinstance(action, dict)
    assert action["size_scalar"] == 1.2
    assert action["allow_entry"] is True
    assert action["agent_confidence"] == 0.8
    assert action["source"] == "ppo_live"
