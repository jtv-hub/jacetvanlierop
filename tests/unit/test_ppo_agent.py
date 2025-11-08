"""
Unit tests for PPOAgent inference and fallback behavior.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from crypto_trading_bot.rl import state_builder as state_builder_module
from crypto_trading_bot.rl.ppo_agent import PPOAgent


def mock_context() -> dict:
    """Provide a representative PPO context without hitting live systems."""
    return {
        "strategy_id": "SimpleRSI",
        "confidence": 0.5,
        "indicators": {
            "rsi": 60,
            "ema_diff": 0.1,
            "macd": 0.05,
            "boll_width": 0.02,
        },
        "regime": {
            "trend_strength": 0.7,
            "volatility_regime": 0.6,
        },
        "portfolio": {
            "drawdown_pct": 0.02,
            "capital_buffer": 0.25,
            "win_streak": 3,
        },
        "timestamp": datetime.now(UTC),
    }


def test_ppo_agent_disabled_mode():
    """Verify the disabled PPO agent returns safe default actions."""
    agent = PPOAgent(model_path="nonexistent_model.zip")
    loaded = agent.load_model()

    # Expect the model to not load (disabled)
    assert loaded is False
    assert agent.enabled is False

    context = mock_context()
    action = agent.get_action(state=None, context=context)

    # Expected default output
    assert isinstance(action, dict)
    assert action["source"] == "ppo_disabled"
    assert action["size_scalar"] == 1.0
    assert action["allow_entry"] is True
    assert action["agent_confidence"] == 0.0


def test_predict_fallback_when_model_unavailable():
    """Fallback confidence should be 0.0 when PPO is disabled."""
    agent = PPOAgent(model_path="missing_model.zip")
    agent.enabled = False
    confidence = agent.predict(context=mock_context())
    assert confidence == 0.0


def test_predict_fallback_when_model_errors():
    """Enabled agent returns 0.5 fallback if prediction raises."""
    agent = PPOAgent(model_path="dummy.zip")
    agent.enabled = True
    agent.model = MagicMock()
    agent.model.predict.side_effect = ValueError("bad vector")
    confidence = agent.predict(context=mock_context())
    assert confidence == 0.5


def test_predict_returns_model_confidence():
    """Ensure predict() returns a clipped float from the PPO model output."""
    agent = PPOAgent(model_path="dummy.zip")
    agent.enabled = True
    built_state = state_builder_module.build_state(mock_context())
    agent.expected_state_size = len(built_state["vector"])

    mock_model = MagicMock()
    mock_model.predict.return_value = ([0.2, 0.9, 1.5], None)
    agent.model = mock_model

    confidence = agent.predict(observation=built_state)
    assert confidence == 1.0  # clipped from 1.5
    mock_model.predict.assert_called_once()


def test_agent_uses_approved_model_path():
    """PPOAgent should prefer the approved model path when provided."""
    approved_path = "models/approved_model.zip"
    with patch("crypto_trading_bot.rl.ppo_agent.get_latest_approved_model_path", return_value=approved_path):
        agent = PPOAgent()
    assert agent.model_path == approved_path


def test_get_action_disabled_returns_zero_confidence():
    """Disabled PPO agents must report 0.0 confidence."""
    agent = PPOAgent(model_path="missing_model.zip")
    action = agent.get_action(state=None, context=mock_context())
    assert action["agent_confidence"] == 0.0
    assert action["source"] == "ppo_disabled"


def test_get_action_inference_failure_uses_default_confidence():
    """When inference fails, agent_confidence should fall back to 0.5 with error source."""
    agent = PPOAgent(model_path="dummy.zip")
    agent.enabled = True
    built_state = state_builder_module.build_state(mock_context())
    agent.expected_state_size = len(built_state["vector"])
    mock_model = MagicMock()
    mock_model.predict.side_effect = ValueError("boom")
    agent.model = mock_model

    action = agent.get_action(state=built_state, context=mock_context())
    assert action["agent_confidence"] == 0.5
    assert action["source"] == "ppo_error"


def test_get_action_valid_inference_returns_live_confidence():
    """Successful inference should produce live action output."""
    agent = PPOAgent(model_path="dummy.zip")
    agent.enabled = True
    built_state = state_builder_module.build_state(mock_context())
    agent.expected_state_size = len(built_state["vector"])
    mock_model = MagicMock()
    mock_model.predict.return_value = ([1.2, 1.0, 0.8], None)
    agent.model = mock_model

    action = agent.get_action(state=built_state, context=mock_context())
    assert action["source"] == "ppo_live"
    assert action["agent_confidence"] == 0.8
    assert action["size_scalar"] == 1.2
