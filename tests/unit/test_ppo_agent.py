from __future__ import annotations

import numpy as np

from crypto_trading_bot.context.trading_context import TradingContext
from crypto_trading_bot.learning.ppo_agent import STATE_DIM, PPOAgent, get_agent


def test_get_agent_returns_singleton():
    agent_one = get_agent()
    agent_two = get_agent()
    assert isinstance(agent_one, PPOAgent)
    assert agent_one is agent_two


def test_predict_outputs_action_and_confidence():
    agent = get_agent()
    dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
    action, confidence = agent.predict(dummy_state)
    assert isinstance(action, int)
    assert 0 <= action < 3
    assert isinstance(confidence, float)
    assert 0.0 <= confidence <= 1.0


def test_train_on_batch_executes_and_returns_metrics():
    agent = get_agent()
    rng = np.random.default_rng(123)
    states = rng.standard_normal((10, STATE_DIM), dtype=np.float32)
    rewards = rng.standard_normal(10).astype(np.float32)
    metrics = agent.train_on_batch(states, rewards, timesteps=50)
    assert "avg_reward" in metrics
    assert "timesteps" in metrics
    assert metrics["timesteps"] == 50


def test_train_with_context():
    agent = get_agent()
    state = np.zeros(STATE_DIM, dtype=np.float32)
    context = TradingContext(pair="BTC/USDC", position_size=0.01)
    metrics = agent.train_on_batch(
        states=np.array([state]),
        rewards=np.array([0.001], dtype=np.float32),
        timesteps=10,
        context=context,
    )
    assert "avg_reward" in metrics
    assert agent.model_path.exists()
