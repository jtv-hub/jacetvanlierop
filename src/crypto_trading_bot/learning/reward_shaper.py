"""
reward_shaper.py

Utility helpers for translating trading outcomes into scalar rewards that
can be consumed by the PPO training loop.
"""

from __future__ import annotations

from crypto_trading_bot.config import CONFIG

DEFAULT_DRAW_DOWN_PENALTY = float(CONFIG.get("DRAW_DOWN_PENALTY_MULTIPLIER", 10.0))


def compute_reward(
    roi: float | None,
    fee: float | None,
    slippage: float | None,
    drawdown: float | None,
    risk_alloc: float | None,
    *,
    reward_weight: float = 1.0,
    drawdown_penalty_multiplier: float | None = None,
) -> float:
    """
    Combine ROI, fees, slippage, drawdown, and risk allocation into a single reward.

    Args:
        roi: Trade return on investment (fractional).
        fee: Total fees paid, expressed as fractional cost relative to position size.
        slippage: Slippage cost as fractional loss.
        drawdown: Observed drawdown fraction during trade lifetime.
        risk_alloc: Fraction of capital allocated to the trade.
        reward_weight: Optional multiplier applied to the ROI term.
        drawdown_penalty_multiplier: Override for drawdown penalty scaling.

    Returns:
        float: Scalar reward suitable for PPO training.
    """

    net_roi = float(roi or 0.0) * float(reward_weight)
    fee_cost = abs(float(fee or 0.0))
    slippage_cost = abs(float(slippage or 0.0))
    drawdown_value = abs(float(drawdown or 0.0))
    risk_value = abs(float(risk_alloc or 0.0))

    penalty_multiplier = (
        float(drawdown_penalty_multiplier) if drawdown_penalty_multiplier is not None else DEFAULT_DRAW_DOWN_PENALTY
    )

    drawdown_penalty = penalty_multiplier * drawdown_value
    risk_bonus = 0.0
    if risk_value > 0:
        # Encourage meaningful capital deployment without dominating the reward.
        risk_bonus = min(risk_value, 0.1)

    reward = net_roi - fee_cost - slippage_cost - drawdown_penalty + risk_bonus
    return float(reward)


__all__ = ["compute_reward"]
