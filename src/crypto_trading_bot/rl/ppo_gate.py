"""
Shadow test gate for PPO agent: decides whether to trust PPO action or defer to strategy signal.
"""

from typing import Any, Dict


def decide(strategy_output: Dict[str, Any], ppo_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare PPO output with strategy signal and decide which to follow.

    Returns:
        {
            "use_ppo": bool,
            "reason": str,
            "strategy_action": Dict,
            "ppo_action": Dict,
        }
    """
    try:
        # Compare confidence and action vectors
        strat_conf = strategy_output.get("confidence", 0)
        ppo_conf = ppo_output.get("confidence", 0)

        if abs(ppo_conf - strat_conf) < 0.1:
            return {
                "use_ppo": False,
                "reason": "Confidence delta too small",
                "strategy_action": strategy_output,
                "ppo_action": ppo_output,
            }

        if ppo_conf < 0.4:
            return {
                "use_ppo": False,
                "reason": "PPO confidence too low",
                "strategy_action": strategy_output,
                "ppo_action": ppo_output,
            }

        return {
            "use_ppo": True,
            "reason": "PPO confidence clearly stronger",
            "strategy_action": strategy_output,
            "ppo_action": ppo_output,
        }

    except (KeyError, TypeError, ValueError) as e:
        return {
            "use_ppo": False,
            "reason": f"Gate error: {e}",
            "strategy_action": strategy_output,
            "ppo_action": ppo_output,
        }
