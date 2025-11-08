"""
policy_manager.py
Controls whether PPO can override, shadow, or remain disabled.
Ensures PPO never bypasses existing validators or risk guards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

LOGGER = logging.getLogger(__name__)


class PolicyManager:
    """
    Determines whether a PPO action is allowed to override the rule-based decision.
    Modes:
      - "disabled": ignore PPO entirely
      - "shadow": log PPO action but do not override
      - "live": allow controlled overrides
    """

    def __init__(self):
        self.mode = "disabled"  # disabled | shadow | live

    def set_mode(self, mode: str) -> None:
        """Validate and update the operating mode."""
        assert mode in ("disabled", "shadow", "live"), f"Invalid mode: {mode}"
        self.mode = mode

    def apply_policy(
        self,
        *,
        rule_decision: Dict[str, Any],
        ppo_action: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Merge rule-based and PPO decisions safely.
        - In disabled mode → return rule_decision unchanged.
        - In shadow mode → log PPO action but do not override.
        - In live mode → allow PPO to modify size and entry permission.
        """
        ctx = context or {}
        if self.mode == "disabled":
            result = {**rule_decision, "ppo_used": False, "ppo_action": ppo_action}
            self._log_merge(result, ctx)
            return result

        if self.mode == "shadow":
            result = {**rule_decision, "ppo_used": False, "ppo_action": ppo_action}
            self._log_merge(result, ctx)
            return result

        # LIVE mode: controlled override
        if ppo_action.get("agent_confidence", 0.0) < 0.7:
            result = {**rule_decision, "ppo_used": False, "ppo_action": ppo_action}
            self._log_merge(result, ctx)
            return result

        merged = rule_decision.copy()
        if not ppo_action.get("allow_entry", True):
            merged["enter_trade"] = False

        # Cap size scalar to prevent over-leverage
        size_scalar = ppo_action.get("size_scalar", 1.0)
        merged["size_scalar"] = min(size_scalar, 1.2)

        merged["ppo_used"] = True
        merged["ppo_action"] = ppo_action
        self._log_merge(merged, ctx)
        return merged

    def _log_merge(self, merged: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Emit debug details for downstream observability."""
        LOGGER.debug(
            "PolicyManager mode=%s trade_id=%s merged=%s",
            self.mode,
            context.get("trade_id"),
            merged,
        )
