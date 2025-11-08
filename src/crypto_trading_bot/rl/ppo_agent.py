"""
ppo_agent.py
Interface wrapper for PPO policy inference.
Loads PPO model (when available) and returns safe trading actions.
Works in 3 modes:
- disabled (default)
- shadow (log actions, no execution override)
- live (actions modify trade size or entry/exit decisions)
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np

from crypto_trading_bot.rl import state_builder as state_builder_module
from crypto_trading_bot.rl.deploy_gate import get_latest_approved_model_path
from crypto_trading_bot.rl.experience_logger import log_experience
from crypto_trading_bot.utils.system_logger import get_system_logger

LOGGER = get_system_logger().getChild("ppo_agent")
LOG_PATH = Path("logs/rl_actions.log")
DEFAULT_MODEL_PATH = "models/ppo_agent_v1.zip"
MODEL_METADATA_PATH = Path("models/ppo_agent_v1_meta.json")
DISABLED_CONFIDENCE = 0.0  # PPO disabled or unavailable
DEFAULT_CONFIDENCE = 0.5  # PPO enabled but inference failed


class PPOAgent:
    """Thin wrapper around a PPO policy with safe fallbacks."""

    def __init__(self, model_path: str | None = None):
        resolved_path = model_path or get_latest_approved_model_path(DEFAULT_MODEL_PATH)
        self.model_path = resolved_path
        self.model = None
        self.vec_env = None
        self.enabled = False
        self.state_version: str | None = None
        self.expected_state_size = state_builder_module.STATE_VECTOR_SIZE

    def load_model(self) -> bool:
        """Load the latest approved PPO model and set enabled state."""
        if not os.path.exists(self.model_path):
            LOGGER.warning("PPO model path not found: %s", self.model_path)
            self.model = None
            self.vec_env = None
            self.enabled = False
            return False
        try:
            sb3 = importlib.import_module("stable_baselines3")
            ppo_cls = getattr(sb3, "PPO")
            self.model = ppo_cls.load(self.model_path)
            metadata = getattr(self.model, "metadata", {})
            self.state_version = metadata.get("state_version")
            self.enabled = self.state_version == state_builder_module.STATE_VERSION
            self.expected_state_size = self._load_expected_state_size(metadata)
            if self.enabled:
                LOGGER.info(
                    "Loaded PPO model path=%s state_version=%s",
                    self.model_path,
                    self.state_version,
                )
            else:
                LOGGER.warning(
                    "PPO model %s state_version=%s does not match builder version=%s",
                    self.model_path,
                    self.state_version,
                    state_builder_module.STATE_VERSION,
                )
            return self.enabled
        except ModuleNotFoundError:
            LOGGER.warning(
                "stable_baselines3 not installed; PPO agent disabled for now.",
            )
            self.model = None
            self.vec_env = None
            self.enabled = False
            return False
        except (ImportError, AttributeError, ValueError, OSError) as exc:
            LOGGER.error("Failed loading PPO model %s: %s", self.model_path, exc)
            self.model = None
            self.vec_env = None
            self.enabled = False
            return False

    def _load_expected_state_size(self, metadata: Dict[str, Any]) -> int:
        """Derive the required observation length from metadata or fallback."""
        candidate = metadata.get("state_size")
        if isinstance(candidate, int) and candidate > 0:
            return candidate
        try:
            with MODEL_METADATA_PATH.open("r", encoding="utf-8") as handle:
                meta = json.load(handle)
            meta_size = meta.get("state_size")
            if isinstance(meta_size, int) and meta_size > 0:
                return meta_size
        except (OSError, json.JSONDecodeError):
            LOGGER.debug("Unable to load PPO metadata file for state size")
        return state_builder_module.STATE_VECTOR_SIZE

    def _prepare_observation(self, built_state: Dict[str, Any]) -> np.ndarray | None:
        vector = built_state.get("vector")
        if isinstance(vector, np.ndarray):
            vector = vector.astype(np.float32, copy=False)
        elif isinstance(vector, Iterable):
            try:
                vector = np.asarray(vector, dtype=np.float32)
            except (TypeError, ValueError):
                return None
        else:
            return None
        if vector.ndim != 1 or vector.shape[0] != self.expected_state_size:
            LOGGER.warning(
                "PPO observation length mismatch: got %s expected %s",
                vector.shape,
                self.expected_state_size,
            )
            return None
        if not np.all(np.isfinite(vector)):
            LOGGER.warning("Non-finite PPO observation detected; skipping inference")
            return None
        return vector

    @staticmethod
    def _extract_confidence(action_prob: Any) -> float:
        value = 0.0
        if isinstance(action_prob, np.ndarray):
            if action_prob.size >= 3:
                value = float(action_prob[2])
            elif action_prob.size:
                value = float(action_prob[0])
        elif isinstance(action_prob, (list, tuple)):
            if len(action_prob) >= 3:
                value = float(action_prob[2])
            elif action_prob:
                value = float(action_prob[0])
        elif isinstance(action_prob, (int, float, np.floating, np.integer)):
            value = float(action_prob)
        return float(np.clip(value, 0.0, 1.0))

    def predict(
        self,
        observation: Dict[str, Any] | None = None,
        *,
        context: Dict[str, Any] | None = None,
    ) -> float:
        """Return deterministic PPO confidence, falling back safely when needed."""
        built_state = observation or state_builder_module.build_state(context or {})
        obs_vector = self._prepare_observation(built_state)
        if obs_vector is None:
            LOGGER.warning("Invalid PPO observation; returning fallback confidence")
            return DISABLED_CONFIDENCE if not self.enabled else DEFAULT_CONFIDENCE
        if not self.enabled or self.model is None:
            LOGGER.info("PPO model disabled; returning base confidence")
            return DISABLED_CONFIDENCE
        try:
            prediction = self.model.predict(obs_vector, deterministic=True)
        except (ValueError, AttributeError) as exc:
            LOGGER.warning("PPO predict failed: %s", exc)
            return DEFAULT_CONFIDENCE
        confidence = self._extract_confidence(prediction[0])
        LOGGER.debug(
            "PPO confidence=%.3f source=%s",
            confidence,
            self.model_path,
        )
        return confidence

    def get_action(
        self,
        state: Dict[str, Any] | None,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return PPO-informed trade action or safe defaults when unavailable."""
        action = {
            "size_scalar": 1.0,
            "allow_entry": True,
            "agent_confidence": DISABLED_CONFIDENCE,
            "source": "ppo_disabled",
        }

        built_state = state or state_builder_module.build_state(context or {})
        obs_vector = self._prepare_observation(built_state)

        if not self.enabled or self.model is None:
            action["agent_confidence"] = DISABLED_CONFIDENCE
            action["source"] = "ppo_disabled"
        elif obs_vector is None:
            LOGGER.warning("Missing PPO observation vector; using fallback action")
            action["agent_confidence"] = DEFAULT_CONFIDENCE
            action["source"] = "ppo_error"
        else:
            try:
                predictions = self.model.predict(obs_vector, deterministic=True)
                action_prob = predictions[0]
                size_scalar = float(np.clip(action_prob[0], 0.5, 1.5))
                allow_entry = bool(action_prob[1] > 0.5)
                agent_confidence = self._extract_confidence(action_prob)

                action.update(
                    {
                        "size_scalar": size_scalar,
                        "allow_entry": allow_entry,
                        "agent_confidence": agent_confidence,
                        "source": "ppo_live",
                    }
                )
                LOGGER.debug(
                    "PPO action source=%s size_scalar=%.3f allow_entry=%s confidence=%.3f",
                    self.model_path,
                    size_scalar,
                    allow_entry,
                    agent_confidence,
                )
            except (ValueError, AttributeError) as exc:
                LOGGER.error("Failed in get_action: %s", exc)
                action["agent_confidence"] = DEFAULT_CONFIDENCE
                action["source"] = "ppo_error"

        context_payload = context or {}
        self._log_experience(context_payload, action)
        self._log_action(built_state, action, context_payload)
        return action

    def is_approved(self) -> bool:
        """Return True if the loaded model matches the latest approved path."""
        try:
            from crypto_trading_bot.rl import deploy_gate

            latest = deploy_gate.get_latest_approved_model_path(deploy_gate.DEFAULT_MODEL_CANDIDATE)
            return bool(self.enabled and isinstance(latest, str) and latest and self.model_path == latest)
        except Exception:  # pragma: no cover - approval must fail closed
            return False

    def _log_experience(self, context: Dict[str, Any], action: Dict[str, Any]) -> None:
        """Record experience tuples for later PPO training."""
        try:
            log_experience(
                {
                    "strategy_id": context.get("strategy_id"),
                    "confidence": context.get("confidence"),
                    "indicators": context.get("indicators"),
                    "regime": context.get("regime"),
                    "portfolio": context.get("portfolio"),
                    "size_scalar": action.get("size_scalar"),
                    "allow_entry": action.get("allow_entry"),
                    "confidence_out": action.get("agent_confidence"),
                    "source": action.get("source", "unknown"),
                }
            )
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.debug("Experience logging failed", exc_info=True)

    def _log_action(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        context: Dict[str, Any],
    ) -> None:
        """Persist PPO action decisions for offline review with sanitized payloads."""
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = self._sanitize_payload(
                {
                    "state_version": state.get("version"),
                    "state_vector": state.get("vector"),
                    "action": action,
                    "context": context,
                }
            )
            with LOG_PATH.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            LOGGER.error("Failed in _log_action: %s", exc)

    def _sanitize_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return JSON-safe deep copy of payload to avoid circular references."""
        seen: set[int] = set()

        def _sanitize(obj):
            obj_id = id(obj)
            if obj_id in seen:
                return "<recursion>"
            if isinstance(obj, dict):
                seen.add(obj_id)
                return {str(k): _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                seen.add(obj_id)
                return [_sanitize(v) for v in obj]
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            return str(obj)

        return _sanitize(payload)

    @staticmethod
    def _serialize_numpy(value: Any) -> Any:
        """Convert numpy values to native Python types for JSON logging."""
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer)):
            return value.item()
        return value
