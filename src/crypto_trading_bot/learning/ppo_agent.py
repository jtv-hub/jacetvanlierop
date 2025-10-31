"""
ppo_agent.py

Lightweight wrapper around Stable-Baselines3 PPO tailored for the crypto
trading bot. Provides a singleton-style accessor as well as convenience
methods for inference and mini-batch retraining using precomputed
state/reward samples.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

try:  # gym compatibility shim
    import gym
except ImportError:  # pragma: no cover - gymnasium fallback
    import gymnasium as gym  # type: ignore[assignment]

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from crypto_trading_bot.config import CONFIG
from crypto_trading_bot.context.trading_context import TradingContext
from crypto_trading_bot.learning.reward_shaper import compute_reward
from crypto_trading_bot.utils.sqlite_logger import log_learning_feedback, update_learning_feedback
from crypto_trading_bot.utils.system_logger import get_system_logger

STATE_DIM = 60
ACTION_DIM = 3
DEFAULT_MODEL_DIR = Path("models/ppo")
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "ppo_agent.zip"
DEFAULT_TENSORBOARD_DIR = Path("tensorboard/ppo")
LOGGER = get_system_logger().getChild("ppo_agent")


def _log_ppo_metrics(suggestion_id: str, metrics: Dict[str, float]) -> None:
    payload = {
        "suggestion_id": suggestion_id,
        "strategy": "ppo_agent",
        "parameters": metrics,
        "accepted": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        update_learning_feedback(suggestion_id, payload)
    except Exception as exc:  # pragma: no cover - logging fallback
        LOGGER.debug("update_learning_feedback failed: %s", exc)
        try:
            log_learning_feedback(payload)
        except Exception as log_exc:  # pragma: no cover - logging fallback
            LOGGER.warning("PPO metrics logging failed: %s (error=%s)", payload, log_exc)


def _ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class _StaticStateEnv(gym.Env):
    """Minimal environment used for inference-only vectorisation."""

    metadata: Dict[str, Any] = {}

    def __init__(self, state_dim: int = STATE_DIM) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self._state = np.zeros(self.state_dim, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore[override]
        if seed is not None:
            super().reset(seed=seed)
        self._state[:] = 0.0
        return self._state.copy(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        # Single-step episode with zero reward; suitable for deterministic inference wrappers.
        return self._state.copy(), 0.0, True, False, {}

    def close(self) -> None:  # pragma: no cover - no resources to release
        return


class _BatchReplayEnv(gym.Env):
    """Replay environment that cycles through a fixed batch of states/rewards."""

    metadata: Dict[str, Any] = {}

    def __init__(self, states: np.ndarray, rewards: np.ndarray):
        super().__init__()
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError(f"states must be shaped (N, {STATE_DIM})")
        if rewards.shape[0] != states.shape[0]:
            raise ValueError("rewards must align with states length")
        self.states = states.astype(np.float32, copy=False)
        self.rewards = rewards.astype(np.float32, copy=False)
        self.action_space = gym.spaces.Discrete(ACTION_DIM)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(STATE_DIM,),
            dtype=np.float32,
        )
        self._index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:  # type: ignore[override]
        if seed is not None:
            super().reset(seed=seed)
        self._index = 0
        return self.states[self._index], {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:  # type: ignore[override]
        reward = float(self.rewards[self._index])
        self._index += 1
        done = self._index >= len(self.states)
        obs = self.states[self._index % len(self.states)]
        info: Dict[str, Any] = {}
        return obs, reward, done, False, info

    def close(self) -> None:  # pragma: no cover - no external handles
        return


class PPOAgent:
    """Encapsulates Stable-Baselines3 PPO with convenience helpers."""

    def __init__(
        self,
        *,
        model_path: Path | None = None,
        tensorboard_path: Path | None = None,
        seed: int | None = None,
        deterministic: bool | None = None,
        load_existing: bool = True,
    ) -> None:
        cfg = CONFIG.get("ppo", {})
        base_path = Path(model_path or DEFAULT_MODEL_PATH)
        if base_path.suffix:
            self.model_dir = base_path.parent
            self.model_path = base_path
        else:
            self.model_dir = base_path
            self.model_path = base_path / "ppo_agent.zip"

        self.tensorboard_dir = tensorboard_path or DEFAULT_TENSORBOARD_DIR
        self.seed = seed if seed is not None else int(cfg.get("seed", 12345))
        self.deterministic = bool(cfg.get("deterministic_inference", True)) if deterministic is None else deterministic

        _ensure_directory(self.model_dir)
        _ensure_directory(Path(self.tensorboard_dir))

        base_env = DummyVecEnv([lambda: _StaticStateEnv()])
        self._base_env = base_env

        if load_existing and self.model_path.exists():
            self.model = PPO.load(str(self.model_path), env=base_env, seed=self.seed)
        else:
            n_steps = int(CONFIG.get("ppo_n_steps", 2048))
            batch_size = int(CONFIG.get("ppo_batch_size", 64))
            learning_rate = float(CONFIG.get("ppo_learning_rate", 3e-4))
            n_epochs = int(CONFIG.get("ppo_n_epochs", 10))
            self.model = PPO(
                policy="MlpPolicy",
                env=base_env,
                verbose=0,
                seed=self.seed,
                tensorboard_log=str(self.tensorboard_dir),
                n_steps=n_steps,
                batch_size=batch_size,
                learning_rate=learning_rate,
                n_epochs=n_epochs,
            )
            # When starting fresh, remove any stale checkpoint to avoid accidental reload.
            if not load_existing and self.model_path.exists():
                try:
                    self.model_path.unlink()
                except OSError:
                    pass

    # --------------------------------------------------------------------- #
    # Public API                                                            #
    # --------------------------------------------------------------------- #

    def predict(self, state: np.ndarray, *, deterministic: bool | None = None) -> Tuple[int, float]:
        """Return (action, confidence) for a single 60-dim state vector."""

        if state.shape != (STATE_DIM,):
            raise ValueError(f"State must be shape ({STATE_DIM},), got {state.shape}")
        deterministic_inference = self.deterministic if deterministic is None else deterministic
        action, _ = self.model.predict(state, deterministic=deterministic_inference)

        obs_tensor, _ = self.model.policy.obs_to_tensor(state.reshape(1, -1))
        dist = self.model.policy.get_distribution(obs_tensor)
        try:
            probs = dist.distribution.probs  # type: ignore[attr-defined]
        except AttributeError:
            probs = dist.probs  # type: ignore[attr-defined]
        confidence = float(probs.max().item())
        return int(action), confidence

    def train_on_batch(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        timesteps: int,
        context: TradingContext | None = None,
    ) -> Dict[str, float]:
        """Perform a short PPO update using replayed state/reward samples."""

        if timesteps <= 0:
            raise ValueError("timesteps must be positive")
        states_array = np.asarray(states, dtype=np.float32)
        rewards_array = np.asarray(rewards, dtype=np.float32)
        states_array = np.nan_to_num(states_array.astype(np.float32))
        rewards_array = np.nan_to_num(rewards_array.astype(np.float32))
        if states_array.ndim != 2 or states_array.shape[1] != STATE_DIM:
            raise ValueError(f"states must be shaped (N, {STATE_DIM}); got {states_array.shape}")
        if rewards_array.shape[0] != states_array.shape[0]:
            raise ValueError(
                f"rewards length {rewards_array.shape[0]} does not match states length {states_array.shape[0]}"
            )

        if context is not None:
            shaped_rewards: list[float] = []
            timestamp_obj = getattr(context, "timestamp", None)
            timestamp_value = timestamp_obj.isoformat() if timestamp_obj else None
            for reward_value in rewards_array:
                trade = {
                    "roi": float(reward_value),
                    "pair": getattr(context, "pair", None),
                    "size": getattr(context, "position_size", None),
                    "entry_price": getattr(context, "entry_price", None),
                    "exit_price": getattr(context, "exit_price", None),
                    "timestamp": timestamp_value,
                }
                try:
                    shaped_value = compute_reward(
                        roi=trade["roi"],
                        fee=getattr(context, "fee", None),
                        slippage=getattr(context, "slippage", None),
                        drawdown=getattr(context, "drawdown", None),
                        risk_alloc=getattr(context, "risk_alloc", None),
                    )
                except Exception as exc:  # pragma: no cover - fallback to raw reward
                    LOGGER.debug("Reward shaping failed for trade %s: %s", trade, exc)
                    shaped_value = float(reward_value)
                shaped_rewards.append(float(shaped_value))
            rewards_array = np.nan_to_num(np.asarray(shaped_rewards, dtype=np.float32))

        timesteps = int(CONFIG.get("ppo_train_timesteps", timesteps))
        replay_env = DummyVecEnv([lambda: _BatchReplayEnv(states_array, rewards_array)])
        suggestion_id = f"ppo_train_{int(time.time())}"
        metrics: Dict[str, float]
        logger_values: Dict[str, Any] = {}
        with _AGENT_LOCK:
            self.model.set_env(replay_env)
            try:
                self.model.learn(
                    total_timesteps=timesteps,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
                logger_values = getattr(self.model.logger, "name_to_value", {}) or {}
                self.save()
                metrics = {
                    "avg_reward": float(np.mean(rewards_array)) if rewards_array.size else 0.0,
                    "kl_divergence": float(logger_values.get("train/approx_kl", 0.0)),
                    "entropy_loss": float(logger_values.get("train/entropy_loss", 0.0)),
                    "clip_fraction": float(logger_values.get("train/clip_fraction", 0.0)),
                    "timesteps": int(timesteps),
                }
                _log_ppo_metrics(suggestion_id, metrics)
            except Exception as exc:
                LOGGER.exception("PPO training failed during learn: %s", exc)
                raise
            finally:
                self.model.set_env(self._base_env)
                replay_env.close()
        return metrics

    def save(self, path: Path | None = None) -> Path:
        """Persist current model to disk."""

        target = path or self.model_path
        if target.is_dir():
            target = target / "ppo_agent.zip"
        _ensure_directory(target.parent)
        self.model.save(str(target))
        self.model_path = target
        return target

    def load(self, path: Path | None = None) -> None:
        """Load model weights from disk."""

        load_path = path or self.model_path
        if load_path.is_dir():
            load_path = load_path / "ppo_agent.zip"
        if not load_path.exists():
            raise FileNotFoundError(load_path)
        self.model = PPO.load(str(load_path), env=self._base_env, seed=self.seed)
        self.model_path = load_path

    @property
    def model_version(self) -> str:
        """Return a simple identifier for the currently loaded model."""

        return str(self.model_path)


_AGENT_SINGLETON: PPOAgent | None = None
_AGENT_LOCK = threading.Lock()


def get_agent() -> PPOAgent:
    """Return the shared PPOAgent instance (lazy-created)."""

    global _AGENT_SINGLETON  # pylint: disable=global-statement
    with _AGENT_LOCK:
        if _AGENT_SINGLETON is None:
            _AGENT_SINGLETON = PPOAgent()
        return _AGENT_SINGLETON
