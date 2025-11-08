"""
train_ppo.py
============
Production-ready PPO training harness for the trading bot.

This script:
- Loads PPO experience data (created by ppo_agent + shadow logger)
- Builds daily-episode reinforcement learning environments
- Uses Stable Baselines 3 PPO with proper 15‑dim observation space
- Implements a realistic reward function using ROI, confidence,
  drawdown, regime correctness, and slippage penalties
- Performs an 80/20 train/validation split
- Saves model + metadata for PPOAgent to load in live trading
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

# === PARAMETERS ===
STATE_DIM = 15  # Must match state_builder.STATE_VECTOR_SIZE - EMBEDDING_DIM
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODEL_DIR / "ppo_agent_v1.zip"
META_PATH = MODEL_DIR / "ppo_agent_v1_meta.json"
EXPERIENCE_LOG = Path("logs/ppo_experience.jsonl")
USE_DUMMY_EXPERIENCE = True

logger = logging.getLogger("train_ppo")
logger.setLevel(logging.INFO)


# ==============================================================================
#  ENVIRONMENT: Each trading day = one PPO episode
# ==============================================================================


class TradingDayEnv(Env):
    """
    A single episode representing one full trading day.

    Each step processes one PPO experience record:
      - observation: 15-dim state vector
      - reward: computed from ROI, drawdown, slippage, and confidence
      - done: true at end of day
    """

    metadata = {"render_modes": []}

    def __init__(self, day_records: List[Dict[str, Any]]):
        super().__init__()
        self.records = day_records
        self.index = 0

        # Match state_builder vector size (WITHOUT embedding)
        self.observation_space = spaces.Box(low=-5, high=5, shape=(STATE_DIM,), dtype=np.float32)
        # Three PPO outputs: size_scalar, allow_entry, agent_confidence
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.0, 0.0]),
            high=np.array([1.5, 1.0, 1.0]),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        del options  # gym compatibility; environment has no configurable options
        self.index = 0
        first_record = self.records[0]
        return self._extract_observation(first_record), {}

    def render(self):
        """Gym render stub; environment has no visual output."""
        return None

    def _extract_observation(self, rec: Dict[str, Any]) -> np.ndarray:
        vector = rec.get("state_vector") or [0.0] * STATE_DIM
        vector = np.array(vector[:STATE_DIM], dtype=np.float32)
        return vector

    def _compute_reward(self, rec: Dict[str, Any]) -> float:
        """
        Reward uses:
        - Positive: ROI (USD), confidence alignment
        - Penalty: large drawdown, wrong regime, slippage
        """
        roi = float(rec.get("roi_usd", 0.0))
        confidence = float(rec.get("confidence", 0.0))
        drawdown_pct = float(rec.get("drawdown_pct", 0.0))
        slippage_bps = float(rec.get("slippage_bps", 0.0))
        correct_regime = bool(rec.get("correct_regime", False))

        reward = (
            roi * confidence
            - 0.5 * max(0, drawdown_pct - 5) ** 2
            - 0.0001 * slippage_bps
            + (1.0 if correct_regime else 0.0)
        )
        return float(reward)

    def step(self, action: np.ndarray):
        rec = self.records[self.index]

        reward = self._compute_reward(rec)

        self.index += 1
        terminated = self.index >= len(self.records)

        if terminated:
            next_obs = self._extract_observation(rec)
        else:
            next_obs = self._extract_observation(self.records[self.index])

        return next_obs, reward, terminated, False, {}


# ==============================================================================
#  EXPERIENCE LOG LOADING
# ==============================================================================


def load_experience() -> Dict[str, List[Dict[str, Any]]]:
    """Load PPO experience rows grouped into a dict keyed by trading day."""
    if not EXPERIENCE_LOG.exists():
        logger.error(
            "PPO training aborted: experience log is missing. " "Run the trading bot to collect experience first.",
        )
        sys.exit(1)

    daily: Dict[str, List[Dict[str, Any]]] = {}

    with EXPERIENCE_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            ts = rec.get("timestamp") or datetime.now(timezone.utc).isoformat()
            date_key = ts.split("T")[0]

            daily.setdefault(date_key, []).append(rec)

    return daily


def _load_dummy_experience() -> Dict[str, List[Dict[str, Any]]]:
    """Return a minimal deterministic dataset for developer experiments."""
    base_date = datetime.now(timezone.utc).date()
    dummy: Dict[str, List[Dict[str, Any]]] = {}
    for offset in range(10):
        day = (base_date - timedelta(days=offset)).isoformat()
        record = {
            "timestamp": f"{day}T12:00:00+00:00",
            "state_vector": [0.01 * (offset + 1)] * STATE_DIM,
            "roi_usd": float(5 - offset * 0.1),
            "confidence": 0.5 + 0.02 * offset,
            "drawdown_pct": 1.0 + offset * 0.2,
            "slippage_bps": 5.0 + offset,
            "correct_regime": offset % 2 == 0,
        }
        dummy[day] = [record]
    return dummy


# ==============================================================================
#  TRAINING PIPELINE
# ==============================================================================


def build_envs(daily_records: Dict[str, List[Dict[str, Any]]]):
    """
    Creates Stable Baselines VecEnvs for training and validation.
    """
    days = sorted(daily_records.keys())
    if len(days) < 10:
        raise RuntimeError("Not enough PPO experience to train (need at least 10 days).")

    split = int(len(days) * 0.8)
    train_days = days[:split]
    val_days = days[split:]

    train_env_fns = [lambda day=d: TradingDayEnv(daily_records[day]) for d in train_days]
    val_env_fns = [lambda day=d: TradingDayEnv(daily_records[day]) for d in val_days]

    train_env = DummyVecEnv(train_env_fns)
    val_env = DummyVecEnv(val_env_fns)

    return train_env, val_env, train_days, val_days


def save_metadata(meta: Dict[str, Any]):
    """Persist PPO training metadata alongside the model artifact."""
    with META_PATH.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def train():
    """
    Train a PPO agent using daily episode environments built from logged experience.

    Loads the experience dataset, constructs training and validation environments,
    trains a PPO model on the training set, evaluates it on the validation set,
    saves the model and training netadata, and log all major steps.
    """
    logger.info("Loading PPO experience...")
    if USE_DUMMY_EXPERIENCE:
        logger.warning("Using dummy PPO experience for development/testing.")
        daily_records = _load_dummy_experience()
    else:
        daily_records = load_experience()

    logger.info("Building environments...")
    train_env, val_env, train_days, val_days = build_envs(daily_records)

    logger.info("Initializing PPO model...")
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./tb_logs/",
        learning_rate=3e-4,
        batch_size=64,
        n_steps=512,
    )

    logger.info("Training PPO...")
    model.learn(total_timesteps=200_000)

    logger.info("Evaluating PPO on validation split...")
    mean_reward, std_reward = evaluate_policy(model, val_env, n_eval_episodes=5)

    logger.info("Validation Reward Mean: %.4f, Std: %.4f", mean_reward, std_reward)

    logger.info("Saving PPO model + metadata...")
    model.save(MODEL_PATH)

    metadata = {
        "state_version": "v1",
        "train_days": train_days,
        "val_days": val_days,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validation_mean_reward": mean_reward,
        "validation_std_reward": std_reward,
    }
    save_metadata(metadata)

    logger.info("PPO training completed successfully.")


if __name__ == "__main__":
    train()
