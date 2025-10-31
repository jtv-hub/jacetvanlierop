"""
train_ppo.py

Batch trainer for the PPO agent using historical trades stored in db/trades.db.
Builds short replay sequences from trades that contain a 60-dim state vector
and updates the PPO policy accordingly.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from crypto_trading_bot.learning.ppo_agent import PPOAgent
from crypto_trading_bot.learning.reward_shaper import compute_reward
from crypto_trading_bot.utils.sqlite_logger import log_learning_feedback

TRADES_DB_PATH = Path("db/trades.db")
EPISODE_LENGTH = 100


def _load_samples(db_path: Path) -> Sequence[Tuple[str, np.ndarray, float]]:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT timestamp, payload, state_vector
            FROM trades
            WHERE state_vector IS NOT NULL
            ORDER BY timestamp ASC
            """
        ).fetchall()
    finally:
        conn.close()

    samples: List[Tuple[str, np.ndarray, float]] = []
    for row in rows:
        payload_raw = row["payload"]
        state_raw = row["state_vector"]
        if not payload_raw or not state_raw:
            continue
        try:
            trade = json.loads(payload_raw)
            state_vals = json.loads(state_raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(state_vals, list) or len(state_vals) != 60:
            continue
        try:
            state = np.asarray(state_vals, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        roi = trade.get("roi") or 0.0
        fee = trade.get("fee") or 0.0
        slippage_entry = trade.get("entry_slippage_amount") or 0.0
        slippage_exit = trade.get("exit_slippage_amount") or 0.0
        drawdown = trade.get("drawdown") or 0.0
        risk_alloc = trade.get("capital_buffer") or 0.0
        reward = compute_reward(
            roi=float(roi or 0.0),
            fee=float(fee or 0.0),
            slippage=float(slippage_entry or 0.0) + float(slippage_exit or 0.0),
            drawdown=float(drawdown or 0.0),
            risk_alloc=float(risk_alloc or 0.0),
        )
        samples.append((row["timestamp"], state, reward))
    return samples


def _chunk_sequences(samples: Sequence[Tuple[str, np.ndarray, float]], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    if not samples:
        raise ValueError("No samples available for training.")
    states: List[np.ndarray] = []
    rewards: List[float] = []
    idx = 0
    while idx < len(samples):
        chunk = samples[idx : idx + max_len]
        for _, state, reward in chunk:
            states.append(state)
            rewards.append(float(reward))
        idx += max_len
    return np.stack(states, axis=0), np.asarray(rewards, dtype=np.float32)


def _log_training_event(metrics: Dict[str, float], params: Dict[str, object], model_path: Path) -> None:
    suggestion_id = f"ppo-train-{int(time.time())}"
    payload = {
        "suggestion_id": suggestion_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "strategy": "ppo_agent",
        "status": "training_completed",
        "parameters": params,
        "accepted": False,
        "action": None,
        "actual_roi": None,
        "reward": metrics.get("avg_reward"),
        "samples_used": params.get("samples_used"),
        "timesteps": metrics.get("timesteps"),
        "kl_divergence": metrics.get("kl_divergence"),
        "entropy": metrics.get("entropy_loss"),
        "clip_fraction": metrics.get("clip_fraction"),
        "model_version": str(model_path),
    }
    log_learning_feedback(payload)


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent on historical trades.")
    parser.add_argument("--timesteps", type=int, default=50000, help="Total PPO timesteps to learn.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint if present.")
    parser.add_argument("--dry-run", action="store_true", help="Load data but skip PPO updates.")
    parser.add_argument("--seed", type=int, default=None, help="Override PPO seed.")
    parser.add_argument("--db-path", type=Path, default=TRADES_DB_PATH, help="Path to trades SQLite database.")
    return parser.parse_args(list(argv))


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    try:
        samples = _load_samples(args.db_path)
    except FileNotFoundError as exc:
        print(f"[PPO] trades database missing: {exc}", file=sys.stderr)
        return 1
    if not samples:
        print("[PPO] No training samples with state vectors available.", file=sys.stderr)
        return 1

    samples_sorted = sorted(samples, key=lambda item: item[0] or "")
    states, rewards = _chunk_sequences(samples_sorted, EPISODE_LENGTH)
    samples_used = len(states)
    print(f"[PPO] Loaded {samples_used} samples from {args.db_path}")

    if args.dry_run:
        print("[PPO] Dry-run complete (no training executed).")
        return 0

    agent = PPOAgent(seed=args.seed, load_existing=args.resume)
    metrics = agent.train_on_batch(states, rewards, timesteps=max(args.timesteps, 1))
    save_path = agent.save()
    metrics.setdefault("kl_divergence", None)
    metrics.setdefault("entropy_loss", None)
    metrics.setdefault("clip_fraction", None)
    print(
        "[PPO] Training finished | timesteps={timesteps} avg_reward={avg:.6f} "
        "kl={kl:.6f} entropy={entropy:.6f} clip={clip:.6f}".format(
            timesteps=metrics.get("timesteps"),
            avg=metrics.get("avg_reward", 0.0),
            kl=metrics.get("kl_divergence", float("nan")) or 0.0,
            entropy=metrics.get("entropy_loss", float("nan")) or 0.0,
            clip=metrics.get("clip_fraction", float("nan")) or 0.0,
        )
    )
    params = {
        "timesteps": args.timesteps,
        "resume": args.resume,
        "seed": args.seed,
        "samples_used": samples_used,
    }
    _log_training_event(metrics, params, save_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
