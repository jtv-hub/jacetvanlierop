"""
Defines parameter encoding and helper methods for NSGA-3 individuals.
"""

from __future__ import annotations

import random
from typing import Any, Dict


class Individual:
    """Encapsulates a single candidate parameter set."""

    PARAM_BOUNDS = {
        "rsi_low": {"min": 10, "max": 40},
        "rsi_high": {"min": 60, "max": 90},
        "ema_fast": {"min": 10, "max": 50},
        "ema_slow": {"min": 50, "max": 200},
        "trailing_sl_atr": {"min": 0.5, "max": 3.0},
        "confidence_min": {"min": 0.3, "max": 0.8},
        "ppo_lr": {"min": 1e-5, "max": 1e-3},
        "ppo_clip": {"min": 0.1, "max": 0.3},
        "risk_per_trade_pct": {"min": 0.5, "max": 3.0},
    }

    def __init__(  # pylint: disable=too-many-arguments
        self,
        params: Dict[str, float],
        objectives: Dict[str, float] | None = None,
        *,
        rank: int = 0,
        crowding_distance: float = 0.0,
        constraint_violation: float = 0.0,
        metadata: Dict[str, Any] | None = None,
    ):
        self.params = params
        self.objectives = objectives or {"roi": 0.0, "drawdown": 0.0, "win_rate": 0.0}
        self.rank = rank
        self.crowding_distance = crowding_distance
        self.constraint_violation = constraint_violation
        self.metadata: Dict[str, Any] = metadata or {}

    @classmethod
    def random(
        cls,
        bounds: Dict[str, Dict[str, float]] | None = None,
        rng: random.Random | None = None,
    ) -> "Individual":
        """Return an individual sampled uniformly from parameter bounds."""
        rng = rng or random.Random()
        rng_bounds = bounds or cls.PARAM_BOUNDS
        params = {key: rng.uniform(value["min"], value["max"]) for key, value in rng_bounds.items()}
        return cls(params, metadata={"origin": "random"})

    def clamp(self, bounds: Dict[str, Dict[str, float]] | None = None) -> None:
        """Clamp params to stay within bounds."""
        rng_bounds = bounds or self.PARAM_BOUNDS
        for key, limit in rng_bounds.items():
            if key in self.params:
                self.params[key] = max(limit["min"], min(limit["max"], self.params[key]))

    def copy(self) -> "Individual":
        """Return a deep copy of the individual."""
        return Individual(
            self.params.copy(),
            self.objectives.copy(),
            rank=self.rank,
            crowding_distance=self.crowding_distance,
            constraint_violation=self.constraint_violation,
            metadata=self.metadata.copy(),
        )

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "params": self.params,
            "objectives": self.objectives,
            "rank": self.rank,
            "crowding_distance": self.crowding_distance,
            "constraint_violation": self.constraint_violation,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Individual":
        """Rehydrate an individual from checkpoint data."""
        return cls(
            params=dict(payload.get("params") or {}),
            objectives=dict(payload.get("objectives") or {}),
            rank=int(payload.get("rank", 0)),
            crowding_distance=float(payload.get("crowding_distance", 0.0)),
            constraint_violation=float(payload.get("constraint_violation", 0.0)),
            metadata=dict(payload.get("metadata") or {}),
        )
