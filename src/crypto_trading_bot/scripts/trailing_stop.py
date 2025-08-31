#!/usr/bin/env python3
"""
Reusable trailing-stop helper for the bot.

Supports two modes:
- pct: percent of entry (e.g., 0.004 = 0.4%)
- atr: ATR multiple (requires you to feed an ATR value)

Usage (long example):
    from trailing_stop import TrailingStop, TrailConfig

    cfg = TrailConfig(mode="pct", pct=0.004, activate=0.0015, step=0.0)
    trail = TrailingStop(side="long", entry=100.0, cfg=cfg)

    # on each new price:
    decision = trail.update(price=101.0)
    if decision.exit:
        print("Exit due to", decision.exit_reason)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Side = Literal["long", "short"]
Mode = Literal["pct", "atr"]


@dataclass
class TrailConfig:
    """Configuration for a trailing stop."""

    mode: Mode = "pct"
    pct: float = 0.004  # 0.4% trail if mode="pct"
    atr_mult: float = 2.0  # used if mode="atr"
    activate: float = 0.0  # e.g., 0.0015: arm trail after +0.15% move
    step: float = 0.0  # snap stop to nearest step (0.0 = no snap)


@dataclass
class TrailDecision:
    """Result of a single update step."""

    exit: bool
    exit_reason: Optional[str]
    stop: float
    max_fav: float


class TrailingStop:
    """
    Stateful trailing stop.

    For longs:
      - We track the best favorable price (max_fav).
      - Stop is either entry*(1-pct) or max_fav*(1-pct) once activated, or
        for ATR mode: entry-ATR*mult or max_fav-ATR*mult.
    For shorts: symmetric.
    """

    def __init__(self, side: Side, entry: float, cfg: TrailConfig, atr: float | None = None):
        if side not in ("long", "short"):
            raise ValueError("side must be 'long' or 'short'")
        if cfg.mode == "atr" and (atr is None or atr <= 0):
            raise ValueError("ATR mode requires a positive atr value")

        self.side = side
        self.entry = float(entry)
        self.cfg = cfg
        self.atr = float(atr) if atr is not None else None

        self.max_fav = entry  # best favorable price seen
        self.armed = cfg.activate <= 0.0
        self.stop = self._initial_stop()

    def _initial_stop(self) -> float:
        if self.cfg.mode == "pct":
            if self.side == "long":
                return self.entry * (1.0 - self.cfg.pct)
            return self.entry * (1.0 + self.cfg.pct)
        # ATR
        assert self.atr is not None
        if self.side == "long":
            return self.entry - self.atr * self.cfg.atr_mult
        return self.entry + self.atr * self.cfg.atr_mult

    def _snap(self, x: float) -> float:
        step = self.cfg.step
        if step and step > 0:
            # snap to nearest step; for safety, bias slightly in stop's favor
            return round(x / step) * step
        return x

    def _update_activation(self, price: float) -> None:
        if self.armed:
            return
        if self.cfg.activate <= 0.0:
            self.armed = True
            return

        if self.side == "long":
            if price >= self.entry * (1.0 + self.cfg.activate):
                self.armed = True
        else:
            if price <= self.entry * (1.0 - self.cfg.activate):
                self.armed = True

    def _update_max_fav(self, price: float) -> None:
        if self.side == "long":
            self.max_fav = max(self.max_fav, price)
        else:
            self.max_fav = min(self.max_fav, price)

    def _compute_stop(self) -> float:
        if not self.armed:
            return self.stop  # unchanged until armed

        if self.cfg.mode == "pct":
            if self.side == "long":
                base = self.max_fav * (1.0 - self.cfg.pct)
                return max(self.stop, base)
            base = self.max_fav * (1.0 + self.cfg.pct)
            return min(self.stop, base)

        # ATR mode
        assert self.atr is not None
        if self.side == "long":
            base = self.max_fav - self.atr * self.cfg.atr_mult
            return max(self.stop, base)
        base = self.max_fav + self.atr * self.cfg.atr_mult
        return min(self.stop, base)

    def update(self, price: float) -> TrailDecision:
        """Feed the next price; returns decision with possibly-updated stop."""
        price = float(price)
        self._update_activation(price)
        self._update_max_fav(price)
        new_stop = self._snap(self._compute_stop())
        self.stop = new_stop

        # Check exit
        if self.side == "long":
            hit = price <= self.stop
        else:
            hit = price >= self.stop

        exit_reason = "trail_stop" if hit and self.armed else None
        return TrailDecision(
            exit=bool(hit),
            exit_reason=exit_reason,
            stop=self.stop,
            max_fav=self.max_fav,
        )
