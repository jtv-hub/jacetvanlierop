"""Shared utilities for Kraken asset pair normalization.

This module exposes the central mapping between human-friendly trading
pairs (e.g. ``BTC/USDC``) and Kraken's altnames (e.g. ``XBTUSDC``). All
Kraken API helpers should import the mapping from here to avoid circular
imports between public and private client modules.
"""

from __future__ import annotations

from typing import Dict

PAIR_MAP: Dict[str, str] = {
    "BTC/USDC": "XBTUSDC",
    "ETH/USDC": "ETHUSDC",
    "SOL/USDC": "SOLUSDC",
    "XRP/USDC": "XRPUSDC",
    "LINK/USDC": "LINKUSDC",
}


def normalize_pair(pair: str) -> str:
    """Return Kraken's altname for a human-readable trading pair."""

    if not isinstance(pair, str) or "/" not in pair:
        raise ValueError(f"Invalid pair format: {pair!r}; expected like 'BTC/USD'")
    up = pair.upper()
    mapped = PAIR_MAP.get(up)
    if mapped:
        return mapped
    base, quote = up.split("/", 1)
    if base == "BTC":
        base = "XBT"
    return f"{base}{quote}"


__all__ = ["PAIR_MAP", "normalize_pair"]
