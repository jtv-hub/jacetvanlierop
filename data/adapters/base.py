#!/usr/bin/env python3
"""
Base interface for market data adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any


class MarketDataAdapter(ABC):
    """Abstract interface for pulling market data."""

    @abstractmethod
    def ping(self) -> bool:
        """Lightweight health check for the adapter (e.g., API status)."""

    @abstractmethod
    def get_ticker(self, pair: str) -> Dict[str, Any]:
        """
        Return the latest best bid/ask/last trade price and timestamp for a pair.
        Pair format is adapter-specific (we normalize user input when feasible).
        """
