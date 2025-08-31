"""
Base strategy interface for crypto trading strategies.
"""

from abc import ABC, abstractmethod


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """

    @abstractmethod
    def generate_trade_signal(self, market_data: dict) -> dict:
        """
        Given market data, return a trade signal dictionary.
        Must include at least:
        - 'executed': bool
        - 'confidence_score': float (0.0 to 1.0)
        - Any additional metrics used for decision-making
        """
        raise NotImplementedError("Subclasses must implement generate_trade_signal.")
