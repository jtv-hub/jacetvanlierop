"""
Signal Validator
Ensures multiple indicators align before confirming a trade.
"""

import logging


def validate_signals(signals: dict, min_confirmations: int = 2) -> tuple[bool, float]:
    """
    Validate whether a trade should be executed based on signal confirmations.

    Args:
        signals (dict): Example:
            {
                "RSI": {"signal": "buy", "confidence": 0.7},
                "MACD": {"signal": "buy", "confidence": 0.6},
                "VWAP": {"signal": "neutral", "confidence": 0.0}
            }
        min_confirmations (int): Minimum number of aligned signals required.

    Returns:
        (bool, float): (is_confirmed, avg_confidence)
    """
    try:
        if not signals:
            return False, 0.0

        # Count aligned signals
        votes = {"buy": 0, "sell": 0}
        confidences = {"buy": [], "sell": []}

        for data in signals.values():
            sig = data.get("signal", "neutral")
            conf = data.get("confidence", 0.0)
            if sig in ["buy", "sell"]:
                votes[sig] += 1
                confidences[sig].append(conf)

        # Decide on majority direction
        if votes["buy"] >= min_confirmations:
            avg_conf_buy = sum(confidences["buy"]) / len(confidences["buy"])
            return True, avg_conf_buy
        elif votes["sell"] >= min_confirmations:
            avg_conf_sell = sum(confidences["sell"]) / len(confidences["sell"])
            return True, avg_conf_sell
        else:
            return False, 0.0
    except (KeyError, TypeError, ZeroDivisionError) as e:
        logging.error("❌ Signal validation failed: %s", e)
        return False, 0.0


if __name__ == "__main__":
    # Quick test
    test_signals = {
        "RSI": {"signal": "buy", "confidence": 0.7},
        "MACD": {"signal": "buy", "confidence": 0.65},
        "VWAP": {"signal": "neutral", "confidence": 0.0},
    }
    confirmed, avg_conf = validate_signals(test_signals, min_confirmations=2)
    print("Confirmed?", confirmed, "Confidence:", avg_conf)
