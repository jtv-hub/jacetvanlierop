"""
dual_threshold_strategies.py

Implements the DualThresholdStrategy class using absolute price thresholds.
"""


class DualThresholdStrategy:
    """
    Strategy that generates buy/sell/hold signals based on a recent average baseline.
    """

    def __init__(self, window: int = 20):
        self.window = window

    def validate_volume(self, volume, min_volume: int = 100) -> bool:
        """
        Validates whether the given volume meets a minimum threshold.
        Returns True if valid, False otherwise.
        """
        if volume is None or volume < min_volume:
            print(f"[DEBUG] Low volume: {volume} < {min_volume}")
            return False
        return True

    def generate_signal(self, price_data, volume):
        """
        Signals:
        - If close <= (1 - 0.02) * recent_avg -> buy, confidence 0.7
        - If close >= (1 + 0.02) * recent_avg -> sell, confidence 0.7
        - Else hold, confidence 0.0
        """
        if (
            not price_data
            or len(price_data) < max(self.window, 5)
            or any(p is None or p <= 0 for p in price_data)
            or not self.validate_volume(volume)
        ):
            print(f"[DEBUG] Invalid input: prices_len={len(price_data) if price_data else 0}, " f"volume={volume}")
            return {"signal": "hold", "confidence": 0.0}

        print(f"[DEBUG] Price data: {price_data[-5:]}")
        close_price = price_data[-1]
        recent = price_data[-self.window :]
        avg = sum(recent) / len(recent)

        lower = avg * 0.98
        upper = avg * 1.02
        print(
            f"[DEBUG] Close: {close_price}, " f"RecentAvg: {avg:.4f}, " f"Bands: lower={lower:.4f}, upper={upper:.4f}"
        )

        # Shared confidence scale 0.4..1.0 based on distance beyond band
        def scale(min_v: float, max_v: float, ratio: float) -> float:
            r = max(0.0, min(1.0, ratio))
            return min_v + (max_v - min_v) * r

        if close_price <= lower:
            ratio = (avg - close_price) / max(avg * 0.02, 1e-9)
            return {"signal": "buy", "confidence": scale(0.4, 1.0, ratio)}
        if close_price >= upper:
            ratio = (close_price - avg) / max(avg * 0.02, 1e-9)
            return {
                "signal": "sell",
                "confidence": scale(0.4, 1.0, ratio),
            }
        return {"signal": "hold", "confidence": 0.0}
