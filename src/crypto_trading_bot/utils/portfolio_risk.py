"""
Portfolio Risk Logic

Determines which trades to execute based on exposure limits,
signal strength, and correlation between assets.
"""


def compute_correlations(signals, lookback="1h"):  # pylint: disable=unused-argument
    """
    Placeholder correlation logic. Replace with actual correlation
    from historical price data.

    Args:
        signals (list): List of signal dictionaries.
        lookback (str): Time window for correlation (e.g., '1h').

    Returns:
        dict: Correlation values between asset pairs.
    """
    correlations = {
        "BTC_ETH": 0.82,
        "BTC_SOL": 0.65,
        "BTC_XRP": 0.38,
        "BTC_LINK": 0.41,
        "ETH_SOL": 0.70,
        "ETH_XRP": 0.52,
        "ETH_LINK": 0.69,
        "SOL_XRP": 0.40,
        "SOL_LINK": 0.53,
        "XRP_LINK": 0.50,
    }
    return correlations


def portfolio_risk_logic(
    equity: float,
    signals: list,
    prices: dict,
    max_exposure: float = 0.06,
    max_trades: int = 3,
    correlations: dict | None = None,
):
    """
    Simple portfolio construction:
    - Sort signals by strength descending.
    - Add up to `max_trades` positions.
    - Skip assets highly correlated (> 0.7) with already-selected ones if a correlations map is provided.
    - Position size = equity * max_exposure / price (floor at tiny > 0).
    Returns a list of proposed trade dicts. (Keeps API stable for tests.)
    """
    correlations = correlations or {}
    picked = []
    picked_assets = set()

    # Normalize signals to have 'asset' and 'signal_score'
    norm = []
    for s in signals or []:
        a = s.get("asset")
        sc = s.get("signal_score", 0.0)
        if not a or a not in prices:
            continue
        norm.append((a, float(sc), s))
    norm.sort(key=lambda x: x[1], reverse=True)

    def _corr(a, b) -> float:
        if a == b:
            return 1.0
        key1 = f"{a}_{b}"
        key2 = f"{b}_{a}"
        return correlations.get(key1) or correlations.get(key2) or 0.0

    for asset, score, s in norm:
        # correlation screen against already picked assets
        ok = True
        for pa in picked_assets:
            if _corr(asset, pa) > 0.7:
                ok = False
                break
        if not ok:
            continue

        px = float(prices[asset])
        if px <= 0:
            continue
        units = (equity * float(max_exposure)) / px
        if units <= 0:
            continue

        picked.append(
            {
                "asset": asset,
                "units": units,
                "price": px,
                "action": "BUY",
                "signal_score": score,
                "strategy": s.get("strategy", "unknown"),
            }
        )
        picked_assets.add(asset)
        if len(picked) >= int(max_trades):
            break

    return picked
