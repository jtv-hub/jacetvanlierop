"""
Real-time price feed utilities for external market data providers.

Currently supports fetching ticker data from Kraken's public API.
"""

# Pylance may warn in some environments where the source isn't available.
try:
    import requests  # pylint: disable=import-error  # type: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    requests = None  # type: ignore[assignment]


def get_kraken_price(pair="XBTUSD"):
    """
    Fetches the current price for a Kraken trading pair.
    Example: 'XBTUSD' for BTC/USD
    """
    url = "https://api.kraken.com/0/public/Ticker"
    if requests is None:  # pylint: disable=import-error
        raise ImportError("The 'requests' package is required to fetch prices.")
    response = requests.get(url, params={"pair": pair})
    if response.status_code == 200:
        data = response.json()
        key = list(data["result"].keys())[0]
        return float(data["result"][key]["c"][0])
    raise ConnectionError(f"Failed to fetch price: {response.status_code}")


def get_current_price(pair: str = "BTC/USD") -> float | None:
    """
    Convenience wrapper that accepts a human-friendly pair like "BTC/USD",
    converts it to Kraken format (e.g., "XBTUSD"), and returns the latest price.

    Returns None if fetching fails so callers can skip gracefully.
    """
    if not isinstance(pair, str) or "/" not in pair:
        print(f"⚠️ Invalid pair string: {pair}")
        return None
    # Kraken uses XBT for BTC, others map 1:1; remove slash for Kraken API
    kraken_pair = pair.replace("BTC", "XBT").replace("/", "")
    try:
        price = get_kraken_price(kraken_pair)
        print(f"[FEED] get_kraken_price called for {pair} as {kraken_pair}: {price}")
        return price
    except Exception as e:  # pylint: disable=broad-exception-caught
        print(f"⚠️ Failed to fetch real price for {pair}: {e}")
        return None
