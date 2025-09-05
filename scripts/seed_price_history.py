"""
Seed historical daily close prices from Kraken OHLC API.

Pairs:
  BTC/USD -> XBTUSD
  ETH/USD -> ETHUSD
  XRP/USD -> XRPUSD
  SOL/USD -> SOLUSD
  LINK/USD -> LINKUSD

Writes JSON file at `data/seeded_prices.json` in the format:

price_history_cache = {
  "ETH/USD": [
      ["2025-08-01T00:00:00", 2821.45],
      ["2025-08-02T00:00:00", 2877.21],
      ...
  ],
  ...
}

Notes:
- Uses UTC ISO timestamps via datetime.utcfromtimestamp(ts).isoformat()
- Fetches at 1-day interval for the last 30 candles
- Sleeps 1s between requests to respect rate limits
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List

try:
    # Prefer stdlib to avoid external dependency
    from urllib.parse import urlencode
    from urllib.request import urlopen
except ImportError:  # pragma: no cover
    urlopen = None  # type: ignore[assignment]
    urlencode = None  # type: ignore[assignment]


API_BASE = "https://api.kraken.com/0/public/OHLC"
OUT_PATH = "data/seeded_prices.json"

# Mapping from display pair to Kraken API pair
PAIR_MAP: Dict[str, str] = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
    "XRP/USD": "XRPUSD",
    "SOL/USD": "SOLUSD",
    "LINK/USD": "LINKUSD",
}


def _build_url(pair_code: str, interval: int = 1440) -> str:
    params = {"pair": pair_code, "interval": interval}
    return f"{API_BASE}?{urlencode(params)}"


def _fetch_ohlc(pair_code: str) -> List[List[Any]]:
    if urlopen is None:
        raise RuntimeError("urllib not available in this environment")
    url = _build_url(pair_code)
    with urlopen(url, timeout=30) as resp:  # nosec - trusted public API
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("error"):
        raise RuntimeError(f"Kraken API error for {pair_code}: {data['error']}")
    result = data.get("result", {})
    # Result key may vary; attempt direct then fallback to first non-'last'
    rows = result.get(pair_code)
    if rows is None:
        rows = next((v for k, v in result.items() if k != "last"), [])
    if not isinstance(rows, list):
        raise RuntimeError(f"Unexpected OHLC format for {pair_code}")
    return rows


def _to_iso(ts: float | int) -> str:
    # Use naive UTC ISO string without offset for readability
    return datetime.utcfromtimestamp(int(ts)).isoformat()


def seed_prices() -> Dict[str, List[List[float]]]:
    """Fetch and persist 30 daily close prices per pair.

    Uses Kraken's public OHLC endpoint at 1-day interval and writes a
    JSON mapping of display pair to a list of ``[iso_timestamp, close]``
    items at ``data/seeded_prices.json``.
    """
    os.makedirs("data", exist_ok=True)
    cache: Dict[str, List[List[float]]] = {}
    for display_pair, api_pair in PAIR_MAP.items():
        try:
            rows = _fetch_ohlc(api_pair)
            # Each row: [time, open, high, low, close, vwap, volume, count]
            last_30 = rows[-30:]
            out: List[List[float]] = []
            for r in last_30:
                if not isinstance(r, list) or len(r) < 5:
                    continue
                ts = r[0]
                close = r[4]
                try:
                    iso = _to_iso(ts)
                    price = float(close)
                except (ValueError, TypeError):
                    continue
                out.append([iso, price])
            cache[display_pair] = out
            print(f"[SEED] {display_pair} seeded with {len(out)} candles")
        except Exception as e:  # pylint: disable=broad-exception-caught
            print(f"[SEED] Failed for {display_pair}: {e}")
        finally:
            time.sleep(1)

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)
    return cache


if __name__ == "__main__":
    seed_prices()
