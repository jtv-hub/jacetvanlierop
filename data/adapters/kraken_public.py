"""
Kraken public REST adapter (no API keys needed).

- Uses urllib with an SSL context pinned to certifi's CA bundle so HTTPS verification
  succeeds on macOS/Python.
- Minimal endpoints: Time, Ticker.
"""

from __future__ import annotations

import json
import ssl
import time
from typing import Any, Dict, Optional
from urllib import error, parse, request  # noqa: S310 (urllib is fine for simple GETs)

import certifi


class KrakenPublicAdapter:
    """Lightweight client for Kraken's public API (public endpoints only)."""

    def __init__(
        self,
        base: str | None = None,
        timeout: float = 10.0,
        retries: int = 2,
        backoff: float = 0.5,
    ) -> None:
        self.base = base or "https://api.kraken.com/0/public"
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        # SSL context using certifi's CA store to avoid certificate issues
        self._ssl_ctx = ssl.create_default_context(cafile=certifi.where())

    # ------------------------------- helpers -------------------------------- #

    @staticmethod
    def _map_pair(pair: str) -> str:
        """
        Map a friendly pair like 'BTC/USD' to Kraken's pair code.
        Extend this mapping as needed.
        """
        mapping = {
            "BTC/USD": "XXBTZUSD",
            "ETH/USD": "XETHZUSD",
        }
        return mapping.get(pair.upper(), pair.replace("/", ""))

    def _request(
        self,
        endpoint: str,
        params: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        GET {base}/{endpoint}?{params}, return parsed JSON 'result' object.
        Retries a few times with exponential backoff on transient errors.
        """
        url = f"{self.base}/{endpoint}"
        if params:
            url = f"{url}?{parse.urlencode(params)}"

        attempt = 0
        while True:
            try:
                with request.urlopen(
                    url, timeout=self.timeout, context=self._ssl_ctx
                ) as resp:
                    if resp.status != 200:
                        raise error.HTTPError(
                            url, resp.status, "non-200", resp.headers, None
                        )
                    payload = json.loads(resp.read().decode("utf-8"))

                if payload.get("error"):
                    # Kraken returns a list of error strings; surface them.
                    raise RuntimeError(f"Kraken error(s): {payload['error']}")

                result = payload.get("result")
                if not isinstance(result, dict):
                    raise ValueError("Malformed response: 'result' not a dict")
                return result

            except (
                error.URLError,
                error.HTTPError,
                ssl.SSLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                if attempt >= self.retries:
                    raise RuntimeError(f"HTTP error calling {url}: {exc}") from exc
                sleep_for = self.backoff * (2**attempt)
                time.sleep(sleep_for)
                attempt += 1

    # ------------------------------- endpoints ------------------------------ #

    def ping(self) -> bool:
        """Return True if the public API responds to /Time."""
        try:
            _ = self._request("Time")
            return True
        except (
            RuntimeError,
            error.URLError,
            error.HTTPError,
            ssl.SSLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ):
            # We deliberately catch the specific network/parse errors we can get from _request.
            return False

    def get_ticker(self, pair: str) -> Dict[str, Any]:
        """
        Return a normalized ticker snapshot:
        {
          'ts_exchange': <epoch int>,
          'pair': 'BTC/USD',
          'exchange_pair': 'XXBTZUSD',
          'last':  ..., 'bid': ..., 'ask': ..., 'volume_24h': ...,
          'source': 'kraken'
        }
        """
        kpair = self._map_pair(pair)
        result = self._request("Ticker", {"pair": kpair})
        entry = result.get(kpair)
        if not isinstance(entry, dict):
            raise ValueError(f"No ticker data for {kpair}")

        # Per Kraken docs:
        # 'c' last trade [price, volume], 'b' best bid [price, lot vol],
        # 'a' best ask [price, lot vol], 'v' volume [today, last 24h]
        def _first_f(x: Any) -> float:
            try:
                return float(x[0])
            except (TypeError, ValueError, IndexError) as exc:
                raise ValueError(f"Unexpected field format: {x}") from exc

        last = _first_f(entry.get("c", ["nan"]))
        bid = _first_f(entry.get("b", ["nan"]))
        ask = _first_f(entry.get("a", ["nan"]))
        vol_24h = float(entry.get("v", [0.0, 0.0])[1])

        return {
            "ts_exchange": int(time.time()),
            "pair": pair,
            "exchange_pair": kpair,
            "last": last,
            "bid": bid,
            "ask": ask,
            "volume_24h": vol_24h,
            "source": "kraken",
        }
