#!/usr/bin/env python3
"""
Kraken PUBLIC ticker capture → data/live/kraken_<SYMBOL>_YYYYMMDD.jsonl

Usage:
    python scripts/capture_kraken_public.py BTCUSD [ETHUSD ...]

Writes one JSON line per run, e.g.:
    {"ts": "2025-08-14T17:20:47Z", "pair": "BTCUSD", "price": 64231.2}

This uses only Kraken's *public* REST endpoint. No API keys required.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import ssl

# Prefer certifi if available to avoid SSL issues, otherwise fall back.
try:
    import certifi  # type: ignore

    _SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    _SSL_CTX = ssl.create_default_context()

# Map our symbols to Kraken's pairs
PAIR_MAP: dict[str, str] = {
    "BTCUSD": "XBTUSD",
    "ETHUSD": "ETHUSD",
    "SOLUSD": "SOLUSD",
}


def now_utc_iso() -> str:
    """Return current UTC time as an ISO-8601 string with a trailing 'Z'."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def out_path(sym: str) -> Path:
    """Return the output path for today's JSONL file for the given symbol."""
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = Path("data/live")
    path.mkdir(parents=True, exist_ok=True)
    return path / f"kraken_{sym}_{day}.jsonl"


def _fetch_once(sym: str) -> Tuple[float | None, str | None]:
    """Fetch once and return (price, error_message). Uses a single return for lint cleanliness."""
    price: float | None = None
    error: str | None = None

    pair = PAIR_MAP[sym]
    url = f"https://api.kraken.com/0/public/Ticker?pair={pair}"

    try:
        with urlopen(url, timeout=10, context=_SSL_CTX) as resp:
            data = json.load(resp)
    except HTTPError as e:
        error = f"HTTP {e.code}"
    except URLError as e:
        error = f"URL error: {getattr(e, 'reason', e)}"
    except ssl.SSLError as e:
        error = f"SSL error: {e}"
    except TimeoutError:
        error = "timeout"
    except json.JSONDecodeError as e:
        error = f"json decode error: {e}"
    else:
        if data.get("error"):
            error = f"kraken error: {data['error']}"
        else:
            result = data.get("result", {})
            if not result:
                error = "empty result"
            else:
                inner = next(iter(result.values()))
                try:
                    # "c" = last trade price
                    price = float(inner["c"][0])
                except (KeyError, TypeError, ValueError) as e:
                    error = f"parse error: {e}"

    return price, error


def fetch_last(sym: str) -> Tuple[float | None, str | None]:
    """Fetch last price with a brief retry and return (price, error_message)."""
    price, _ = _fetch_once(sym)
    if price is not None:
        return price, None
    time.sleep(0.5)  # quick retry
    return _fetch_once(sym)


def main(argv: list[str]) -> int:
    """Entry point for command-line execution.

    Args:
        argv: Command-line arguments where argv[1:] are symbols like BTCUSD.

    Returns:
        0 on success, 2 on usage error.
    """
    if len(argv) < 2:
        print("usage: capture_kraken_public.py BTCUSD [ETHUSD ...]", file=sys.stderr)
        return 2

    syms = argv[1:]
    for sym in syms:
        if sym not in PAIR_MAP:
            print(f"[WARN] unsupported symbol: {sym}", file=sys.stderr)
            continue

        price, error_msg = fetch_last(sym)
        if price is None:
            print(f"[WARN] {sym}: no price fetched ({error_msg})", file=sys.stderr)
            continue

        line = {"ts": now_utc_iso(), "pair": sym, "price": price}
        path = out_path(sym)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"[OK] {sym} {price} → {path}")
        time.sleep(0.2)  # light pacing if multiple symbols

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
