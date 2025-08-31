#!/usr/bin/env python3
"""
Capture spot prices from Kraken's public REST API and write newline-delimited JSON
so the rest of the pipeline can treat it like a 'live' feed file.

Usage:
  python scripts/capture_kraken.py --pair BTCUSD --seconds 60 --interval 5 [--verbose]
Output:
  data/live/kraken_BTCUSD_YYYYMMDD.jsonl
Each line:
  {"ts":"2025-08-15T21:10:03+00:00","pair":"BTCUSD","price":118123.45,"source":"kraken"}
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

API = "https://api.kraken.com/0/public/Ticker?pair={pair}"

# Map “human” pair to Kraken’s preferred query code.
# (Kraken accepts alt names sometimes, but being explicit improves reliability.)
PAIR_MAP = {
    "BTCUSD": "XBTUSD",  # Kraken uses XBT, not BTC
    "ETHUSD": "ETHUSD",  # altname accepted
    "SOLUSD": "SOLUSD",  # altname accepted
}


def kraken_query_pair(user_pair: str) -> str:
    """Return the Kraken query code for a user-supplied pair."""
    up = user_pair.upper()
    return PAIR_MAP.get(up, up)


def dbg(verbose: bool, msg: str) -> None:
    """Print a debug line if verbose."""
    if verbose:
        print(msg)


def fetch_price(pair_for_user: str, *, verbose: bool = False) -> Optional[float]:
    """
    Fetch last price from Kraken Ticker for the given user-facing pair.
    Retries a couple times with a short delay to smooth over transient hiccups.
    Returns None on failure.
    """
    query_pair = kraken_query_pair(pair_for_user)
    url = API.format(pair=query_pair)

    # Build a request with a UA header; some endpoints are picky without it.
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "crypto-bot/1.0 (+https://example.com)",
            "Accept": "application/json",
        },
        method="GET",
    )

    attempts = 3
    for i in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                if r.status != 200:
                    dbg(
                        verbose,
                        f"[capture] HTTP {r.status} for {query_pair} " f"(attempt {i}/{attempts})",
                    )
                    time.sleep(0.6)
                    continue

                data = json.loads(r.read().decode("utf-8"))

            # Kraken returns {"error": [...], "result": {...}}
            if data.get("error"):
                dbg(
                    verbose,
                    f"[capture] Kraken error {data['error']} for {query_pair} " f"(attempt {i}/{attempts})",
                )
                time.sleep(0.6)
                continue

            result = data.get("result") or {}
            if not result:
                dbg(
                    verbose,
                    f"[capture] Empty result for {query_pair} " f"(attempt {i}/{attempts})",
                )
                time.sleep(0.6)
                continue

            # Pull the first item; payload has e.g. {"XBTUSD": {"c": ["117333.1", "1.0"], ...}}
            first = next(iter(result.values()))
            last_trade = first.get("c")
            if not last_trade or not last_trade[0]:
                dbg(
                    verbose,
                    f"[capture] Missing 'c' field for {query_pair} " f"(attempt {i}/{attempts})",
                )
                time.sleep(0.6)
                continue

            return float(last_trade[0])

        except urllib.error.HTTPError as err:
            dbg(
                verbose,
                f"[capture] HTTPError for {query_pair}: {err} (attempt {i}/{attempts})",
            )
            time.sleep(0.6)
        except urllib.error.URLError as err:
            dbg(
                verbose,
                f"[capture] URLError for {query_pair}: {err} (attempt {i}/{attempts})",
            )
            time.sleep(0.6)
        except (ValueError, KeyError) as err:
            dbg(
                verbose,
                f"[capture] Parse error for {query_pair}: {err} (attempt {i}/{attempts})",
            )
            time.sleep(0.6)

    return None


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", required=True, help="Kraken pair like BTCUSD, ETHUSD, SOLUSD")
    ap.add_argument("--seconds", type=int, default=60, help="How long to capture")
    ap.add_argument("--interval", type=int, default=5, help="Seconds between polls")
    ap.add_argument("--verbose", action="store_true", help="Print diagnostics")
    args = ap.parse_args()

    root = Path(".").resolve()
    out_dir = root / "data" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
    out_path = out_dir / f"kraken_{args.pair.upper()}_{day}.jsonl"

    end = time.time() + args.seconds
    wrote = 0
    with out_path.open("a", encoding="utf-8") as fh:
        while time.time() < end:
            px = fetch_price(args.pair, verbose=args.verbose)
            if px is not None:
                obj = {
                    "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "pair": args.pair.upper(),
                    "price": px,
                    "source": "kraken",
                }
                fh.write(json.dumps(obj) + "\n")
                fh.flush()
                wrote += 1
            else:
                dbg(args.verbose, f"[capture] {args.pair.upper()} poll returned no data")
            time.sleep(max(1, args.interval))

    print(f"[capture] {args.pair.upper()} wrote {wrote} rows -> {out_path}")


if __name__ == "__main__":
    main()
