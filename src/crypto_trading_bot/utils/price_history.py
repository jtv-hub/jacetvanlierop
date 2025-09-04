"""
price_history.py

Lightweight in-memory price history cache with seeded fallback.

Loads historical close prices from ``data/seeded_prices.json`` on demand
and exposes helpers to ensure a minimum window length for indicators
like RSI. Live prices can be appended and will not be overwritten.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Tuple

SEED_PATH = "data/seeded_prices.json"

# pair -> list of (iso_timestamp, close)
_history: Dict[str, List[Tuple[str, float]]] = {}
_seed_cache: Dict[str, List[Tuple[str, float]]] = {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_seed_file() -> None:
    if _seed_cache or not os.path.exists(SEED_PATH):
        return
    try:
        with open(SEED_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for pair, rows in (data or {}).items():
            out: List[Tuple[str, float]] = []
            for r in rows or []:
                if not isinstance(r, (list, tuple)) or len(r) < 2:
                    continue
                ts = str(r[0])
                try:
                    price = float(r[1])
                except (ValueError, TypeError):
                    continue
                out.append((ts, price))
            if out:
                _seed_cache[pair.upper()] = out
    except (OSError, json.JSONDecodeError):
        # Fail-open: simply keep empty cache
        _seed_cache.clear()


def ensure_min_history(pair: str, min_len: int = 14) -> None:
    """Ensure ``_history[pair]`` has at least ``min_len`` entries.

    If not, load from the seed file and inject the seeded rows that are not
    already present by timestamp. Logs once per injection.
    """
    if not isinstance(pair, str) or "/" not in pair:
        return
    key = pair.upper()
    cur = _history.get(key, [])
    if len(cur) >= min_len:
        _history[key] = cur
        return

    _load_seed_file()
    seeded = list(_seed_cache.get(key, []))
    if not seeded:
        _history[key] = cur
        return

    # Avoid duplicates by timestamp
    seen = {ts for ts, _ in cur}
    injected = [(ts, px) for ts, px in seeded if ts not in seen]
    if injected:
        # Keep chronological order; seed data assumed oldest->newest
        cur = (cur + injected)[-max(min_len, len(cur) + len(injected)) :]
        _history[key] = cur
        print(f"[SEED FALLBACK] Injected seeded history for {key} ({len(injected)} candles)")
    else:
        _history[key] = cur


def append_live_price(pair: str, price: float, ts: str | None = None) -> None:
    """Append a live price with timestamp; avoid simple duplicates.

    If the last timestamp equals ``ts`` or the last price equals ``price``,
    do not append a duplicate entry.
    """
    if not pair or price is None:
        return
    key = pair.upper()
    ts_iso = ts or _now_iso()
    cur = _history.setdefault(key, [])
    if cur:
        last_ts, last_px = cur[-1]
        if last_ts == ts_iso or (last_px == price):
            # Update last timestamp if same price but different time
            if last_ts != ts_iso and last_px == price:
                cur[-1] = (ts_iso, price)
            return
    cur.append((ts_iso, float(price)))


def get_history_prices(pair: str, min_len: int = 14) -> List[float]:
    """Return a list of closes for ``pair`` ensuring ``min_len`` via seeds."""
    ensure_min_history(pair, min_len=min_len)
    rows = _history.get(pair.upper(), [])
    return [px for _, px in rows]


__all__ = [
    "ensure_min_history",
    "append_live_price",
    "get_history_prices",
]
