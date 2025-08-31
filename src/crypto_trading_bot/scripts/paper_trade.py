#!/usr/bin/env python3
"""
Paper trading simulation engine for Kraken JSONL files using mock price data.
"""
import argparse
import glob
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

# --- Mock price data for strategies -------------------------------------------
assets = ["BTC", "ETH", "XRP", "LINK"]
mock_price_data = {
    "BTC": [30000 + i for i in range(100)],
    "ETH": [2000 + i for i in range(100)],
    "XRP": [0.5 + (i * 0.01) for i in range(100)],
    "LINK": [7 + (i * 0.05) for i in range(100)],
}

# --- Params loader (lint-friendly optional import) ----------------------------
try:
    from scripts import params_loader as PARAMS_LOADER  # type: ignore[import-not-found]
except (ImportError, ModuleNotFoundError):
    PARAMS_LOADER = None  # type: ignore[assignment]


def _default_params() -> dict:
    """Return default trading parameters if external loader fails."""
    return {
        "rsi_th": 55,
        "tp": 0.002,
        "sl": 0.003,
        "trail": {
            "mode": "pct",
            "pct": 0.002,
            "activate": 0.0,
        },
    }


def load_params_for_symbol(symbol: str) -> dict:
    """Load trading parameters for a specific symbol using an optional external loader."""

    def try_load_params():
        try:
            candidate = getattr(PARAMS_LOADER, "load_params_for_symbol", None)
            if candidate is None:
                candidate = getattr(PARAMS_LOADER, "load_params", None)
            if candidate is not None and callable(candidate):
                cfg = candidate(symbol)
                if isinstance(cfg, dict):
                    return cfg
        except (
            ImportError,
            ModuleNotFoundError,
            KeyError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
        ):
            return None

    loaded = try_load_params()
    return loaded if loaded is not None else _default_params()


@dataclass
class Trail:
    """Class representing trailing stop parameters and logic."""

    mode: str = field(default="pct")
    pct: float = field(default=0.002)
    activate: float = field(default=0.0)

    def armed(self, up_pct: float) -> bool:
        """Check if the trailing stop condition is activated based on current profit percentage."""
        return self.mode == "pct" and up_pct >= self.activate


@dataclass
class TradeParams:
    """Container for trade parameters such as TP, SL, and trailing stop."""

    tp: float = field(default=0.002)
    sl: float = field(default=0.003)
    trail: Trail = field(default_factory=Trail)


@dataclass
class Tick:
    """A single price tick with timestamp and price."""

    ts: datetime
    price: float


def _expand_patterns(patterns: Iterable[str]) -> List[Path]:
    """Expand glob patterns and return a sorted Path list."""
    out: List[Path] = []
    for pat in patterns:
        matches = glob.glob(pat)
        for m in matches:
            out.append(Path(m))
    return sorted(out)


def _read_ticks(path: Path) -> List[Tick]:
    """Read a JSONL file and return a list of Tick objects parsed from it."""
    ticks: List[Tick] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                price = float(obj["price"])
                ts_raw = obj.get("ts")
                if isinstance(ts_raw, str):
                    ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
                else:
                    ts = datetime.fromtimestamp(float(obj["ts"]), tz=timezone.utc)
                ticks.append(Tick(ts=ts, price=price))
            except (KeyError, ValueError, TypeError, json.JSONDecodeError):
                continue
    return ticks


def _symbol_from_filename(p: Path) -> str:
    """Attempt to extract the trading symbol from the filename, falling back to file contents."""
    name = p.name
    if name.startswith("kraken_") and "_" in name:
        try:
            symbol_parts = name.split("_", maxsplit=2)
            return symbol_parts[1]
        except IndexError:
            pass
    try:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if "pair" in obj and isinstance(obj["pair"], str):
                        return obj["pair"].upper()
                except (json.JSONDecodeError, TypeError, ValueError, KeyError):
                    continue
    except FileNotFoundError:
        pass
    return "UNKNOWN"


def _paper_trades_from_ticks(
    symbol: str, ticks: List[Tick], params: TradeParams, file_source: Path
) -> Iterator[dict]:
    """Simulate paper trades from price ticks using given trade parameters."""
    if len(ticks) < 2:
        return

    open_i: Optional[int] = None
    entry_price = 0.0
    trail_high = 0.0

    def emit_exit(idx: int, reason: str) -> dict:  # pylint: disable=redefined-builtin
        """Helper to log a simulated exit trade with metadata."""
        nonlocal open_i, entry_price, trail_high
        exit_price = ticks[idx].price
        pnl_pct = (exit_price / entry_price) - 1.0
        trade = {
            "symbol": symbol,
            "side": "long",
            "ts_start": ticks[open_i].ts.isoformat(),
            "ts_end": ticks[idx].ts.isoformat(),
            "entry": round(entry_price, 8),
            "exit": round(exit_price, 8),
            "pnl_pct": round(pnl_pct, 6),
            "exit_reason": reason,
            "file_source": str(file_source.resolve()),
        }
        open_i = None
        entry_price = 0.0
        trail_high = 0.0
        return trade

    for i, tick in enumerate(ticks):
        px = tick.price

        if open_i is None:
            if i == 0:
                continue
            open_i = i
            entry_price = px
            trail_high = px
            continue

        if px > trail_high:
            trail_high = px

        up_pct = (px / entry_price) - 1.0
        down_pct = 1.0 - (px / entry_price)

        if up_pct >= params.tp:
            yield emit_exit(i, "take_profit")
            continue
        if down_pct >= params.sl:
            yield emit_exit(i, "stop_loss")
            continue

        if params.trail.mode == "pct" and params.trail.armed(
            up_pct
        ):  # pylint: disable=no-member
            trail_stop = trail_high * (
                1.0 - params.trail.pct
            )  # pylint: disable=no-member
            if px <= trail_stop:
                yield emit_exit(i, "trailing_stop")
                continue

    if open_i is not None:
        yield emit_exit(len(ticks) - 1, "eod")


def _params_for(symbol: str) -> TradeParams:
    """Build TradeParams from config dictionary with defaults."""
    cfg = load_params_for_symbol(symbol) or {}
    trail_cfg = (cfg.get("trail") or {}) if isinstance(cfg, dict) else {}
    return TradeParams(
        tp=cfg.get("tp", 0.002),
        sl=cfg.get("sl", 0.003),
        trail=Trail(
            mode=trail_cfg.get("mode", "pct"),
            pct=trail_cfg.get("pct", 0.002),
            activate=trail_cfg.get("activate", 0.0),
        ),
    )


def _today_tag() -> str:
    """Return today's date in YYYYMMDD format (UTC)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry for running the paper trading engine."""
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="e.g. data/live/kraken_*_*.jsonl")
    ap.add_argument("--size", type=float, default=25.0)
    ap.add_argument("--trail-mode", choices=["pct"], default=None)
    ap.add_argument("--trail-pct", type=float, default=None)
    ap.add_argument("--trail-activate", type=float, default=None)
    args = ap.parse_args(argv)

    out_dir = Path("logs") / "paper"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _today_tag()
    out_path = out_dir / f"paper_trades_{tag}.jsonl"

    patterns = [str(p) for p in args.files]
    paths = _expand_patterns(patterns)

    if not paths:
        print("[paper] no matching input files")
        return 0

    wrote = 0
    with out_path.open("a", encoding="utf-8") as out:
        for path in paths:
            ticks = _read_ticks(path)
            if len(ticks) < 2:
                print(f"[paper] {path} has <2 valid rows; skipping")
                continue

            symbol = _symbol_from_filename(path)
            params = _params_for(symbol)

            if any(
                [
                    args.trail_mode,
                    args.trail_pct,
                    args.trail_activate,
                ]
            ):
                trail = Trail(
                    mode=(
                        args.trail_mode if args.trail_mode else params.trail.mode
                    ),  # pylint: disable=no-member
                    pct=(
                        args.trail_pct
                        if args.trail_pct is not None
                        else params.trail.pct
                    ),  # pylint: disable=no-member
                    activate=(
                        args.trail_activate
                        if args.trail_activate is not None
                        else params.trail.activate
                    ),  # pylint: disable=no-member
                )
                params = TradeParams(tp=params.tp, sl=params.sl, trail=trail)

            for trade in _paper_trades_from_ticks(symbol, ticks, params, path):
                trade_json = json.dumps(trade)
                out.write(trade_json + "\n")

                wrote += 1

    print("[paper] wrote " f"{wrote} trade(s) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
