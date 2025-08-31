#!/usr/bin/env python3
"""
Minimal playback shim.

It accepts the same CLI flags your pipeline passes and exits 0 so the
pipeline can continue (review + nightly email). Replace this later with
your real playback logic when you’re ready.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone


def main() -> int:
    """Run a tiny, no-op playback that logs a stub trade row and exits cleanly.

    This shim mirrors the CLI your pipeline uses so it can be swapped for
    real logic later without changing any shell scripts. It also keeps output
    lines short to satisfy linters.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="JSONL price file")
    p.add_argument("--rsi-th", type=float, default=60.0)
    p.add_argument("--tp", type=float, default=0.002)
    p.add_argument("--sl", type=float, default=0.003)
    p.add_argument("--max-hold", type=int, default=10)
    p.add_argument("--size", type=float, default=100.0)
    p.add_argument("--trades-out", default="logs/live_playback_trades.jsonl")
    args = p.parse_args()

    rel = os.path.relpath(args.file, start=os.getcwd())
    print(f"▶️  Playback (shim) file: {rel}")

    params = f"    Params: rsi_th={args.rsi_th} tp={args.tp} " f"sl={args.sl} max_hold={args.max_hold} size={args.size}"
    print(params)
    print(f"    Logging trades → {args.trades_out}")

    os.makedirs(os.path.dirname(args.trades_out), exist_ok=True)
    stub = {
        # timezone-aware ISO8601; ends with "+00:00" (UTC)
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategy": "ShimPlayback",
        "file": rel,
        "note": "shim ran; no trades simulated",
    }
    with open(args.trades_out, "a", encoding="utf-8") as f:
        f.write(json.dumps(stub) + "\n")

    # Optionally show a couple of lines from the input file, if present
    try:
        with open(args.file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 2:
                    break
                try:
                    obj = json.loads(line)
                    price = obj.get("price", "NA")
                    print(f"sample price={price}")
                except json.JSONDecodeError:
                    # Ignore non-JSON lines gracefully
                    pass
    except FileNotFoundError:
        print(f"[WARN] Playback file not found: {args.file}", file=sys.stderr)

    print("\n✅ Playback (shim) done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
