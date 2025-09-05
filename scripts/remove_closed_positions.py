"""
Remove Closed Positions

Reads a cleaned trades log and positions file, removes any positions whose
trade_id appears as closed in the trades log, and writes a cleaned
positions JSONL file.

Usage:
  python scripts/remove_closed_positions.py --dry-run
  python scripts/remove_closed_positions.py

Defaults:
  trades file:    logs/trades_fixed.log
  positions file: logs/positions_fixed.jsonl
  output file:    logs/positions_cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, Set, Tuple

try:
    from colorama import Fore, Style  # type: ignore
    from colorama import init as colorama_init
except ImportError:  # pragma: no cover - optional

    class _Dummy:
        RESET_ALL = ""

    class _Fore(_Dummy):
        RED = GREEN = YELLOW = CYAN = ""

    class _Style(_Dummy):
        BRIGHT = NORMAL = ""

    Fore, Style = _Fore(), _Style()  # type: ignore

    def colorama_init(*_args, **_kwargs):  # type: ignore
        return None


def _read_jsonl_ids_of_closed(trades_path: str) -> Tuple[Set[str], int, int, int]:
    """Parse a trades JSONL file and collect IDs of closed trades.

    Returns (closed_ids, parsed_count, malformed_count, skipped_count)

    - parsed_count: number of valid JSON objects parsed
    - malformed_count: number of JSON decode failures
    - skipped_count: valid JSON but not counted (missing trade_id or status != 'closed')
    """
    closed: Set[str] = set()
    malformed = 0
    parsed = 0
    skipped = 0
    if not os.path.exists(trades_path):
        print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} missing {trades_path}; nothing to remove.")
        return closed, parsed, malformed, skipped
    with open(trades_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} malformed JSON at {trades_path}:{ln}; skipped")
                continue
            parsed += 1
            status = str(obj.get("status") or "").lower()
            if status == "closed":
                tid = obj.get("trade_id")
                if isinstance(tid, str) and tid:
                    closed.add(tid)
                else:
                    skipped += 1
            else:
                skipped += 1
    return closed, parsed, malformed, skipped


def _filter_positions(positions_path: str, closed_ids: Set[str]) -> Tuple[int, int, Iterable[str]]:
    """Return (parsed_positions, removed_count, kept_lines).

    - parsed_positions: count of valid JSON objects parsed from the positions file
    - removed_count: number of entries dropped due to closed trade_ids
    - kept_lines: iterable of JSONL strings to write to the cleaned file
    """
    removed = 0
    parsed = 0
    kept_lines: list[str] = []

    if not os.path.exists(positions_path):
        print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} missing {positions_path}; nothing to clean.")
        return parsed, removed, kept_lines

    with open(positions_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                parsed += 1
            except json.JSONDecodeError:
                print(f"{Fore.YELLOW}Warning:{Style.RESET_ALL} malformed JSON at " f"{positions_path}:{ln}; kept")
                kept_lines.append(line)
                continue
            tid = obj.get("trade_id")
            if isinstance(tid, str) and tid in closed_ids:
                removed += 1
                continue
            kept_lines.append(json.dumps(obj, separators=(",", ":")) + "\n")

    return parsed, removed, kept_lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove closed trades from positions file")
    parser.add_argument("--trades", default="logs/trades_fixed.log", help="Path to trades JSONL")
    parser.add_argument("--positions", default="logs/positions_fixed.jsonl", help="Path to positions JSONL")
    parser.add_argument("--output", default="logs/positions_cleaned.jsonl", help="Output JSONL path")
    parser.add_argument("--dry-run", action="store_true", help="Only report; do not write output")
    args = parser.parse_args()

    colorama_init(autoreset=True)

    closed_ids, parsed_count, malformed_trades, skipped_count = _read_jsonl_ids_of_closed(args.trades)
    total_pos, removed, kept_lines = _filter_positions(args.positions, closed_ids)

    status_color = Fore.GREEN if removed == 0 else Fore.CYAN
    print(f"{Style.BRIGHT}=== Remove Closed Positions ==={Style.RESET_ALL}")
    print(f"Trades parsed: {parsed_count}")
    print(f"Malformed lines skipped: {malformed_trades}")
    print(f"Valid but not closed/missing trade_id: {skipped_count}")
    print(f"Positions scanned: {total_pos}")
    print(f"Closed trade_ids found: {len(closed_ids)}")
    print(f"{status_color}Positions to remove: {removed}{Style.RESET_ALL}")

    if args.dry_run:
        print("--dry-run enabled; no output written.")
        return

    # Write cleaned positions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as out:
        for line in kept_lines:
            out.write(line)
    print(f"✅ Wrote cleaned positions to {args.output}")


if __name__ == "__main__":
    main()
