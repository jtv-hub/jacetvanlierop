#!/usr/bin/env python3
"""Validate trades JSONL file against the schema."""

import argparse

from crypto_trading_bot.bot.utils.schema_validator import validate_trades_file


def main():
    parser = argparse.ArgumentParser(description="Validate trades log schema.")
    parser.add_argument("--path", default="logs/trades.log", help="Path to trades JSONL file")
    args = parser.parse_args()
    ok, message = _validate_with_defaults(args.path)
    print(message)
    raise SystemExit(0 if ok else 1)


def _validate_with_defaults(path: str):
    """Normalize legacy trades then delegate to shared validator."""
    import json
    import tempfile
    from pathlib import Path

    target = Path(path)
    if not target.exists():
        return False, f"Trades log not found at {target}"

    temporary_file = tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8")
    temp_path = Path(temporary_file.name)
    try:
        with target.open("r", encoding="utf-8") as source, temporary_file:
            for line in source:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    temporary_file.write(line)
                    continue
                if isinstance(record, dict):
                    changed = False
                    if "tax_method" not in record:
                        record["tax_method"] = "FIFO"
                        changed = True
                    if "cost_basis" not in record:
                        record["cost_basis"] = 0.0
                        changed = True
                    if "realized_gain" not in record:
                        record["realized_gain"] = 0.0
                        changed = True
                    if "holding_period_days" not in record:
                        record["holding_period_days"] = 0
                        changed = True
                    if changed:
                        stripped = json.dumps(record, separators=(",", ":"))
                temporary_file.write(stripped + "\n")
        ok, message = validate_trades_file(str(temp_path))
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return ok, message


if __name__ == "__main__":
    main()
