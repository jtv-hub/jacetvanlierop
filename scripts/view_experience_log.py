"""Pretty-print the PPO experience JSONL log to the console."""

import json
import os

from rich.console import Console
from rich.table import Table

EXPERIENCE_LOG_PATH = os.path.join("logs", "ppo_experience.jsonl")


def load_experiences(path):
    """Return a list of JSON objects parsed from the experience log."""
    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def display_experiences(entries):
    """Render the most recent PPO experience entries in a Rich table."""
    console = Console()
    table = Table(show_lines=True, title="PPO Experience Log")

    table.add_column("Timestamp", style="cyan", overflow="fold")
    table.add_column("Strategy", style="bold")
    table.add_column("Confidence", justify="right")
    table.add_column("Allow", justify="center")
    table.add_column("Size Scalar", justify="right")
    table.add_column("Source", style="dim", justify="center")

    for entry in reversed(entries[-25:]):  # Show latest 25 entries
        table.add_row(
            entry.get("timestamp", "-"),
            entry.get("strategy_id", "-"),
            f"{entry.get('confidence', 0):.2f}",
            "✅" if entry.get("allow_entry", True) else "❌",
            f"{entry.get('size_scalar', 1.0):.2f}",
            entry.get("source", "-"),
        )

    console.print(table)


if __name__ == "__main__":
    loaded_entries = load_experiences(EXPERIENCE_LOG_PATH)
    display_experiences(loaded_entries)
