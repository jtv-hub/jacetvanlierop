#!/usr/bin/env python3
"""
Build a lightweight HTML dashboard showing the latest paper results (per symbol)
and a quick at-a-glance health section.

- Reads logs/paper/paper_summary_*.json (latest per symbol)
- Optionally filters out the UNKNOWN bucket (default: hide)
- Writes logs/reports/dashboard.html
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
LOG_PAPER = REPO / "logs" / "paper"
LOG_REPORTS = REPO / "logs" / "reports"
LIVE_DIR = REPO / "data" / "live"


def _iso_to_dt(s: str) -> datetime:
    """Parse an ISO timestamp into a timezone-aware datetime."""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        # Fallback: parse basic form (YYYY-mm-ddTHH:MM:SS) and assume UTC.
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)


def _age_secs(p: Path) -> int:
    """Return how many seconds old a file is; large number if it does not exist."""
    if not p.exists():
        return 10**9
    return int(datetime.now(timezone.utc).timestamp() - p.stat().st_mtime)


def _pct(val: float) -> str:
    """Format a 0..1 value as percentage text."""
    try:
        return f"{(float(val) * 100):.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _live_ages_text() -> str:
    """Return a human string describing age of the latest BTC/ETH/SOL live files."""
    parts: List[str] = []
    for sym in ("BTCUSD", "ETHUSD", "SOLUSD"):
        files = sorted(LIVE_DIR.glob(f"kraken_{sym}_*.jsonl"))
        if files:
            parts.append(f"{sym}:{_age_secs(files[-1])}s old")
        else:
            parts.append(f"{sym}:n/a")
    return " · ".join(parts)


def _equity_html() -> str:
    """Return an <img> tag for the newest equity PNG (if any), else a placeholder."""
    pngs = sorted((REPO / "logs" / "backtests").glob("equity_*.png"))
    if not pngs:
        return '<div style="color:#777">No equity plots yet.</div>'
    rel = pngs[-1].relative_to(REPO)
    return f'<img alt="latest equity" src="{rel}" style="max-width:720px;">'


def _link_to_summary(sym: str, ts: str) -> str:
    """
    Return an HTML <a> tag pointing to the summary JSON for (sym, ts),
    or to the newest summary for the symbol if exact match is missing.
    """
    # First: try exact timestamp match.
    for f in sorted(LOG_PAPER.glob(f"paper_summary_{sym}_*.json")):
        try:
            j = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if j.get("timestamp") == ts:
            return f'<a href="{f.relative_to(REPO)}">json</a>'

    # Fallback: newest summary for this symbol.
    candidates = sorted(LOG_PAPER.glob(f"paper_summary_{sym}_*.json"))
    if candidates:
        return f'<a href="{candidates[-1].relative_to(REPO)}">json</a>'
    return ""


def _row_html(sym: str, data: Dict[str, Any]) -> str:
    """Render a single <tr> for the table."""
    trades = int(data.get("trades", 0))
    wr = _pct(data.get("win_rate", 0.0))
    retpct = _pct(data.get("ret_pct", 0.0))
    link_html = _link_to_summary(sym, data.get("timestamp", ""))
    return (
        "<tr>"
        f"<td>{sym}</td>"
        f"<td>{trades}</td>"
        f"<td>{wr}</td>"
        f"<td>{retpct}</td>"
        f"<td>{link_html}</td>"
        "</tr>"
    )


def load_latest_summaries(exclude_unknown: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Collect the latest summary per symbol.

    Args:
        exclude_unknown: If True, drop summaries whose 'symbol' resolves to 'UNKNOWN'.

    Returns:
        A mapping of SYMBOL -> summary_dict.
    """
    per_symbol: Dict[str, Tuple[datetime, Dict[str, Any], Path]] = {}

    for f in sorted(LOG_PAPER.glob("paper_summary_*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue

        symbol = str(data.get("symbol", "UNKNOWN")).upper()
        if exclude_unknown and symbol == "UNKNOWN":
            continue

        ts = data.get("timestamp")
        if isinstance(ts, str):
            dt = _iso_to_dt(ts)
        else:
            dt = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)

        prev = per_symbol.get(symbol)
        if (prev is None) or (dt > prev[0]):
            per_symbol[symbol] = (dt, data, f)

    return {sym: tup[1] for sym, tup in per_symbol.items()}


def read_position_size() -> float:
    """Peek at the most recent paper summary to show the last position_size (or 0.0)."""
    latest: Path | None = None
    for f in sorted(LOG_PAPER.glob("paper_summary_*.json")):
        latest = f
    if latest:
        try:
            data = json.loads(latest.read_text())
            return float(data.get("position_size", 0))
        except (json.JSONDecodeError, OSError, ValueError, UnicodeDecodeError):
            return 0.0
    return 0.0


def build_html(rows: List[Tuple[str, Dict[str, Any]]]) -> str:
    """
    Render the complete dashboard HTML using small helpers to keep locals minimal.
    """
    now_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    live_text = _live_ages_text()
    pos_size = read_position_size()
    table_rows = "\n".join(_row_html(sym, data) for sym, data in rows) or (
        "<tr><td colspan='5' style='color:#777'>No per‑symbol results yet.</td></tr>"
    )
    equity_block = _equity_html()

    return (
        "<!doctype html>\n"
        "<html>\n"
        "<head>\n"
        '  <meta charset="utf-8">\n'
        "  <title>Crypto Bot — Dashboard</title>\n"
        "  <style>\n"
        "    body { font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,\n"
        "           Helvetica, Arial, sans-serif; margin: 24px; }\n"
        "    h1 { font-size: 32px; margin-bottom: 8px; }\n"
        "    .meta { color:#666; font-size: 14px; margin-bottom: 16px; }\n"
        "    table { border-collapse: collapse; width: 100%; max-width: 820px; }\n"
        "    th, td { border: 1px solid #e5e5e5; padding: 8px 10px; text-align: left; }\n"
        "    th { background: #fafafa; }\n"
        "    .pill { display:inline-block; margin-left:6px; padding:2px 6px;\n"
        "            background:#f3f4f6; border-radius:6px; font-size:12px; color:#374151; }\n"
        "    .section { margin-top: 24px; }\n"
        "  </style>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>Crypto Bot — Dashboard</h1>\n"
        f'  <div class="meta">Last tick: {now_text} · Live files: {live_text}</div>\n'
        '  <div class="meta">Position size (current): '
        f'<span class="pill">{pos_size}</span></div>\n'
        "\n"
        "  <h3>Latest Paper Results (per symbol)</h3>\n"
        "  <table>\n"
        "    <thead>\n"
        "      <tr><th>Symbol</th><th>Trades</th><th>Win Rate</th>"
        "<th>Return</th><th>Summary</th></tr>\n"
        "    </thead>\n"
        "    <tbody>\n"
        f"      {table_rows}\n"
        "    </tbody>\n"
        "  </table>\n"
        "\n"
        '  <div class="section">\n'
        "    <h3>Equity</h3>\n"
        f"    {equity_block}\n"
        "  </div>\n"
        "</body>\n"
        "</html>\n"
    )


def main() -> None:
    """Entry point: build the dashboard HTML file."""
    LOG_REPORTS.mkdir(parents=True, exist_ok=True)

    # Allow SHOW_UNKNOWN=1 to keep the UNKNOWN row (defaults to hidden).
    show_unknown = os.environ.get("SHOW_UNKNOWN", "0") == "1"
    summaries = load_latest_summaries(exclude_unknown=not show_unknown)

    # Stable order: SOLUSD, ETHUSD, BTCUSD, then alphabetical others.
    preferred = {"BTCUSD": 2, "ETHUSD": 1, "SOLUSD": 0}
    rows = sorted(summaries.items(), key=lambda kv: (preferred.get(kv[0], 99), kv[0]))

    html = build_html(rows)
    out = LOG_REPORTS / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"[dashboard] wrote {out}")


if __name__ == "__main__":
    main()
