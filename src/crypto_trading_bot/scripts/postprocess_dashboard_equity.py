#!/usr/bin/env python3
"""Ensure dashboard shows the latest equity PNG right under the ‘Equity’ header.

- Matches <h2> or <h3> 'Equity' headers (case/whitespace tolerant).
- Removes any prior equity_latest.png <img>, then inserts a fresh one.
- Uses absolute file:// URL with cache-busting ?ts=<mtime>.
- Prints clear diagnostics and exits nonzero on failure (unless --force).
"""

from __future__ import annotations
import argparse
import sys
import time
import re
from pathlib import Path

HTML_PATH = Path("logs/reports/dashboard.html")
PNG_PATH = Path("logs/reports/equity_latest.png").resolve()


def ensure_equity_img(html: str, png_abs: Path, *, verbose: bool) -> tuple[str, int]:
    """Return (updated_html, changes_count)."""

    # Remove any previous equity_latest reference (idempotent)
    before = html
    html = re.sub(r"<img[^>]*equity_latest\.png[^>]*>\s*", "", html, flags=re.I)
    removed = 0 if html == before else 1

    # Build file:// URL with cache-buster
    ts = int(png_abs.stat().st_mtime) if png_abs.exists() else int(time.time())
    src = f"file://{png_abs}?ts={ts}"

    # Insert immediately after the Equity header (<h2> or <h3>)
    header_pat = re.compile(r"(<h[23]>\s*Equity\s*</h[23]>)", re.I)
    img_tag = (
        f'\n<p><img alt="latest equity" src="{src}" '
        'style="max-width:880px;width:100%;height:auto;"></p>\n'
    )

    header_found = bool(header_pat.search(html))
    if header_found:
        html = header_pat.sub(rf"\1{img_tag}", html, count=1)
        inserted = 1
    else:
        # Fallback: append at end if there is no visible header match
        html += f"\n<h3>Equity</h3>{img_tag}"
        inserted = 1

    if verbose:
        print(f"[postprocess] reading={HTML_PATH}")
        print(f"[postprocess] png_exists={png_abs.exists()} path={png_abs}")
        print(f"[postprocess] header_found={header_found} removed_old={removed}")
        print(f"[postprocess] inserted_new={inserted}")
        print(f"[postprocess] src={src}")

    return html, (removed + inserted)


def main() -> None:
    """CLI entry point to inject the latest equity PNG into dashboard.html.
    Exits non‑zero on failure unless --force is provided.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--force", action="store_true", help="Always exit 0 (don’t fail pipeline)."
    )
    ap.add_argument("--verbose", action="store_true", help="Print diagnostics.")
    args = ap.parse_args()

    if not HTML_PATH.exists():
        print(f"[postprocess] ERROR: {HTML_PATH} not found", file=sys.stderr)
        sys.exit(0 if args.force else 2)

    s = HTML_PATH.read_text(encoding="utf-8")
    updated, changes = ensure_equity_img(s, PNG_PATH, verbose=args.verbose)

    if changes <= 0:
        print("[postprocess] ERROR: nothing inserted/replaced", file=sys.stderr)
        sys.exit(0 if args.force else 3)

    HTML_PATH.write_text(updated, encoding="utf-8")
    print(f"[postprocess] wrote → {HTML_PATH}")
    sys.exit(0)


if __name__ == "__main__":
    main()
