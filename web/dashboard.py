"""Lightweight Flask dashboard for the crypto trading bot.

Reads data from logs/ and serves a mobile-friendly dashboard at /
and raw metrics at /metrics.json.

Run locally:
    PYTHONPATH=. FLASK_APP=web/dashboard.py flask run --host=0.0.0.0 --port=5000
or:
    python web/dashboard.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, render_template_string

LOG_DIR = "logs"
TRADES_LOG = os.path.join(LOG_DIR, "trades.log")
POSITIONS_LOG = os.path.join(LOG_DIR, "positions.jsonl")

app = Flask(__name__)


def load_json_lines(path: str) -> list[dict]:
    """Read a JSONL file and return list[dict]; returns [] on error/missing."""
    if not os.path.exists(path):
        return []
    rows: list[dict] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines gracefully
                    continue
    except OSError:
        return []
    return rows


def to_dt_iso(s: str | None) -> datetime | None:
    """Parse ISO8601 string (supports trailing 'Z') into datetime or None."""
    if not s or not isinstance(s, str):
        return None
    try:
        # Support both with/without timezone
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def latest_trades(n: int = 20) -> list[dict]:
    """Return most recent n trades from trades.log sorted by timestamp desc."""
    trades = load_json_lines(TRADES_LOG)
    # Sort by timestamp desc if available
    trades_sorted = sorted(
        trades,
        key=lambda t: to_dt_iso(t.get("timestamp")) or datetime.fromtimestamp(0, tz=timezone.utc),
        reverse=True,
    )
    return trades_sorted[:n]


def open_positions() -> list[dict]:
    """Return current open positions from positions.jsonl (best-effort)."""
    return load_json_lines(POSITIONS_LOG)


def compute_performance_summary(trades: list[dict]) -> dict:
    """Compute performance metrics over closed trades list."""
    # Use only closed trades with numeric ROI
    closed = [t for t in trades if t.get("status") == "closed" and isinstance(t.get("roi"), (int, float))]
    if not closed:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_roi": 0.0,
            "cumulative_return": 0.0,
            "max_drawdown": 0.0,
            "conf_weighted_roi": 0.0,
            "conf_weighted_success": 0.0,
            "balance": 1000.0,
            "reason_counts": {},
        }

    rois = np.array([float(t.get("roi", 0.0)) for t in closed], dtype=float)

    # Wins/losses
    wins = int((rois > 0).sum())
    total = len(rois)
    win_rate = float(wins / total) if total else 0.0
    avg_roi = float(np.mean(rois)) if total else 0.0
    cumulative_return = float(np.prod(1 + rois) - 1)

    # Max drawdown using equity curve
    equity_curve = np.cumprod(1 + rois)
    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve - running_max) / running_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0

    # Confidence-weighted metrics
    confs = np.array([float(t.get("confidence") or 0.0) for t in closed], dtype=float)
    conf_sum = float(np.sum(confs))
    if conf_sum > 0:
        conf_weighted_roi = float(np.sum(confs * rois) / conf_sum)
        conf_weighted_success = float(np.sum(confs * (rois > 0)) / conf_sum)
    else:
        conf_weighted_roi = 0.0
        conf_weighted_success = 0.0

    # Account balance starting from $1000
    balance = float(1000.0 * np.prod(1 + rois))

    # Exit reason counts (map similar reasons)
    reason_counts: dict[str, int] = {}
    for t in closed:
        r = str(t.get("reason") or "unknown").upper()
        key = (
            "STOP_LOSS"
            if "STOP" in r and "TRAIL" not in r
            else (
                "TRAILING_STOP"
                if "TRAIL" in r
                else (
                    "TAKE_PROFIT"
                    if "TAKE" in r or "PROFIT" in r
                    else "RSI_EXIT" if "RSI" in r else "MAX_HOLD" if "MAX_HOLD" in r or "HOLD" in r else r
                )
            )
        )
        reason_counts[key] = reason_counts.get(key, 0) + 1

    return {
        "total_trades": total,
        "win_rate": round(win_rate, 4),
        "avg_roi": round(avg_roi, 6),
        "cumulative_return": round(cumulative_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "conf_weighted_roi": round(conf_weighted_roi, 6),
        "conf_weighted_success": round(conf_weighted_success, 6),
        "balance": round(balance, 2),
        "reason_counts": reason_counts,
    }


def compute_equity_curve(trades: list[dict]) -> list[dict]:
    """Build a cumulative equity curve from closed trades' ROI.

    Starts at 1000 and multiplies by (1 + roi) for each closed trade ordered by timestamp.
    Returns a list of {ts: str, balance: float} sampled to at most ~100 points.
    """
    closed = []
    for t in trades:
        roi = t.get("roi")
        if t.get("status") == "closed" and isinstance(roi, (int, float)):
            ts = to_dt_iso(t.get("timestamp"))
            if ts is None:
                continue
            closed.append((ts, float(roi), t.get("timestamp")))
    if not closed:
        return []
    closed.sort(key=lambda x: x[0])
    bal = 1000.0
    points: list[dict] = []
    for _ts_dt, roi, ts_raw in closed:
        bal *= 1.0 + roi
        points.append({"ts": ts_raw, "balance": round(bal, 2)})
    # Sample to ~100 points to keep chart light
    n = len(points)
    stride = (n + 99) // 100  # ceil(n/100), at least 1
    return points[:: max(1, stride)]


def build_metrics() -> dict:
    """Aggregate summary, positions, latest trades, and equity curve."""
    trades = load_json_lines(TRADES_LOG)
    positions = open_positions()
    summary = compute_performance_summary(trades)
    equity = compute_equity_curve(trades)
    return {
        "summary": summary,
        "positions": positions,
        "latest_trades": latest_trades(20),
        "equity_curve": equity,
    }


BASE_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>Crypto Bot Dashboard</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <style>
    body { padding: 1rem; }
    .card { margin-bottom: 1rem; }
    .roi-pos { color: #198754; font-weight: 600; }
    .roi-neg { color: #dc3545; font-weight: 600; }
    .pill { font-size: .9rem; }
    pre { white-space: pre-wrap; word-wrap: break-word; }
    #equityChart { max-height: 220px; }
  </style>
  <script>
    async function refreshNow(){ location.reload(); }
  </script>
  </head>
<body>
  <div class="container-fluid">
    <div class="d-flex align-items-center mb-3">
      <h3 class="me-auto">📊 Crypto Bot Dashboard</h3>
      <button class="btn btn-outline-primary" onclick="refreshNow()">Refresh</button>
    </div>

    <div class="row">
      <div class="col-12 col-lg-4">
        <div class="card">
          <div class="card-header">Strategy Performance Summary</div>
          <div class="card-body">
            {% if summary.total_trades == 0 %}
              <p>No data</p>
            {% else %}
              <ul class="list-group list-group-flush">
                <li class="list-group-item d-flex justify-content-between">
                  <span>Total trades</span>
                  <strong>{{ summary.total_trades }}</strong>
                </li>
                <li class="list-group-item d-flex justify-content-between">
                  <span>Win rate</span>
                  <strong>{{ (summary.win_rate*100)|round(2) }}%</strong>
                </li>
                <li class="list-group-item d-flex justify-content-between">
                  <span>Avg ROI</span>
                  <strong>{{ (summary.avg_roi*100)|round(2) }}%</strong>
                </li>
                <li class="list-group-item d-flex justify-content-between">
                  <span>Cumulative return</span>
                  <strong>{{ (summary.cumulative_return*100)|round(2) }}%</strong>
                </li>
                <li class="list-group-item d-flex justify-content-between">
                  <span>Max drawdown</span>
                  <strong>{{ (summary.max_drawdown*100)|round(2) }}%</strong>
                </li>
                <li class="list-group-item d-flex justify-content-between">
                  <span>Conf-weighted ROI</span>
                  <strong>{{ (summary.conf_weighted_roi*100)|round(2) }}%</strong>
                </li>
                <li class="list-group-item d-flex justify-content-between">
                  <span>Conf-weighted success</span>
                  <strong>{{ (summary.conf_weighted_success*100)|round(2) }}%</strong>
                </li>
                <li class="list-group-item d-flex justify-content-between">
                  <span>Balance</span>
                  <strong>${{ summary.balance }}</strong>
                </li>
              </ul>
            {% endif %}
            <div class="mt-3">
              <canvas id="equityChart"></canvas>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">Exit Reason Counts</div>
          <div class="card-body">
            {% if not summary.reason_counts %}
              <p>No data</p>
            {% else %}
              <ul class="list-group list-group-flush">
                {% for k, v in summary.reason_counts.items() %}
                  <li class="list-group-item d-flex justify-content-between">
                    <span>{{ k }}</span>
                    <span class="badge text-bg-secondary pill">{{ v }}</span>
                  </li>
                {% endfor %}
              </ul>
            {% endif %}
          </div>
        </div>
      </div>

      <div class="col-12 col-lg-8">
        <div class="card">
          <div class="card-header">Live Trade History (latest 20)</div>
          <div class="card-body table-responsive">
            {% if not latest_trades %}
              <p>No data</p>
            {% else %}
            <table class="table table-sm table-striped align-middle">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Pair</th>
                  <th>Size</th>
                  <th>Strategy</th>
                  <th>Status</th>
                  <th>Exit</th>
                  <th>ROI</th>
                  <th>Reason</th>
                </tr>
              </thead>
              <tbody>
                {% for t in latest_trades %}
                {% set roi = t.get('roi') %}
                <tr>
                  <td>{{ t.get('timestamp','') }}</td>
                  <td>{{ t.get('pair','') }}</td>
                  <td>{{ t.get('size','') }}</td>
                  <td>{{ t.get('strategy','') }}</td>
                  <td>{{ t.get('status','') }}</td>
                  <td>{{ t.get('exit_price','') }}</td>
                  <td>
                    {% if roi is not none %}
                      <span class="{{ 'roi-pos' if roi > 0 else 'roi-neg' }}">{{ (roi*100)|round(2) }}%</span>
                    {% else %}
                      —
                    {% endif %}
                  </td>
                  <td>{{ t.get('reason','') }}</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
            {% endif %}
          </div>
        </div>

        <div class="card">
          <div class="card-header">Open Positions</div>
          <div class="card-body table-responsive">
            {% if not positions %}
              <p>No data</p>
            {% else %}
            <table class="table table-sm table-striped align-middle">
              <thead>
                <tr>
                  <th>Trade ID</th>
                  <th>Pair</th>
                  <th>Size</th>
                  <th>Entry</th>
                  <th>Timestamp</th>
                  <th>Strategy</th>
                  <th>Confidence</th>
                </tr>
              </thead>
              <tbody>
                {% for p in positions %}
                <tr>
                  <td>{{ p.get('trade_id','') }}</td>
                  <td>{{ p.get('pair','') }}</td>
                  <td>{{ p.get('size','') }}</td>
                  <td>{{ p.get('entry_price','') }}</td>
                  <td>{{ p.get('timestamp','') }}</td>
                  <td>{{ p.get('strategy','') }}</td>
                  <td>{{ (p.get('confidence') or 0)|round(2) }}</td>
                </tr>
                {% endfor %}
              </tbody>
            </table>
            {% endif %}
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
<script>
  // Render equity curve when data is available
  (function(){
    const ec = {{ equity_curve|tojson }};
    const el = document.getElementById('equityChart');
    if (!el || !ec || ec.length === 0) return;
    const labels = ec.map(p => p.ts);
    const data = ec.map(p => p.balance);
    const ctx = el.getContext('2d');
    const gradient = ctx.createLinearGradient(0,0,0,220);
    gradient.addColorStop(0,'rgba(25,135,84,0.35)');
    gradient.addColorStop(1,'rgba(25,135,84,0.0)');
    new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Equity',
          data: data,
          fill: true,
          backgroundColor: gradient,
          borderColor: '#198754',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.25
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { maxTicksLimit: 5, color: '#6c757d' },
            grid: { display: false }
          },
          y: {
            ticks: { color: '#6c757d' },
            grid: { color: 'rgba(0,0,0,0.05)' }
          }
        }
      }
    });
  })();
</script>
</html>
"""


@app.route("/")
def index():
    """Render dashboard HTML with current metrics."""
    data = build_metrics()
    return render_template_string(
        BASE_HTML,
        summary=data["summary"],
        latest_trades=data["latest_trades"],
        positions=data["positions"],
        equity_curve=data["equity_curve"],
    )


@app.route("/metrics.json")
def metrics():
    """Return metrics JSON for external use."""
    return jsonify(build_metrics())


if __name__ == "__main__":
    # Run directly for quick local use
    app.run(host="0.0.0.0", port=5000, debug=False)
