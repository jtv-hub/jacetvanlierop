"""Lightweight Flask dashboard for the crypto trading bot.

Reads data from logs/ and serves a mobile-friendly dashboard at /
and raw metrics at /metrics.json.

Run locally:
    PYTHONPATH=. FLASK_APP=web/dashboard.py flask run --host=0.0.0.0 --port=5000
or:
    python web/dashboard.py

Remote access (developer tips):

- A) Tailscale (preferred)
  1. Install Tailscale on your machine and phone.
  2. Log in on both devices so they join the same tailnet.
  3. Run this dashboard with host=0.0.0.0 (already configured) and note your
     device's Tailscale IP (e.g., 100.x.x.x). On your phone, open:
       http://100.x.x.x:5000

- B) Ngrok (fallback)
  1. Install ngrok and sign in (ngrok config add-authtoken <token>).
  2. Start the dashboard locally (port 5000).
  3. In a separate terminal run: ngrok http 5000
  4. Use the forwarded HTTPS URL shown by ngrok to access the dashboard remotely.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone

import numpy as np
from flask import Flask, jsonify, render_template_string

LOG_DIR = "logs"
TRADES_LOG = os.path.join(LOG_DIR, "trades.log")
POSITIONS_LOG = os.path.join(LOG_DIR, "positions.jsonl")
LEARN_FEEDBACK_LOG = os.path.join(LOG_DIR, "learning_feedback.jsonl")
SHADOW_RESULTS_LOG = os.path.join(LOG_DIR, "shadow_test_results.jsonl")

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
        if "STOP" in r and "TRAIL" not in r:
            key = "STOP_LOSS"
        elif "TRAIL" in r:
            key = "TRAILING_STOP"
        elif "TAKE" in r or "PROFIT" in r:
            key = "TAKE_PROFIT"
        elif "RSI" in r:
            key = "RSI_EXIT"
        elif "MAX_HOLD" in r or "HOLD" in r:
            key = "MAX_HOLD"
        else:
            key = r
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


# ---- Additional summaries reused by new endpoints ----


def _count_exit_reasons(path: str) -> dict:
    counts: dict[str, int] = {}
    for row in load_json_lines(path):
        if row.get("status") != "closed":
            continue
        reason = row.get("reason")
        if isinstance(reason, str) and reason.strip():
            key = reason.strip().upper()
            counts[key] = counts.get(key, 0) + 1
    return counts


def _is_numeric(val) -> bool:
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False


def _validate_closed_trades(path: str) -> dict:
    total = 0
    valid = 0
    invalid = 0
    missing: dict[str, int] = {}
    for row in load_json_lines(path):
        if row.get("status") != "closed":
            continue
        total += 1
        errors = []
        if not _is_numeric(row.get("capital_buffer")):
            errors.append("capital_buffer")
        side = row.get("side")
        if not (isinstance(side, str) and side.lower() in {"long", "short"}):
            errors.append("side")
        if not _is_numeric(row.get("roi")):
            errors.append("roi")
        reason = row.get("reason")
        if not (isinstance(reason, str) and reason.strip()):
            errors.append("reason")
        if errors:
            invalid += 1
            for k in errors:
                missing[k] = missing.get(k, 0) + 1
        else:
            valid += 1
    return {
        "total_closed_trades": total,
        "valid": valid,
        "invalid": invalid,
        "missing_fields": missing,
    }


def _live_stats_summary(path: str) -> dict:
    trades = [r for r in load_json_lines(path) if r.get("status") == "closed" and _is_numeric(r.get("roi"))]
    total = len(trades)
    wins = sum(1 for t in trades if float(t.get("roi", 0.0)) > 0)
    losses = total - wins
    win_rate = (wins / total) if total else 0.0
    cum_roi = sum(float(t.get("roi", 0.0)) for t in trades)
    avg_roi = (cum_roi / total) if total else 0.0
    # leaderboard by strategy (wins, roi)
    wins_by: dict[str, int] = {}
    roi_by: dict[str, float] = {}
    for t in trades:
        s = str(t.get("strategy") or "Unknown")
        if float(t.get("roi", 0.0)) > 0:
            wins_by[s] = wins_by.get(s, 0) + 1
        roi_by[s] = roi_by.get(s, 0.0) + float(t.get("roi", 0.0))
    strategies = set(wins_by) | set(roi_by)
    combo = [{"strategy": s, "wins": wins_by.get(s, 0), "roi": roi_by.get(s, 0.0)} for s in strategies]
    top = sorted(combo, key=lambda x: (x["wins"], x["roi"]), reverse=True)[:5]
    # biggest winner/loser
    biggest_winner = None
    biggest_loser = None
    for trade in trades:
        roi = trade.get("roi")
        if roi is None:
            continue
        if biggest_winner is None or roi > biggest_winner.get("roi", -float("inf")):
            biggest_winner = trade
        if biggest_loser is None or roi < biggest_loser.get("roi", float("inf")):
            biggest_loser = trade

    def _brief(t: dict) -> dict:
        return (
            {
                "pair": t.get("pair"),
                "roi": float(t.get("roi", 0.0)) if _is_numeric(t.get("roi")) else None,
                "exit_reason": t.get("reason"),
                "strategy": t.get("strategy"),
                "timestamp": t.get("timestamp"),
            }
            if t
            else None
        )

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "cumulative_roi": round(cum_roi, 6),
        "average_roi": round(avg_roi, 6),
        "top_strategies": top,
        "biggest_winner": _brief(biggest_winner),
        "biggest_loser": _brief(biggest_loser),
    }


def _first(obj: dict, *keys: str):
    for k in keys:
        if k in obj:
            return obj.get(k)
    return None


def _learning_feedback_summary(path: str) -> dict:
    entries = []
    for row in load_json_lines(path):
        strat = _first(row, "strategy", "strategy_name") or "Unknown"
        status = _first(row, "status", "result", "outcome")
        confidence = _first(row, "confidence", "confidence_score")
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            pass
        parameter = _first(row, "parameter")
        value = _first(row, "suggested_value", "value")
        if not parameter and isinstance(row.get("parameters"), dict):
            try:
                parameter = next(iter(row["parameters"].keys()))
                if value is None:
                    value = row["parameters"].get(parameter)
            except StopIteration:
                pass
        entries.append(
            {
                "strategy": strat,
                "status": status,
                "confidence": confidence,
                "parameter": parameter,
                "value": value,
            }
        )
    total = len(entries)
    strat_counts = Counter(e["strategy"] for e in entries)
    top_strategy = strat_counts.most_common(1)[0][0] if strat_counts else None

    def _cat(s):
        s = (s or "").lower()
        if s in {"approved", "accepted", "applied"}:
            return "passed"
        if s in {"rejected", "declined", "denied"}:
            return "failed"
        return "other"

    passed = sum(1 for e in entries if _cat(e["status"]) == "passed")
    failed = sum(1 for e in entries if _cat(e["status"]) == "failed")
    recent = entries[-3:]
    return {
        "total_suggestions": total,
        "unique_strategies": len(strat_counts),
        "top_strategy": top_strategy,
        "passed": passed,
        "failed": failed,
        "recent": recent,
    }


def _shadow_results_summary(path: str) -> dict:
    def to_float(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    entries = []
    for row in load_json_lines(path):
        strategy = _first(row, "strategy", "strategy_name") or "Unknown"
        status = _first(row, "result", "status", "outcome")
        success = to_float(_first(row, "win_rate", "success_rate"))
        trades = _first(row, "trades_tested", "sample_size")
        try:
            trades = int(trades) if trades is not None else None
        except (TypeError, ValueError):
            trades = None
        entries.append(
            {
                "strategy": strategy,
                "status": status,
                "success_rate": success,
                "trades_tested": trades,
            }
        )
    total = len(entries)
    unique = len(Counter(e["strategy"] for e in entries))
    passed = sum(1 for e in entries if isinstance(e.get("success_rate"), float) and e["success_rate"] >= 0.6)
    failed = total - passed
    recent = entries[-3:]
    return {
        "total": total,
        "unique_strategies": unique,
        "passed": passed,
        "failed": failed,
        "recent": recent,
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
      <!-- New dynamic summaries row -->
      <div class="row mt-3">
        <div class="col-12 col-lg-6">
          <div class="card mb-3">
            <div class="card-header">📈 Live Stats</div>
            <div class="card-body" id="liveStatsBody">Loading…</div>
          </div>
          <div class="card mb-3">
            <div class="card-header">🩺 Trade Health</div>
            <div class="card-body" id="tradeHealthBody">Loading…</div>
          </div>
        </div>
        <div class="col-12 col-lg-6">
          <div class="card mb-3">
            <div class="card-header">📤 Exit Reasons</div>
            <div class="card-body">
              <ul class="list-group list-group-flush" id="exitSummaryList">
                <li class="list-group-item">Loading…</li>
              </ul>
            </div>
          </div>
          <div class="card mb-3">
            <div class="card-header">🧠 Learning Feedback</div>
            <div class="card-body" id="learningFeedbackBody">Loading…</div>
          </div>
          <div class="card mb-3">
            <div class="card-header">🧪 Shadow Test Results</div>
            <div class="card-body" id="shadowResultsBody">Loading…</div>
          </div>
        </div>
      </div>
      <div class="row mt-4">
        <div class="col-md-6">
          <div class="card border-success">
            <div class="card-header bg-success text-white">📈 Biggest Winner</div>
            <div class="card-body" id="biggestWinnerBody">Loading...</div>
          </div>
        </div>
        <div class="col-md-6">
          <div class="card border-danger">
            <div class="card-header bg-danger text-white">📉 Biggest Loser</div>
            <div class="card-body" id="biggestLoserBody">Loading...</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
<script>
  // Utility: fetch JSON with graceful failure
  async function fetchJson(url){
    try{
      const r = await fetch(url, {cache:'no-store'});
      if(!r.ok) return null;
      return await r.json();
    }catch(_e){ return null; }
  }
  function pct(x, digits=2){
    if(typeof x !== 'number') return 'n/a';
    return (x*100).toFixed(digits) + '%';
  }
  async function refreshLiveStats(){
    const el = document.getElementById('liveStatsBody');
    const data = await fetchJson('/live-stats');
    if(!data){ el.textContent = 'Data unavailable'; return; }
    el.innerHTML = `
      <ul class="list-group list-group-flush">
        <li class="list-group-item d-flex justify-content-between">
          <span>Total trades</span>
          <strong>${data.total_trades}</strong>
        </li>
        <li class="list-group-item d-flex justify-content-between">
          <span>Wins</span>
          <strong>${data.wins}</strong>
        </li>
        <li class="list-group-item d-flex justify-content-between">
          <span>Losses</span>
          <strong>${data.losses}</strong>
        </li>
        <li class="list-group-item d-flex justify-content-between">
          <span>Win rate</span>
          <strong>${pct(data.win_rate,1)}</strong>
        </li>
        <li class="list-group-item d-flex justify-content-between">
          <span>Cumulative ROI</span>
          <strong>${pct(data.cumulative_roi,2)}</strong>
        </li>
        <li class="list-group-item d-flex justify-content-between">
          <span>Average ROI</span>
          <strong>${pct(data.average_roi,2)}</strong>
        </li>
      </ul>`;
    const formatTrade = (t) => {
      if (!t) return "No data";
      return `
        <strong>Pair:</strong> ${t.pair || "N/A"}<br>
        <strong>ROI:</strong> ${(t.roi || 0).toFixed(2)}%<br>
        <strong>Exit:</strong> ${t.exit_reason || "N/A"}<br>
        <strong>Strategy:</strong> ${t.strategy || "N/A"}
      `;
    };
    document.getElementById("biggestWinnerBody").innerHTML = formatTrade(data.biggest_winner);
    document.getElementById("biggestLoserBody").innerHTML = formatTrade(data.biggest_loser);
  }
  async function refreshExitSummary(){
    const ul = document.getElementById('exitSummaryList');
    const data = await fetchJson('/exit-summary');
    if(!data || !data.counts){ ul.innerHTML = '<li class="list-group-item">No data</li>'; return; }
    const items = Object.entries(data.counts)
      .sort((a,b)=>b[1]-a[1])
      .map(([k,v])=>
        `
        <li class="list-group-item d-flex justify-content-between">
          <span>${k}</span>
          <span class="badge text-bg-secondary">${v}</span>
        </li>`
      ).join('');
    ul.innerHTML = items || '<li class="list-group-item">No data</li>';
  }
  async function refreshTradeHealth(){
    const el = document.getElementById('tradeHealthBody');
    const d = await fetchJson('/trade-health');
    if(!d){ el.textContent='Data unavailable'; return; }
    const missing = d.missing_fields || {};
    const top = Object.entries(missing).sort((a,b)=>b[1]-a[1]).slice(0,3)
      .map(([k,v])=>`${k}: ${v}`).join(', ');
    el.innerHTML = `
      <ul class="list-group list-group-flush">
        <li class="list-group-item d-flex justify-content-between">
          <span>Total closed</span>
          <strong>${d.total_closed_trades}</strong>
        </li>
        <li class="list-group-item d-flex justify-content-between">
          <span>Valid</span>
          <strong>${d.valid}</strong>
        </li>
        <li class="list-group-item d-flex justify-content-between">
          <span>Invalid</span>
          <strong>${d.invalid}</strong>
        </li>
        <li class="list-group-item">
          <span>Top missing fields</span><br>
          <small>${top || '—'}</small>
        </li>
      </ul>`;
  }
  function fmtRecentItems(items, mapper){
    if(!items || !items.length) return '<em>No recent items</em>';
    return '<ul class="list-group list-group-flush">' + items.map(mapper).join('') + '</ul>';
  }
  async function refreshLearningFeedback(){
    const el = document.getElementById('learningFeedbackBody');
    const d = await fetchJson('/learning-feedback');
    if(!d){ el.textContent='Data unavailable'; return; }
    const recent = fmtRecentItems(d.recent, r => {
      const conf = (typeof r.confidence === 'number') ? ` (conf: ${r.confidence.toFixed(2)})` : '';
      const param = r.parameter || '(n/a)';
      const val = (r.value !== undefined) ? r.value : 'n/a';
      const stat = r.status || '';
      return `<li class="list-group-item">${r.strategy}: ${param} → ${val}${conf} — ${stat}</li>`;
    });
    el.innerHTML = `
      <div class="mb-2">
        Total: <strong>${d.total_suggestions}</strong>
        | Top: <strong>${d.top_strategy || 'n/a'}</strong>
      </div>
      <div class="mb-2">Passed: <strong>${d.passed}</strong> | Failed: <strong>${d.failed}</strong></div>
      ${recent}`;
  }
  async function refreshShadowResults(){
    const el = document.getElementById('shadowResultsBody');
    const d = await fetchJson('/shadow-results');
    if(!d){ el.textContent='Data unavailable'; return; }
    const recent = fmtRecentItems(d.recent, r => {
      const sr = (typeof r.success_rate === 'number') ? r.success_rate.toFixed(2) : 'n/a';
      const n = (typeof r.trades_tested === 'number') ? r.trades_tested : 'n/a';
      const stat = r.status || '';
      return `<li class="list-group-item">${r.strategy}: success=${sr}, trades=${n} — ${stat}</li>`;
    });
    el.innerHTML = `
      <div class="mb-2">Total: <strong>${d.total}</strong> | Unique: <strong>${d.unique_strategies}</strong></div>
      <div class="mb-2">Passed: <strong>${d.passed}</strong> | Failed: <strong>${d.failed}</strong></div>
      ${recent}`;
  }
  async function refreshAll(){
    refreshLiveStats();
    refreshExitSummary();
    refreshTradeHealth();
    refreshLearningFeedback();
    refreshShadowResults();
  }
  refreshAll();
  setInterval(refreshAll, 60000);
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
            ticks: {
              autoSkip: true,
              maxTicksLimit: 10,
              callback: function(value, index, values) {
                const label = this.getLabelForValue(value);
                return label.length > 10 ? label.substring(5) : label;
              },
              maxRotation: 45,
              minRotation: 20,
              color: '#6c757d'
            },
            grid: { display: false }
          },
          y: {
            beginAtZero: true,
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


@app.route("/exit-summary")
def exit_summary():
    """Return a summary of exit reasons from the trades log."""
    return jsonify({"counts": _count_exit_reasons(TRADES_LOG)})


@app.route("/trade-health")
def trade_health():
    """Return a health summary of closed trades."""
    return jsonify(_validate_closed_trades(TRADES_LOG))


@app.route("/live-stats")
def live_stats():
    """Return live trading statistics from the trades log."""
    return jsonify(_live_stats_summary(TRADES_LOG))


@app.route("/learning-feedback")
def learning_feedback():
    """Return a summary of learning feedback suggestions."""
    if not os.path.exists(LEARN_FEEDBACK_LOG):
        return jsonify(
            {
                "total_suggestions": 0,
                "unique_strategies": 0,
                "passed": 0,
                "failed": 0,
                "recent": [],
            }
        )
    return jsonify(_learning_feedback_summary(LEARN_FEEDBACK_LOG))


@app.route("/shadow-results")
def shadow_results():
    """Return a summary of shadow test results."""
    if not os.path.exists(SHADOW_RESULTS_LOG):
        return jsonify(
            {
                "total": 0,
                "unique_strategies": 0,
                "passed": 0,
                "failed": 0,
                "recent": [],
            }
        )
    return jsonify(_shadow_results_summary(SHADOW_RESULTS_LOG))


if __name__ == "__main__":
    # Run directly for quick local use
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
