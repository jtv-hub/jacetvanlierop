# Go-Live Checklist for `crypto_trading_bot`

Use this checklist before enabling live trading with real funds.

---

## 1. Credentials & Secrets
- [ ] Confirm `KRAKEN_API_KEY` / `KRAKEN_API_SECRET` are set via environment or `.env` and **never** hard-coded.
- [ ] Verify API key permissions are limited to *Trading* (`Trade`) – **no withdrawal rights**.
- [ ] Store secrets in a secure location (e.g., encrypted secrets manager or locked-down `.env`); restrict file permissions.
- [ ] Run `python scripts/verify_kraken_credentials.py` (or equivalent) to confirm the new key pair authenticates.

## 2. Mode Toggles
- [ ] Confirm `crypto_trading_bot.config.IS_LIVE` defaults to `False`.
- [ ] Document the exact steps to switch to live mode:
  1. Set environment export: `export IS_LIVE=true`
  2. Update `.env` if required (and re-source).
  3. Restart services / processes that rely on the environment.
- [ ] Keep a rollback plan: `export IS_LIVE=false` to force paper mode quickly.

## 3. Logging & Monitoring
- [ ] Ensure log rotation is active (`bot/utils/log_rotation.py`); check logs stay < 10 MB.
- [ ] Confirm alerts (email, Slack, etc.) trigger on trade failures, API disconnects, or critical log messages.
- [ ] Tail `logs/system.log` and `logs/anomalies.log`—ensure no unhandled errors remain.
- [ ] Archive historical logs and ensure timestamped backups exist.

## 4. Risk Controls
- [ ] Validate `capital_buffer` logic in the ledger populates every closed trade entry.
- [ ] Confirm max drawdown guard (`risk_guard.py`) is enabled and thresholds match risk appetite.
- [ ] Review position-sizing limits, confidence thresholds, and volume filters in strategies.
- [ ] Confirm emergency tools (`scripts/kill_switch.sh`, `scripts/self_heal.sh`) are ready and documented.

## 5. Health Audits
- [ ] Run a final dry-run audit: `export IS_LIVE=false && bash scripts/final_health_audit.sh --dry-run`
  - Expect no reconciliation mismatches, credential errors, or balance diffs.
- [ ] Inspect reconciliation output for any “legacy_unmatched” entries; resolve if recent.
- [ ] Check ROI audit (`scripts/audit_roi.py`) and reconciliation reports in `logs/`.

## 6. Paper Mode → Live Transition
- [ ] Define paper-trading success criteria (e.g., ≥ *N* trades, ROI > *X%*, zero critical alerts).
- [ ] Document who approves the switch to live mode and how the sign-off is recorded.
- [ ] Verify automation / crons are paused or adjusted during the transition window.

## 7. Backup & Versioning
- [ ] Tag the release in Git (e.g., `git tag v1.0-go-live`) and push tags to remote.
- [ ] Back up configuration files (`.env`, strategy configs, risk thresholds).
- [ ] Snapshot important logs / reports before switching modes.

## 8. Monitoring & Dashboards
- [ ] Run `python scripts/show_live_stats.py` to verify metrics align with expectations.
- [ ] Confirm dashboards or external monitoring display trade counts, ROI, and drawdown.
- [ ] Ensure schedule for manual review (daily/weekly) is communicated to the team.

## 9. Escalation & Incident Response
- [ ] Document on-call / escalation contacts and expected response SLAs.
- [ ] Outline steps to pause trading (`scripts/kill_switch.sh`) or revert to paper mode quickly.
- [ ] Maintain a checklist for manual intervention (cancel open orders, reconcile ledger, notify stakeholders).

---

## Common Terminal Commands

| Action | Command |
| --- | --- |
| Final paper audit | `export IS_LIVE=false && bash scripts/final_health_audit.sh --dry-run` |
| Enable live mode & audit | `export IS_LIVE=true && bash scripts/final_health_audit.sh` |
| Monitor live stats | `python scripts/show_live_stats.py` |
| Validate closed trades | `python scripts/validate_closed_trades.py` |

---

## Deployment Risk Summary

| Risk | Description | Mitigation |
| --- | --- | --- |
| Kraken client complexity | `kraken_client.py` still has complex flows; edge cases may be hard to maintain. | Incremental refactors post-launch; keep unit tests up-to-date (`tests/unit/test_kraken_integration.py`). |
| Reconciliation edge cases | Legacy unmatched trades can appear if Kraken history is incomplete. | Review audit logs; use legacy partitioning; resolve any unmatched trades that are recent. |
| Market conditions | Live slippage/latency can diverge from paper assumptions. | Start with smaller capital exposure; monitor fills closely on initial sessions. |
| External dependencies | Email/alerting or log rotation may fail silently. | Periodically test alert channel and confirm log archives are produced. |
| Human error during mode switch | Incorrect environment toggle or outdated `.env` file. | Follow checklist, have at least two-person sign-off, maintain rollback commands. |

Review this file before each deployment to ensure all prerequisites are satisfied and that risk mitigations are understood.
