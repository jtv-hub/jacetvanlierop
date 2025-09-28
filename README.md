# jacetvanlierop

Crypto trading bot (paper/live-ready).  
Status: ✅ tests green, ✅ lint clean, ✅ pre-commit hooks installed.

## Quick Start
```bash
python -m venv venv && source venv/bin/activate
pip install -U pip
pip install -e .  # or: pip install -r requirements.txt
pytest -q
pre-commit install && pre-commit run -a
# Ruff 0.4+ requires an explicit subcommand
ruff check .
```

## Autonomous Execution

To keep the trading pipeline running while VS Code is closed, load the provided launchd agents:

1. Copy the plist files from `scripts/launch/` into `~/Library/LaunchAgents/`.
2. Run `launchctl load ~/Library/LaunchAgents/com.jace.crypto.pipeline.plist` to schedule the nightly pipeline (runs at load and every day at 00:05 UTC).
3. Run `launchctl load ~/Library/LaunchAgents/com.jace.crypto.watchdog.plist` to enable the 15-minute watchdog that restarts the pipeline if logs stop updating.

Both agents use absolute paths and write logs to `logs/pipeline.launchd.{out,err}`.

## Live Launch Procedure

1. Ensure the environment is configured (API keys present, paper mode stable).
2. Run the dry-run and safety guard wrapper: `./scripts/launch_live.sh` (the script auto-archives large `logs/alerts.log`; you can also run `python3 scripts/clear_alerts.py --archive-if-over 5000` manually).
   - The script backs up logs, clears any kill-switch flag, performs a dry-run, executes the prelaunch guard, and then starts the scheduler with `CRYPTO_TRADING_BOT_LIVE=1`.
3. Monitor `logs/system.log` and `logs/alerts.log` for the first cycle to confirm no emergency halts triggered the guard logic (disk space, missing trades, price fallbacks).

To allow the guard to auto-archive the alerts log during startup, export `CRYPTO_TRADING_BOT_AUTO_ARCHIVE_ALERTS=1` and run `python3 -m src.crypto_trading_bot.safety.prelaunch_guard --auto-archive-alerts`.

## Go Live Checklist

- [ ] Disable `CRYPTO_TRADING_BOT_TEST_MODE`; dry-run should show real Kraken volumes with no mock fallbacks in logs.
- [ ] Verify `python -m src.crypto_trading_bot.scripts.verify_kraken_credentials` succeeds and API key lacks withdraw rights.
- [ ] Run `pytest tests/unit/` — RSI exit and correlation guards must pass the new live-readiness tests.
- [ ] Inspect `logs/system.log` after a scheduler cycle to confirm: no phantom trades, correlation blocks logged for highly aligned pairs, minimum volume adjustments recorded.
- [ ] Confirm per-pair minimum order sizes in `CONFIG["trade_size"]` match `kraken_get_asset_pair_meta` output (see `src/crypto_trading_bot/config/__init__.py`).
- [ ] Execute `python -m src.crypto_trading_bot.safety.prelaunch_guard --auto-archive-alerts` immediately before `./scripts/launch_live.sh`.
