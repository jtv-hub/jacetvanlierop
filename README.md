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
```

## Autonomous Execution

To keep the trading pipeline running while VS Code is closed, load the provided launchd agents:

1. Copy the plist files from `scripts/launch/` into `~/Library/LaunchAgents/`.
2. Run `launchctl load ~/Library/LaunchAgents/com.jace.crypto.pipeline.plist` to schedule the nightly pipeline (runs at load and every day at 00:05 UTC).
3. Run `launchctl load ~/Library/LaunchAgents/com.jace.crypto.watchdog.plist` to enable the 15-minute watchdog that restarts the pipeline if logs stop updating.

Both agents use absolute paths and write logs to `logs/pipeline.launchd.{out,err}`.
