from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from crypto_trading_bot.safety.prelaunch_guard import _count_recent_high_severity


def _write_alert(path: Path, *, level: str, timestamp: datetime) -> None:
    entry = {
        "timestamp": timestamp.isoformat(),
        "level": level,
        "message": f"test-{level.lower()}",
    }
    with path.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle)
        handle.write("\n")


def test_count_recent_high_severity(tmp_path: Path) -> None:
    alerts_log = tmp_path / "alerts.log"
    now = datetime.now(timezone.utc)

    _write_alert(alerts_log, level="INFO", timestamp=now)
    _write_alert(alerts_log, level="ERROR", timestamp=now - timedelta(hours=1))
    _write_alert(alerts_log, level="CRITICAL", timestamp=now - timedelta(hours=10))
    _write_alert(alerts_log, level="ERROR", timestamp=now - timedelta(hours=80))  # outside window

    count = _count_recent_high_severity(alerts_log, window_hours=72, now=now)
    assert count == 2  # INFO ignored, one ERROR outside window ignored


@pytest.mark.parametrize("threshold", [3])
def test_clear_alerts_archives_when_threshold_exceeded(tmp_path: Path, threshold: int) -> None:
    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()
    alerts_log = logs_dir / "alerts.log"
    alerts_log.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    script = Path(__file__).resolve().parents[2] / "scripts" / "clear_alerts.py"
    result = subprocess.run(
        [sys.executable, str(script), "--archive-if-over", str(threshold)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    archive_dir = logs_dir / "archive"
    archived_files = list(archive_dir.glob("alerts_*.log"))
    assert len(archived_files) == 1
    assert alerts_log.read_text(encoding="utf-8").count("\n") == 1  # header line only

    system_log = logs_dir / "system.log"
    assert system_log.exists(), "clear_alerts should log to system.log"
