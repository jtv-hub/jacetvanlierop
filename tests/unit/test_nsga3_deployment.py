"""CLI smoke tests for NSGA-III deployment tooling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

CLI_PATH = Path("scripts/nsga3_cli.py")


def _run_cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute the CLI with the given arguments."""
    cmd = [sys.executable, str(CLI_PATH), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_cli_healthcheck_runs():
    """The CLI healthcheck flag should emit PASS/FAIL text."""
    result = _run_cli(["--healthcheck"])
    assert "PASS" in result.stdout or "FAIL" in result.stdout


def test_cli_rotate_runs():
    """The CLI rotate command should succeed."""
    result = _run_cli(["--rotate"])
    assert "Logs rotated" in result.stdout or result.returncode == 0
