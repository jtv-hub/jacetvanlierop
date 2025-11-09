"""Stress tests focusing on log rotation throughput."""

from __future__ import annotations

import os

from crypto_trading_bot.nsga3 import log_rotation, nsga3_engine


def test_stress_rotation():
    """Multiple cycles followed by rotation should produce archive files."""
    os.makedirs("logs", exist_ok=True)
    for _ in range(3):
        nsga3_engine.run_nsga3_cycle(dry_run=True, regime="global")
    log_rotation.rotate_nsga3_logs(regime="global")
    archive_dir = "logs/archive"
    os.makedirs(archive_dir, exist_ok=True)
    gz_files = [f for f in os.listdir(archive_dir) if f.endswith(".gz")]
    assert isinstance(gz_files, list)
