"""
nsga3_cli.py — Manual control CLI for NSGA-III automation workflows.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from crypto_trading_bot.nsga3.regime_utils import normalize_regime_label

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

HEARTBEAT_PATH = Path("logs/nsga3_scheduler.log")
CHECKPOINT_PATH = Path("logs/nsga3_checkpoint_global.json")
_COMPONENTS: dict[str, Any] | None = None


def _get_components() -> dict[str, Any]:
    """Load NSGA-III modules lazily to keep CLI import-safe."""
    global _COMPONENTS  # noqa: PLW0603
    if _COMPONENTS is None:
        rotation_mod = importlib.import_module("crypto_trading_bot.nsga3.log_rotation")
        engine_mod = importlib.import_module("crypto_trading_bot.nsga3.nsga3_engine")
        health_mod = importlib.import_module("crypto_trading_bot.nsga3.nsga3_healthcheck")
        promo_mod = importlib.import_module("crypto_trading_bot.nsga3.promotion_manager")
        sim_mod = importlib.import_module("crypto_trading_bot.nsga3.shadow_simulation")
        _COMPONENTS = {
            "rotate_logs": rotation_mod.rotate_nsga3_logs,
            "default_targets": rotation_mod.DEFAULT_TARGETS,
            "run_cycle": engine_mod.run_nsga3_cycle,
            "healthcheck": health_mod.run_healthcheck,
            "promotion": promo_mod.run_promotion_cycle,
            "shadow": sim_mod.run_shadow_tests,
        }
    return _COMPONENTS


def _append_heartbeat(status: str, extra: dict | None = None, regime: str = "global") -> None:
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": "nsga3_cycle",
        "status": status,
        "regime": normalize_regime_label(regime),
    }
    if extra:
        entry.update(extra)
    HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HEARTBEAT_PATH.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle)
        handle.write("\n")
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass


def _read_last_json_line(path: Path) -> dict | None:
    if not path.exists():
        return None
    last_line = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def show_status(regime: str) -> None:
    """Display the most recent heartbeat entry and checkpoint generation."""
    heartbeat = _read_last_json_line(HEARTBEAT_PATH)
    checkpoint = None
    label = normalize_regime_label(regime)
    checkpoint_path = Path(f"logs/nsga3_checkpoint_{label}.json")
    candidate_paths = [checkpoint_path, CHECKPOINT_PATH]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            checkpoint = json.loads(path.read_text(encoding="utf-8"))
            break
        except json.JSONDecodeError:
            checkpoint = None

    if heartbeat:
        print(f"Heartbeat: status={heartbeat.get('status')} " f"timestamp={heartbeat.get('timestamp')}")
    else:
        print("Heartbeat: no entries recorded.")

    if checkpoint:
        print(
            f"Checkpoint: generation={checkpoint.get('generation')} "
            f"pareto_size={len(checkpoint.get('pareto_front', []))}"
        )
    else:
        print("Checkpoint: not found.")


def clear_logs(regime: str) -> bool:
    """Rotate and truncate NSGA-III logs after user confirmation."""
    response = input("Type YES to confirm clearing NSGA-III logs: ").strip().upper()
    if response != "YES":
        print("Aborted.")
        return False

    components = _get_components()
    rotate_logs = components["rotate_logs"]
    targets = [Path(p) for p in components["default_targets"]]
    label = normalize_regime_label(regime)
    regime_targets = [
        Path(f"logs/nsga3_checkpoint_{label}.json"),
        Path(f"logs/nsga3_promotions_{label}.jsonl"),
        Path(f"logs/nsga3_front_{label}.jsonl"),
        Path(f"logs/nsga3_top_candidates_{label}.jsonl"),
        Path(f"logs/nsga3_shadow_results_{label}.jsonl"),
    ]

    rotate_logs(regime=regime)
    for path in targets + regime_targets + [HEARTBEAT_PATH]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("", encoding="utf-8")
    print("NSGA-III logs cleared.")
    return True


def run_full_cycle(regime: str) -> bool:
    """Run healthcheck + NSGA-III + shadow + promotion sequence."""
    components = _get_components()
    rotate_logs = components["rotate_logs"]
    run_cycle = components["run_cycle"]
    healthcheck = components["healthcheck"]
    promote = components["promotion"]
    shadow = components["shadow"]

    _append_heartbeat("pending", regime=regime)
    if not healthcheck():
        LOGGER.warning("NSGA-III cycle skipped: healthcheck failed.")
        _append_heartbeat("skipped", {"reason": "healthcheck_failed"}, regime=regime)
        return False
    try:
        population, generation = run_cycle(regime=regime)
        shadow_results = shadow(population, max_trades=500, regime=regime)
        promote(
            population=population,
            shadow_results=shadow_results,
            generation=generation,
            regime=regime,
        )
        rotate_logs(regime=regime)
        LOGGER.info("NSGA-III cycle completed via CLI.")
        _append_heartbeat("success", regime=regime)
        return True
    except Exception as exc:  # pylint: disable=broad-exception-caught
        LOGGER.exception("NSGA-III cycle failed: %s", exc)
        _append_heartbeat("failed", {"error": str(exc)}, regime=regime)
        return False


def main() -> int:
    """Parse CLI arguments and execute requested NSGA-III action."""
    parser = argparse.ArgumentParser(description="NSGA-III control CLI")
    parser.add_argument("--run-now", action="store_true", help="Execute one NSGA-III cycle.")
    parser.add_argument("--healthcheck", action="store_true", help="Run healthcheck only.")
    parser.add_argument("--rotate", action="store_true", help="Rotate NSGA-III logs.")
    parser.add_argument("--status", action="store_true", help="Show last heartbeat and checkpoint.")
    parser.add_argument("--clear-logs", action="store_true", help="Archive and truncate NSGA logs.")
    parser.add_argument("--regime", default="global", help="Regime label for targeted actions.")
    args = parser.parse_args()

    if args.run_now:
        return 0 if run_full_cycle(args.regime) else 1
    if args.healthcheck:
        ok = _get_components()["healthcheck"]()
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    if args.rotate:
        _get_components()["rotate_logs"](regime=args.regime)
        print("Logs rotated.")
        return 0
    if args.status:
        show_status(args.regime)
        return 0
    if args.clear_logs:
        return 0 if clear_logs(args.regime) else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
