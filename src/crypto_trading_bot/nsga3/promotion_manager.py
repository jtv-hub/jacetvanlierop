# pylint: disable=duplicate-code
"""
promotion_manager.py
Phase 4: Integration + Promotion System (Final Version)

Handles:
- Top-N candidate selection (from NSGA-3 generation)
- Matching with 500-trade shadow validation results
- Safety gating and duplicate prevention
- Promotion logging + audit trail
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from crypto_trading_bot.nsga3.integration_hook import promote_to_learning_machine
from crypto_trading_bot.nsga3.regime_utils import normalize_regime_label

try:
    import fcntl  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

# === Configurable constants ===
TOP_N = 5
LOG_DIR = "logs"


def _regime_paths(regime: str) -> dict[str, str]:
    """Return regime-specific log file paths."""
    label = normalize_regime_label(regime)
    return {
        "top": os.path.join(LOG_DIR, f"nsga3_top_candidates_{label}.jsonl"),
        "front": os.path.join(LOG_DIR, f"nsga3_front_{label}.jsonl"),
        "promotions": os.path.join(LOG_DIR, f"nsga3_promotions_{label}.jsonl"),
    }


@contextlib.contextmanager
def _locked_file(path: str, mode: str, lock_flag: int | None = None):
    """Context manager that acquires an advisory lock if supported."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode, encoding="utf-8") as handle:
        if fcntl and lock_flag is not None:
            try:
                fcntl.flock(handle.fileno(), lock_flag)
            except OSError:
                pass
        try:
            yield handle
        finally:
            if fcntl and lock_flag is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass


# === Utility helpers ===
def _hash_params(params: Dict[str, Any]) -> str:
    """Return deterministic SHA256 hash of parameter dictionary."""
    j = json.dumps(params, sort_keys=True)
    return hashlib.sha256(j.encode("utf-8")).hexdigest()


def _load_existing_hashes(promo_file: str) -> set[str]:
    """Load hashes of already promoted parameter sets to avoid duplicates."""
    if not os.path.exists(promo_file):
        return set()
    hashes: set[str] = set()
    lock_flag = fcntl.LOCK_SH if fcntl else None
    with _locked_file(promo_file, "r", lock_flag) as handle:
        for line in handle:
            try:
                entry = json.loads(line)
                if "param_hash" in entry:
                    hashes.add(entry["param_hash"])
            except json.JSONDecodeError:
                continue
    return hashes


# === Core selection logic ===
def select_top_candidates(population: List[Dict[str, Any]], regime: str = "global") -> List[Dict[str, Any]]:
    """
    Select the top N individuals by NSGA-3 risk-adjusted ROI.
    This uses short-horizon fitness, *not* shadow validation.
    """
    ranked = sorted(
        population,
        key=lambda ind: (ind["objectives"]["roi"] * ind["objectives"]["win_rate"])
        / (ind["objectives"]["drawdown"] + 1e-6),
        reverse=True,
    )
    top = ranked[:TOP_N]

    paths = _regime_paths(regime)
    os.makedirs(LOG_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with open(paths["top"], "a", encoding="utf-8") as f:
        for ind in top:
            rec = {
                "timestamp": now,
                "params": ind["params"],
                "objectives": ind["objectives"],
                "rar_nsga3": (ind["objectives"]["roi"] * ind["objectives"]["win_rate"])
                / (ind["objectives"]["drawdown"] + 1e-6),
                "regime": normalize_regime_label(regime),
            }
            f.write(json.dumps(rec) + "\n")
        # Flush/fsync for durability in case of crash mid-write
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
    return top


# === Promotion + Shadow Validation Integration ===
def run_promotion_cycle(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    population: List[Dict[str, Any]],
    shadow_results: Optional[List[Dict[str, Any]]] = None,
    generation: int = 0,
    regime: str = "global",
) -> None:
    """
    Core Phase 4 routine:
    1. Select top-N NSGA-3 individuals
    2. Match them with 500-trade shadow results (via param_hash)
    3. Apply safety gates and promote successful candidates
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    paths = _regime_paths(regime)
    existing = _load_existing_hashes(paths["promotions"])
    now = datetime.now(timezone.utc).isoformat()

    # Build quick-lookup map from shadow sim results (if provided)
    shadow_map: dict[str, dict[str, Any]] = {}
    if shadow_results:
        for res in shadow_results:
            try:
                # ensure consistent hashing
                params = res.get("params", {})
                shadow_map[_hash_params(params)] = res
            except (AttributeError, TypeError, ValueError):
                continue

    top = select_top_candidates(population, regime=regime)
    promoted_count = 0

    for ind in top:
        params = ind["params"]
        obj = ind["objectives"]
        param_hash = _hash_params(params)
        rar_nsga3 = (obj["roi"] * obj["win_rate"]) / (obj["drawdown"] + 1e-6)

        # === Shadow validation ===
        shadow_data = shadow_map.get(param_hash)
        shadow_metrics = shadow_data.get("metrics") if shadow_data else None
        shadow_rar = None
        if shadow_metrics:
            roi = shadow_metrics.get("roi", 0.0)
            drawdown = shadow_metrics.get("drawdown", 1.0)
            win_rate = shadow_metrics.get("win_rate", 0.0)
            shadow_rar = roi * win_rate / (drawdown + 1e-6)

        # === Safety gates ===
        if param_hash in existing:
            status = "duplicate"
        elif obj["win_rate"] < 0.55:
            status = "low_win_rate"
        elif obj["drawdown"] > 0.15:
            status = "high_drawdown"
        elif shadow_rar is None:
            status = "no_shadow_data"
        elif shadow_rar < 0.18:
            status = "rar_below_threshold"
        else:
            status = "passed"

        # === Audit log to front file ===
        record = {
            "timestamp": now,
            "generation": generation,
            "param_hash": param_hash,
            "params": params,
            "objectives": obj,
            "rar_nsga3": rar_nsga3,
            "shadow_metrics": shadow_metrics,
            "shadow_rar": shadow_rar,
            "status": status,
            "regime": normalize_regime_label(regime),
        }
        with open(paths["front"], "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            # Flush/fsync for durability in case of crash mid-write
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass

        # === Actual promotion ===
        if status == "passed":
            promoted_count += 1
            payload = {
                "timestamp": now,
                "generation": generation,
                "param_hash": param_hash,
                "params": params,
                "objectives": obj,
                "shadow_metrics": shadow_metrics,
                "shadow_rar": shadow_rar,
                "rar_nsga3": rar_nsga3,
                "source": "nsga3_shadow",
                "reason": "shadow_pass",
                "regime": normalize_regime_label(regime),
            }
            promote_to_learning_machine(payload, generation=generation, reason="shadow_pass", regime=regime)

    print(
        f"[PromotionManager] Generation {generation}: "
        f"Evaluated {len(top)} top candidates → Promoted {promoted_count}."
    )
