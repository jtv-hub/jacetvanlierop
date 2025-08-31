"""
Parameter loader for paper trading simulations.

This module provides per-symbol trading parameters, such as
take-profit (tp), stop-loss (sl), and trailing stop configuration.

It is designed to be imported dynamically by paper_trade.py:
    from scripts import params_loader as PARAMS_LOADER

paper_trade.py expects a function named `load_params_for_symbol(symbol: str) -> dict`
so we expose that here.
"""

import json
from pathlib import Path
from typing import Dict

# Default parameters (used if no config file is found for a symbol)
_DEFAULTS: Dict = {
    "tp": 0.002,  # 0.2% take profit
    "sl": 0.003,  # 0.3% stop loss
    "trail": {"mode": "pct", "pct": 0.002, "activate": 0.0},
}


def load_params_for_symbol(symbol: str, defaults: Dict = None) -> Dict:
    """
    Return configuration for the given symbol.

    Looks for a JSON file named params_<SYMBOL>.json in the configs/ directory.
    If not found or malformed, returns the defaults.
    """
    defaults = defaults or _DEFAULTS
    config_dir = Path("configs")
    config_path = config_dir / f"params_{symbol.upper()}.json"

    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                return cfg
        except (OSError, json.JSONDecodeError, ValueError, TypeError):
            # fall back to defaults if file is malformed
            pass

    return defaults
