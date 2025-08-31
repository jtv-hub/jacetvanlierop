"""
Script to generate and export top optimization parameter suggestions from backtest results.
Part of Step 4.6.5.2.
"""

import json
from pathlib import Path
from typing import List, Dict
from crypto_trading_bot.learning.optimization import detect_outliers, export_suggestions

SUGGESTIONS_DIR = Path("reports")
SUGGESTIONS_DIR.mkdir(exist_ok=True)


def generate_parameter_suggestions(top_configs: List[Dict]) -> List[Dict]:
    """Generate optimization suggestions from top parameter configurations."""
    suggestions = []
    for config in top_configs:
        strategy_name, params = config["strategy_config"].split("::", 1)
        suggestions.append(
            {
                "category": "parameter_tuning",
                "suggestion": f"Use params {params} for {strategy_name} (score: {config['score']})",
                "confidence": max(0.0, min(1.0, config["win_rate"])),
                "reason": (
                    f"Backtest shows ROI={config['avg_roi']:.2f}, "
                    f"Win Rate={config['win_rate']:.2f}, Sharpe={config['sharpe']:.2f}"
                ),
            }
        )
    return suggestions


if __name__ == "__main__":
    detected_configs = detect_outliers(min_trades=25, top_n=3)
    if detected_configs:
        parameter_suggestions = generate_parameter_suggestions(detected_configs)
        export_suggestions(parameter_suggestions)

        # Write a separate file for shadow_test_runner compatibility
        latest_path = SUGGESTIONS_DIR / "suggestions_latest.json"
        with latest_path.open("w", encoding="utf-8") as f:
            for suggestion in parameter_suggestions:
                f.write(json.dumps(suggestion) + "\n")

        print("\n✅ Parameter optimization suggestions generated and exported.")
    else:
        print("\n⚠️ No eligible configurations found for suggestion.")
