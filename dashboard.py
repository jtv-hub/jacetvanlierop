from __future__ import annotations

import os

from flask import Flask, render_template

# Helper analytics module
from src.crypto_trading_bot.analytics.learning_summary import (
    get_recent_anomalies,
    get_strategy_leaderboard,
    get_top_confidence_suggestions,
    get_top_shadow_tests,
)


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates")

    # Data sources
    LEARN_FEEDBACK = os.path.join("logs", "learning_feedback.jsonl")
    SHADOW_RESULTS = os.path.join("logs", "shadow_test_results.jsonl")
    ANOMALIES_LOG = os.path.join("logs", "anomalies.log")
    TRADES_LOG = os.path.join("logs", "trades.log")

    @app.route("/")
    def index():
        # Assemble learning section data
        top_suggestions = get_top_confidence_suggestions(LEARN_FEEDBACK, TRADES_LOG, top_n=5)
        top_shadow = get_top_shadow_tests(SHADOW_RESULTS, top_n=3)
        anomalies = get_recent_anomalies(ANOMALIES_LOG, limit=3)
        leaderboard = get_strategy_leaderboard(TRADES_LOG, window=30, top_n=5)

        return render_template(
            "index.html",
            learning={
                "suggestions": top_suggestions,
                "shadow": top_shadow,
                "anomalies": anomalies,
                "leaderboard": leaderboard,
            },
        )

    @app.route("/learning-summary")
    def learning_summary():
        top_suggestions = get_top_confidence_suggestions(LEARN_FEEDBACK, TRADES_LOG, top_n=5)
        top_shadow = get_top_shadow_tests(SHADOW_RESULTS, top_n=3)
        anomalies = get_recent_anomalies(ANOMALIES_LOG, limit=3)
        leaderboard = get_strategy_leaderboard(TRADES_LOG, window=30, top_n=5)

        return render_template(
            "learning_summary.html",
            suggestions=top_suggestions,
            shadow=top_shadow,
            anomalies=anomalies,
            leaderboard=leaderboard,
            standalone=True,
        )

    return app


if __name__ == "__main__":
    app = create_app()
    # Bind to PORT if defined, otherwise 5000.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
