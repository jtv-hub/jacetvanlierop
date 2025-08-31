"""
Test script for the learning machine module.
Runs the learning cycle against trades.log and prints metrics.
"""

from crypto_trading_bot.learning.learning_machine import run_learning_cycle


def main():
    """Run the learning cycle and display metrics."""
    print("🚀 Running learning machine test...\n")
    metrics = run_learning_cycle()

    if "message" in metrics:
        print(metrics["message"])
    else:
        print("📊 Learning Machine Metrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
