"""
Test Nightly Cycle
Runs the full nightly pipeline: review ledger -> shadow test runner.
"""

import json
import os
import subprocess


def run_script(path):
    """Run a Python script and return its exit code + output."""
    print(f"\n▶ Running {path}...")
    result = subprocess.run(
        ["python", path],
        capture_output=True,
        text=True,
        check=False,  # explicitly set check to avoid Pylint warning
    )
    print(result.stdout)
    if result.stderr:
        print("⚠️ STDERR:", result.stderr)
    return result.returncode


def verify_file(path, sample_lines=3):
    """Verify that a file exists and show first few lines."""
    if not os.path.exists(path):
        print(f"❌ Missing expected file: {path}")
        return False
    print(f"✅ Found file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            try:
                print("   ", json.loads(line.strip()))
            except json.JSONDecodeError:
                print("   (invalid JSON)", line.strip())
    return True


def main():
    """Run the full nightly cycle: review ledger -> shadow test runner -> verify outputs."""
    # Step 1: Run nightly review
    ret1 = run_script("scripts/review_learning_ledger.py")

    # Step 2: Run shadow test runner
    ret2 = run_script("learning/shadow_test_runner.py")

    # Step 3: Verify outputs
    print("\n=== Verifying Outputs ===")
    ok1 = verify_file("learning/suggestions.jsonl")
    ok2 = verify_file("learning/shadow_test_results.jsonl")

    if ret1 == 0 and ret2 == 0 and ok1 and ok2:
        print("\n🌙 Nightly cycle completed successfully!")
    else:
        print("\n⚠️ Nightly cycle had issues. Check logs/ for details.")


if __name__ == "__main__":
    main()
