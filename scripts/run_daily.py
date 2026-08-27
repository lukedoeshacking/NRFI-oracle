"""
Daily scheduler entry point for NRFI Oracle.

Railway (or any cron-style job) should run this once a day at noon:

    python scripts/run_daily.py

It pulls today's games, then asks the Flask API to generate predictions.
If a step fails, we print the error and keep going instead of crashing.
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import requests

# This file lives in /scripts. pull_daily.py should be right next to it.
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
PULL_DAILY = SCRIPTS_DIR / "pull_daily.py"
GENERATE_URL = "http://localhost:5000/api/predictions/generate"


def print_step_error(step_number, error):
    """Show a failed step without stopping the rest of the run."""
    print(f"Step {step_number} failed: {error}")


def main():
    # --- Step 1 ---
    try:
        print("=== NRFI Oracle Daily Run Starting ===")
    except Exception as error:
        print_step_error(1, error)

    # --- Step 2 ---
    try:
        now = datetime.now()
        print(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as error:
        print_step_error(2, error)

    # --- Step 3: run pull_daily.py and wait until it finishes ---
    try:
        print(f"Running {PULL_DAILY.name}...")
        # sys.executable = the same Python that started this script
        completed = subprocess.run(
            [sys.executable, str(PULL_DAILY)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        # Show whatever pull_daily.py printed so you can follow along
        if completed.stdout:
            print(completed.stdout.rstrip())
        if completed.stderr:
            print(completed.stderr.rstrip())
        if completed.returncode != 0:
            raise RuntimeError(
                completed.stderr.strip()
                or f"{PULL_DAILY.name} exited with code {completed.returncode}"
            )
    except Exception as error:
        print_step_error(3, error)

    # --- Step 4 ---
    try:
        print("Daily pull complete")
    except Exception as error:
        print_step_error(4, error)

    # --- Step 5: ask the API to score today's games ---
    predictions = None
    try:
        print(f"POST {GENERATE_URL}")
        response = requests.post(GENERATE_URL, timeout=120)
        # Raise if the server returned 4xx/5xx
        try:
            response.raise_for_status()
        except requests.HTTPError as http_error:
            # Include the JSON error message from Flask when we can
            detail = response.text
            raise RuntimeError(f"{http_error}; body: {detail}") from http_error
        predictions = response.json()
        if isinstance(predictions, dict) and predictions.get("status") == "error":
            raise RuntimeError(predictions.get("message", predictions))
    except Exception as error:
        print_step_error(5, error)

    # --- Step 6: print a short summary (teams + pick only) ---
    try:
        if predictions is None:
            raise RuntimeError("No predictions to print because step 5 did not succeed")
        rows = predictions
        # The API usually returns a list. If it ever returns {"predictions": [...]}
        # we still handle that.
        if isinstance(predictions, dict):
            rows = predictions.get("predictions", [])
        if not isinstance(rows, list):
            raise RuntimeError(f"Unexpected API response: {predictions!r}")

        print("Today's predictions:")
        if not rows:
            print("  (none)")
        for row in rows:
            away = row.get("away_team", "?")
            home = row.get("home_team", "?")
            pick = row.get("prediction", "?")
            print(f"  {away} @ {home} — {pick}")
    except Exception as error:
        print_step_error(6, error)

    # --- Step 7 ---
    try:
        print("=== Daily Run Complete ===")
    except Exception as error:
        print_step_error(7, error)


if __name__ == "__main__":
    # Catch anything we missed so the scheduled job exits cleanly.
    try:
        main()
    except Exception as error:
        print_step_error("unknown", error)
