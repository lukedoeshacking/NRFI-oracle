"""
NRFI Oracle Flask API.

Start locally (from the /backend folder):
    python app.py

The Procfile runs the same app in production:
    gunicorn app:app

Routes at a glance:
    GET  /api/health                  — is the server up?
    POST /api/predictions/generate    — run Monte Carlo simulation, save & return results
    GET  /api/predictions/today       — fetch today's saved predictions from the DB
    GET  /api/predictions/history     — recent predictions + accuracy stats
"""

import os
import subprocess
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from db import execute_query, insert_row

# ----------------------------------------------------------
# Config: file paths and environment variables
# ----------------------------------------------------------
# This file lives in /backend. Everything else is resolved from there.
BACKEND_DIR      = Path(__file__).resolve().parent
PROJECT_ROOT     = BACKEND_DIR.parent
TODAY_GAMES_PATH = PROJECT_ROOT / "model" / "today_games.json"
SIMULATE_SCRIPT  = PROJECT_ROOT / "scripts" / "simulate.py"

# Load DATABASE_URL (and any other secrets) from the .env file in the
# project root so they're available to execute_query() below.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# ----------------------------------------------------------
# Fields we return to the frontend for every prediction row.
# Adding a field here automatically includes it in every response.
# ----------------------------------------------------------
PREDICTION_FIELDS = [
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "nrfi_probability",
    "prediction",        # "NRFI" or "YRFI"
    "confidence",        # "High", "Medium", or "Low"
    "game_date",
    "is_pick_of_the_day",
]

app = Flask(__name__)

# flask-cors lets the Vercel frontend (a different domain) call this API
# from the browser without being blocked by the same-origin policy.
CORS(app)


# ----------------------------------------------------------
# Helper functions
# ----------------------------------------------------------

def json_safe(value):
    """
    Turn database values into types that Flask's jsonify() can handle.

    jsonify() crashes on Decimal, datetime, and bytes objects, so we
    convert them to their plain-Python equivalents first.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()   # e.g. "2026-09-01"
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def row_to_prediction(row):
    """
    Convert one database row (a dict) to a clean prediction dict.

    Only the fields in PREDICTION_FIELDS are included — this prevents
    accidental leakage of internal columns.
    """
    return {field: json_safe(row.get(field)) for field in PREDICTION_FIELDS}


def error_response(message, status_code=500):
    """
    Return a JSON error body instead of Flask's default HTML error page.

    Example response:
        {"status": "error", "message": "Could not find today_games.json ..."}
    """
    return jsonify({"status": "error", "message": message}), status_code


def actual_is_nrfi(value):
    """
    Convert a stored outcome value to True (NRFI), False (YRFI), or None.

    The games table stores outcomes in several formats depending on how
    the data was entered, so we handle all of them.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "nrfi"}:
        return True
    if text in {"false", "f", "0", "yrfi"}:
        return False
    return None


def fetch_history_rows(days):
    """
    Load recent predictions from the DB, joining to actual outcomes when
    possible.

    Tries three queries in order of preference:
      1. predictions.actual_nrfi column (if it exists)
      2. JOIN to the games table (older schema)
      3. Predictions only (no accuracy possible)
    """
    since_clause = "p.game_date >= CURRENT_DATE - (%s * INTERVAL '1 day')"
    select_fields = """
        p.home_team,
        p.away_team,
        p.home_pitcher,
        p.away_pitcher,
        p.nrfi_probability,
        p.prediction,
        p.confidence,
        p.game_date,
        COALESCE(p.is_pick_of_the_day, false) AS is_pick_of_the_day
    """

    # Attempt 1: use predictions.actual_nrfi
    try:
        return execute_query(
            f"""
            SELECT {select_fields}, p.actual_nrfi AS actual_nrfi
            FROM   predictions p
            WHERE  {since_clause}
            ORDER  BY p.game_date DESC, p.home_team
            """,
            (days,),
        )
    except Exception:
        pass

    # Attempt 2: join games table for outcomes
    try:
        return execute_query(
            f"""
            SELECT {select_fields}, g.nrfi AS actual_nrfi
            FROM   predictions p
            LEFT JOIN games g
              ON  g.home_team  = p.home_team
             AND  g.away_team  = p.away_team
             AND  g.game_date  = p.game_date
            WHERE {since_clause}
            ORDER BY p.game_date DESC, p.home_team
            """,
            (days,),
        )
    except Exception:
        pass

    # Attempt 3: predictions only, no accuracy
    return execute_query(
        f"""
        SELECT {select_fields}
        FROM   predictions p
        WHERE  {since_clause}
        ORDER  BY p.game_date DESC, p.home_team
        """,
        (days,),
    )


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """
    GET /api/health

    Returns a simple OK response. Used by Railway/Vercel health checks
    and as a quick "is the server up?" test.
    """
    return jsonify({"status": "ok", "message": "NRFI Oracle is running"})


@app.route("/api/predictions/today", methods=["GET"])
def predictions_today():
    """
    GET /api/predictions/today

    Returns today's saved predictions from the database.

    Sorting:
      • Pick-of-the-day games appear first (is_pick_of_the_day = true).
      • Within each tier, games are sorted by how far the probability is
        from 50% (the most confident calls come first).

    Each prediction includes:
      matchup           — "Away Team @ Home Team" (pre-built for the frontend)
      home_team         — string
      away_team         — string
      home_pitcher      — string
      away_pitcher      — string
      nrfi_probability  — 0.0–1.0 float
      prediction        — "NRFI" or "YRFI"
      confidence        — "High", "Medium", or "Low"
      game_date         — "YYYY-MM-DD"
      is_pick_of_the_day — true / false

    Response envelope:
      {
        "date": "2026-09-01",
        "total_games": 12,
        "picks_of_the_day": 2,
        "predictions": [...]
      }
    """
    try:
        rows = execute_query(
            """
            SELECT
                home_team,
                away_team,
                home_pitcher,
                away_pitcher,
                nrfi_probability,
                prediction,
                confidence,
                game_date,
                COALESCE(is_pick_of_the_day, false) AS is_pick_of_the_day
            FROM  predictions
            WHERE game_date = CURRENT_DATE
            ORDER BY
                is_pick_of_the_day DESC,               -- picks first
                ABS(nrfi_probability - 0.5) DESC        -- then most confident
            """
        )

        predictions = []
        for row in rows:
            entry = row_to_prediction(row)
            # Add a convenience "matchup" string so the frontend doesn't
            # have to concatenate away_team + " @ " + home_team itself.
            entry["matchup"] = f"{row.get('away_team')} @ {row.get('home_team')}"
            predictions.append(entry)

        return jsonify({
            "date":             date.today().isoformat(),
            "total_games":      len(predictions),
            "picks_of_the_day": sum(1 for p in predictions if p.get("is_pick_of_the_day")),
            "predictions":      predictions,
        })

    except Exception as error:
        return error_response(f"Could not load today's predictions: {error}")


@app.route("/api/predictions/generate", methods=["POST"])
def generate_predictions():
    """
    POST /api/predictions/generate

    Runs the full Monte Carlo simulation pipeline and saves results to
    the predictions table (including the is_pick_of_the_day flag), then
    returns those results as JSON.

    How it works:
      1. Calls scripts/simulate.py as a subprocess.  All the simulation
         logic (game loading, lineup lookup, pitcher stats, two-half
         Monte Carlo, pick-of-the-day selection) lives in that one file,
         so any improvements there automatically flow through this endpoint
         without touching app.py.
      2. simulate.py writes its results directly to the DB.
      3. This endpoint reads those rows back and returns them in the same
         format as GET /api/predictions/today.

    Response on success:
      {
        "status": "ok",
        "games_simulated": 12,
        "predictions": [...],
        "simulation_log": "..."    — last 3 000 chars of simulate.py stdout
      }

    Error responses:
      404 — today_games.json not found (run scripts/pull_daily.py first)
      500 — simulate.py crashed (check "simulation_log" in the response)
      504 — simulation timed out after 5 minutes
    """
    # Guard: make sure today's game slate is available before invoking simulate.py.
    # On Railway, the cron service and the API service run in separate containers,
    # so today_games.json written by the cron container is never visible here.
    # We therefore check the today_games DB table first (populated by pull_daily.py
    # alongside the JSON file), and only fall back to the file check for local dev.
    slate_ready = False
    try:
        rows = execute_query(
            "SELECT COUNT(*) AS n FROM today_games WHERE game_date = CURRENT_DATE"
        )
        if rows and int(rows[0].get("n", 0)) > 0:
            slate_ready = True
    except Exception:
        # Table may not exist yet on first run — fall back to file check below.
        pass

    if not slate_ready and TODAY_GAMES_PATH.exists():
        slate_ready = True   # local dev: JSON file is enough

    if not slate_ready:
        return error_response(
            "No games found for today. "
            "Run scripts/pull_daily.py first to fetch today's slate "
            "(it writes to both today_games.json and the today_games DB table).",
            status_code=404,
        )

    if not SIMULATE_SCRIPT.exists():
        return error_response(
            f"Simulation script not found at {SIMULATE_SCRIPT}.",
            status_code=500,
        )

    try:
        # Run simulate.py from the project root so its relative path
        # references (model/, backend/) resolve correctly.
        # We pass the current process environment so DATABASE_URL and
        # any other secrets are available inside the subprocess.
        result = subprocess.run(
            [sys.executable, str(SIMULATE_SCRIPT)],
            capture_output=True,
            text=True,
            timeout=300,             # 5-minute ceiling for a large slate
            cwd=str(PROJECT_ROOT),
            env={**os.environ},
        )

    except subprocess.TimeoutExpired:
        return error_response(
            "Simulation timed out after 5 minutes. "
            "Check that the database is reachable and try again.",
            status_code=504,
        )

    # A non-zero exit code means simulate.py crashed.
    if result.returncode != 0:
        return error_response(
            f"Simulation script exited with code {result.returncode}. "
            f"Stderr: {result.stderr[-1000:]}",
            status_code=500,
        )

    # Read the predictions simulate.py saved — same sort order as /today.
    rows = execute_query(
        """
        SELECT
            home_team,
            away_team,
            home_pitcher,
            away_pitcher,
            nrfi_probability,
            prediction,
            confidence,
            game_date,
            COALESCE(is_pick_of_the_day, false) AS is_pick_of_the_day
        FROM  predictions
        WHERE game_date = CURRENT_DATE
        ORDER BY
            is_pick_of_the_day DESC,
            ABS(nrfi_probability - 0.5) DESC
        """
    )

    predictions = []
    for row in rows:
        entry = row_to_prediction(row)
        entry["matchup"] = f"{row.get('away_team')} @ {row.get('home_team')}"
        predictions.append(entry)

    return jsonify({
        "status":          "ok",
        "games_simulated": len(predictions),
        "predictions":     predictions,
        # Include the tail of simulate.py's output so the caller can see
        # quality_factor values, pick-of-the-day selection, etc.
        "simulation_log":  result.stdout[-3000:],
    })


@app.route("/api/predictions/history", methods=["GET"])
def predictions_history():
    """
    GET /api/predictions/history?days=30

    Returns predictions from the last N days (default 30, max 365)
    along with accuracy stats when actual first-inning outcomes are stored.

    Response:
      {
        "predictions":       [...],
        "accuracy_pct":      72.5,   — null if no graded games yet
        "total_predictions": 45
      }
    """
    try:
        days = request.args.get("days", default=30, type=int)
        if not days or days < 1:
            days = 30
        days = min(days, 365)   # safety cap

        rows        = fetch_history_rows(days)
        predictions = [row_to_prediction(row) for row in rows]

        # Count how many predictions can be graded and how many were correct.
        graded  = 0
        correct = 0
        for row in rows:
            actual = actual_is_nrfi(row.get("actual_nrfi"))
            if actual is None:
                continue
            graded += 1
            predicted_nrfi = str(row.get("prediction", "")).upper() == "NRFI"
            if predicted_nrfi == actual:
                correct += 1

        accuracy_pct = round((correct / graded) * 100, 2) if graded else None

        return jsonify({
            "predictions":       predictions,
            "accuracy_pct":      accuracy_pct,
            "total_predictions": len(predictions),
        })

    except Exception as error:
        return error_response(f"Could not load prediction history: {error}")


# ----------------------------------------------------------
# Error handler
# ----------------------------------------------------------

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """
    Catch any unhandled exception and return JSON instead of an HTML page.

    Flask's default error page is useless to a JSON API client, so we
    intercept everything here.  Built-in HTTP errors (404, 405, etc.) are
    passed through unchanged.
    """
    if isinstance(error, HTTPException):
        return error
    return error_response(f"Unexpected server error: {error}")


if __name__ == "__main__":
    # debug=True auto-reloads when you save a file.
    # host="0.0.0.0" makes it reachable on your local network, not just localhost.
    app.run(debug=True, host="0.0.0.0", port=5000)
