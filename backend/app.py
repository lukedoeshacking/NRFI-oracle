"""
NRFI Oracle Flask API.

Start locally (from the /backend folder):
    python app.py

The Procfile runs the same app in production:
    gunicorn app:app
"""

import json
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import joblib
import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from db import execute_query, insert_row

# ----------------------------------------------------------
# Config: folders, secrets, and the trained model
# ----------------------------------------------------------
# This file lives in /backend. The project root is one folder up.
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
MODEL_PATH = PROJECT_ROOT / "model" / "nrfi_model.pkl"
TODAY_GAMES_PATH = PROJECT_ROOT / "model" / "today_games.json"

# Load DATABASE_URL from .env (db.py does this too; we do it here so
# it is obvious that this API needs that variable).
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# These names MUST stay in the same order the model was trained on
# (see scripts/train_model.py). XGBoost reads columns by position.
FEATURE_COLUMNS = [
    "home_pitcher_era",
    "home_pitcher_fi_era",
    "home_pitcher_whip",
    "home_pitcher_k_per_9",
    "home_pitcher_bb_per_9",
    "away_pitcher_era",
    "away_pitcher_fi_era",
    "away_pitcher_whip",
    "away_pitcher_k_per_9",
    "away_pitcher_bb_per_9",
    "home_team_fi_rate",
    "away_team_fi_rate",
]

# Fields we return to the frontend for a single prediction.
PREDICTION_FIELDS = [
    "home_team",
    "away_team",
    "home_pitcher",
    "away_pitcher",
    "nrfi_probability",
    "prediction",
    "confidence",
    "game_date",
]

# Load the model ONCE when Flask starts — not on every HTTP request.
# If the file is missing, the health route still works; generate will
# return a clear error until you run scripts/train_model.py.
MODEL = None
MODEL_LOAD_ERROR = None
try:
    MODEL = joblib.load(MODEL_PATH)
    print(f"Loaded trained model from {MODEL_PATH}")
except Exception as error:
    MODEL_LOAD_ERROR = str(error)
    print(f"Could not load model at startup: {MODEL_LOAD_ERROR}")

app = Flask(__name__)
# Allow the frontend (any origin) to call this API from the browser.
CORS(app)


# ----------------------------------------------------------
# Small helpers
# ----------------------------------------------------------
def json_safe(value):
    """Turn database/numpy values into types Flask can convert to JSON."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def row_to_prediction(row):
    """Keep only the fields the frontend expects, in a JSON-safe form."""
    return {field: json_safe(row.get(field)) for field in PREDICTION_FIELDS}


def error_response(message, status_code=500):
    """Return a JSON error body instead of a stack trace page."""
    return jsonify({"status": "error", "message": message}), status_code


def parse_today_games(payload):
    """
    Accept either a raw list of games or an object like {"games": [...]}.
    """
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("games", "data", "today_games"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(
        "today_games.json must be a list of games or an object with a 'games' array."
    )


def game_has_features(game):
    """True when every model input is already present on the JSON object."""
    return all(game.get(column) is not None for column in FEATURE_COLUMNS)


def lookup_features_from_db(game):
    """
    If today_games.json only has teams/pitchers (no stats), look them up
    the same way training did: pitchers + teams for that season.
    """
    season = game.get("season") or date.today().year
    home_pitcher = game.get("home_pitcher")
    away_pitcher = game.get("away_pitcher")
    home_team = game.get("home_team")
    away_team = game.get("away_team")

    rows = execute_query(
        """
        SELECT
            hp.era              AS home_pitcher_era,
            hp.first_inning_era AS home_pitcher_fi_era,
            hp.whip             AS home_pitcher_whip,
            hp.k_per_9          AS home_pitcher_k_per_9,
            hp.bb_per_9         AS home_pitcher_bb_per_9,
            ap.era              AS away_pitcher_era,
            ap.first_inning_era AS away_pitcher_fi_era,
            ap.whip             AS away_pitcher_whip,
            ap.k_per_9          AS away_pitcher_k_per_9,
            ap.bb_per_9         AS away_pitcher_bb_per_9,
            (ht.first_inning_runs_scored::float
                / NULLIF(ht.first_inning_games, 0)) AS home_team_fi_rate,
            (at.first_inning_runs_scored::float
                / NULLIF(at.first_inning_games, 0)) AS away_team_fi_rate
        FROM pitchers hp
        JOIN pitchers ap
            ON ap.name = %s AND ap.season = %s
        JOIN teams ht
            ON ht.name = %s AND ht.season = %s
        JOIN teams at
            ON at.name = %s AND at.season = %s
        WHERE hp.name = %s AND hp.season = %s
        LIMIT 1
        """,
        (
            away_pitcher,
            season,
            home_team,
            season,
            away_team,
            season,
            home_pitcher,
            season,
        ),
    )
    if not rows:
        raise ValueError(
            f"Could not find stats for {away_team} @ {home_team} "
            f"({away_pitcher} vs {home_pitcher}, season {season}). "
            "Put the 12 feature columns in today_games.json or load pitcher/team stats."
        )
    return rows[0]


def feature_vector(game):
    """Build a 1 x 12 list of numbers in FEATURE_COLUMNS order."""
    source = game if game_has_features(game) else lookup_features_from_db(game)
    try:
        return [float(source[column]) for column in FEATURE_COLUMNS]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Missing or invalid model features for "
            f"{game.get('away_team')} @ {game.get('home_team')}: {error}"
        ) from error


def confidence_label(nrfi_probability):
    """High > 0.70, Medium > 0.58, otherwise Low."""
    if nrfi_probability > 0.70:
        return "High"
    if nrfi_probability > 0.58:
        return "Medium"
    return "Low"


def prediction_label(nrfi_probability):
    """NRFI if the model thinks 'no runs' is more likely than not."""
    return "NRFI" if nrfi_probability > 0.5 else "YRFI"


def actual_is_nrfi(value):
    """
    Convert a stored outcome into True (NRFI), False (YRFI), or None (unknown).
    Accepts booleans, 0/1, and the strings NRFI/YRFI.
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
    Load recent predictions. Prefer an actual_nrfi column on predictions;
    if that column does not exist yet, fall back to joining the games table.
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
        p.game_date
    """
    try:
        return execute_query(
            f"""
            SELECT {select_fields}, p.actual_nrfi AS actual_nrfi
            FROM predictions p
            WHERE {since_clause}
            ORDER BY p.game_date DESC, p.home_team
            """,
            (days,),
        )
    except Exception:
        # actual_nrfi may not be on the predictions table yet.
        pass

    try:
        return execute_query(
            f"""
            SELECT {select_fields}, g.nrfi AS actual_nrfi
            FROM predictions p
            LEFT JOIN games g
              ON g.home_team = p.home_team
             AND g.away_team = p.away_team
             AND g.game_date = p.game_date
            WHERE {since_clause}
            ORDER BY p.game_date DESC, p.home_team
            """,
            (days,),
        )
    except Exception:
        # Last resort: predictions only, no accuracy possible.
        return execute_query(
            f"""
            SELECT {select_fields}
            FROM predictions p
            WHERE {since_clause}
            ORDER BY p.game_date DESC, p.home_team
            """,
            (days,),
        )


# ----------------------------------------------------------
# Routes
# ----------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    """Simple check so you (and a host) can see the API is up."""
    return jsonify({"status": "ok", "message": "NRFI Oracle is running"})


@app.route("/api/predictions/today", methods=["GET"])
def predictions_today():
    """Return every saved prediction whose game_date is today."""
    try:
        rows = execute_query(
            """
            SELECT home_team, away_team, home_pitcher, away_pitcher,
                   nrfi_probability, prediction, confidence, game_date
            FROM predictions
            WHERE game_date = CURRENT_DATE
            ORDER BY home_team
            """
        )
        return jsonify([row_to_prediction(row) for row in rows])
    except Exception as error:
        return error_response(
            f"Could not load today's predictions. {error}"
        )


@app.route("/api/predictions/generate", methods=["POST"])
def generate_predictions():
    """
    Score today's slate, save rows to PostgreSQL, and return them as JSON.
    """
    try:
        if MODEL is None:
            return error_response(
                "The trained model is not loaded. Run scripts/train_model.py "
                f"so {MODEL_PATH.name} exists. Details: {MODEL_LOAD_ERROR}",
                status_code=503,
            )

        if not TODAY_GAMES_PATH.exists():
            return error_response(
                f"Could not find {TODAY_GAMES_PATH}. "
                "Save today's slate as model/today_games.json first.",
                status_code=404,
            )

        with TODAY_GAMES_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        games = parse_today_games(payload)
        if not games:
            return error_response(
                "today_games.json has no games to score.",
                status_code=400,
            )

        today = date.today()
        # Re-running generate should replace today's rows, not duplicate them.
        execute_query(
            "DELETE FROM predictions WHERE game_date = %s",
            (today,),
        )

        saved = []
        for game in games:
            vector = feature_vector(game)
            # predict_proba returns [P(YRFI), P(NRFI)] because class 1 = NRFI.
            nrfi_probability = float(MODEL.predict_proba([vector])[0][1])
            record = {
                "home_team": game.get("home_team"),
                "away_team": game.get("away_team"),
                "home_pitcher": game.get("home_pitcher"),
                "away_pitcher": game.get("away_pitcher"),
                "nrfi_probability": round(nrfi_probability, 4),
                "prediction": prediction_label(nrfi_probability),
                "confidence": confidence_label(nrfi_probability),
                "game_date": game.get("game_date") or today.isoformat(),
            }
            inserted = insert_row("predictions", record)
            saved.append(row_to_prediction(inserted[0] if inserted else record))

        return jsonify(saved)
    except Exception as error:
        return error_response(f"Could not generate predictions. {error}")


@app.route("/api/predictions/history", methods=["GET"])
def predictions_history():
    """
    GET /api/predictions/history?days=30

    Returns recent predictions plus accuracy when actual 1st-inning
    results are already stored (predictions.actual_nrfi or games.nrfi).
    """
    try:
        days = request.args.get("days", default=30, type=int)
        if days is None or days < 1:
            days = 30
        # Keep the query bounded so a typo cannot request decades of rows.
        days = min(days, 365)

        rows = fetch_history_rows(days)
        predictions = [row_to_prediction(row) for row in rows]

        graded = 0
        correct = 0
        for row in rows:
            actual = actual_is_nrfi(row.get("actual_nrfi"))
            if actual is None:
                continue
            graded += 1
            predicted_nrfi = str(row.get("prediction", "")).upper() == "NRFI"
            if predicted_nrfi == actual:
                correct += 1

        accuracy_pct = None
        if graded:
            accuracy_pct = round((correct / graded) * 100, 2)

        return jsonify(
            {
                "predictions": predictions,
                "accuracy_pct": accuracy_pct,
                "total_predictions": len(predictions),
            }
        )
    except Exception as error:
        return error_response(f"Could not load prediction history. {error}")


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Last-resort handler so uncaught bugs still return JSON, not HTML."""
    if isinstance(error, HTTPException):
        return error
    return error_response(f"Unexpected server error. {error}")


if __name__ == "__main__":
    # debug=True auto-reloads when you save a file. Turn this off in production.
    app.run(debug=True, host="0.0.0.0", port=5000)
