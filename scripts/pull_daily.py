"""
Pull today's MLB games and probable starters, attach stats from our
PostgreSQL database, and write model/today_games.json.

How to run (from any folder):
    python scripts/pull_daily.py

What it does:
    1. Gets today's date automatically.
    2. Hits the MLB Stats API schedule endpoint with probablePitcher data.
    3. For each game, extracts teams, probable pitchers, and game time.
    4. Looks up each pitcher and team in our PostgreSQL database.
    5. Falls back to league-average values when a pitcher/team is missing.
    6. Writes model/today_games.json with all stats embedded so the Flask
       API can score the games immediately without extra DB queries.

run_daily.py calls this script automatically every day.
"""

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
# Paths, .env, and importing db.py from /backend
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
MODEL_DIR = PROJECT_ROOT / "model"
TODAY_GAMES_PATH = MODEL_DIR / "today_games.json"

# Add /backend to the import path so we can do "from db import ..."
# Guard prevents duplicate entries when this module is imported more than once.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load DATABASE_URL from .env in the project root
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

if not os.getenv("DATABASE_URL"):
    raise SystemExit(
        "DATABASE_URL is missing. Add it to the .env file in the project root."
    )

from db import execute_query  # noqa: E402  (import after path/env setup)

# ----------------------------------------------------------
# Database table setup
# ----------------------------------------------------------
def ensure_today_games_table():
    """
    Create the today_games table if it doesn't exist yet.

    Each row holds one game's worth of stats as a JSONB blob so any
    Railway service (cron, API) can read it — no shared filesystem needed.
    Safe to call every run; CREATE TABLE IF NOT EXISTS is a no-op when the
    table already exists.
    """
    execute_query(
        """
        CREATE TABLE IF NOT EXISTS today_games (
            id         SERIAL    PRIMARY KEY,
            game_date  DATE      NOT NULL,
            game_data  JSONB     NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
        """
    )


def save_games_to_db(games_out, today_str):
    """
    Upsert today's game slate into the today_games table.

    We delete the existing rows for today first so re-running the script
    gives a fresh slate rather than duplicate rows.
    """
    try:
        execute_query(
            "DELETE FROM today_games WHERE game_date = %s",
            (today_str,),
        )
        for game in games_out:
            execute_query(
                "INSERT INTO today_games (game_date, game_data) VALUES (%s, %s::jsonb)",
                (today_str, json.dumps(game)),
            )
        print(f"Saved {len(games_out)} game(s) to today_games table in DB.")
    except Exception as error:
        # Not fatal — the JSON file is still available for local use.
        print(f"Warning: could not save games to DB today_games table — {error}")


# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
HTTP_HEADERS = {"User-Agent": "nrfi-oracle-daily-pull"}

# League-average fallback numbers used when a pitcher is not in our database.
# These are roughly the MLB-wide means over recent seasons.
LEAGUE_AVG_PITCHER = {
    "era": 4.30,
    "first_inning_era": 4.73,   # slightly higher than full-game ERA
    "whip": 1.30,
    "k_per_9": 8.50,
    "bb_per_9": 3.20,
}

# League-average first-inning runs-per-game (used when a team is missing).
LEAGUE_AVG_TEAM_FI_RATE = 0.50


# ----------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------
def fetch_json(url, params=None, retries=3):
    """GET a URL and return parsed JSON, retrying on transient failures."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HTTP_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            print(f"  Request failed (attempt {attempt}/{retries}): {error}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


# ----------------------------------------------------------
# Database lookups
# ----------------------------------------------------------
def lookup_pitcher(name):
    """
    Find a pitcher's most recent season row in our pitchers table.

    Returns a dict of stats, or None if the pitcher is not in our database.
    We sort by season DESC so the newest stats are used for players with
    multiple seasons stored.
    """
    if not name:
        return None
    try:
        rows = execute_query(
            """
            SELECT name, era, whip, k_per_9, bb_per_9, first_inning_era, season
            FROM   pitchers
            WHERE  name = %s
            ORDER  BY season DESC
            LIMIT  1
            """,
            (name,),
        )
        return rows[0] if rows else None
    except Exception as error:
        print(f"  DB error looking up pitcher '{name}': {error}")
        return None


def lookup_team(name):
    """
    Find a team's most recent season row in our teams table.

    Returns a dict of stats, or None if the team is not found.
    """
    if not name:
        return None
    try:
        rows = execute_query(
            """
            SELECT team, first_inning_runs_scored, first_inning_games, season
            FROM   teams
            WHERE  team = %s
            ORDER  BY season DESC
            LIMIT  1
            """,
            (name,),
        )
        return rows[0] if rows else None
    except Exception as error:
        print(f"  DB error looking up team '{name}': {error}")
        return None


# ----------------------------------------------------------
# Feature builders
# ----------------------------------------------------------
def pitcher_features(pitcher_row, side, pitcher_name):
    """
    Turn a DB row (or None) into the 5 pitcher feature columns the model
    expects, prefixed with 'home_' or 'away_'.

    If the row is None, league-average values are substituted and a
    warning is printed so you know the fallback was used.
    """
    prefix = f"{side}_pitcher"

    if pitcher_row is None:
        # Pitcher not in our database — fall back to league averages.
        label = pitcher_name or "Unknown pitcher"
        print(f"  '{label}' not found in DB — using league-average stats.")
        return {
            f"{prefix}_era":    LEAGUE_AVG_PITCHER["era"],
            f"{prefix}_fi_era": LEAGUE_AVG_PITCHER["first_inning_era"],
            f"{prefix}_whip":   LEAGUE_AVG_PITCHER["whip"],
            f"{prefix}_k_per_9":  LEAGUE_AVG_PITCHER["k_per_9"],
            f"{prefix}_bb_per_9": LEAGUE_AVG_PITCHER["bb_per_9"],
        }

    # Use the stored value, falling back to the league average for any
    # individual stat that is NULL in the database.
    era = float(pitcher_row.get("era") or LEAGUE_AVG_PITCHER["era"])
    fi_era_stored = pitcher_row.get("first_inning_era")
    return {
        f"{prefix}_era":    era,
        f"{prefix}_fi_era": float(fi_era_stored) if fi_era_stored is not None else round(era * 1.1, 3),
        f"{prefix}_whip":   float(pitcher_row.get("whip") or LEAGUE_AVG_PITCHER["whip"]),
        f"{prefix}_k_per_9":  float(pitcher_row.get("k_per_9") or LEAGUE_AVG_PITCHER["k_per_9"]),
        f"{prefix}_bb_per_9": float(pitcher_row.get("bb_per_9") or LEAGUE_AVG_PITCHER["bb_per_9"]),
    }


def team_fi_rate(team_row, team_name):
    """
    Calculate a team's first-inning run rate (runs ÷ games).

    Returns the league average if the team is missing from our database or
    if the numbers needed to compute the rate are NULL.
    """
    if team_row is None:
        label = team_name or "Unknown team"
        print(f"  '{label}' not found in DB — using league-average fi_rate.")
        return LEAGUE_AVG_TEAM_FI_RATE

    runs = team_row.get("first_inning_runs_scored")
    games = team_row.get("first_inning_games")

    if runs is not None and games:
        return round(float(runs) / float(games), 4)

    return LEAGUE_AVG_TEAM_FI_RATE


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    today = date.today()
    today_str = today.isoformat()

    print(f"NRFI Oracle daily pull — {today_str}")

    # Make sure the today_games table exists before we try to write to it.
    ensure_today_games_table()

    print("Fetching today's schedule from MLB Stats API...")

    # Hit the schedule endpoint, asking it to include probablePitcher data
    # for each game so we get starter names in a single request.
    schedule_data = fetch_json(
        f"{MLB_STATS_BASE}/schedule",
        params={
            "sportId": 1,
            "date": today_str,
            "hydrate": "probablePitcher",
        },
    )
    for day in schedule_data.get("dates") or []:
        print("MLB itself says these games are officially for:", day.get("date"))
    # The schedule response nests games under a "dates" list.
    raw_games = []
    for day in schedule_data.get("dates") or []:
        raw_games.extend(day.get("games") or [])

    if not raw_games:
        print(f"No games found for {today_str}.")
        # Write an empty list so the Flask API gets a clean response.
        MODEL_DIR.mkdir(exist_ok=True)
        TODAY_GAMES_PATH.write_text(json.dumps([], indent=2), encoding="utf-8")
        print(f"Wrote empty game list to {TODAY_GAMES_PATH}")
        # Clear any stale DB rows for today so the API also sees an empty slate.
        save_games_to_db([], today_str)
        return

    print(f"Found {len(raw_games)} game(s) on the schedule. Looking up stats...\n")

    games_out = []

    for raw_game in raw_games:
        teams = raw_game.get("teams") or {}
        home_side = teams.get("home") or {}
        away_side = teams.get("away") or {}

        # Team names
        home_team_name = (home_side.get("team") or {}).get("name")
        away_team_name = (away_side.get("team") or {}).get("name")

        # Probable pitcher names (may be None if not announced yet)
        home_pitcher_info = home_side.get("probablePitcher") or {}
        away_pitcher_info = away_side.get("probablePitcher") or {}
        home_pitcher_name = home_pitcher_info.get("fullName") or None
        away_pitcher_name = away_pitcher_info.get("fullName") or None

        # gameDate is an ISO datetime string, e.g. "2026-08-31T18:10:00Z"
        game_time = raw_game.get("gameDate")

        print(f"{away_team_name} @ {home_team_name}  ({game_time})")
        print(f"  Starters: {away_pitcher_name or 'TBD'} vs {home_pitcher_name or 'TBD'}")

        # --- Database lookups ---
        home_pitcher_row = lookup_pitcher(home_pitcher_name)
        away_pitcher_row = lookup_pitcher(away_pitcher_name)
        home_team_row    = lookup_team(home_team_name)
        away_team_row    = lookup_team(away_team_name)

        # --- Build game object ---
        # We embed all 12 model-feature columns directly so the Flask API's
        # generate endpoint can score the games without extra DB queries.
        game_obj = {
            "game_date":      today_str,
            "game_time":      game_time,
            "home_team":      home_team_name,
            "away_team":      away_team_name,
            "home_pitcher":   home_pitcher_name,
            "away_pitcher":   away_pitcher_name,
            "season":         today.year,
        }

        # Pitcher stats (5 features each, prefixed home_/away_)
        game_obj.update(pitcher_features(home_pitcher_row, "home", home_pitcher_name))
        game_obj.update(pitcher_features(away_pitcher_row, "away", away_pitcher_name))

        # Team first-inning run rates (1 feature each)
        game_obj["home_team_fi_rate"] = team_fi_rate(home_team_row, home_team_name)
        game_obj["away_team_fi_rate"] = team_fi_rate(away_team_row, away_team_name)

        games_out.append(game_obj)
        print()

    # --- Write output file (used locally and by simulate.py when present) ---
    MODEL_DIR.mkdir(exist_ok=True)
    TODAY_GAMES_PATH.write_text(
        json.dumps(games_out, indent=2),
        encoding="utf-8",
    )

    print(f"Found {len(games_out)} games today.")
    print(f"Saved to {TODAY_GAMES_PATH}")

    # --- Also write to the DB so cross-service calls work on Railway ---
    # On Railway each service has its own isolated filesystem, so the API
    # container can't read the file written by the cron container.  Storing
    # the games in the shared PostgreSQL database fixes that.
    save_games_to_db(games_out, today_str)


if __name__ == "__main__":
    main()
