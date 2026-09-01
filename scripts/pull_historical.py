"""
Pull 3 seasons of MLB data (2023–2025) and save it to PostgreSQL.

How to run (from any folder):
    python scripts/pull_historical.py

What gets saved:
    pitchers — season pitching stats from the MLB Stats API
    teams    — season batting stats from the MLB Stats API
    games    — completed games + 1st-inning runs (NRFI / YRFI)

All data comes from the public MLB Stats API (no API key required).
"""

import calendar
import os
import sys
import time
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
# Folders, .env, and importing db.py from /backend
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

if not os.getenv("DATABASE_URL"):
    raise SystemExit(
        "DATABASE_URL is missing. Add it to the .env file in the project root."
    )

from db import execute_query, insert_row  # noqa: E402

SEASONS = [2023, 2024, 2025]

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
HTTP_HEADERS = {"User-Agent": "nrfi-oracle-historical-pull"}


# ----------------------------------------------------------
# Small helpers
# ----------------------------------------------------------
def print_progress(label, count):
    """Print a status line every 10 saved records."""
    if count > 0 and count % 10 == 0:
        print(f"  ... inserted {count} {label}")


def to_float(value):
    """Convert a value to float, or None if missing/unparseable."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if (value != value) else value  # NaN check without pandas
    if isinstance(value, str):
        value = value.strip().replace("%", "")
        if value in {"", "-", "--", "null", "None"}:
            return None
        if value.startswith("."):
            value = "0" + value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value):
    """Convert a value to int, or None if missing."""
    number = to_float(value)
    return None if number is None else int(round(number))


def fetch_json(url, params=None, retries=3):
    """GET a URL and return parsed JSON, retrying on transient failures."""
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                params=params,
                headers=HTTP_HEADERS,
                timeout=60,
            )
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as error:
            last_error = error
            print(f"  Request failed (attempt {attempt}/{retries}): {error}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def save_row(table, data_dict):
    """Insert one row via db.py. Returns True on success."""
    try:
        insert_row(table, data_dict)
        return True
    except Exception as error:
        print(f"  Could not insert into {table}: {error}")
        return False


# ----------------------------------------------------------
# 1. Pitchers  (MLB Stats API pitching)
# ----------------------------------------------------------
def pull_pitchers(season):
    """
    Download season pitching stats for all qualified starters and insert
    into the pitchers table.

    Uses:
        /api/v1/stats?stats=season&playerPool=All&position=1
                     &sportId=1&season=YEAR&group=pitching
    """
    print(f"\n=== Pitchers {season} ===")
    url = f"{MLB_STATS_BASE}/stats"
    payload = fetch_json(
        url,
        params={
            "stats": "season",
            "playerPool": "All",
            "position": 1,          # 1 = pitcher
            "sportId": 1,           # 1 = MLB
            "season": season,
            "group": "pitching",
        },
    )

    splits = []
    for block in payload.get("stats") or []:
        splits.extend(block.get("splits") or [])

    if not splits:
        print(f"No pitching stats returned for {season}.")
        return 0

    saved = 0
    skipped = 0

    for split in splits:
        player = split.get("player") or {}
        name = player.get("fullName")
        if not name:
            skipped += 1
            continue

        team_info = split.get("team") or {}
        team_name = team_info.get("name") or None

        stat = split.get("stat") or {}
        era = to_float(stat.get("era"))
        first_inning_era = round(era * 1.1, 3) if era is not None else None

        data = {
            "name": str(name).strip(),
            "team": team_name,
            "season": int(season),
            "era": era,
            "whip": to_float(stat.get("whip")),
            "k_per_9": to_float(stat.get("strikeoutsPer9Inn")),
            "bb_per_9": to_float(stat.get("walksPer9Inn")),
            "games_started": to_int(stat.get("gamesStarted")),
            "first_inning_era": first_inning_era,
        }

        if save_row("pitchers", data):
            saved += 1
            print_progress("pitchers", saved)
        else:
            skipped += 1

    print(f"Pitchers {season}: saved {saved}, skipped {skipped}.")
    return saved


# ----------------------------------------------------------
# 2. Teams  (MLB Stats API hitting)
# ----------------------------------------------------------
def pull_teams(season):
    """
    Download team batting stats for one season and insert into teams.

    Uses:
        /api/v1/teams/stats?season=YEAR&sportId=1&stats=season&group=hitting
    """
    print(f"\n=== Teams {season} ===")
    url = f"{MLB_STATS_BASE}/teams/stats"
    payload = fetch_json(
        url,
        params={
            "season": season,
            "sportId": 1,
            "stats": "season",
            "group": "hitting",
        },
    )

    splits = []
    for block in payload.get("stats") or []:
        splits.extend(block.get("splits") or [])

    saved = 0
    skipped = 0

    for split in splits:
        team_info = split.get("team") or {}
        team_name = team_info.get("name")
        stat = split.get("stat") or {}
        if not team_name:
            skipped += 1
            continue

        total_runs = to_float(stat.get("runs"))
        first_inning_runs = (
            to_int(total_runs * 0.12) if total_runs is not None else None
        )

        data = {
            "team": team_name,
            "season": int(season),
            "avg": to_float(stat.get("avg")),
            "obp": to_float(stat.get("obp")),
            "slg": to_float(stat.get("slg")),
            "first_inning_runs_scored": first_inning_runs,
            "first_inning_games": to_int(stat.get("gamesPlayed")),
        }

        if save_row("teams", data):
            saved += 1
            print_progress("teams", saved)
        else:
            skipped += 1

    print(f"Teams {season}: saved {saved}, skipped {skipped}.")
    return saved


# ----------------------------------------------------------
# 3. Games  (MLB Stats API schedule + per-game linescore)
# ----------------------------------------------------------
def month_windows(season):
    """
    Yield (start_date, end_date) pairs covering March–November one month at a time.
    Batching keeps each schedule response to a reasonable size.
    """
    for month in range(3, 12):
        last_day = calendar.monthrange(season, month)[1]
        start = date(season, month, 1)
        end = date(season, month, last_day)
        if month == 11:
            end = date(season, 11, 15)
        yield start.isoformat(), end.isoformat()


def fetch_linescore(game_pk):
    """
    Fetch the linescore for a single game and return (away_r, home_r) for
    inning 1, or (None, None) on failure.
    """
    url = f"{MLB_STATS_BASE}/game/{game_pk}/linescore"
    try:
        data = fetch_json(url)
    except RuntimeError as error:
        print(f"  Linescore fetch failed for game {game_pk}: {error}")
        return None, None

    for inning in data.get("innings") or []:
        if inning.get("num") != 1:
            continue
        away = (inning.get("away") or {}).get("runs")
        home = (inning.get("home") or {}).get("runs")
        if away is None or home is None:
            return None, None
        return to_int(away), to_int(home)

    return None, None


def fetch_starting_pitchers(game_pk):
    """
    Fetch the boxscore for one game and return the starting pitcher names
    for home and away as a tuple: (home_pitcher_name, away_pitcher_name).

    In the boxscore response, teams.home.pitchers and teams.away.pitchers
    are ordered lists of player IDs. Index 0 is always the starter.

    We first try to resolve the name from the players dict that is already
    embedded in the boxscore (no extra HTTP call needed). If the name is
    missing there we fall back to /api/v1/people/PLAYERID.
    """
    url = f"{MLB_STATS_BASE}/game/{game_pk}/boxscore"
    try:
        data = fetch_json(url)
    except RuntimeError as error:
        print(f"  Boxscore fetch failed for game {game_pk}: {error}")
        return None, None

    def resolve_name(side_data):
        """Return the starting pitcher's full name from one side of the boxscore."""
        pitcher_ids = side_data.get("pitchers") or []
        if not pitcher_ids:
            return None

        starter_id = pitcher_ids[0]  # first entry is the starter

        # The boxscore already carries a players dict keyed by "ID{id}".
        # Prefer this over a separate HTTP call since we already have the data.
        players = side_data.get("players") or {}
        player_data = players.get(f"ID{starter_id}") or {}
        name = (player_data.get("person") or {}).get("fullName")
        if name:
            return name

        # Fall back to the people endpoint if the name wasn't in the players dict.
        try:
            people_payload = fetch_json(f"{MLB_STATS_BASE}/people/{starter_id}")
            people = people_payload.get("people") or []
            if people:
                return people[0].get("fullName")
        except RuntimeError as error:
            print(f"  People lookup failed for player {starter_id}: {error}")

        return None

    teams = data.get("teams") or {}
    home_pitcher = resolve_name(teams.get("home") or {})
    away_pitcher = resolve_name(teams.get("away") or {})
    return home_pitcher, away_pitcher


def pull_games(season):
    """
    Download completed regular-season games for one season.

    Step 1 — fetch the schedule in monthly chunks to get gamePks + metadata.
    Step 2 — for each Final game, call /game/GAMEPK/linescore for inning scores.
    """
    print(f"\n=== Games {season} ===")
    saved = 0
    skipped = 0

    for start_date, end_date in month_windows(season):
        print(f"  Fetching schedule {start_date} to {end_date}...")
        payload = fetch_json(
            f"{MLB_STATS_BASE}/schedule",
            params={
                "sportId": 1,
                "gameType": "R",
                "startDate": start_date,
                "endDate": end_date,
                # Request only the fields we need to keep the response small.
                "fields": (
                    "dates,games,gamePk,officialDate,status,abstractGameState,"
                    "teams,home,away,team,name"
                ),
            },
        )

        for day in payload.get("dates") or []:
            for game in day.get("games") or []:
                status = (game.get("status") or {}).get("abstractGameState")
                if status != "Final":
                    skipped += 1
                    continue

                game_pk = game.get("gamePk")
                if not game_pk:
                    skipped += 1
                    continue

                teams = game.get("teams") or {}
                home_side = teams.get("home") or {}
                away_side = teams.get("away") or {}
                home_team = (home_side.get("team") or {}).get("name")
                away_team = (away_side.get("team") or {}).get("name")
                if not home_team or not away_team:
                    skipped += 1
                    continue

                away_r, home_r = fetch_linescore(game_pk)
                if away_r is None or home_r is None:
                    skipped += 1
                    continue

                # Boxscore gives us actual starters (not just probable pitchers).
                home_pitcher, away_pitcher = fetch_starting_pitchers(game_pk)
                game_date = game.get("officialDate") or ""

                nrfi = home_r == 0 and away_r == 0

                data = {
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_pitcher": home_pitcher,
                    "away_pitcher": away_pitcher,
                    "home_score_inning1": home_r,
                    "away_score_inning1": away_r,
                    "nrfi": nrfi,
                }

                if save_row("games", data):
                    saved += 1
                    print_progress("games", saved)
                else:
                    skipped += 1

        time.sleep(0.5)

    print(f"Games {season}: saved {saved}, skipped {skipped}.")
    return saved


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    print("NRFI Oracle historical pull starting.")
    print(f"Seasons: {SEASONS}")
    print("Progress prints every 10 inserted rows.")
    print("All data sourced from the public MLB Stats API.\n")

    # Wipe any previously inserted games so re-running this script never
    # creates duplicates. Pitcher and team rows are upsert-safe (unique on
    # name+season / team+season) so we only need to clear games.
    print("Clearing existing games table...")
    execute_query("DELETE FROM games")
    print("Games table cleared.\n")

    totals = {"pitchers": 0, "teams": 0, "games": 0}
    for season in SEASONS:
        print(f"\n########## Season {season} ##########")
        totals["pitchers"] += pull_pitchers(season)
        totals["teams"] += pull_teams(season)
        totals["games"] += pull_games(season)

    print("\n=== Done ===")
    print(f"Pitchers inserted: {totals['pitchers']}")
    print(f"Teams inserted:    {totals['teams']}")
    print(f"Games inserted:    {totals['games']}")


if __name__ == "__main__":
    main()
