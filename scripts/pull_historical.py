"""
Pull 3 seasons of MLB data (2023–2025) and save it to PostgreSQL.

How to run (from any folder):
    python scripts/pull_historical.py

What gets saved:
    pitchers — season pitching stats from pybaseball
    teams    — season batting stats from the MLB Stats API
    games    — completed games + 1st-inning runs (NRFI / YRFI)

This can take a while because every row is inserted one at a time
through db.insert_row(), and because pybaseball downloads FanGraphs data.
"""

import calendar
import os
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from pybaseball import pitching_stats

# ----------------------------------------------------------
# Folders, .env, and importing db.py from /backend
# ----------------------------------------------------------
# This file lives in /scripts. The project root is one folder up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

# db.py is inside /backend, which is NOT on Python's default import path.
# Adding that folder lets us write: from db import insert_row
sys.path.insert(0, str(BACKEND_DIR))

# Load DATABASE_URL from the project-root .env file.
# db.py also calls load_dotenv(), but we do it here so this script is explicit.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

if not os.getenv("DATABASE_URL"):
    raise SystemExit(
        "DATABASE_URL is missing. Add it to the .env file in the project root."
    )

from db import insert_row  # noqa: E402  (import AFTER sys.path change)

# Seasons to download. Change this list if you want more/fewer years.
SEASONS = [2023, 2024, 2025]

# Public MLB Stats API (no API key needed).
MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
# Some hosts block requests with no User-Agent, so we send a simple one.
HTTP_HEADERS = {"User-Agent": "nrfi-oracle-historical-pull"}


# ----------------------------------------------------------
# Small helpers
# ----------------------------------------------------------
def print_progress(label, count):
    """Print a status line every 10 saved records so you can see it working."""
    if count > 0 and count % 10 == 0:
        print(f"  ... inserted {count} {label}")


def to_float(value):
    """Turn a number/string into a float, or None if it is missing."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        value = value.strip().replace("%", "")
        if value in {"", "-", "--", "null", "None"}:
            return None
        # MLB batting AVG often looks like ".263"
        if value.startswith("."):
            value = "0" + value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def to_int(value):
    """Turn a number into an integer, or None if it is missing."""
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def series_get(row, *column_names):
    """
    Read the first matching column from a pandas row.

    FanGraphs / pybaseball sometimes rename columns slightly
    (for example 'K/9' vs 'SO/9'), so we try a few options.
    """
    for name in column_names:
        if name in row.index:
            value = row[name]
            if value is not None and not (isinstance(value, float) and pd.isna(value)):
                return value
    return None


def fetch_json(url, params=None, retries=3):
    """GET a URL and return parsed JSON. Retry a few times if the network hiccups."""
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
    """
    Insert one dict into a table using db.py.

    Returns True on success so the caller can count saved rows.
    One bad row should not stop the whole download.
    """
    try:
        insert_row(table, data_dict)
        return True
    except Exception as error:
        print(f"  Could not insert into {table}: {error}")
        return False


# ----------------------------------------------------------
# 1. Pitchers  (pybaseball / FanGraphs)
# ----------------------------------------------------------
def pull_pitchers(season):
    """
    Download pitching stats for one season and insert into pitchers.

    pybaseball.pitching_stats(year, qual=10) means:
        year  = MLB season
        qual  = minimum innings pitched (we skip tiny samples)
    """
    print(f"\n=== Pitchers {season} ===")
    print(f"Downloading FanGraphs pitching stats for {season} (qual=10)...")

    # This call can take a minute the first time (it downloads a large file).
    stats = pitching_stats(season, qual=10)
    if stats is None or stats.empty:
        print(f"No pitching stats returned for {season}.")
        return 0

    saved = 0
    skipped = 0
    for _, row in stats.iterrows():
        name = series_get(row, "Name", "name")
        if not name:
            skipped += 1
            continue

        era = to_float(series_get(row, "ERA", "era"))
        # First-inning ERA is not in this table, so we approximate:
        # first innings are a little worse than overall ERA on average.
        first_inning_era = round(era * 1.1, 3) if era is not None else None

        data = {
            "name": str(name).strip(),
            "team": str(series_get(row, "Team", "team") or "").strip() or None,
            "season": int(season),
            "era": era,
            "whip": to_float(series_get(row, "WHIP", "whip")),
            "k_per_9": to_float(series_get(row, "K/9", "SO/9", "K9", "k_per_9")),
            "bb_per_9": to_float(series_get(row, "BB/9", "BB9", "bb_per_9")),
            "games_started": to_int(series_get(row, "GS", "gs", "games_started")),
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

    first_inning_runs_scored is approximated as 12% of total runs
    (roughly the share of runs that score in inning 1).
    first_inning_games uses gamesPlayed so later we can compute:
        first-inning run rate = first_inning_runs_scored / first_inning_games
    """
    print(f"\n=== Teams {season} ===")
    url = f"{MLB_STATS_BASE}/teams/stats"
    payload = fetch_json(
        url,
        params={
            "season": season,
            "sportId": 1,       # 1 = Major League Baseball
            "stats": "season",
            "group": "hitting",
        },
    )

    # The JSON looks like: { "stats": [ { "splits": [ { "team": ..., "stat": ... } ] } ] }
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
# 3. Games  (MLB Stats API schedule + linescore)
# ----------------------------------------------------------
def month_windows(season):
    """
    Yield (start_date, end_date) pairs covering a season, one month at a time.

    We request the schedule in chunks so each API response stays a reasonable size.
    Regular season + postseason usually runs late March through early November.
    """
    for month in range(3, 12):  # 3 = March, 11 = November
        last_day = calendar.monthrange(season, month)[1]
        start = date(season, month, 1)
        end = date(season, month, last_day)
        if month == 11:
            end = date(season, 11, 15)
        yield start.isoformat(), end.isoformat()


def first_inning_runs(linescore):
    """
    Return (away_runs, home_runs) from inning 1 of a linescore, or (None, None).

    MLB stores innings like:
        { "num": 1, "away": { "runs": 0 }, "home": { "runs": 1 } }
    """
    if not linescore:
        return None, None
    for inning in linescore.get("innings") or []:
        if inning.get("num") != 1:
            continue
        away = (inning.get("away") or {}).get("runs")
        home = (inning.get("home") or {}).get("runs")
        # A missing "runs" key usually means that half-inning never happened.
        if away is None or home is None:
            return None, None
        return to_int(away), to_int(home)
    return None, None


def pitcher_name(team_side):
    """Read the probable / listed starting pitcher name from a schedule team object."""
    pitcher = team_side.get("probablePitcher") or {}
    name = pitcher.get("fullName")
    if name:
        return name
    return None


def pull_games(season):
    """
    Download completed games for one season and insert into games.

    NRFI is TRUE only when BOTH teams scored 0 runs in inning 1.
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
                "startDate": start_date,
                "endDate": end_date,
                # hydrate=linescore puts inning-by-inning runs on each game
                # hydrate=probablePitcher adds the listed starter names
                "hydrate": "linescore,probablePitcher",
            },
        )

        for day in payload.get("dates") or []:
            for game in day.get("games") or []:
                status = (game.get("status") or {}).get("abstractGameState")
                # Skip games that have not finished (or were postponed).
                if status != "Final":
                    skipped += 1
                    continue

                away_r, home_r = first_inning_runs(game.get("linescore"))
                if away_r is None or home_r is None:
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

                # NRFI = No Runs First Inning → both sides scored zero.
                nrfi = (home_r == 0 and away_r == 0)
                game_date = game.get("officialDate") or (game.get("gameDate") or "")[:10]

                data = {
                    "game_date": game_date,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_pitcher": pitcher_name(home_side),
                    "away_pitcher": pitcher_name(away_side),
                    "home_score_inning1": home_r,
                    "away_score_inning1": away_r,
                    "nrfi": nrfi,
                }

                if save_row("games", data):
                    saved += 1
                    print_progress("games", saved)
                else:
                    skipped += 1

        # Brief pause between months so we do not hammer the public API.
        time.sleep(0.5)

    print(f"Games {season}: saved {saved}, skipped {skipped}.")
    return saved


# ----------------------------------------------------------
# Main: pitchers → teams → games, for each season
# ----------------------------------------------------------
def main():
    print("NRFI Oracle historical pull starting.")
    print(f"Seasons: {SEASONS}")
    print("Progress prints every 10 inserted rows.\n")

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
