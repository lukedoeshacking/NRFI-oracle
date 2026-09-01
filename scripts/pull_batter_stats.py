"""
Pull season hitting stats for all MLB batters (2023–2025) and save them
to the batters table in PostgreSQL.

How to run (from any folder):
    python scripts/pull_batter_stats.py

Approach (3 steps per season):
    1. Fetch all 30 MLB teams for the season.
    2. For each team, fetch their active roster and collect every
       position player (anyone whose position code is not "1" / Pitcher).
    3. For each unique position player, call the per-player stats endpoint
       to get season hitting totals, then a second call for vs-RHP / vs-LHP
       splits.  Players are deduplicated so mid-season trades don't cause
       double inserts.

Only batters with at least 50 plate appearances are saved.
Expected output: 300-400+ batters per season.
"""

import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
# Paths, .env, and importing db.py from /backend
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

from db import insert_row  # noqa: E402

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
SEASONS = [2023, 2024, 2025]
MIN_PLATE_APPEARANCES = 50

# Small pause between per-player API calls to be polite to the MLB API.
REQUEST_DELAY = 0.15  # seconds

MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
HTTP_HEADERS = {"User-Agent": "nrfi-oracle-batter-pull"}


# ----------------------------------------------------------
# Small helpers
# ----------------------------------------------------------
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
            print(f"    Request failed (attempt {attempt}/{retries}): {error}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


def save_row(table, data_dict):
    """Insert one row via db.py. Returns True on success."""
    try:
        insert_row(table, data_dict)
        return True
    except Exception as error:
        print(f"    Could not insert into {table}: {error}")
        return False


# ----------------------------------------------------------
# Step 1 — Fetch all MLB teams for a season
# ----------------------------------------------------------
def fetch_teams(season):
    """
    Return a list of (team_id, team_name) tuples for all active MLB teams.

    Uses: /api/v1/teams?sportId=1&season=YEAR
    """
    payload = fetch_json(
        f"{MLB_STATS_BASE}/teams",
        params={"sportId": 1, "season": season},
    )
    teams = []
    for team in payload.get("teams") or []:
        team_id = team.get("id")
        team_name = team.get("name")
        if team_id and team_name:
            teams.append((team_id, team_name))
    return teams


# ----------------------------------------------------------
# Step 2 — Fetch a team's active roster, return position players only
# ----------------------------------------------------------
def fetch_position_players(team_id, team_name, season):
    """
    Return a list of dicts for every position player on the active roster.

    Each dict has: player_id, name, team.
    Pitchers (position code "1") are excluded.

    Uses: /api/v1/teams/TEAMID/roster?season=YEAR&rosterType=active
    """
    try:
        payload = fetch_json(
            f"{MLB_STATS_BASE}/teams/{team_id}/roster",
            params={"season": season, "rosterType": "active"},
        )
    except RuntimeError as error:
        print(f"  Roster fetch failed for {team_name}: {error}")
        return []

    players = []
    for entry in payload.get("roster") or []:
        position = entry.get("position") or {}
        # Position code "1" means pitcher — skip them.
        if position.get("code") == "1":
            continue

        person = entry.get("person") or {}
        player_id = person.get("id")
        player_name = person.get("fullName")
        if player_id and player_name:
            players.append({
                "player_id": player_id,
                "name": player_name,
                "team": team_name,
            })

    return players


# ----------------------------------------------------------
# Step 3a — Fetch one player's season hitting stats
# ----------------------------------------------------------
def fetch_player_season_stats(player_id, season):
    """
    Return the season hitting stat dict for one player, or None if unavailable.

    The API can return multiple splits for a traded player (one per team).
    We pick the split with the most plate appearances — that is the combined
    season total.

    Uses: /api/v1/people/PLAYERID/stats?stats=season&season=YEAR&group=hitting
    """
    try:
        payload = fetch_json(
            f"{MLB_STATS_BASE}/people/{player_id}/stats",
            params={"stats": "season", "season": season, "group": "hitting"},
        )
    except RuntimeError:
        return None

    best_stat = None
    best_pa = -1

    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            pa = to_int(stat.get("plateAppearances")) or 0
            if pa > best_pa:
                best_pa = pa
                best_stat = stat

    return best_stat


# ----------------------------------------------------------
# Step 3b — Fetch one player's vs-RHP and vs-LHP OBP splits
# ----------------------------------------------------------
def fetch_player_splits(player_id, season):
    """
    Return a dict with vs_rhp_obp and vs_lhp_obp for one player.
    Values are None when the split data is unavailable.

    Uses: /api/v1/people/PLAYERID/stats?stats=statSplits&season=YEAR
                                        &group=hitting&sitCodes=vr,vl
    """
    result = {"vs_rhp_obp": None, "vs_lhp_obp": None}
    try:
        payload = fetch_json(
            f"{MLB_STATS_BASE}/people/{player_id}/stats",
            params={
                "stats": "statSplits",
                "season": season,
                "group": "hitting",
                "sitCodes": "vr,vl",
            },
        )
    except RuntimeError:
        return result

    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            code = (split.get("split") or {}).get("code", "")
            obp = to_float((split.get("stat") or {}).get("obp"))
            if code == "vr":
                result["vs_rhp_obp"] = obp
            elif code == "vl":
                result["vs_lhp_obp"] = obp

    return result


# ----------------------------------------------------------
# Main season pull
# ----------------------------------------------------------
def pull_batters(season):
    """
    Pull all position-player hitting stats for one season using the
    team → roster → per-player pipeline.
    """
    print(f"\n=== Batters {season} ===")

    # Step 1: get all MLB teams.
    print(f"  Fetching MLB teams for {season}...")
    teams = fetch_teams(season)
    print(f"  Found {len(teams)} teams.")

    # Step 2: collect unique position players across all rosters.
    # Use a dict keyed by player_id so traded players are not duplicated.
    # The last team seen wins (doesn't matter much — stats are season-wide).
    players_by_id = {}
    for team_id, team_name in teams:
        roster_players = fetch_position_players(team_id, team_name, season)
        for p in roster_players:
            players_by_id[p["player_id"]] = p
        time.sleep(REQUEST_DELAY)

    print(f"  Found {len(players_by_id)} unique position players across all rosters.")
    print(f"  Fetching individual stats (this may take a few minutes)...\n")

    # Step 3: fetch stats + splits for each player and insert.
    saved = 0
    skipped_no_pa = 0
    skipped_error = 0

    for player_id, player_info in players_by_id.items():
        player_name = player_info["name"]
        team_name = player_info["team"]

        # Season totals
        stat = fetch_player_season_stats(player_id, season)
        time.sleep(REQUEST_DELAY)

        if stat is None:
            skipped_error += 1
            continue

        plate_appearances = to_int(stat.get("plateAppearances"))
        if not plate_appearances or plate_appearances < MIN_PLATE_APPEARANCES:
            skipped_no_pa += 1
            continue

        strikeouts = to_int(stat.get("strikeOuts"))
        walks = to_int(stat.get("baseOnBalls"))

        k_rate = round(strikeouts / plate_appearances, 4) if strikeouts is not None else None
        bb_rate = round(walks / plate_appearances, 4) if walks is not None else None

        # vs-RHP / vs-LHP splits
        splits = fetch_player_splits(player_id, season)
        time.sleep(REQUEST_DELAY)

        data = {
            "player_id": player_id,
            "name": str(player_name).strip(),
            "team": team_name,
            "season": int(season),
            "avg": to_float(stat.get("avg")),
            "obp": to_float(stat.get("obp")),
            "slg": to_float(stat.get("slg")),
            "k_rate": k_rate,
            "bb_rate": bb_rate,
            "plate_appearances": plate_appearances,
            "vs_rhp_obp": splits["vs_rhp_obp"],
            "vs_lhp_obp": splits["vs_lhp_obp"],
        }

        if save_row("batters", data):
            saved += 1
            if saved % 50 == 0:
                print(f"  ... inserted {saved} batters so far")
        else:
            skipped_error += 1

    print(
        f"\nBatters {season}: saved {saved}, "
        f"skipped (< {MIN_PLATE_APPEARANCES} PA): {skipped_no_pa}, "
        f"skipped (error/no stats): {skipped_error}."
    )
    return saved


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    print("NRFI Oracle batter stats pull starting.")
    print(f"Seasons: {SEASONS}")
    print(f"Minimum plate appearances required: {MIN_PLATE_APPEARANCES}")
    print("Progress prints every 50 inserted batters.")
    print("All data sourced from the public MLB Stats API.\n")

    total = 0
    for season in SEASONS:
        print(f"\n########## Season {season} ##########")
        total += pull_batters(season)

    print(f"\n=== Done ===")
    print(f"Total batters inserted: {total}")


if __name__ == "__main__":
    main()
