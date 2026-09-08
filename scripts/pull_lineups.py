"""
Pull today's confirmed MLB batting lineups from the MLB Stats API
and store the top 5 batters per team in the lineups table.

How to run (from any folder):
    python scripts/pull_lineups.py

Three-tier fallback system
--------------------------
Tier 1 — Confirmed lineup (preferred)
    Today's lineup has been officially posted by the team. Use it as-is.

Tier 2 — Projected lineup
    Today's lineup hasn't been posted yet. Find the team's most recent
    completed game in the last 7 days, pull the boxscore batting order,
    and use those same 5 batters prefixed with "[PROJ]".

Tier 3 — Injury override
    After building a Tier 1 or Tier 2 lineup, check the injury_overrides
    table for any players on that team who are marked OUT. Remove them
    and shift the remaining players up so batting_order stays 1–5.
"""

import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
# Paths, .env, and importing db.py from /backend
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

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

from db import execute_query, insert_row  # noqa: E402  (import after path/env setup)

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
HTTP_HEADERS = {"User-Agent": "nrfi-oracle-lineup-pull"}

TOP_ORDER_SLOTS = 5      # only keep batters 1–5
LOOKBACK_DAYS = 7        # how far back to search for a projected lineup


# ----------------------------------------------------------
# HTTP helper
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
            print(f"    Request failed (attempt {attempt}/{retries}): {error}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_error}")


# ----------------------------------------------------------
# Tier 2 — projected lineup from most recent boxscore
# ----------------------------------------------------------
def get_projected_lineup(team_id, team_name, today_str):
    """
    Look up the team's most recent completed game in the last LOOKBACK_DAYS
    days, pull the batting order from that boxscore, and return a list of
    dicts: [{"id": int, "fullName": "[PROJ] Player Name"}, ...]

    Returns an empty list if no suitable game is found.
    """
    start_date = (date.today() - timedelta(days=LOOKBACK_DAYS)).isoformat()

    # Fetch the team's recent schedule (completed games only)
    schedule_data = fetch_json(
        f"{MLB_STATS_BASE}/schedule",
        params={
            "sportId": 1,
            "teamId": team_id,
            "startDate": start_date,
            "endDate": today_str,
            "gameType": "R",          # regular season only
        },
    )

    # Collect all games, newest first
    all_games = []
    for day in schedule_data.get("dates") or []:
        for game in day.get("games") or []:
            all_games.append(game)

    # Reverse so index 0 is the most recent date (dates come oldest-first)
    all_games.reverse()

    # Find the most recent FINAL game
    recent_game = None
    for game in all_games:
        status = (game.get("status") or {}).get("abstractGameState", "")
        if status == "Final":
            recent_game = game
            break

    if recent_game is None:
        print(f"    No completed games found in the last {LOOKBACK_DAYS} days for {team_name}.")
        return []

    game_pk   = recent_game["gamePk"]
    game_date = recent_game.get("officialDate") or recent_game.get("gameDate", "")[:10]

    # Determine whether our team was home or away in that game
    g_teams   = recent_game.get("teams") or {}
    home_id   = (g_teams.get("home", {}).get("team") or {}).get("id")
    side      = "home" if home_id == team_id else "away"

    # Fetch the boxscore for that game
    boxscore = fetch_json(f"{MLB_STATS_BASE}/game/{game_pk}/boxscore")

    team_box     = (boxscore.get("teams") or {}).get(side) or {}
    batting_order = team_box.get("battingOrder") or []   # list of player IDs in order
    players_map   = team_box.get("players") or {}        # "ID12345" -> player dict

    if not batting_order:
        print(f"    Boxscore for game {game_pk} has no batting order for {team_name}.")
        return []

    projected = []
    for player_id in batting_order[:TOP_ORDER_SLOTS]:
        # The players map uses string keys like "ID592450"
        player_info = players_map.get(f"ID{player_id}") or {}
        person      = player_info.get("person") or {}
        full_name   = person.get("fullName") or f"Player {player_id}"
        projected.append({"id": player_id, "fullName": f"[PROJ] {full_name}"})

    print(f"    Using projected lineup for {team_name} based on {game_date} game")
    return projected


# ----------------------------------------------------------
# Tier 3 — injury override
# ----------------------------------------------------------
def apply_injury_overrides(players, team_name):
    """
    Query the injury_overrides table for any player on this team marked OUT.
    Remove them from the lineup list and return the cleaned list.

    'players' is a list of dicts: [{"id": ..., "fullName": ...}, ...]
    """
    try:
        out_rows = execute_query(
            """
            SELECT player_name
            FROM   injury_overrides
            WHERE  team = %s
              AND  status = 'OUT'
            """,
            (team_name,),
        )
    except Exception as error:
        # If the table doesn't exist yet, warn but don't crash
        print(f"    Warning: could not query injury_overrides — {error}")
        return players

    if not out_rows:
        return players

    # Build a set of lowercase names for fast lookup.
    # Strip [PROJ] when comparing so projected players are caught too.
    out_names = {row["player_name"].lower() for row in out_rows}

    cleaned = []
    for player in players:
        raw_name = player["fullName"]
        # Remove the [PROJ] prefix before comparing
        compare_name = raw_name.replace("[PROJ] ", "").strip().lower()
        if compare_name in out_names:
            print(f"    Removed {raw_name} from {team_name} lineup — marked OUT")
        else:
            cleaned.append(player)

    return cleaned


# ----------------------------------------------------------
# Save lineup to DB
# ----------------------------------------------------------
def save_lineup(today_str, team_name, players):
    """
    Insert up to TOP_ORDER_SLOTS players into the lineups table.
    batting_order is re-assigned 1..N after any injury removals.
    """
    for batting_order, player in enumerate(players[:TOP_ORDER_SLOTS], start=1):
        insert_row(
            "lineups",
            {
                "game_date":     today_str,
                "team":          team_name,
                "batting_order": batting_order,
                "player_name":   player["fullName"],
                "player_id":     player.get("id"),
            },
        )


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    today = date.today()
    today_str = today.isoformat()

    print(f"NRFI Oracle lineup pull — {today_str}")
    print("Fetching today's schedule + lineups from MLB Stats API...")

    schedule_data = fetch_json(
        f"{MLB_STATS_BASE}/schedule",
        params={
            "sportId": 1,
            "date": today_str,
            "hydrate": "lineups",
        },
    )

    # The schedule response nests games under a "dates" list.
    raw_games = []
    for day in schedule_data.get("dates") or []:
        raw_games.extend(day.get("games") or [])

    if not raw_games:
        print(f"No games found for {today_str}.")
        return

    print(f"Found {len(raw_games)} game(s). Processing lineups...\n")

    # Clear today's existing lineup rows so we can insert fresh ones.
    print(f"Clearing existing lineup rows for {today_str}...")
    execute_query(
        "DELETE FROM lineups WHERE game_date = %s",
        (today_str,),
    )
    print("  Done.\n")

    teams_posted = 0

    for raw_game in raw_games:
        g_teams   = raw_game.get("teams") or {}
        home_side = g_teams.get("home") or {}
        away_side = g_teams.get("away") or {}

        home_team_name = (home_side.get("team") or {}).get("name") or "Unknown"
        away_team_name = (away_side.get("team") or {}).get("name") or "Unknown"
        home_team_id   = (home_side.get("team") or {}).get("id")
        away_team_id   = (away_side.get("team") or {}).get("id")

        print(f"{away_team_name} @ {home_team_name}")

        # Lineup data lives inside raw_game["lineups"] when hydrated
        lineups      = raw_game.get("lineups") or {}
        home_players_raw = lineups.get("homePlayers") or []
        away_players_raw = lineups.get("awayPlayers") or []

        for team_name, team_id, players_raw in [
            (home_team_name, home_team_id, home_players_raw),
            (away_team_name, away_team_id, away_players_raw),
        ]:
            # --------------------------------------------------
            # Tier 1: confirmed lineup
            # --------------------------------------------------
            if players_raw:
                tier = 1
                # Normalize to the same dict shape used by Tier 2
                players = [
                    {
                        "id":       p.get("id"),
                        "fullName": p.get("fullName") or p.get("name") or "Unknown",
                    }
                    for p in players_raw[:TOP_ORDER_SLOTS]
                ]
                print(f"  [T1] Confirmed lineup for {team_name}")

            # --------------------------------------------------
            # Tier 2: projected lineup from recent boxscore
            # --------------------------------------------------
            else:
                tier = 2
                print(f"  [T2] Lineup not yet posted for {team_name} — fetching projected...")
                if not team_id:
                    print(f"    Cannot project lineup: team ID unknown for {team_name}")
                    continue
                players = get_projected_lineup(team_id, team_name, today_str)
                if not players:
                    print(f"    No projected lineup available for {team_name} — skipping")
                    continue

            # --------------------------------------------------
            # Tier 3: remove injured (OUT) players
            # --------------------------------------------------
            players = apply_injury_overrides(players, team_name)

            if not players:
                print(f"    All batters removed by injury overrides for {team_name} — skipping")
                continue

            # --------------------------------------------------
            # Save to database
            # --------------------------------------------------
            save_lineup(today_str, team_name, players)
            label = "confirmed" if tier == 1 else "projected"
            print(f"  Saved {label} lineup for {team_name} ({min(len(players), TOP_ORDER_SLOTS)} batters)")
            teams_posted += 1

        print()

    print(f"Posted lineups for {teams_posted} team(s) today.")


if __name__ == "__main__":
    main()
