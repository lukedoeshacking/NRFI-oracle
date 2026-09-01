"""
Quickly flag a player as injured in the injury_overrides table.

Usage:
    python scripts/add_injury.py "Aaron Judge" "New York Yankees" OUT
    python scripts/add_injury.py "Shohei Ohtani" "Los Angeles Dodgers" QUESTIONABLE
    python scripts/add_injury.py "Freddie Freeman" "Los Angeles Dodgers" DAY_TO_DAY

Valid statuses: OUT, QUESTIONABLE, DAY_TO_DAY

The pull_lineups.py script checks this table and removes any OUT players
from the lineup before saving it to the database.
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------
# Paths, .env, and importing db.py from /backend
# ----------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

from db import execute_query, insert_row  # noqa: E402

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
VALID_STATUSES = {"OUT", "QUESTIONABLE", "DAY_TO_DAY"}


def main():
    # -------------------------------------------------------
    # Validate command-line arguments
    # -------------------------------------------------------
    if len(sys.argv) != 4:
        print("Usage: python scripts/add_injury.py \"Player Name\" \"Team Name\" STATUS")
        print(f"Valid statuses: {', '.join(sorted(VALID_STATUSES))}")
        sys.exit(1)

    player_name = sys.argv[1].strip()
    team_name   = sys.argv[2].strip()
    status      = sys.argv[3].strip().upper()

    if status not in VALID_STATUSES:
        print(f"Error: '{status}' is not a valid status.")
        print(f"Valid statuses: {', '.join(sorted(VALID_STATUSES))}")
        sys.exit(1)

    if not player_name:
        print("Error: player name cannot be empty.")
        sys.exit(1)

    if not team_name:
        print("Error: team name cannot be empty.")
        sys.exit(1)

    # -------------------------------------------------------
    # Check if there's already an entry for this player/team
    # and update it instead of inserting a duplicate.
    # -------------------------------------------------------
    existing = execute_query(
        """
        SELECT id, status
        FROM   injury_overrides
        WHERE  player_name = %s
          AND  team        = %s
        ORDER  BY noted_at DESC
        LIMIT  1
        """,
        (player_name, team_name),
    )

    if existing:
        old_status = existing[0]["status"]
        row_id     = existing[0]["id"]
        execute_query(
            """
            UPDATE injury_overrides
            SET    status   = %s,
                   noted_at = NOW()
            WHERE  id = %s
            """,
            (status, row_id),
        )
        print(f"Updated: {player_name} ({team_name})  {old_status} → {status}")
    else:
        insert_row(
            "injury_overrides",
            {
                "player_name": player_name,
                "team":        team_name,
                "status":      status,
            },
        )
        print(f"Added:   {player_name} ({team_name})  → {status}")

    # Show the current injury list for that team so you can verify
    team_injuries = execute_query(
        """
        SELECT player_name, status, noted_at
        FROM   injury_overrides
        WHERE  team = %s
        ORDER  BY noted_at DESC
        """,
        (team_name,),
    )

    print(f"\nCurrent injury overrides for {team_name}:")
    if team_injuries:
        for row in team_injuries:
            ts = str(row["noted_at"])[:16]   # trim to "YYYY-MM-DD HH:MM"
            print(f"  {row['player_name']:<30} {row['status']:<15} (noted {ts})")
    else:
        print("  (none)")


if __name__ == "__main__":
    main()
