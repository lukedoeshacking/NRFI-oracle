"""
Monte Carlo simulation engine for NRFI Oracle.

For each game today, this script runs 1,000 simulated first innings
and calculates the probability of no runs scoring (NRFI).

How to run (from any folder):
    python scripts/simulate.py

What it does:
    1. Loads today_games.json from /model/  (built by pull_daily.py)
    2. For each game, fetches the away team's lineup from the lineups table
       (away team bats in the top of the 1st inning)
    3. Fetches the home pitcher's WHIP and K/9 from the pitchers table
    4. Runs 1,000 Monte Carlo simulations of the first inning
    5. Calculates NRFI probability = simulations with 0 runs / 1,000
    6. Saves results to the predictions table
    7. Prints a one-line summary per game
"""

import json
import os
import random
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# ----------------------------------------------------------
# Paths, .env, and importing db.py from /backend
# ----------------------------------------------------------
# This script lives in /scripts. The project root is one folder up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR  = PROJECT_ROOT / "backend"
MODEL_DIR    = PROJECT_ROOT / "model"
TODAY_GAMES_PATH = MODEL_DIR / "today_games.json"

# Add /backend to Python's module search path so "from db import ..." works
# whether this file is run directly, called as a subprocess, or imported as
# a module from app.py.  The guard prevents duplicate entries on repeated imports.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load DATABASE_URL from the .env file at the project root
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

if not os.getenv("DATABASE_URL"):
    raise SystemExit(
        "DATABASE_URL is missing. Add it to the .env file in the project root."
    )

from db import execute_query, insert_row, get_connection  # noqa: E402  (import after path/env setup)

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
NUM_SIMULATIONS = 1000   # number of first-inning simulations per game

# League-average fallback values (used when a player or pitcher is missing)
LEAGUE_AVG_OBP  = 0.320  # average batter on-base percentage across MLB
LEAGUE_AVG_SLG  = 0.410  # average batter slugging percentage across MLB
LEAGUE_AVG_WHIP = 1.30   # league-average walks + hits per inning pitched
LEAGUE_AVG_K9   = 8.50   # league-average strikeouts per 9 innings

# Ratio used to scale each batter's double rate relative to league average.
# A batter with a higher SLG/OBP ratio than this has above-average power.
LEAGUE_SLG_PER_OBP = LEAGUE_AVG_SLG / LEAGUE_AVG_OBP  # ≈ 1.28


# ----------------------------------------------------------
# Database helpers
# ----------------------------------------------------------
def get_lineup(game_date_str, team_name, conn=None):
    """
    Fetch the top 5 batters for a team on a given date from the lineups table.

    The lineups table is populated by pull_lineups.py every morning.
    Returns a list of player name strings in batting order (1–5).
    Returns an empty list when no lineup exists for that team today.

    conn — optional already-open psycopg2 connection (passed from main).
    """
    try:
        rows = execute_query(
            """
            SELECT player_name
            FROM   lineups
            WHERE  game_date = %s
              AND  team = %s
            ORDER  BY batting_order
            LIMIT  5
            """,
            (game_date_str, team_name),
            conn,
        )
        return [row["player_name"] for row in rows]
    except Exception as error:
        print(f"    DB error fetching lineup for {team_name}: {error}")
        return []


def get_batter_stats(player_name, conn=None):
    """
    Return {"obp": float, "slg": float} for a batter, checking three sources
    in priority order:

      1. batter_recent_form table  — blended stats (65% last-30-days + 35% season)
         populated by pull_recent_form.py each morning
      2. batters table             — full-season average
      3. League-average fallback

    Falls back to the next source automatically if a table is missing or the
    batter is not found.

    conn — optional already-open psycopg2 connection (passed from main).
    """
    clean_name = player_name.replace("[PROJ] ", "").strip()

    # --- 1. Blended recent-form stats (most up-to-date) ---
    try:
        rows = execute_query(
            """
            SELECT blended_obp, blended_slg
            FROM   batter_recent_form
            WHERE  name = %s AND as_of_date = CURRENT_DATE
            ORDER  BY id DESC
            LIMIT  1
            """,
            (clean_name,),
            conn,
        )
        if rows and rows[0].get("blended_obp") is not None:
            return {
                "obp": float(rows[0]["blended_obp"]),
                "slg": float(rows[0]["blended_slg"] or LEAGUE_AVG_SLG),
            }
    except Exception:
        pass   # table might not exist yet — continue to next source

    # --- 2. Season stats from the batters table ---
    try:
        rows = execute_query(
            """
            SELECT obp, slg
            FROM   batters
            WHERE  name = %s
            ORDER  BY season DESC
            LIMIT  1
            """,
            (clean_name,),
            conn,
        )
        if rows and rows[0].get("obp") is not None:
            return {
                "obp": float(rows[0]["obp"]),
                "slg": float(rows[0]["slg"] or LEAGUE_AVG_SLG),
            }
    except Exception as error:
        print(f"    DB error fetching stats for '{clean_name}': {error}")

    # --- 3. League-average fallback ---
    print(f"    '{clean_name}' not in batters table — using league-avg OBP/SLG")
    return {"obp": LEAGUE_AVG_OBP, "slg": LEAGUE_AVG_SLG}


def get_pitcher_stats(pitcher_name, game, side="home", conn=None):
    """
    Return (whip, k_per_9) for a pitcher, checking four sources in order:

      1. pitcher_recent_form table — blended stats (65% last-5-starts + 35% season)
         populated by pull_recent_form.py each morning
      2. pitchers table            — full-season average
      3. today_games.json values   — stats embedded by pull_daily.py
      4. League-average fallback

    Falls back to the next source automatically at each step.

    side : "home" or "away" — controls which fallback key is read from the
           game dict (e.g. "home_pitcher_whip" vs "away_pitcher_whip").
    conn : optional already-open psycopg2 connection (passed from main).
    """
    if pitcher_name and pitcher_name != "TBD":

        # --- 1. Blended recent-form stats (most up-to-date) ---
        try:
            rows = execute_query(
                """
                SELECT blended_whip, blended_k_per_9
                FROM   pitcher_recent_form
                WHERE  name = %s AND as_of_date = CURRENT_DATE
                ORDER  BY id DESC
                LIMIT  1
                """,
                (pitcher_name,),
                conn,
            )
            if rows and rows[0].get("blended_whip") is not None:
                return float(rows[0]["blended_whip"]), float(rows[0]["blended_k_per_9"])
        except Exception:
            pass   # table might not exist yet — continue to next source

        # --- 2. Season stats from the pitchers table ---
        try:
            rows = execute_query(
                """
                SELECT whip, k_per_9
                FROM   pitchers
                WHERE  name = %s
                ORDER  BY season DESC
                LIMIT  1
                """,
                (pitcher_name,),
                conn,
            )
            if rows:
                whip    = rows[0].get("whip")
                k_per_9 = rows[0].get("k_per_9")
                if whip is not None and k_per_9 is not None:
                    return float(whip), float(k_per_9)
        except Exception as error:
            print(f"    DB error fetching stats for '{pitcher_name}': {error}")

    # --- 3. Stats embedded in today_games.json by pull_daily.py ---
    # Use the correct side prefix so home and away pitchers get their own values.
    whip    = game.get(f"{side}_pitcher_whip")    or LEAGUE_AVG_WHIP
    k_per_9 = game.get(f"{side}_pitcher_k_per_9") or LEAGUE_AVG_K9
    return float(whip), float(k_per_9)


# ----------------------------------------------------------
# Pitcher quality factor
# ----------------------------------------------------------
def calc_quality_factor(whip, k_per_9):
    """
    Turn a pitcher's WHIP and K/9 into a single scaling factor for batter OBP.

    How each component works:
      WHIP component  = pitcher_whip / league_avg_whip
        * Low WHIP  (good pitcher) → factor < 1 → batters reach base less often
        * High WHIP (bad pitcher)  → factor > 1 → batters reach base more often

      K/9 component   = league_avg_k9 / pitcher_k_per_9
        * High K rate (good pitcher) → factor < 1 → batters reach less often
        * Low K rate  (bad pitcher)  → factor > 1 → batters reach more often

    We average the two so neither stat dominates.

    Example — great pitcher (WHIP 1.00, K/9 11.0):
        whip_component = 1.00 / 1.30 = 0.77
        k_component    = 8.50 / 11.0 = 0.77
        quality_factor = 0.77  (batters have 77% of their normal OBP)

    Example — bad pitcher (WHIP 1.60, K/9 6.0):
        whip_component = 1.60 / 1.30 = 1.23
        k_component    = 8.50 /  6.0 = 1.42
        quality_factor = 1.32  (batters have 132% of their normal OBP)
    """
    whip_component = whip / LEAGUE_AVG_WHIP
    # Guard against a pitcher with 0 strikeouts (avoid divide-by-zero)
    k_component    = LEAGUE_AVG_K9 / max(k_per_9, 0.1)
    return (whip_component + k_component) / 2.0


# ----------------------------------------------------------
# At-bat event classification
# ----------------------------------------------------------
def on_base_event_type(obp, slg):
    """
    Given that a batter has already reached base, decide HOW they got on:
    'walk', 'single', or 'double'.

    Fractions are calibrated to real 2024 MLB averages:

      Walk  ≈ 27%   BB/PA (8.5%) divided by OBP (31.5%)
      Double≈ 14%   2B/PA (4.4%) divided by OBP (31.5%)
      Single≈ 59%   remaining hits (triples/HRs are rare, folded into singles
                    for simplicity — this slightly over-counts singles but
                    avoids the complexity of simulating HR separately)

    The double fraction is further scaled by the batter's power (SLG/OBP)
    relative to the league average, so sluggers get proportionally more
    doubles and contact hitters get more singles.

    Example — power hitter (SLG=0.520, OBP=0.380):
        power_ratio = (0.520/0.380) / 1.28 = 1.07
        double_rate ≈ 15%  (slightly above average)

    Example — contact hitter (SLG=0.340, OBP=0.350):
        power_ratio = (0.340/0.350) / 1.28 = 0.76
        double_rate ≈ 11%  (below average)
    """
    # ~27% of all OBP events are walks — calibrated to real MLB BB/OBP ratio
    walk_rate = 0.27

    # Base double rate ~14%; scale by batter's power relative to league avg.
    # Cap between 5% and 35% so extreme inputs stay sensible.
    power_ratio = (slg / max(obp, 0.001)) / LEAGUE_SLG_PER_OBP
    double_rate = min(max(0.14 * power_ratio, 0.05), 0.35)

    # Singles fill whatever fraction is left after walks and doubles.
    # single_rate = 1 - walk_rate - double_rate (implicit from the thresholds below)

    roll = random.random()
    if roll < walk_rate:
        return "walk"
    if roll < walk_rate + double_rate:
        return "double"
    return "single"


# ----------------------------------------------------------
# Baserunner advancement helpers
# ----------------------------------------------------------
def apply_walk(bases):
    """
    Award batter 1st base.  Only advance runners who are 'forced'
    (their path forward is blocked because the batter is pushing
    the chain from behind).

    Force chain:  batter → 1st  forces  1st → 2nd  forces  2nd → 3rd
                  forces  3rd → home  (RUN SCORES).

    bases : [bool, bool, bool]  — [1st, 2nd, 3rd]
    Returns (new_bases, run_scored).
    """
    new_1st = True           # batter always takes 1st
    new_2nd = bases[1]       # 2nd runner stays unless forced
    new_3rd = bases[2]       # 3rd runner stays unless forced
    run_scored = False

    if bases[0]:             # 1st was occupied → runner forced to 2nd
        new_2nd = True
        if bases[1]:         # 2nd was occupied → runner forced to 3rd
            new_3rd = True
            if bases[2]:     # 3rd was occupied → runner forced home
                run_scored = True

    return [new_1st, new_2nd, new_3rd], run_scored


def apply_single(bases):
    """
    All runners advance 1 base.

    Scoring rules (calibrated to real MLB):
      Runner on 3rd  → scores virtually every time (100%)
      Runner on 2nd  → scores ~45% of the time on a single
                       (depends on runner speed, hit direction, outfield depth)
      Runner on 1st  → advances to 2nd; does NOT score
      Batter         → takes 1st base

    bases : [bool, bool, bool]  — [1st, 2nd, 3rd]
    Returns (new_bases, run_scored).
    """
    # Runner on 3rd always scores; runner on 2nd scores roughly half the time
    run_scored = bases[2] or (bases[1] and random.random() < 0.45)
    # After the single: old-1st → 2nd, batter → 1st, 2nd+3rd are now empty
    new_bases = [True, bases[0], False]
    return new_bases, run_scored


def apply_double(bases):
    """
    All runners score (any runner on any base reaches home on a double).
    Batter ends up on 2nd base.

    bases : [bool, bool, bool]  — [1st, 2nd, 3rd]
    Returns (new_bases, run_scored).
    """
    run_scored = any(bases)             # every runner on base scores
    new_bases  = [False, True, False]   # only the batter, now on 2nd
    return new_bases, run_scored


# ----------------------------------------------------------
# Single inning simulation
# ----------------------------------------------------------
def simulate_inning(batters_stats, quality_factor):
    """
    Simulate one half-inning and return True (NRFI) or False (YRFI).

    Parameters
    ----------
    batters_stats  : list of 5 dicts — each has "obp" and "slg" for one batter
    quality_factor : float — pitcher strength multiplier (from calc_quality_factor)

    How the simulation works
    ------------------------
    We cycle through the batting order until 3 outs are recorded.

    Each plate appearance has two stages:
      1. Does the batter reach base?
         effective_obp = batter_obp x quality_factor (capped at 0.95).
         random.random() < effective_obp  → batter reaches base.

      2. If on base, what kind of event was it?
         on_base_event_type() returns 'walk', 'single', or 'double'
         based on the batter's SLG/OBP power ratio.

    Baserunner advancement:
      walk   → only force runners who are blocked (see apply_walk)
      single → runners on 2nd/3rd score; runner on 1st advances to 2nd
      double → all runners score; batter ends on 2nd

    Returns True when 3 outs are recorded with 0 runs scored (NRFI).
    """
    outs  = 0
    bases = [False, False, False]   # [1st, 2nd, 3rd]
    batter_idx = 0

    while outs < 3:
        batter        = batters_stats[batter_idx % len(batters_stats)]
        raw_obp       = batter["obp"]
        slg           = batter["slg"]
        effective_obp = min(raw_obp * quality_factor, 0.95)

        if random.random() < effective_obp:
            # ---- Batter reaches base — classify the event ----
            event = on_base_event_type(raw_obp, slg)

            if event == "walk":
                bases, run_scored = apply_walk(bases)
            elif event == "double":
                bases, run_scored = apply_double(bases)
            else:                              # single
                bases, run_scored = apply_single(bases)

            if run_scored:
                return False    # YRFI — stop as soon as a run scores
        else:
            # ---- Batter makes an out ----
            outs += 1

        batter_idx += 1

    return True   # 3 outs, 0 runs → NRFI


# ----------------------------------------------------------
# Debug simulation (verbose, for diagnosing probability issues)
# ----------------------------------------------------------
def debug_simulate_inning(batters_stats, quality_factor, batter_names, sim_num):
    """
    Run one simulated inning and print every plate appearance in detail.

    This is only called for the FIRST game, for 3 sample simulations.
    It is never called during the real 1,000-simulation loop.

    Shows for each PA:
      - Batter name, raw OBP, and SLG
      - The effective OBP after the pitcher quality factor is applied
      - The random number that was drawn
      - The event type (walk / single / double) if the batter reached
      - The state of the bases after the PA
      - The running out count
    """
    print(f"  -- Simulation #{sim_num} --------------------------------------------------")

    outs      = 0
    bases     = [False, False, False]
    batter_idx = 0
    outcome   = "NRFI"

    while outs < 3:
        slot      = batter_idx % len(batters_stats)
        batter    = batters_stats[slot]
        raw_obp   = batter["obp"]
        slg       = batter["slg"]
        eff_obp   = min(raw_obp * quality_factor, 0.95)
        roll      = random.random()
        reached   = roll < eff_obp
        name      = batter_names[slot] if batter_names else f"Batter {slot + 1}"
        short_name = name.replace("[PROJ] ", "")[:20]

        if reached:
            event = on_base_event_type(raw_obp, slg)

            if event == "walk":
                bases, run_scored = apply_walk(bases)
            elif event == "double":
                bases, run_scored = apply_double(bases)
            else:
                bases, run_scored = apply_single(bases)

            bases_str  = "".join(["1B" if bases[0] else "__",
                                   " 2B" if bases[1] else " __",
                                   " 3B" if bases[2] else " __"])
            run_tag    = " (RUN!)" if run_scored else ""
            result_tag = f"REACHED/{event.upper()}{run_tag}"

            print(
                f"    #{slot+1} {short_name:<20s}  "
                f"OBP={raw_obp:.3f} SLG={slg:.3f}  eff={eff_obp:.3f}  "
                f"roll={roll:.3f}  ->  {result_tag:<24s} | {bases_str}  outs={outs}"
            )

            if run_scored:
                outcome = "YRFI"
                break
        else:
            outs += 1
            bases_str = "".join(["1B" if bases[0] else "__",
                                  " 2B" if bases[1] else " __",
                                  " 3B" if bases[2] else " __"])
            print(
                f"    #{slot+1} {short_name:<20s}  "
                f"OBP={raw_obp:.3f} SLG={slg:.3f}  eff={eff_obp:.3f}  "
                f"roll={roll:.3f}  ->  OUT                        | {bases_str}  outs={outs}"
            )

        batter_idx += 1

    print(f"  -> Inning result: {outcome}")
    print()
    return outcome == "NRFI"


# ----------------------------------------------------------
# Prediction labels (same thresholds as the rest of the app)
# ----------------------------------------------------------
def confidence_label(nrfi_prob):
    """Convert a probability to a confidence tier: High / Medium / Low."""
    if nrfi_prob > 0.70:
        return "High"
    if nrfi_prob > 0.58:
        return "Medium"
    return "Low"


def prediction_label(nrfi_prob):
    """NRFI if the probability is above 50%, otherwise YRFI."""
    return "NRFI" if nrfi_prob > 0.5 else "YRFI"


# ----------------------------------------------------------
# Pick of the day
# ----------------------------------------------------------

def ensure_pick_column(conn=None):
    """
    Add the is_pick_of_the_day column to predictions if it doesn't exist yet.

    Uses ADD COLUMN IF NOT EXISTS so this is safe to run every time the
    script starts — it does nothing when the column is already there.

    conn — optional already-open psycopg2 connection (passed from main).
    """
    try:
        execute_query(
            """
            ALTER TABLE predictions
            ADD COLUMN IF NOT EXISTS is_pick_of_the_day BOOLEAN DEFAULT FALSE
            """,
            conn=conn,
        )
    except Exception as error:
        print(f"  Warning: could not add is_pick_of_the_day column — {error}")


def get_pick_of_the_day(results):
    """
    Return the 1–2 most confident predictions from today's simulation.

    Confidence is measured by how far the probability is from 50%:
        distance = abs(nrfi_probability - 0.5)

    A strong YRFI (15% NRFI probability, distance 0.35) ranks just as
    confidently as a strong NRFI (85%, distance 0.35).  This avoids
    always favouring high-NRFI games over low-NRFI games.

    Number of picks returned:
        • 1 pick  — when 1–3 games are on the slate (small card)
        • 2 picks — when 4+ games are on the slate

    Returns a list of result dicts (subset of `results`).
    """
    if not results:
        return []

    sorted_by_confidence = sorted(
        results,
        key=lambda r: abs(r["nrfi_probability"] - 0.5),
        reverse=True,
    )

    n_picks = 2 if len(results) >= 4 else 1
    return sorted_by_confidence[:n_picks]


# ----------------------------------------------------------
# Game loading (file → DB fallback)
# ----------------------------------------------------------
def load_today_games(today_str, conn=None):
    """
    Return today's game list, trying two sources in order:

      1. model/today_games.json — only trusted if the games inside it are
         actually dated for today. A leftover file from a previous day
         (e.g. accidentally committed to git) is ignored instead of silently
         reused.
      2. today_games database table — used on Railway where the cron service
         and the API service each have their own isolated filesystem.

    Returns a list of game dicts (may be empty if no games today).
    Raises SystemExit if neither source can be read.

    conn — optional already-open psycopg2 connection (passed from main).
    """
    # --- Source 1: JSON file, but only if it's actually for TODAY ---
    if TODAY_GAMES_PATH.exists():
        with TODAY_GAMES_PATH.open(encoding="utf-8") as fh:
            file_games = json.load(fh)
        if file_games and file_games[0].get("game_date") == today_str:
            print(f"Loading games from {TODAY_GAMES_PATH} (confirmed fresh for {today_str})")
            return file_games
        else:
            stale_date = file_games[0].get("game_date") if file_games else "unknown"
            print(
                f"{TODAY_GAMES_PATH} exists but is dated {stale_date}, not {today_str} — "
                "ignoring stale file and checking the database instead."
            )

    # --- Source 2: today_games DB table ---
    print("Loading games from today_games database table.")
    try:
        rows = execute_query(
            "SELECT game_data FROM today_games WHERE game_date = %s ORDER BY id",
            (today_str,),
            conn,
        )
        if rows is None:
            raise RuntimeError("execute_query returned None")
        return [row["game_data"] for row in rows]
    except Exception as error:
        raise SystemExit(
            f"Could not load today's games from DB: {error}\n"
            "Run scripts/pull_daily.py first to populate the today_games table."
        )


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
def main():
    today = date.today()
    today_str = today.isoformat()

    print(f"NRFI Oracle Monte Carlo simulation — {today_str}")
    print(f"Running {NUM_SIMULATIONS:,} simulations per game...\n")

    # ------------------------------------------------------------------
    # Open ONE database connection for the entire simulation run.
    # Every DB call below receives this same connection instead of
    # opening (and closing) its own, saving ~100+ round-trips to Supabase.
    # The try/finally guarantees the connection is closed even if an
    # exception is raised partway through.
    # ------------------------------------------------------------------
    print("Opening database connection...")
    conn = get_connection()

    try:
        # Make sure the predictions table has the is_pick_of_the_day column.
        # This is a no-op if the column already exists.
        ensure_pick_column(conn)

        # ---- Step 1: Load today's game slate ----
        # On Railway each service runs in its own container, so the JSON file
        # written by the cron container won't exist on the API container.
        # We try the file first (works locally and in the same container), then
        # fall back to the today_games database table written by pull_daily.py.
        games = load_today_games(today_str, conn)

        if not games:
            print("No games found for today. Nothing to simulate.")
            return

        print(f"Found {len(games)} game(s). Starting simulations...\n")

        # Clear any existing predictions for today so re-running this script
        # replaces the old rows rather than duplicating them.
        execute_query(
            "DELETE FROM predictions WHERE game_date = %s",
            (today_str,),
            conn,
        )

        results = []
        quality_factors = []   # collects every pitcher quality_factor this run

        for game_idx, game in enumerate(games):
            away_team    = game.get("away_team")   or "Unknown"
            home_team    = game.get("home_team")   or "Unknown"
            home_pitcher = game.get("home_pitcher") or "TBD"
            away_pitcher = game.get("away_pitcher") or "TBD"

            print(f"{away_team} @ {home_team}")
            print(f"  TOP: {away_team} bat vs {home_pitcher}  |  BOT: {home_team} bat vs {away_pitcher}")

            # ---- Step 2a: Away lineup (top of 1st — away team bats vs home pitcher) ----
            away_lineup = get_lineup(today_str, away_team, conn)
            if not away_lineup:
                print(f"  No lineup found for {away_team} — using 5 league-avg batters")
                away_stats = [{"obp": LEAGUE_AVG_OBP, "slg": LEAGUE_AVG_SLG}] * 5
            else:
                preview = ", ".join(away_lineup[:3])
                print(f"  {away_team} lineup: {preview}{'...' if len(away_lineup) > 3 else ''}")
                away_stats = [get_batter_stats(name, conn) for name in away_lineup]
                while len(away_stats) < 5:
                    away_stats.append({"obp": LEAGUE_AVG_OBP, "slg": LEAGUE_AVG_SLG})

            # ---- Step 2b: Home lineup (bottom of 1st — home team bats vs away pitcher) ----
            home_lineup = get_lineup(today_str, home_team, conn)
            if not home_lineup:
                print(f"  No lineup found for {home_team} — using 5 league-avg batters")
                home_stats = [{"obp": LEAGUE_AVG_OBP, "slg": LEAGUE_AVG_SLG}] * 5
            else:
                preview = ", ".join(home_lineup[:3])
                print(f"  {home_team} lineup: {preview}{'...' if len(home_lineup) > 3 else ''}")
                home_stats = [get_batter_stats(name, conn) for name in home_lineup]
                while len(home_stats) < 5:
                    home_stats.append({"obp": LEAGUE_AVG_OBP, "slg": LEAGUE_AVG_SLG})

            # ---- Step 3a: Home pitcher stats (faces away batters in top half) ----
            h_whip, h_k9 = get_pitcher_stats(home_pitcher, game, side="home", conn=conn)
            h_quality     = calc_quality_factor(h_whip, h_k9)
            h_whip_comp   = h_whip / LEAGUE_AVG_WHIP
            h_k_comp      = LEAGUE_AVG_K9 / max(h_k9, 0.1)
            print(
                f"  [TOP] {home_pitcher}: "
                f"WHIP={h_whip:.2f} (comp={h_whip_comp:.3f})  "
                f"K/9={h_k9:.1f} (comp={h_k_comp:.3f})  "
                f"quality={h_quality:.3f}"
            )
            quality_factors.append(h_quality)

            # ---- Step 3b: Away pitcher stats (faces home batters in bottom half) ----
            a_whip, a_k9 = get_pitcher_stats(away_pitcher, game, side="away", conn=conn)
            a_quality     = calc_quality_factor(a_whip, a_k9)
            a_whip_comp   = a_whip / LEAGUE_AVG_WHIP
            a_k_comp      = LEAGUE_AVG_K9 / max(a_k9, 0.1)
            print(
                f"  [BOT] {away_pitcher}: "
                f"WHIP={a_whip:.2f} (comp={a_whip_comp:.3f})  "
                f"K/9={a_k9:.1f} (comp={a_k_comp:.3f})  "
                f"quality={a_quality:.3f}"
            )
            quality_factors.append(a_quality)

            # ---- DEBUG: 3 sample simulations for the first game (top half only) ----
            if game_idx == 0:
                print()
                print("  ══════════════════════════════════════════════════════════")
                print(f"  DEBUG — 3 sample TOP-half simulations for: {away_team} @ {home_team}")
                print()
                print(f"  Home pitcher quality factor breakdown:")
                print(f"    WHIP: {h_whip:.2f}/{LEAGUE_AVG_WHIP} = {h_whip_comp:.3f}  |  K/9: {LEAGUE_AVG_K9}/{h_k9:.1f} = {h_k_comp:.3f}  |  avg = {h_quality:.3f}")
                print()
                print(f"  {away_team} batter OBP/SLG (top-half batters):")
                names_debug = away_lineup if away_lineup else [f"Batter {i+1}" for i in range(5)]
                for i, (name, stats) in enumerate(zip(names_debug, away_stats), start=1):
                    eff = min(stats["obp"] * h_quality, 0.95)
                    print(f"    #{i}  {name.replace('[PROJ] ', ''):<22s}  OBP={stats['obp']:.3f}  SLG={stats['slg']:.3f}  eff_OBP={eff:.3f}")
                print()
                for sample_num in range(1, 4):
                    debug_simulate_inning(away_stats, h_quality, names_debug, sample_num)
                print("  ══════════════════════════════════════════════════════════")
                print()

            # ---- Step 4: Run NUM_SIMULATIONS, track top half, bottom half, and both ----
            # NRFI requires BOTH halves of the first inning to be scoreless.
            top_nrfi   = 0   # simulations where away team scored 0 runs
            bot_nrfi   = 0   # simulations where home team scored 0 runs
            both_nrfi  = 0   # simulations where NEITHER team scored (true NRFI)

            for _ in range(NUM_SIMULATIONS):
                top_scoreless = simulate_inning(away_stats, h_quality)
                bot_scoreless = simulate_inning(home_stats, a_quality)
                if top_scoreless:
                    top_nrfi += 1
                if bot_scoreless:
                    bot_nrfi += 1
                if top_scoreless and bot_scoreless:
                    both_nrfi += 1

            top_prob  = top_nrfi  / NUM_SIMULATIONS
            bot_prob  = bot_nrfi  / NUM_SIMULATIONS
            nrfi_prob = both_nrfi / NUM_SIMULATIONS

            # ---- Step 5: Derive prediction and confidence ----
            pick = prediction_label(nrfi_prob)
            conf = confidence_label(nrfi_prob)

            # ---- Step 6: Save to the predictions table ----
            record = {
                "game_date":           today_str,
                "home_team":           home_team,
                "away_team":           away_team,
                "home_pitcher":        home_pitcher,
                "away_pitcher":        away_pitcher,
                "nrfi_probability":    round(nrfi_prob, 4),
                "prediction":          pick,
                "confidence":          conf,
                "is_pick_of_the_day":  False,
            }

            try:
                insert_row("predictions", record, conn)
            except Exception as error:
                print(f"  Warning: could not save prediction to DB — {error}")

            results.append(record)

            # ---- Step 7: Print the per-game result ----
            print(
                f"  Top-half scoreless : {top_nrfi:>4}/{NUM_SIMULATIONS}  ({top_prob * 100:.1f}%)"
            )
            print(
                f"  Bot-half scoreless : {bot_nrfi:>4}/{NUM_SIMULATIONS}  ({bot_prob * 100:.1f}%)"
            )
            print(
                f"  Combined NRFI      : {both_nrfi:>4}/{NUM_SIMULATIONS}  "
                f"→ {nrfi_prob * 100:.1f}% — {conf.upper()} confidence"
            )
            print()

        # ---- Pick of the day ----
        # Print the average quality_factor so we can immediately spot whether
        # high NRFI rates are data-driven (strong pitchers) or a calibration bug.
        if quality_factors:
            avg_q = sum(quality_factors) / len(quality_factors)
            below = sum(1 for q in quality_factors if q < 1.0)
            print(
                f"Pitcher quality_factor  avg={avg_q:.3f}  "
                f"({below}/{len(quality_factors)} appearances below 1.0 = above-avg pitchers)\n"
                f"  (1.00 = league avg  |  <1.00 = pitcher suppresses offence  |  >1.00 = hitter-friendly)\n"
            )
        # Select the 1–2 most confident games (furthest from 50%) and mark them
        # in both the in-memory results list and the predictions table.
        picks = get_pick_of_the_day(results)
        pick_keys = {(p["game_date"], p["home_team"], p["away_team"]) for p in picks}

        for pick in picks:
            # Flip the flag in-memory so the summary print can show the star marker
            pick["is_pick_of_the_day"] = True
            try:
                execute_query(
                    """
                    UPDATE predictions
                    SET    is_pick_of_the_day = TRUE
                    WHERE  game_date = %s
                      AND  home_team = %s
                      AND  away_team = %s
                    """,
                    (pick["game_date"], pick["home_team"], pick["away_team"]),
                    conn,
                )
            except Exception as error:
                print(f"  Warning: could not mark pick of the day in DB — {error}")

        # ---- Summary ----
        print("=" * 60)
        print(f"Done. Simulated {len(results)} game(s) for {today_str}.\n")
        for r in results:
            star = " ★" if r.get("is_pick_of_the_day") else "  "
            print(
                f"{star} {r['away_team']} @ {r['home_team']:20s}  "
                f"{r['nrfi_probability'] * 100:.1f}%  {r['prediction']:4s}  {r['confidence']}"
            )

        if picks:
            print()
            print("  PICK(S) OF THE DAY:")
            for p in picks:
                dist = abs(p["nrfi_probability"] - 0.5) * 100
                print(
                    f"    ★  {p['away_team']} @ {p['home_team']}"
                    f"  →  {p['prediction']}  "
                    f"({p['nrfi_probability'] * 100:.1f}%,  {dist:.1f}pp from 50%)"
                )

    finally:
        # Close the single shared connection — runs even if an exception occurred.
        conn.close()
        print("\nDatabase connection closed.")


if __name__ == "__main__":
    main()
