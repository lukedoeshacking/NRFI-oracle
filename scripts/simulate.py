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

# Add /backend to Python's search path so we can import db.py
sys.path.insert(0, str(BACKEND_DIR))

# Load DATABASE_URL from the .env file at the project root
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
def get_lineup(game_date_str, team_name):
    """
    Fetch the top 5 batters for a team on a given date from the lineups table.

    The lineups table is populated by pull_lineups.py every morning.
    Returns a list of player name strings in batting order (1–5).
    Returns an empty list when no lineup exists for that team today.
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
        )
        return [row["player_name"] for row in rows]
    except Exception as error:
        print(f"    DB error fetching lineup for {team_name}: {error}")
        return []


def get_batter_stats(player_name):
    """
    Return {"obp": float, "slg": float} for a batter, checking three sources
    in priority order:

      1. batter_recent_form table  — blended stats (65% last-30-days + 35% season)
         populated by pull_recent_form.py each morning
      2. batters table             — full-season average
      3. League-average fallback

    Falls back to the next source automatically if a table is missing or the
    batter is not found.
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


def get_pitcher_stats(pitcher_name, game):
    """
    Return (whip, k_per_9) for a pitcher, checking four sources in order:

      1. pitcher_recent_form table — blended stats (65% last-5-starts + 35% season)
         populated by pull_recent_form.py each morning
      2. pitchers table            — full-season average
      3. today_games.json values   — stats embedded by pull_daily.py
      4. League-average fallback

    Falls back to the next source automatically at each step.
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
            )
            if rows:
                whip    = rows[0].get("whip")
                k_per_9 = rows[0].get("k_per_9")
                if whip is not None and k_per_9 is not None:
                    return float(whip), float(k_per_9)
        except Exception as error:
            print(f"    DB error fetching stats for '{pitcher_name}': {error}")

    # --- 3. Stats embedded in today_games.json by pull_daily.py ---
    whip    = game.get("home_pitcher_whip")    or LEAGUE_AVG_WHIP
    k_per_9 = game.get("home_pitcher_k_per_9") or LEAGUE_AVG_K9
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

    Base split (league average):
        10% walks  — advance runners only if forced
        70% singles — all runners advance 1 base; runner on 2nd or 3rd scores
        20% doubles — all runners score; batter ends on 2nd

    The double fraction is scaled by the batter's power relative to the
    league average (SLG/OBP ratio).  A slugger gets more doubles; a
    slap hitter gets more singles.

    Example — power hitter (SLG=0.520, OBP=0.380):
        power_ratio = (0.520/0.380) / 1.28 = 1.07
        double_rate ≈ 21%  (slightly above average)

    Example — contact hitter (SLG=0.340, OBP=0.350):
        power_ratio = (0.340/0.350) / 1.28 = 0.76
        double_rate ≈ 15%  (below average)
    """
    walk_rate = 0.10   # fixed fraction — roughly matches MLB walk/OBP ratio

    # Scale the double rate by the batter's power (SLG relative to OBP).
    # Cap between 5% and 40% so extreme inputs stay sensible.
    power_ratio = (slg / max(obp, 0.001)) / LEAGUE_SLG_PER_OBP
    double_rate = min(max(0.20 * power_ratio, 0.05), 0.40)

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
    Runners on 2nd or 3rd score; runner on 1st moves to 2nd; batter takes 1st.

    bases : [bool, bool, bool]  — [1st, 2nd, 3rd]
    Returns (new_bases, run_scored).
    """
    run_scored = bases[1] or bases[2]   # 2nd or 3rd runner scores
    # After the single: old-1st → 2nd, batter → 1st, 2nd+3rd vacated (scored)
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
# Main
# ----------------------------------------------------------
def main():
    today = date.today()
    today_str = today.isoformat()

    print(f"NRFI Oracle Monte Carlo simulation — {today_str}")
    print(f"Running {NUM_SIMULATIONS:,} simulations per game...\n")

    # ---- Step 1: Load today's game slate ----
    if not TODAY_GAMES_PATH.exists():
        raise SystemExit(
            f"Could not find {TODAY_GAMES_PATH}.\n"
            "Run  python scripts/pull_daily.py  first to create today_games.json."
        )

    with TODAY_GAMES_PATH.open(encoding="utf-8") as fh:
        games = json.load(fh)

    if not games:
        print("No games in today_games.json. Nothing to simulate.")
        return

    print(f"Found {len(games)} game(s). Starting simulations...\n")

    # Clear any existing predictions for today so re-running this script
    # replaces the old rows rather than duplicating them.
    execute_query(
        "DELETE FROM predictions WHERE game_date = %s",
        (today_str,),
    )

    results = []

    for game_idx, game in enumerate(games):
        away_team    = game.get("away_team")   or "Unknown"
        home_team    = game.get("home_team")   or "Unknown"
        home_pitcher = game.get("home_pitcher") or "TBD"
        away_pitcher = game.get("away_pitcher") or "TBD"

        print(f"{away_team} @ {home_team}  |  Home pitcher: {home_pitcher}")

        # ---- Step 2: Get the away team's lineup ----
        # The away team bats in the top of the 1st inning, so we need their lineup
        # facing the home pitcher.
        lineup_names = get_lineup(today_str, away_team)

        if not lineup_names:
            print(f"  No lineup found for {away_team} — using 5 league-avg batters")
            batters_stats = [{"obp": LEAGUE_AVG_OBP, "slg": LEAGUE_AVG_SLG}] * 5
        else:
            preview = ", ".join(lineup_names[:3])
            print(f"  Lineup: {preview}{'...' if len(lineup_names) > 3 else ''}")

            # Look up each batter's OBP and SLG from the batters table
            batters_stats = [get_batter_stats(name) for name in lineup_names]

            # Pad to 5 slots with league averages if fewer batters were stored
            while len(batters_stats) < 5:
                batters_stats.append({"obp": LEAGUE_AVG_OBP, "slg": LEAGUE_AVG_SLG})

        # ---- Step 3: Get the home pitcher's stats ----
        whip, k_per_9 = get_pitcher_stats(home_pitcher, game)
        quality       = calc_quality_factor(whip, k_per_9)
        # Show the full quality-factor breakdown for every game so we can
        # immediately spot if all pitchers are getting the same fallback stats.
        whip_comp = whip / LEAGUE_AVG_WHIP
        k_comp    = LEAGUE_AVG_K9 / max(k_per_9, 0.1)
        print(
            f"  Pitcher stats → "
            f"WHIP={whip:.2f} (component={whip_comp:.3f})  "
            f"K/9={k_per_9:.1f} (component={k_comp:.3f})  "
            f"quality_factor={quality:.3f}"
        )

        # ---- DEBUG: Print 3 sample simulations for the first game only ----
        # This shows exactly what the math is doing so we can spot problems.
        # It runs BEFORE the real 1,000-simulation loop and does not affect results.
        if game_idx == 0:
            print()
            print("  ══════════════════════════════════════════════════════════")
            print(f"  DEBUG — 3 sample simulations for: {away_team} @ {home_team}")
            print()

            # Show how the quality factor was calculated so we can verify it
            whip_component = whip / LEAGUE_AVG_WHIP
            k_component    = LEAGUE_AVG_K9 / max(k_per_9, 0.1)
            print(f"  Quality factor breakdown:")
            print(f"    WHIP component  = pitcher_whip / league_avg  = {whip:.2f} / {LEAGUE_AVG_WHIP} = {whip_component:.3f}")
            print(f"    K/9  component  = league_avg / pitcher_k_per_9 = {LEAGUE_AVG_K9} / {k_per_9:.1f} = {k_component:.3f}")
            print(f"    Average of both = ({whip_component:.3f} + {k_component:.3f}) / 2 = {quality:.3f}")
            print()

            # Show the full batter stats table so we know what values are feeding in
            print(f"  Batter OBP/SLG feeding into the simulation:")
            names_for_debug = lineup_names if lineup_names else [f"Batter {i+1}" for i in range(5)]
            for i, (name, stats) in enumerate(zip(names_for_debug, batters_stats), start=1):
                eff = min(stats["obp"] * quality, 0.95)
                print(f"    #{i}  {name.replace('[PROJ] ', ''):<22s}  OBP={stats['obp']:.3f}  SLG={stats['slg']:.3f}  eff_OBP={eff:.3f}")
            print()

            # Run 3 fully verbose sample simulations
            # NOTE: random.seed() is NOT called here so these samples use the same
            # random stream as the real simulations — no cherry-picking.
            for sample_num in range(1, 4):
                debug_simulate_inning(batters_stats, quality, names_for_debug, sample_num)

            print("  ══════════════════════════════════════════════════════════")
            print()

        # ---- Step 4 & 5: Run 1,000 simulations, count NRFI outcomes ----
        nrfi_count = sum(
            1 for _ in range(NUM_SIMULATIONS)
            if simulate_inning(batters_stats, quality)
        )
        nrfi_prob = nrfi_count / NUM_SIMULATIONS

        # ---- Step 5: Derive prediction and confidence ----
        pick = prediction_label(nrfi_prob)
        conf = confidence_label(nrfi_prob)

        # ---- Step 6: Save to the predictions table ----
        record = {
            "game_date":        today_str,
            "home_team":        home_team,
            "away_team":        away_team,
            "home_pitcher":     home_pitcher,
            "away_pitcher":     away_pitcher,
            "nrfi_probability": round(nrfi_prob, 4),
            "prediction":       pick,
            "confidence":       conf,
        }

        try:
            insert_row("predictions", record)
        except Exception as error:
            print(f"  Warning: could not save prediction to DB — {error}")

        results.append(record)

        # ---- Step 7: Print the per-game result ----
        print(
            f"  {away_team} @ {home_team}: "
            f"{nrfi_count}/{NUM_SIMULATIONS} NRFI "
            f"→ {nrfi_prob * 100:.1f}% — {conf.upper()} confidence"
        )
        print()

    # ---- Summary ----
    print("=" * 60)
    print(f"Done. Simulated {len(results)} game(s) for {today_str}.\n")
    for r in results:
        print(
            f"  {r['away_team']} @ {r['home_team']:20s}  "
            f"{r['nrfi_probability'] * 100:.1f}%  {r['prediction']:4s}  {r['confidence']}"
        )


if __name__ == "__main__":
    main()
