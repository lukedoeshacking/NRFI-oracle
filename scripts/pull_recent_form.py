"""
Pull recent form stats for today's pitchers and batters and store
blended stats (65% recent + 35% season average) in the database.

How to run (after pull_daily.py and pull_lineups.py have been run):
    python scripts/pull_recent_form.py

What it does:
    1. Loads today's pitcher names from model/today_games.json
    2. Loads today's batter names from the lineups table
    3. For each pitcher: fetches their per-start game log for this season,
       takes the last 5 starts, and computes recent ERA / WHIP / K9 / BB9
    4. For each batter: fetches their per-game hitting log, keeps games
       from the last 30 calendar days, and computes recent OBP / SLG / K-rate
    5. Blends each recent stat with the full-season average:
           blended = 65% recent  +  35% season
    6. Stores the blended stats in pitcher_recent_form and batter_recent_form
    7. Prints a side-by-side comparison (season / recent / blended) for
       a sample of pitchers and batters so you can see the difference
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

# ----------------------------------------------------------
# Paths, .env, and importing db.py from /backend
# ----------------------------------------------------------
PROJECT_ROOT     = Path(__file__).resolve().parent.parent
BACKEND_DIR      = PROJECT_ROOT / "backend"
MODEL_DIR        = PROJECT_ROOT / "model"
TODAY_GAMES_PATH = MODEL_DIR / "today_games.json"

# Guard prevents duplicate entries when this module is imported more than once.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

if not os.getenv("DATABASE_URL"):
    raise SystemExit(
        "DATABASE_URL is missing. Add it to the .env file in the project root."
    )

from db import execute_query, insert_row  # noqa: E402

# ----------------------------------------------------------
# Constants
# ----------------------------------------------------------
MLB_STATS_BASE = "https://statsapi.mlb.com/api/v1"
HTTP_HEADERS   = {"User-Agent": "nrfi-oracle-recent-form"}

N_RECENT_STARTS    = 5    # how many of a pitcher's most recent starts to use
RECENT_DAYS        = 30   # how many calendar days back to look for batter stats
MIN_PITCHER_STARTS = 2    # minimum starts needed before we trust recent pitcher data
MIN_BATTER_PA      = 10   # minimum plate appearances needed for recent batter data

# ----------------------------------------------------------
# BLENDING WEIGHTS
# ----------------------------------------------------------
# We combine recent performance with the full-season average.
# This gives more weight to recent trends while keeping the season
# average as a stabilising anchor against small-sample noise.
#
# Why 65 / 35?
#   - 5 starts ≈ 25-35 innings. That's meaningful but not huge.
#   - If a pitcher just returned from injury and had one great start,
#     the blend prevents us from over-reacting.
#   - You can tune RECENT_WEIGHT up (more reactive) or down (more stable).
#
# The two weights must always add up to exactly 1.0.
RECENT_WEIGHT = 0.65   # fraction given to last-5-starts / last-30-days stats
SEASON_WEIGHT = 0.35   # fraction given to full-season average

# League-average fallbacks (same values as the rest of the codebase)
LEAGUE_AVG = {
    "era":            4.30,
    "whip":           1.30,
    "k_per_9":        8.50,
    "bb_per_9":       3.20,
    "first_inning_era": 4.73,
    "obp":            0.320,
    "slg":            0.410,
    "k_rate":         0.230,
}

REQUEST_DELAY = 0.05  # polite pause (seconds) between MLB Stats API calls
                      # Reduced from 0.2 s — parallel workers spread the load,
                      # so each worker only fires once every ~0.4 s at 8 workers.


# ----------------------------------------------------------
# Helpers
# ----------------------------------------------------------
def to_float(value, fallback=None):
    """Convert a value to float, returning fallback if missing or invalid."""
    if value is None:
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def fetch_json(url, params=None, retries=3):
    """GET a URL and return parsed JSON. Retries on transient failures."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, headers=HTTP_HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as err:
            last_err = err
            print(f"    Request failed (attempt {attempt}/{retries}): {err}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not fetch {url}: {last_err}")


def parse_ip(ip_str):
    """
    Convert an MLB innings-pitched string to a decimal number.

    The MLB API uses ".1" to mean one-third of an inning and ".2" to mean
    two-thirds — NOT tenths.  Examples:
        "6.0"  ->  6.000    (six full innings)
        "6.1"  ->  6.333    (six and one-third innings)
        "6.2"  ->  6.667    (six and two-thirds innings)
    """
    if ip_str is None:
        return 0.0
    try:
        parts  = str(ip_str).split(".")
        full   = int(parts[0])
        thirds = int(parts[1]) if len(parts) > 1 else 0
        return full + thirds / 3.0
    except (ValueError, IndexError):
        return 0.0


def blend(recent_val, season_val):
    """
    Combine a recent-form stat and a season-average stat into one blended value.

    Formula:
        blended = recent_val * RECENT_WEIGHT + season_val * SEASON_WEIGHT
                = recent_val * 0.65          + season_val * 0.35

    Example — pitcher who has been pitching well lately:
        recent ERA = 2.80,  season ERA = 4.30
        blended    = 2.80 * 0.65 + 4.30 * 0.35
                   = 1.82  + 1.505
                   = 3.325   (much better than season, but not blindly trusting 5 starts)

    Example — pitcher who has been struggling:
        recent ERA = 6.00,  season ERA = 3.80
        blended    = 6.00 * 0.65 + 3.80 * 0.35
                   = 3.90  + 1.33
                   = 5.23   (worse than season, but not as extreme as recent alone)

    Special cases:
      - If recent_val is None (not enough data): return season_val unchanged.
      - If season_val is None (new pitcher, no DB record): return recent_val.
      - If both are None: return None.
    """
    if recent_val is None and season_val is None:
        return None
    if recent_val is None:
        return season_val   # fall back: no recent data, use season at 100%
    if season_val is None:
        return recent_val   # fall back: no season data, use recent at 100%
    return round(recent_val * RECENT_WEIGHT + season_val * SEASON_WEIGHT, 4)


# ----------------------------------------------------------
# Database — create tables if missing
# ----------------------------------------------------------
def ensure_tables():
    """
    Create pitcher_recent_form and batter_recent_form tables if they don't
    already exist.  Running this multiple times is safe (IF NOT EXISTS).
    """
    execute_query("""
        CREATE TABLE IF NOT EXISTS pitcher_recent_form (
            id               SERIAL PRIMARY KEY,
            name             TEXT NOT NULL,
            player_id        INTEGER,
            as_of_date       DATE NOT NULL,
            recent_starts    INTEGER,
            recent_era       NUMERIC(6,3),
            recent_whip      NUMERIC(5,3),
            recent_k_per_9   NUMERIC(5,2),
            recent_bb_per_9  NUMERIC(5,2),
            recent_fi_era    NUMERIC(6,3),
            blended_era      NUMERIC(6,3),
            blended_whip     NUMERIC(5,3),
            blended_k_per_9  NUMERIC(5,2),
            blended_bb_per_9 NUMERIC(5,2),
            blended_fi_era   NUMERIC(6,3)
        )
    """)
    execute_query("""
        CREATE TABLE IF NOT EXISTS batter_recent_form (
            id              SERIAL PRIMARY KEY,
            name            TEXT NOT NULL,
            player_id       INTEGER,
            as_of_date      DATE NOT NULL,
            recent_pa       INTEGER,
            recent_obp      NUMERIC(5,3),
            recent_slg      NUMERIC(5,3),
            recent_k_rate   NUMERIC(5,3),
            blended_obp     NUMERIC(5,3),
            blended_slg     NUMERIC(5,3),
            blended_k_rate  NUMERIC(5,3)
        )
    """)
    print("Tables ready: pitcher_recent_form, batter_recent_form.")


# ===========================================================
# PITCHER PIPELINE
# ===========================================================

def build_pitcher_id_map(season):
    """
    Fetch all pitchers for the given season from the MLB Stats API and
    return a dict mapping full_name -> player_id.

    We need this because the pitchers table only stores names, not IDs,
    and the game-log endpoint requires a numeric player ID.
    """
    print(f"  Building pitcher name->ID map for {season}...")
    try:
        payload = fetch_json(
            f"{MLB_STATS_BASE}/stats",
            params={
                "stats":      "season",
                "playerPool": "All",
                "position":   1,        # 1 = pitcher
                "sportId":    1,
                "season":     season,
                "group":      "pitching",
                "limit":      1000,
            },
        )
    except RuntimeError as err:
        print(f"  Could not build pitcher ID map: {err}")
        return {}

    id_map = {}
    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            player = split.get("player") or {}
            name   = player.get("fullName")
            pid    = player.get("id")
            if name and pid:
                id_map[name] = pid

    print(f"  Mapped {len(id_map)} pitcher names to IDs.")
    return id_map


def fetch_pitcher_game_log(player_id, season):
    """
    Return a list of per-game pitching stat dicts for the given pitcher
    this season, filtered to starts only (gamesStarted >= 1).

    Each dict contains the raw stat fields from the API plus a '_date' key.
    The list is sorted oldest-first (the API's default order).
    """
    try:
        payload = fetch_json(
            f"{MLB_STATS_BASE}/people/{player_id}/stats",
            params={"stats": "gameLog", "season": season, "group": "pitching"},
        )
    except RuntimeError:
        return []

    starts = []
    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            if int(stat.get("gamesStarted", 0) or 0) >= 1:
                stat["_date"] = split.get("date") or ""
                starts.append(stat)

    return starts   # chronological: oldest first, newest last


def calc_pitcher_recent(starts, n=N_RECENT_STARTS):
    """
    Compute ERA, WHIP, K/9, BB/9 from the last N starts.

    Returns a dict of recent stats, or None if there aren't enough starts.
    We require at least MIN_PITCHER_STARTS to avoid single-game noise.
    """
    last_n = starts[-n:]    # slice from the end = most recent N starts

    if len(last_n) < MIN_PITCHER_STARTS:
        return None   # caller will fall back to season-only stats

    total_ip   = sum(parse_ip(s.get("inningsPitched")) for s in last_n)
    if total_ip < 0.1:
        return None   # defensive: all starts somehow have 0 IP

    total_er   = sum(int(s.get("earnedRuns",  0) or 0) for s in last_n)
    total_hits = sum(int(s.get("hits",        0) or 0) for s in last_n)
    total_bb   = sum(int(s.get("baseOnBalls", 0) or 0) for s in last_n)
    total_k    = sum(int(s.get("strikeOuts",  0) or 0) for s in last_n)

    era    = round((total_er   / total_ip) * 9, 3)
    whip   = round((total_hits + total_bb) / total_ip, 3)
    k_per9 = round((total_k    / total_ip) * 9, 2)
    bb_per9= round((total_bb   / total_ip) * 9, 2)
    # First-inning ERA ≈ 110% of full ERA (same estimate used throughout the codebase)
    fi_era = round(era * 1.1, 3)

    return {
        "starts":  len(last_n),
        "era":     era,
        "whip":    whip,
        "k_per_9": k_per9,
        "bb_per_9":bb_per9,
        "fi_era":  fi_era,
    }


def get_season_pitcher_stats(name):
    """Look up a pitcher's full-season stats row from the pitchers table."""
    try:
        rows = execute_query(
            """
            SELECT era, whip, k_per_9, bb_per_9, first_inning_era
            FROM   pitchers
            WHERE  name = %s
            ORDER  BY season DESC
            LIMIT  1
            """,
            (name,),
        )
        return rows[0] if rows else None
    except Exception:
        return None


def process_pitcher(name, player_id, season, today_str):
    """
    Run the full recent-form pipeline for one pitcher:
      1. Fetch this season's game log, extract the last N starts
      2. Compute recent ERA / WHIP / K9 / BB9 from those starts
      3. Load the full-season averages from the pitchers table
      4. Blend: blended = 65% recent + 35% season
      5. Insert one row into pitcher_recent_form

    Returns a comparison dict used by the summary printer, or None on error.
    """
    # Step 1 & 2 — game log + recent stats
    starts = fetch_pitcher_game_log(player_id, season)
    time.sleep(REQUEST_DELAY)
    recent = calc_pitcher_recent(starts)

    # Step 3 — season stats (fall back to league average per stat if missing)
    season_row = get_season_pitcher_stats(name)
    s_era    = to_float(season_row.get("era")              if season_row else None) or LEAGUE_AVG["era"]
    s_whip   = to_float(season_row.get("whip")             if season_row else None) or LEAGUE_AVG["whip"]
    s_k9     = to_float(season_row.get("k_per_9")          if season_row else None) or LEAGUE_AVG["k_per_9"]
    s_bb9    = to_float(season_row.get("bb_per_9")         if season_row else None) or LEAGUE_AVG["bb_per_9"]
    s_fi_era = to_float(season_row.get("first_inning_era") if season_row else None) or LEAGUE_AVG["first_inning_era"]

    # Step 4 — blend
    if recent:
        b_era    = blend(recent["era"],     s_era)
        b_whip   = blend(recent["whip"],    s_whip)
        b_k9     = blend(recent["k_per_9"], s_k9)
        b_bb9    = blend(recent["bb_per_9"],s_bb9)
        b_fi_era = blend(recent["fi_era"],  s_fi_era)
        tag = f"last {recent['starts']} start(s)"
    else:
        # Not enough recent starts — blended equals season unchanged
        b_era, b_whip, b_k9, b_bb9, b_fi_era = s_era, s_whip, s_k9, s_bb9, s_fi_era
        tag = f"{len(starts)} start(s) found — using season only"

    # Step 5 — persist
    try:
        insert_row("pitcher_recent_form", {
            "name":             name,
            "player_id":        player_id,
            "as_of_date":       today_str,
            "recent_starts":    recent["starts"]   if recent else len(starts),
            "recent_era":       recent["era"]       if recent else None,
            "recent_whip":      recent["whip"]      if recent else None,
            "recent_k_per_9":   recent["k_per_9"]   if recent else None,
            "recent_bb_per_9":  recent["bb_per_9"]  if recent else None,
            "recent_fi_era":    recent["fi_era"]    if recent else None,
            "blended_era":      b_era,
            "blended_whip":     b_whip,
            "blended_k_per_9":  b_k9,
            "blended_bb_per_9": b_bb9,
            "blended_fi_era":   b_fi_era,
        })
    except Exception as err:
        print(f"    DB insert failed for {name}: {err}")
        return None

    return {
        "name":    name,
        "tag":     tag,
        "season":  {"era": s_era,  "whip": s_whip,  "k_per_9": s_k9},
        "recent":  {"era": recent["era"], "whip": recent["whip"], "k_per_9": recent["k_per_9"]} if recent else None,
        "blended": {"era": b_era,  "whip": b_whip,  "k_per_9": b_k9},
    }


# ===========================================================
# BATTER PIPELINE
# ===========================================================

def get_batter_player_id(clean_name):
    """Look up a batter's player_id from the batters or lineups table."""
    for query, params in [
        ("SELECT player_id FROM batters WHERE name = %s ORDER BY season DESC LIMIT 1", (clean_name,)),
        ("SELECT player_id FROM lineups WHERE player_name = %s ORDER BY game_date DESC LIMIT 1", (clean_name,)),
    ]:
        try:
            rows = execute_query(query, params)
            if rows and rows[0].get("player_id"):
                return int(rows[0]["player_id"])
        except Exception:
            pass
    return None


def fetch_batter_game_log(player_id, season):
    """
    Return a list of per-game hitting stat dicts for the given batter this season.
    Each dict includes the MLB stat fields plus a '_date' key for date filtering.
    """
    try:
        payload = fetch_json(
            f"{MLB_STATS_BASE}/people/{player_id}/stats",
            params={"stats": "gameLog", "season": season, "group": "hitting"},
        )
    except RuntimeError:
        return []

    games = []
    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            stat["_date"] = split.get("date") or ""
            games.append(stat)

    return games   # chronological: oldest first


def calc_batter_recent(games, days=RECENT_DAYS):
    """
    Compute OBP, SLG, and K-rate from games played in the last `days` days.

    Returns a dict of recent stats, or None if there aren't enough plate
    appearances (< MIN_BATTER_PA) to produce a reliable estimate.

    The cutoff date is today minus `days` calendar days.
    """
    cutoff = (date.today() - timedelta(days=days)).isoformat()
    recent_games = [g for g in games if g.get("_date", "") >= cutoff]

    total_pa = sum(int(g.get("plateAppearances", 0) or 0) for g in recent_games)
    if total_pa < MIN_BATTER_PA:
        return None   # caller falls back to season stats

    total_ab = sum(int(g.get("atBats",        0) or 0) for g in recent_games)
    total_h  = sum(int(g.get("hits",          0) or 0) for g in recent_games)
    total_bb = sum(int(g.get("baseOnBalls",   0) or 0) for g in recent_games)
    total_k  = sum(int(g.get("strikeOuts",    0) or 0) for g in recent_games)
    total_tb = sum(int(g.get("totalBases",    0) or 0) for g in recent_games)

    obp    = round((total_h + total_bb) / total_pa, 3) if total_pa > 0 else None
    slg    = round(total_tb / total_ab,              3) if total_ab > 0 else None
    k_rate = round(total_k  / total_pa,              3) if total_pa > 0 else None

    return {
        "pa":     total_pa,
        "games":  len(recent_games),
        "obp":    obp,
        "slg":    slg,
        "k_rate": k_rate,
    }


def get_season_batter_stats(clean_name):
    """Look up a batter's full-season stats row from the batters table."""
    try:
        rows = execute_query(
            """
            SELECT obp, slg, k_rate
            FROM   batters
            WHERE  name = %s
            ORDER  BY season DESC
            LIMIT  1
            """,
            (clean_name,),
        )
        return rows[0] if rows else None
    except Exception:
        return None


def process_batter(player_name, season, today_str):
    """
    Run the full recent-form pipeline for one batter:
      1. Resolve player_id from DB
      2. Fetch this season's game log, filter to last 30 days
      3. Compute recent OBP / SLG / K-rate from that window
      4. Load the full-season averages from the batters table
      5. Blend: blended = 65% recent + 35% season
      6. Insert one row into batter_recent_form

    Returns a comparison dict used by the summary printer, or None on error.
    """
    clean = player_name.replace("[PROJ] ", "").strip()

    # Step 1 — player ID
    player_id = get_batter_player_id(clean)
    if not player_id:
        return None   # no ID; skip silently (will fall back to season stats in simulate.py)

    # Step 2 & 3 — game log + recent stats
    games  = fetch_batter_game_log(player_id, season)
    time.sleep(REQUEST_DELAY)
    recent = calc_batter_recent(games)

    # Step 4 — season stats
    season_row = get_season_batter_stats(clean)
    s_obp    = to_float(season_row.get("obp")    if season_row else None) or LEAGUE_AVG["obp"]
    s_slg    = to_float(season_row.get("slg")    if season_row else None) or LEAGUE_AVG["slg"]
    s_k_rate = to_float(season_row.get("k_rate") if season_row else None) or LEAGUE_AVG["k_rate"]

    # Step 5 — blend
    if recent:
        b_obp    = blend(recent["obp"],    s_obp)
        b_slg    = blend(recent["slg"],    s_slg)
        b_k_rate = blend(recent["k_rate"], s_k_rate)
        tag = f"{recent['pa']} PA over last {recent['games']} game(s)"
    else:
        b_obp, b_slg, b_k_rate = s_obp, s_slg, s_k_rate
        cutoff   = (date.today() - timedelta(days=RECENT_DAYS)).isoformat()
        recent_pa = sum(int(g.get("plateAppearances", 0) or 0)
                        for g in games if g.get("_date", "") >= cutoff)
        tag = f"{recent_pa} PA in window — using season only"

    # Step 6 — persist
    try:
        insert_row("batter_recent_form", {
            "name":           clean,
            "player_id":      player_id,
            "as_of_date":     today_str,
            "recent_pa":      recent["pa"]     if recent else None,
            "recent_obp":     recent["obp"]    if recent else None,
            "recent_slg":     recent["slg"]    if recent else None,
            "recent_k_rate":  recent["k_rate"] if recent else None,
            "blended_obp":    b_obp,
            "blended_slg":    b_slg,
            "blended_k_rate": b_k_rate,
        })
    except Exception as err:
        print(f"    DB insert failed for {clean}: {err}")
        return None

    return {
        "name":    clean,
        "tag":     tag,
        "season":  {"obp": s_obp,  "slg": s_slg,  "k_rate": s_k_rate},
        "recent":  {"obp": recent["obp"], "slg": recent["slg"], "k_rate": recent["k_rate"]} if recent else None,
        "blended": {"obp": b_obp,  "slg": b_slg,  "k_rate": b_k_rate},
    }


# ===========================================================
# COMPARISON PRINTER
# ===========================================================

def _fmt(val):
    """Format a stat value as a 3-decimal string, or 'N/A' if missing."""
    return f"{val:.3f}" if val is not None else "  N/A "


def print_pitcher_comparison(comparisons, n=3):
    """Print a side-by-side season / recent / blended table for up to N pitchers."""
    samples = [c for c in comparisons if c is not None][:n]
    if not samples:
        return

    print()
    print("=" * 62)
    print("PITCHER COMPARISON  (season  /  recent  /  blended)")
    print(f"Blending: {int(SEASON_WEIGHT*100)}% season  +  {int(RECENT_WEIGHT*100)}% recent")
    print("=" * 62)

    for c in samples:
        r = c["recent"]
        print(f"\n  {c['name']}  [{c['tag']}]")
        print(f"  {'Stat':<10} {'Season':>9}  {'Recent':>9}  {'Blended':>9}")
        print(f"  {'-'*40}")
        for label, key in [("ERA", "era"), ("WHIP", "whip"), ("K/9", "k_per_9")]:
            r_val = _fmt(r[key]) if r else "  N/A "
            print(f"  {label:<10} {_fmt(c['season'][key]):>9}  {r_val:>9}  {_fmt(c['blended'][key]):>9}")


def print_batter_comparison(comparisons, n=3):
    """Print a side-by-side season / recent / blended table for up to N batters."""
    samples = [c for c in comparisons if c is not None][:n]
    if not samples:
        return

    print()
    print("=" * 62)
    print("BATTER COMPARISON   (season  /  recent  /  blended)")
    print(f"Blending: {int(SEASON_WEIGHT*100)}% season  +  {int(RECENT_WEIGHT*100)}% recent")
    print("=" * 62)

    for c in samples:
        r = c["recent"]
        print(f"\n  {c['name']}  [{c['tag']}]")
        print(f"  {'Stat':<10} {'Season':>9}  {'Recent':>9}  {'Blended':>9}")
        print(f"  {'-'*40}")
        for label, key in [("OBP", "obp"), ("SLG", "slg"), ("K rate", "k_rate")]:
            r_val = _fmt(r[key]) if r else "  N/A "
            print(f"  {label:<10} {_fmt(c['season'][key]):>9}  {r_val:>9}  {_fmt(c['blended'][key]):>9}")


# ===========================================================
# MAIN
# ===========================================================
def main():
    today     = date.today()
    today_str = today.isoformat()
    season    = today.year

    print(f"NRFI Oracle recent form pull — {today_str}")
    print(f"Pitcher window : last {N_RECENT_STARTS} starts  (min {MIN_PITCHER_STARTS} to blend)")
    print(f"Batter  window : last {RECENT_DAYS} days        (min {MIN_BATTER_PA} PA to blend)")
    print(f"Blend weights  : {int(RECENT_WEIGHT*100)}% recent  +  {int(SEASON_WEIGHT*100)}% season\n")

    # Create tables if they don't exist yet
    ensure_tables()

    # Delete today's existing rows so re-running this script is safe
    execute_query("DELETE FROM pitcher_recent_form WHERE as_of_date = %s", (today_str,))
    execute_query("DELETE FROM batter_recent_form   WHERE as_of_date = %s", (today_str,))
    print()

    # ── Load today's pitcher names from today_games.json ──────────────────────
    if not TODAY_GAMES_PATH.exists():
        raise SystemExit(
            f"Could not find {TODAY_GAMES_PATH}.\n"
            "Run scripts/pull_daily.py first."
        )
    games = json.loads(TODAY_GAMES_PATH.read_text(encoding="utf-8"))

    pitcher_names = set()
    for g in games:
        for side in ("home_pitcher", "away_pitcher"):
            name = g.get(side)
            if name and name != "TBD":
                pitcher_names.add(name)

    # ── Load today's batter names from the lineups table ─────────────────────
    lineup_rows  = execute_query(
        "SELECT DISTINCT player_name FROM lineups WHERE game_date = %s",
        (today_str,),
    )
    batter_names = [r["player_name"] for r in lineup_rows]

    print(f"Today: {len(pitcher_names)} unique pitcher(s), {len(batter_names)} batter(s).\n")

    # ── PITCHERS ──────────────────────────────────────────────────────────────
    print(f"--- Processing {len(pitcher_names)} pitcher(s) in parallel ---")
    pitcher_id_map    = build_pitcher_id_map(season)
    pitcher_comps     = []
    pitchers_saved    = 0
    pitchers_skipped  = 0

    # Build the list of (name, player_id) pairs we can actually process.
    # Names with no ID (not found in the season stats map) are skipped now
    # so the thread pool only receives actionable work items.
    pitchers_to_process = []
    for name in sorted(pitcher_names):
        pid = pitcher_id_map.get(name)
        if pid:
            pitchers_to_process.append((name, pid))
        else:
            print(f"  {name}: no player ID found in {season} stats — skipping")
            pitchers_skipped += 1

    # Fire all game-log requests concurrently.
    # MAX_WORKERS caps how many outbound HTTP connections are open at once.
    MAX_WORKERS = 8
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_name = {
            pool.submit(process_pitcher, name, pid, season, today_str): name
            for name, pid in pitchers_to_process
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                comp = future.result()
            except Exception as err:
                print(f"  {name}: error — {err}")
                pitchers_skipped += 1
                continue

            if comp:
                pitcher_comps.append(comp)
                pitchers_saved += 1
                b = comp["blended"]
                print(f"  {name}: blended ERA={b['era']:.2f}  WHIP={b['whip']:.2f}"
                      f"  K/9={b['k_per_9']:.1f}  [{comp['tag']}]")
            else:
                pitchers_skipped += 1

    print(f"\nPitchers: {pitchers_saved} saved, {pitchers_skipped} skipped.\n")

    # ── BATTERS ───────────────────────────────────────────────────────────────
    print(f"--- Processing {len(batter_names)} batter(s) in parallel ---")
    batter_comps    = []
    batters_saved   = 0
    batters_skipped = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_name = {
            pool.submit(process_batter, player_name, season, today_str): player_name
            for player_name in batter_names
        }
        for future in as_completed(future_to_name):
            player_name = future_to_name[future]
            clean = player_name.replace("[PROJ] ", "").strip()
            try:
                comp = future.result()
            except Exception as err:
                print(f"  {clean}: error — {err}")
                batters_skipped += 1
                continue

            if comp:
                batter_comps.append(comp)
                batters_saved += 1
                if batters_saved <= 5:
                    b = comp["blended"]
                    print(f"  {clean}: OBP={b['obp']:.3f} SLG={b['slg']:.3f}  [{comp['tag']}]")
            else:
                batters_skipped += 1

    if batters_saved > 5:
        print(f"  ... ({batters_saved - 5} more batters processed)")
    print(f"\nBatters: {batters_saved} saved, {batters_skipped} skipped.")

    # ── COMPARISON TABLES ─────────────────────────────────────────────────────
    print_pitcher_comparison(pitcher_comps)
    print_batter_comparison(batter_comps)

    print()
    print("=" * 62)
    print("Done.  Run scripts/simulate.py to use these blended stats.")


if __name__ == "__main__":
    main()
