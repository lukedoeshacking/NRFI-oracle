"""
Train an XGBoost model that predicts NRFI (No Runs First Inning).

NRFI means neither team scored in the 1st inning (label = 1).
YRFI means at least one run was scored in the 1st inning (label = 0).

How to run (from any folder):
    python scripts/train_model.py

Expected database tables (filled by scripts/pull_historical.py):
    games    — one row per completed game, including nrfi
    pitchers — season stats for starting pitchers
    teams    — season batting / first-inning rates
"""

import os
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from dotenv import load_dotenv

# ----------------------------------------------------------
# Folders and environment
# ----------------------------------------------------------
# This file lives in /scripts. The project root is one level up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_ROOT / "backend"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "nrfi_model.pkl"

# db.py is inside /backend, not on Python's default import path.
# Inserting that folder first lets us write: from db import execute_query
# Guard prevents duplicate entries when this module is imported more than once.
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Load DATABASE_URL (and any other secrets) from the project-root .env file.
# db.py also loads .env, but we do it here so this script is explicit.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()

from db import execute_query  # noqa: E402  (import after sys.path change)

# Column names the model will learn from. Keep this list in the same order
# everywhere (training, printing importance, and later predicting).
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


def fetch_training_rows():
    """
    Pull completed games joined with home/away pitcher stats and team stats.

    We join pitchers twice (once as home starter, once as away starter)
    and teams twice (home lineup vs away lineup).
    """
    print("Loading completed games joined with pitcher and team stats...")

    # nrfi IS NOT NULL means we already know the 1st-inning outcome.
    # All rows in the games table are completed games, so no extra filter needed.
    query = """
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
            -- first-inning run rate = runs scored in inning 1 / games counted
            (ht.first_inning_runs_scored::float
                / NULLIF(ht.first_inning_games, 0)) AS home_team_fi_rate,
            (at.first_inning_runs_scored::float
                / NULLIF(at.first_inning_games, 0)) AS away_team_fi_rate,
            -- PostgreSQL booleans become True/False; we turn them into 1/0 below
            g.nrfi AS nrfi
        FROM games g
        LEFT JOIN pitchers hp
            ON hp.name = g.home_pitcher
           AND hp.season = EXTRACT(YEAR FROM g.game_date)
        LEFT JOIN pitchers ap
            ON ap.name = g.away_pitcher
           AND ap.season = EXTRACT(YEAR FROM g.game_date)
        LEFT JOIN teams ht
            ON ht.team = g.home_team
           AND ht.season = EXTRACT(YEAR FROM g.game_date)
        LEFT JOIN teams at
            ON at.team = g.away_team
           AND at.season = EXTRACT(YEAR FROM g.game_date)
        WHERE g.nrfi IS NOT NULL
    """

    rows = execute_query(query)
    print(f"Loaded {len(rows)} joined rows from PostgreSQL.")
    return rows


def build_xy(rows):
    """
    Turn SQL rows into a feature matrix X and a label array y.

    X = numbers the model is allowed to look at (pitcher + team stats).
    y = the answer we want it to predict (1 = NRFI, 0 = YRFI).
    """
    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(
            "No training rows came back. Run scripts/pull_historical.py first, "
            "and check that games join to pitchers/teams on name + season."
        )

    # Fill missing stats with league-average fallbacks so every game is usable.
    # LEFT JOINs mean some rows have NULLs where pitcher/team stats didn't match;
    # these defaults keep those games in the training set instead of discarding them.
    LEAGUE_AVERAGES = {
        "home_pitcher_era":    4.50,
        "home_pitcher_fi_era": 4.95,
        "home_pitcher_whip":   1.30,
        "home_pitcher_k_per_9": 8.5,
        "home_pitcher_bb_per_9": 3.2,
        "away_pitcher_era":    4.50,
        "away_pitcher_fi_era": 4.95,
        "away_pitcher_whip":   1.30,
        "away_pitcher_k_per_9": 8.5,
        "away_pitcher_bb_per_9": 3.2,
        "home_team_fi_rate":   0.28,
        "away_team_fi_rate":   0.28,
    }
    df = df.fillna(LEAGUE_AVERAGES)

    # Only drop rows where the label itself is unknown — we need nrfi to train.
    df = df.dropna(subset=["nrfi"])
    print(f"Using {len(df)} rows after applying league-average fallbacks.")

    X = df[FEATURE_COLUMNS].astype(float)

    # True/1/"true"/"t" → 1 (NRFI). Everything else with a value → 0 (YRFI).
    y = df["nrfi"].map(lambda value: 1 if str(value).lower() in {"true", "1", "t"} else 0)
    y = y.astype(int)

    print(f"Label mix — NRFI (1): {(y == 1).sum()}  YRFI (0): {(y == 0).sum()}")
    return X, y


def train_and_evaluate(X, y):
    """Split the data, train XGBoost, and print beginner-friendly metrics."""

    # 80% of rows teach the model; 20% are held out so we can test fairly.
    # stratify=y keeps the NRFI/YRFI ratio similar in both splits.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"Train size: {len(X_train)}   Test size: {len(X_test)}")

    # These settings match the project spec.
    # use_label_encoder=False avoids an old XGBoost warning; newer versions
    # removed that argument, so we fall back if it is not accepted.
    model_kwargs = {
        "n_estimators": 100,   # how many trees to grow
        "max_depth": 4,        # how deep each tree can be
        "learning_rate": 0.1,  # how much each new tree is allowed to change the answer
        "eval_metric": "logloss",
    }
    try:
        model = XGBClassifier(use_label_encoder=False, **model_kwargs)
    except TypeError:
        model = XGBClassifier(**model_kwargs)

    print("Training XGBoost classifier...")
    model.fit(X_train, y_train)

    # Predict on the held-out test games (the model has not seen these).
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    matrix = confusion_matrix(y_test, y_pred)

    print("\n=== Test metrics ===")
    print(f"Accuracy:  {accuracy:.4f}   (share of games predicted correctly)")
    print(f"Precision: {precision:.4f}   (when we say NRFI, how often we are right)")
    print(f"Recall:    {recall:.4f}   (of real NRFIs, how many we caught)")
    print(f"F1 score:  {f1:.4f}   (balance of precision and recall)")
    print("Confusion matrix (rows = actual, columns = predicted):")
    print("              Pred YRFI (0)  Pred NRFI (1)")
    print(f"Actual YRFI   {matrix[0, 0]:<14} {matrix[0, 1]}")
    print(f"Actual NRFI   {matrix[1, 0]:<14} {matrix[1, 1]}")

    # Feature importance: higher = the model relied on that column more.
    print("\n=== Feature importance (highest first) ===")
    importances = model.feature_importances_
    ranked = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda item: item[1],
        reverse=True,
    )
    for name, score in ranked:
        print(f"{name:24} {score:.4f}")

    return model


def save_model(model):
    """Write the trained model to /model/nrfi_model.pkl so the API can load it."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")


def main():
    if not os.getenv("DATABASE_URL"):
        raise ValueError(
            "DATABASE_URL is missing. Add it to the .env file in the project root."
        )

    rows = fetch_training_rows()
    X, y = build_xy(rows)
    model = train_and_evaluate(X, y)
    save_model(model)


if __name__ == "__main__":
    # This block only runs when you execute the file, not when another
    # file imports a function from it.
    main()
