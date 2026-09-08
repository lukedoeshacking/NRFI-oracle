"""
Database helper for the NRFI Oracle backend.

This file is the only place that talks to PostgreSQL.
Other files can import get_connection, execute_query, and insert_row
instead of writing SQL connection code themselves.
"""

import os
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

# ----------------------------------------------------------
# Load environment variables from a .env file
# ----------------------------------------------------------
# python-dotenv reads KEY=value pairs from .env so we never
# hard-code passwords in source code.
#
# We look in two places:
#   1. The current working directory (wherever you ran Python from)
#   2. The project root (one folder above /backend), which is where
#      this repo typically keeps .env
load_dotenv()
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# DATABASE_URL looks like:
# postgresql://username:password@host:5432/database_name
DATABASE_URL = os.getenv("DATABASE_URL")


def get_connection():
    """
    Open a new connection to PostgreSQL.

    Returns a psycopg2 connection object.
    Remember to close it when you are done (execute_query does this for you).
    """
    if not DATABASE_URL:
        # Fail early with a clear message instead of a confusing crash later
        raise ValueError(
            "DATABASE_URL is missing. Add it to your .env file in the project root."
        )

    # psycopg2.connect() opens the actual network connection to the database
    connection = psycopg2.connect(DATABASE_URL)
    print("Database connection successful.")
    return connection


def execute_query(query, params=None, conn=None):
    """
    Run any SQL statement and return rows as a list of dictionaries.

    query  — SQL string, e.g. "SELECT * FROM games WHERE id = %s"
    params — optional tuple/list/dict of values that fill in the %s placeholders.
             Using params (not string concatenation) prevents SQL injection.
    conn   — optional already-open psycopg2 connection.
             When provided, it is reused and NOT closed when done (the caller
             is responsible for closing it).
             When None (default), a new connection is opened and closed
             automatically — the original behaviour for all other callers.

    Example:
        rows = execute_query("SELECT * FROM games WHERE id = %s", (42,))
        # rows might look like: [{"id": 42, "home_team": "NYY"}, ...]

    For INSERT/UPDATE/DELETE that do not return rows, this returns an empty list.
    """
    _own_conn = conn is None      # True when we opened the connection ourselves
    if _own_conn:
        conn = get_connection()
    try:
        # RealDictCursor makes each row a dict keyed by column name
        # instead of a plain tuple like (1, "NYY")
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)

            # cursor.description is None when the statement has no result set
            # (for example a plain INSERT without RETURNING)
            if cursor.description is None:
                conn.commit()  # save the change permanently
                return []

            rows = cursor.fetchall()
            conn.commit()
            # RealDictRow is dict-like; convert to a normal dict for simplicity
            return [dict(row) for row in rows]
    except Exception:
        # If something went wrong, undo the transaction so the DB stays consistent
        conn.rollback()
        raise
    finally:
        # Only close the connection when we opened it ourselves.
        # When the caller passed their own conn, they are responsible for closing it.
        if _own_conn:
            conn.close()


def _safe_identifier(name):
    """
    Allow only letters, numbers, and underscores in table/column names.

    Table and column names cannot use %s placeholders the way values can,
    so we check the name ourselves instead of pasting arbitrary text into SQL.
    """
    if not name or not str(name).replace("_", "").isalnum():
        raise ValueError(f"Invalid table or column name: {name!r}")
    return str(name)


def insert_row(table, data_dict, conn=None):
    """
    Build and run an INSERT for one row.

    table     — table name as a string, e.g. "games"
    data_dict — column names and values, e.g. {"home_team": "NYY", "away_team": "BOS"}
    conn      — optional already-open psycopg2 connection (see execute_query).

    Returns the inserted row as a list of one dict (because of RETURNING *).
    """
    if not data_dict:
        raise ValueError("data_dict cannot be empty.")

    # Column names come from the dictionary keys
    columns = [_safe_identifier(column) for column in data_dict.keys()]
    values = list(data_dict.values())
    table_name = _safe_identifier(table)

    # "%s, %s, %s" — one placeholder per value (psycopg2 fills these in safely)
    placeholders = ", ".join(["%s"] * len(columns))
    column_sql = ", ".join(columns)

    # RETURNING * asks PostgreSQL to send back the row after inserting it
    # (useful for reading default values like an auto-generated id)
    query = f"INSERT INTO {table_name} ({column_sql}) VALUES ({placeholders}) RETURNING *"

    return execute_query(query, values, conn=conn)


if __name__ == "__main__":
    conn = get_connection()
    if conn:
        print("Database connection successful!")
        conn.close()
    else:
        print("Connection failed")