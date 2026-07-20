"""Regression tests for the SQLite export step feeding the Go web app.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import datetime
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.export_web_data import (  # noqa: E402
    build_schema, export_re24, export_win_prob, export_model, export_players,
)
from src.predict import load_tables, load_model  # noqa: E402
from src.train import NUMERIC  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = os.path.join(REPO_ROOT, "backend", "sql", "schema.sql")
FEATURES = os.path.join(REPO_ROOT, "data", "sample", "features_2023_2025.csv")

_ctx = {}


def _get_context():
    """Build one export, in memory, shared across this file's tests --
    rebuilding RE24/win-probability/the model per test would be slow for
    no benefit (same caching pattern as test_predict.py/test_backtest.py).
    """
    if not _ctx:
        conn = sqlite3.connect(":memory:")
        build_schema(conn, SCHEMA)

        tables = load_tables()
        export_re24(conn, tables["re24"])
        export_win_prob(conn, tables["wp_table"], tables["wp_hold_table"])

        model, medians = load_model(features_path=FEATURES, model_kind="logistic")
        export_model(conn, model, medians, {
            "trained_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "train_rows": 10854,
        })

        player_counts = export_players(conn, FEATURES)
        conn.commit()
        _ctx["conn"] = conn
        _ctx["player_counts"] = player_counts
    return _ctx


def test_re24_has_all_24_cells():
    conn = _get_context()["conn"]
    (n,) = conn.execute("SELECT COUNT(*) FROM re24_cells").fetchone()
    assert n == 24


def test_win_prob_cells_split_by_table_kind():
    conn = _get_context()["conn"]
    rows = dict(conn.execute(
        "SELECT table_kind, COUNT(*) FROM win_prob_cells GROUP BY table_kind").fetchall())
    assert rows.get("after", 0) > 1000
    assert rows.get("hold", 0) > 1000
    # The two ranges are built from different season spans (see
    # win_probability.py's module docstring) so they're never the same size.
    assert rows["after"] != rows["hold"]


def test_model_coefficients_match_train_numeric_exactly():
    conn = _get_context()["conn"]
    rows = conn.execute(
        "SELECT feature_name FROM model_coefficients ORDER BY sort_order").fetchall()
    names = [r[0] for r in rows]
    assert names == NUMERIC  # exact order match -- Go reconstructs this via ORDER BY sort_order
    (n,) = conn.execute("SELECT COUNT(*) FROM model_coefficients").fetchone()
    assert n == len(NUMERIC)


def test_model_meta_singleton_row_is_sane():
    conn = _get_context()["conn"]
    row = conn.execute(
        "SELECT intercept, median_runner_sprint_speed, median_runner_age, "
        "median_catcher_pop_time, train_rows FROM model_meta WHERE id = 1").fetchone()
    assert row is not None
    intercept, sprint_med, age_med, pop_med, train_rows = row
    assert 20 < sprint_med < 35          # ft/s, plausible MLB sprint speed range
    assert 20 < age_med < 40
    assert 1.5 < pop_med < 2.5           # seconds, plausible catcher pop time
    assert train_rows > 0


def test_player_counts_are_in_expected_ballpark():
    counts = _get_context()["player_counts"]
    assert 500 < counts["runners"] < 1500
    assert 700 < counts["pitchers"] < 1500
    assert 50 < counts["catchers"] < 300


def test_at_least_99pct_of_players_resolved_to_a_real_name():
    conn = _get_context()["conn"]
    counts = _get_context()["player_counts"]
    total = counts["runners"] + counts["pitchers"] + counts["catchers"]
    assert counts["unresolved_names"] / total < 0.01

    # A resolved name shouldn't just be the raw id echoed back.
    (bad,) = conn.execute(
        "SELECT COUNT(*) FROM runners WHERE full_name = player_id").fetchone()
    assert bad / counts["runners"] < 0.01


def test_search_name_is_lowercased_for_like_search():
    conn = _get_context()["conn"]
    row = conn.execute(
        "SELECT full_name, search_name FROM runners WHERE search_name LIKE '%acuna%' LIMIT 1"
    ).fetchone()
    assert row is not None
    full_name, search_name = row
    assert search_name == full_name.lower()
