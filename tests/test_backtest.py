"""Regression tests for the full-held-out-set backtest (ROADMAP.md, Step 5).

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src.backtest import decide, run_backtest  # noqa: E402
from src.run_expectancy import build_re24  # noqa: E402
from src.win_probability import build_win_prob  # noqa: E402
from src.train import NUMERIC, fit_logistic  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(REPO_ROOT, "data", "sample", "features_2023_2025.csv")
DATA_2023 = os.path.join(REPO_ROOT, "data", "retrosheet_2023")

_ctx = {}


def _get_context():
    """Cache the (small, 2023-only) tables + a tiny fitted model across
    tests in this file -- rebuilding RE24/win-probability per test would be
    slow for no benefit, same pattern as test_predict.py.
    """
    if not _ctx:
        df = pd.read_csv(FEATURES)
        for col in ("runner_sprint_speed", "runner_age", "catcher_pop_time"):
            df[col] = df[col].fillna(df[col].median())
        split_idx = int(len(df) * 0.8)
        train, test = df.iloc[:split_idx], df.iloc[split_idx:].copy()
        model = fit_logistic(train[NUMERIC].fillna(0.0), train["success"].astype(int))

        re24 = build_re24(DATA_2023)
        # Not the real 13-season hold-only baseline -- just enough data to
        # exercise the win-probability path without a slow multi-season build.
        wp_table = build_win_prob(DATA_2023)
        wp_hold_table = build_win_prob(DATA_2023, hold_only=True)

        _ctx.update(model=model, test=test, re24=re24,
                   wp_table=wp_table, wp_hold_table=wp_hold_table)
    return _ctx


def test_decide_returns_valid_decision():
    ctx = _get_context()
    d = decide(ctx["re24"], ctx["wp_table"], ctx["wp_hold_table"],
              inning=3, half=0, outs=1, base_code="1__", score_diff=0,
              target="2", p_model=0.8)
    assert d["decision"] in ("GO", "HOLD")
    assert d["layer"] == "RE24"
    assert 0.0 <= d["break_even"] <= 1.0
    assert d["reward"] > 0 and d["cost"] > 0


def test_decide_swaps_to_win_probability_late_and_close():
    ctx = _get_context()
    d = decide(ctx["re24"], ctx["wp_table"], ctx["wp_hold_table"],
              inning=9, half=1, outs=2, base_code="1__", score_diff=-1,
              target="2", p_model=0.8)
    assert d["layer"] == "WP"


def test_run_backtest_covers_every_row_with_valid_fields():
    ctx = _get_context()
    results = run_backtest(ctx["test"], ctx["model"], ctx["re24"],
                           ctx["wp_table"], ctx["wp_hold_table"])
    assert len(results) == len(ctx["test"])
    scored = results[~results["layer"].str.startswith("EXCLUDED")]
    assert set(scored["decision"]) <= {"GO", "HOLD"}
    assert set(scored["layer"]) <= {"RE24", "WP", "WP (RE24 had no data)"}
    assert scored["break_even"].between(0.0, 1.0).all()


def test_double_steal_into_occupied_base_is_excluded_not_miscomputed():
    ctx = _get_context()
    d = decide(ctx["re24"], ctx["wp_table"], ctx["wp_hold_table"],
              inning=3, half=0, outs=1, base_code="12_", score_diff=0,
              target="2", p_model=0.8)
    assert d["layer"].startswith("EXCLUDED")
    assert d["decision"] is None


def test_actual_value_matches_reward_or_cost_by_outcome():
    ctx = _get_context()
    results = run_backtest(ctx["test"], ctx["model"], ctx["re24"],
                           ctx["wp_table"], ctx["wp_hold_table"])
    # RE24 is built from the full run-expectancy table (no small-sample
    # clipping), so its reward/cost signs are always clean: successful
    # attempts realize a positive reward, caught ones the negative of the
    # cost. The WP layer's win_prob_break_even documents its own exception
    # to this (module docstring: "small-sample cells can disagree by noise
    # alone"), so this check is RE24-only -- see
    # test_double_steal_into_occupied_base_is_excluded_not_miscomputed and
    # this file's WP-specific tests for the other layer.
    re24_rows = results[results["layer"] == "RE24"]
    safe = re24_rows[re24_rows["success"] == 1]
    caught = re24_rows[re24_rows["success"] == 0]
    assert (safe["actual_value"] >= 0).all()
    assert (caught["actual_value"] <= 0).all()


def test_go_calls_have_higher_realized_success_rate_than_hold_calls():
    ctx = _get_context()
    results = run_backtest(ctx["test"], ctx["model"], ctx["re24"],
                           ctx["wp_table"], ctx["wp_hold_table"])
    go, hold = results[results["decision"] == "GO"], results[results["decision"] == "HOLD"]
    if len(go) and len(hold):
        assert go["success"].mean() >= hold["success"].mean()
