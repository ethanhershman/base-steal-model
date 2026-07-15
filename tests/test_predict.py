"""Regression tests for the unified predict_steal_decision interface.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.predict import (  # noqa: E402
    load_tables, load_model, predict_steal_decision, predict_steal_decisions_table,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = os.path.join(REPO_ROOT, "data", "sample", "features_2023_2025.csv")

_tables = None
_model = None
_medians = None


def _get_tables_and_model():
    global _tables, _model, _medians
    if _tables is None:
        _tables = load_tables()
        _model, _medians = load_model(features_path=FEATURES)
    return _tables, _model, _medians


def _row(**overrides):
    tables, model, medians = _get_tables_and_model()
    kwargs = dict(
        inning=9, half=1, outs=2, base_code="1__", score_diff=-1, target="2",
        runner_sprint_speed=29.5, catcher_pop_time=1.95,
        runner_prior_sr=0.82, runner_prior_att=25,
    )
    kwargs.update(overrides)
    return predict_steal_decision(tables, model, medians, **kwargs)


def test_output_has_every_column_the_ui_needs():
    row = _row()
    for key in ("win_prob_current", "win_prob_if_success", "break_even",
               "p_success", "decision", "layer"):
        assert key in row
    assert row["decision"] in ("GO", "HOLD")
    assert 0.0 <= row["break_even"] <= 1.0
    assert 0.0 <= row["p_success"] <= 1.0


def test_high_leverage_uses_win_probability_layer():
    row = _row(inning=9, half=1, outs=2, score_diff=-1)
    assert row["layer"] == "WP"


def test_early_game_uses_re24_layer():
    row = _row(inning=3, half=0, outs=1, score_diff=0)
    assert row["layer"] == "RE24"


def test_leading_break_even_is_certain_hold():
    # Never worth risking a lead late -- break-even should hit the ceiling.
    row = _row(inning=8, half=1, outs=1, score_diff=1)
    assert row["break_even"] == 1.0
    assert row["decision"] == "HOLD"


def test_missing_re24_coverage_falls_back_to_win_probability_instead_of_crashing():
    tables, model, medians = _get_tables_and_model()
    gapped_tables = dict(tables, re24={})
    row = predict_steal_decision(
        gapped_tables, model, medians, inning=3, half=0, outs=1, base_code="1__",
        score_diff=0, target="2", runner_sprint_speed=29.0, catcher_pop_time=2.0,
        runner_prior_sr=0.78, runner_prior_att=40,
    )
    assert "RE24 had no data" in row["layer"]
    assert 0.0 <= row["break_even"] <= 1.0


def test_missing_player_inputs_use_medians_not_crash():
    row = _row(runner_sprint_speed=None, runner_age=None, catcher_pop_time=None)
    assert 0.0 <= row["p_success"] <= 1.0


def test_batch_table_matches_row_count_and_has_decision_column():
    tables, model, medians = _get_tables_and_model()
    situations = [
        dict(inning=3, half=0, outs=1, base_code="1__", score_diff=0, target="2"),
        dict(inning=9, half=1, outs=2, base_code="1__", score_diff=-1, target="2"),
    ]
    df = predict_steal_decisions_table(tables, model, medians, situations)
    assert len(df) == len(situations)
    assert "decision" in df.columns
    assert set(df["decision"]) <= {"GO", "HOLD"}
