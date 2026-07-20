"""Regression tests for the RE24 decision layer against known textbook values.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_expectancy import build_re24, break_even_rate, _state_after_success  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "retrosheet_2023")


def test_re24_matches_textbook():
    re24 = build_re24(DATA)
    # Classic run-expectancy anchors (tolerances are generous).
    assert abs(re24[("___", 0)] - 0.51) < 0.06   # bases empty, 0 out
    assert abs(re24[("123", 0)] - 2.28) < 0.20   # bases loaded, 0 out
    # Run expectancy must fall as outs increase from the same base state.
    for bc in ("___", "1__", "123"):
        assert re24[(bc, 0)] > re24[(bc, 1)] > re24[(bc, 2)]


def test_break_even_reasonable():
    re24 = build_re24(DATA)
    # Steal of 2nd from 1st, textbook break-even is roughly 70-75%.
    for outs in (0, 1, 2):
        be, reward, cost = break_even_rate(re24, "1__", outs, "2")
        assert 0.60 <= be <= 0.85, f"{outs} out: break-even {be:.1%} outside plausible range"
        assert reward > 0
        assert cost > 0


def test_caught_stealing_third_out_ends_inning():
    re24 = build_re24(DATA)
    # With 2 outs, a caught stealing ends the inning (RE=0 after), so the
    # cost of getting caught should equal the full current-state RE.
    be, reward, cost = break_even_rate(re24, "1__", 2, "2")
    assert abs(cost - re24[("1__", 2)]) < 1e-9


def test_state_after_success_normal_single_runner():
    state, runs = _state_after_success("1__", "2")
    assert state == "_2_"
    assert runs == 0


def test_state_after_success_steal_of_home_scores():
    state, runs = _state_after_success("__3", "H")
    assert state == "___"
    assert runs == 1


def test_state_after_success_double_steal_cascades_the_occupied_runner():
    # Runners on 1st and 2nd; the trailing runner steals 2nd, so the
    # runner already on 2nd must simultaneously be advancing to 3rd (two
    # runners can't occupy the same base) -- not overwritten/lost.
    state, runs = _state_after_success("12_", "2")
    assert state == "_23"
    assert runs == 0


def test_state_after_success_double_steal_scores_when_cascade_reaches_home():
    # Runners on 2nd and 3rd; the trailing runner steals 3rd, so the
    # runner already on 3rd is pushed home and scores.
    state, runs = _state_after_success("_23", "3")
    assert state == "__3"
    assert runs == 1


def test_state_after_success_triple_steal_cascades_through_every_base():
    # Bases loaded, runner on 1st steals 2nd: pushes 1st's occupant to
    # 2nd (already occupied) -> 2nd's occupant to 3rd (already occupied)
    # -> 3rd's occupant scores.
    state, runs = _state_after_success("123", "2")
    assert state == "_23"
    assert runs == 1


def test_double_steal_break_even_is_sane_not_corrupted():
    # Before the cascade fix, stealing into an occupied base produced a
    # negative cost / break-even above 100% because the other runner's
    # simultaneous advance was silently dropped.
    re24 = build_re24(DATA)
    be, reward, cost = break_even_rate(re24, "12_", 1, "2")
    assert 0.0 <= be <= 1.0
    assert reward > 0
    assert cost > 0
