"""Regression tests for the win-probability decision layer.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.win_probability import (  # noqa: E402
    build_win_prob, win_prob_lookup, win_prob_break_even, is_high_leverage, MIN_CELL_N,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO_ROOT, "data", "retrosheet_2023")
# A few of the sparser late/close cells need more than one season to give a
# STABLE (not just plausible-looking) answer -- see README.md, "Decision
# layer" for the 3-season-vs-5-season sensitivity finding this came from.
DATA_MULTI = [os.path.join(REPO_ROOT, "data", f"retrosheet_{y}")
             for y in (2021, 2022, 2023, 2024, 2025)]


def test_home_team_trailing_at_game_end_is_certain_loss():
    table = build_win_prob(DATA)
    # Home team's half-inning just ended (outs=3) in the 9th+ while behind:
    # the game is over, by rule -- not an empirical estimate.
    wp, n, source = win_prob_lookup(table, 9, 1, 3, "END", -1)
    assert wp == 0.0
    assert "certain" in source


def test_win_prob_monotonic_in_score():
    table = build_win_prob(DATA)
    # At a fixed inning/half/outs/bases, being further ahead should never
    # hurt your win probability.
    wp_behind, _, _ = win_prob_lookup(table, 9, 1, 1, "1__", -1)
    wp_tied, _, _ = win_prob_lookup(table, 9, 1, 1, "1__", 0)
    wp_ahead, _, _ = win_prob_lookup(table, 9, 1, 1, "1__", 1)
    assert wp_behind <= wp_tied <= wp_ahead


def test_break_even_lower_when_trailing_late_than_tied():
    # Needs multiple seasons -- with only 2023, this specific comparison's
    # cells are too sparse (even after clearing MIN_CELL_N) to reliably
    # land in the right order; see the 3-season-vs-5-season sensitivity
    # finding in README.md.
    table = build_win_prob(DATA_MULTI)
    hold_table = build_win_prob(DATA_MULTI, hold_only=True)
    # The core finding this module exists for: trailing late in the game,
    # a caught stealing that ends your turn is only as bad as "you lose" --
    # which isn't much worse than staying put and likely losing anyway --
    # so the bar for attempting should be lower than when the game is tied
    # (where getting caught costs you a shot at a walk-off without ending
    # your season, so you have more to protect).
    be_trailing, _, _, n_trailing, _ = win_prob_break_even(table, hold_table, 9, 1, 2, "1__", -1, "2")
    be_tied, _, _, n_tied, _ = win_prob_break_even(table, hold_table, 9, 1, 2, "1__", 0, "2")
    assert min(n_trailing, n_tied) >= MIN_CELL_N
    assert be_trailing < be_tied


def test_hold_only_excludes_steal_attempts():
    table = build_win_prob(DATA)
    hold_table = build_win_prob(DATA, hold_only=True)
    # hold_only should never have MORE observations in a cell than the
    # unconditional table -- it's a strict subset (steal attempts removed).
    for key, (rate, n) in hold_table.items():
        all_rate, all_n = table[key]
        assert n <= all_n


def test_is_high_leverage():
    assert is_high_leverage(9, 0)      # tied in the 9th
    assert is_high_leverage(8, -2)     # down 2 in the 8th
    assert not is_high_leverage(3, 0)  # tied in the 3rd -- too early
    assert not is_high_leverage(9, -6) # 9th but a blowout -- not really in doubt
