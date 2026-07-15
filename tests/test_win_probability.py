"""Regression tests for the win-probability decision layer.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.win_probability import (  # noqa: E402
    build_win_prob, win_prob_lookup, win_prob_break_even, is_high_leverage,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "retrosheet_2023")


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
    table = build_win_prob(DATA)
    # The core finding this module exists for: trailing late in the game,
    # a caught stealing that ends your turn is only as bad as "you lose" --
    # which isn't much worse than staying put and likely losing anyway --
    # so the bar for attempting should be lower than when the game is tied
    # (where getting caught costs you a shot at a walk-off without ending
    # your season, so you have more to protect).
    be_trailing, _, _, _, _ = win_prob_break_even(table, 9, 1, 2, "1__", -1, "2")
    be_tied, _, _, _, _ = win_prob_break_even(table, 9, 1, 2, "1__", 0, "2")
    assert be_trailing < be_tied


def test_is_high_leverage():
    assert is_high_leverage(9, 0)      # tied in the 9th
    assert is_high_leverage(8, -2)     # down 2 in the 8th
    assert not is_high_leverage(3, 0)  # tied in the 3rd -- too early
    assert not is_high_leverage(9, -6) # 9th but a blowout -- not really in doubt
