"""Regression tests for the win-probability decision layer.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.win_probability import (  # noqa: E402
    build_win_prob, win_prob_lookup, win_prob_break_even, is_high_leverage, MIN_CELL_N,
    LEGACY_SEASONS, MODERN_SEASONS,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(REPO_ROOT, "data", "retrosheet_2023")
# A few of the sparser late/close cells need more than one season to give a
# STABLE (not just plausible-looking) answer -- see README.md, "Decision
# layer" for the 3-season-vs-5-season sensitivity finding this came from.
DATA_MULTI = [os.path.join(REPO_ROOT, "data", f"retrosheet_{y}")
             for y in (2021, 2022, 2023, 2024, 2025)]
# Full default range (2013-2025), with pre-2020 seasons treated as legacy
# (extra innings excluded -- see build_win_prob's legacy_dirs).
DATA_FULL = [os.path.join(REPO_ROOT, "data", f"retrosheet_{y}")
            for y in LEGACY_SEASONS + MODERN_SEASONS]
LEGACY_DIRS_FULL = [os.path.join(REPO_ROOT, "data", f"retrosheet_{y}")
                   for y in LEGACY_SEASONS]


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


def test_legacy_dirs_excludes_extra_innings():
    # Build from ONE legacy (pre-2020) season, with and without the
    # legacy-dirs exclusion, and confirm no inning-10+ records leak through
    # when it's applied.
    legacy_2019 = [os.path.join(REPO_ROOT, "data", "retrosheet_2019")]
    table_with_extras = build_win_prob(legacy_2019)
    table_no_extras = build_win_prob(legacy_2019, legacy_dirs=legacy_2019)
    # MAX_INNING buckets 9+ together, so this doesn't change which KEYS
    # exist, but it should reduce (or leave equal) the counts in that
    # bucket since some extra-inning plays are now excluded.
    for key in table_no_extras:
        if key[0] == 9:  # inning bucket 9 (includes 9th + extras)
            _, n_with = table_with_extras.get(key, (0, 0))
            _, n_no = table_no_extras[key]
            assert n_no <= n_with


def test_full_range_hold_only_beats_modern_only_on_sample_size():
    # The whole point of extending the hold-only table specifically: does
    # adding 2013-2020 (extra innings excluded from those seasons) actually
    # increase sample size for the sparse late/close cells?
    hold_modern = build_win_prob(DATA_MULTI, hold_only=True)
    hold_full = build_win_prob(DATA_FULL, hold_only=True, legacy_dirs=LEGACY_DIRS_FULL)

    _, n_modern = hold_modern[(9, 1, 2, "1__", -1)]
    _, n_full = hold_full[(9, 1, 2, "1__", -1)]
    assert n_full > n_modern


def test_hold_only_baseline_is_era_consistent_but_table_is_not():
    # Validates the actual production design (see win_probability.py's
    # module docstring): the hold-only "before the decision" baseline is
    # checked to be stable across eras, which is WHY it's safe to extend to
    # 2013-2025 -- but the unconditional after-success/after-caught table
    # is NOT era-stable, which is why it stays on 2021-2025 only, like RE24.
    legacy_only = [os.path.join(REPO_ROOT, "data", f"retrosheet_{y}")
                  for y in LEGACY_SEASONS]
    hold_legacy = build_win_prob(legacy_only, hold_only=True, legacy_dirs=legacy_only)
    hold_modern = build_win_prob(DATA_MULTI, hold_only=True)
    wp_legacy, n_l, _ = win_prob_lookup(hold_legacy, 9, 1, 2, "1__", -1)
    wp_modern, n_m, _ = win_prob_lookup(hold_modern, 9, 1, 2, "1__", -1)
    assert n_l >= MIN_CELL_N and n_m >= MIN_CELL_N
    assert abs(wp_legacy - wp_modern) < 0.02  # era-consistent, small gap

    table_legacy = build_win_prob(legacy_only, legacy_dirs=legacy_only)
    table_modern = build_win_prob(DATA_MULTI)
    succ_legacy, n_sl, _ = win_prob_lookup(table_legacy, 9, 1, 2, "_2_", -1)
    succ_modern, n_sm, _ = win_prob_lookup(table_modern, 9, 1, 2, "_2_", -1)
    assert n_sl >= MIN_CELL_N and n_sm >= MIN_CELL_N
    assert abs(succ_legacy - succ_modern) > 0.02  # NOT era-consistent -- the real gap
