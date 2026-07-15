"""Regression tests for the win-probability decision layer.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.win_probability import (  # noqa: E402
    build_win_prob, win_prob_lookup, win_prob_break_even, is_high_leverage, MIN_CELL_N,
    LEGACY_SEASONS, MODERN_SEASONS, POST_RULE_CHANGE_SEASONS,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dirs(seasons):
    return [os.path.join(REPO_ROOT, "data", f"retrosheet_{y}") for y in seasons]


DATA = _dirs([2023])
# For the after-success/after-caught TABLE: post-rule-change only (2023-2025),
# matching src/features.py and src/run_expectancy.py exactly -- NOT
# MODERN_SEASONS (2021-2025), which still spans the 2023 steal-rule boundary
# and was only proven safe for the HOLD-ONLY baseline, not this table (a real
# bug, caught after the fact -- see README.md, "Decision layer").
POST_RULE_DIRS = _dirs(POST_RULE_CHANGE_SEASONS)
# For the hold-only baseline: the wider ranges checked to be era-consistent.
MODERN_DIRS = _dirs(MODERN_SEASONS)
LEGACY_DIRS = _dirs(LEGACY_SEASONS)
FULL_HOLD_DIRS = LEGACY_DIRS + MODERN_DIRS


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
    # Matches the production design: table on post-rule-change seasons only,
    # hold_table on the wider (era-consistent) range.
    table = build_win_prob(POST_RULE_DIRS)
    hold_table = build_win_prob(FULL_HOLD_DIRS, hold_only=True, legacy_dirs=LEGACY_DIRS)
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
    legacy_2019 = _dirs([2019])
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
    hold_modern = build_win_prob(MODERN_DIRS, hold_only=True)
    hold_full = build_win_prob(FULL_HOLD_DIRS, hold_only=True, legacy_dirs=LEGACY_DIRS)

    _, n_modern = hold_modern[(9, 1, 2, "1__", -1)]
    _, n_full = hold_full[(9, 1, 2, "1__", -1)]
    assert n_full > n_modern


def test_hold_only_baseline_is_era_consistent_but_table_is_not():
    # Validates the actual production design (see win_probability.py's
    # module docstring): the hold-only "before the decision" baseline is
    # checked to be stable across eras, which is WHY it's safe to extend to
    # 2013-2025 -- but the unconditional after-success/after-caught table
    # is NOT era-stable, which is why it stays on 2023-2025 only, like RE24.
    hold_legacy = build_win_prob(LEGACY_DIRS, hold_only=True, legacy_dirs=LEGACY_DIRS)
    hold_modern = build_win_prob(MODERN_DIRS, hold_only=True)
    wp_legacy, n_l, _ = win_prob_lookup(hold_legacy, 9, 1, 2, "1__", -1)
    wp_modern, n_m, _ = win_prob_lookup(hold_modern, 9, 1, 2, "1__", -1)
    assert n_l >= MIN_CELL_N and n_m >= MIN_CELL_N
    assert abs(wp_legacy - wp_modern) < 0.02  # era-consistent, small gap

    table_legacy = build_win_prob(LEGACY_DIRS, legacy_dirs=LEGACY_DIRS)
    table_modern = build_win_prob(MODERN_DIRS)
    succ_legacy, n_sl, _ = win_prob_lookup(table_legacy, 9, 1, 2, "_2_", -1)
    succ_modern, n_sm, _ = win_prob_lookup(table_modern, 9, 1, 2, "_2_", -1)
    assert n_sl >= MIN_CELL_N and n_sm >= MIN_CELL_N
    assert abs(succ_legacy - succ_modern) > 0.02  # NOT era-consistent -- the real gap


def test_table_default_seasons_dont_span_rule_boundary():
    # Regression test for the actual bug found in conversation: the
    # after-success/after-caught table's default season range must not
    # accidentally revert to MODERN_SEASONS (2021-2025, which still spans
    # the 2023 steal-rule boundary that table was proven sensitive to).
    # Inspects the real source of both entry points rather than a parallel
    # reconstruction, so it catches "someone wired the wrong constant back
    # in," not just "the constant's values changed."
    import inspect
    from src import win_probability, demo_decision

    for module, func_name in ((win_probability, "main"), (demo_decision, "main")):
        src = inspect.getsource(getattr(module, func_name))
        # every "--data-dirs" (RE24/table) argument in these two files must
        # default to POST_RULE_CHANGE_SEASONS, and every "--wp-data-dirs" /
        # "--hold-data-dirs" must NOT be the sole source for a bare
        # "--data-dirs" default.
        assert '"--data-dirs"' in src
        data_dirs_block = src.split('"--data-dirs"')[1].split(")")[0]
        assert "POST_RULE_CHANGE_SEASONS" in data_dirs_block, (
            f"{module.__name__}.{func_name}'s --data-dirs default should use "
            f"POST_RULE_CHANGE_SEASONS, not MODERN_SEASONS"
        )
