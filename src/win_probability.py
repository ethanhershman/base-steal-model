"""
Win-probability break-even for late/close situations -- the upgrade
ROADMAP.md calls out under "one important upgrade": RE24 optimizes for
total runs, which is right in the early innings but wrong late and close,
where a run to tie is worth far more than a run in a blowout, and a caught
stealing that ends the game (a walk-off-loss situation) isn't priced at
all by "0 more runs this inning."

Built the same way run_expectancy.py builds RE24: empirically, from real
historical plays, not a formula. `iter_plays_for_win_prob` walks every play
across the given seasons and records the base-out state, score margin, and
inning/half at that moment, plus whether the batting team went on to win
the whole game. Each table cell is the fraction of real historical
instances of that exact situation where the batting team won.

The tricky part is what happens if a caught stealing makes the 3rd out:
that's not "0 more runs" (RE24's answer), it's "this team's turn just
ended at this score, in this inning" -- which can mean a certain loss
(trailing in the bottom of the 9th+) or anything else depending on the
situation. That's handled by a second table, built from the SAME data,
keyed on the moment every half-inning actually ended (regardless of how),
so walk-off/sudden-death dynamics fall out for free instead of needing an
error-prone "flip to the opponent's perspective" formula.

Unlike src/features.py and src/run_expectancy.py, this defaults to
2021-2025 (five seasons, not three): win probability by score/inning/outs/
bases isn't affected by the 2023 steal-specific rule changes the way steal
SUCCESS rates are, so there's no reason to throw away 2021-2022 here --
more data matters a lot for the sparser late/close cells. 2021-2022 do
share the automatic extra-innings runner rule (in effect since 2020) with
2023-2025, so extra-inning dynamics stay consistent; going back further
than 2021 would cross that boundary and mix in a different extra-innings
rule, which isn't done here without deliberately checking for it first.

Usage:
    python -m src.win_probability --data-dirs data/retrosheet_2021 \
        data/retrosheet_2022 data/retrosheet_2023 data/retrosheet_2024 \
        data/retrosheet_2025
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from .retrosheet_parser import iter_plays_for_win_prob
from .run_expectancy import _state_after_success, _state_after_caught

MIN_CELL_N = 20        # below this, fall back to a coarser lookup
SCORE_CLIP = 4          # score margins beyond +-4 are clipped together
MAX_INNING = 9          # 9th and later all bucket together ("9+")


def _inning_bucket(inning: int) -> int:
    return min(inning, MAX_INNING)


def _score_bucket(score_diff: int) -> int:
    return max(-SCORE_CLIP, min(SCORE_CLIP, score_diff))


def build_win_prob(data_dirs, min_n: int = MIN_CELL_N) -> dict:
    """Return {(inning_b, half, outs, base_code, score_b): (win_rate, n)}.

    outs=3 entries use base_code='END' and represent "this team's turn just
    ended (by any means) at this score" -- see module docstring.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    wins = defaultdict(int)
    counts = defaultdict(int)
    for data_dir in data_dirs:
        for rec in iter_plays_for_win_prob(data_dir):
            key = (_inning_bucket(rec["inning"]), rec["half"], rec["outs"],
                  rec["base_code"], _score_bucket(rec["score_diff"]))
            wins[key] += rec["won"]
            counts[key] += 1

    return {k: (wins[k] / counts[k], counts[k]) for k in counts}


def win_prob_lookup(table: dict, inning: int, half: int, outs: int,
                    base_code: str, score_diff: int, min_n: int = MIN_CELL_N):
    """Look up P(win) with a fallback chain for sparse cells. Returns
    (win_prob, n, source) where n is the sample size actually backing the
    answer and source says which fallback stage was used -- callers should
    treat a small n (or a coarse source) as low-confidence, especially for
    extreme/rare situations (e.g. a big lead in the bottom of the 9th).

    Crucially, every fallback step preserves the exact inning bucket and
    half -- inning-lateness is the whole reason this table exists (a team
    trailing when their half-inning ends in the 9th or later has LOST, full
    stop, versus the same score margin in the 3rd where the game is barely
    underway), so a fallback that averages across innings would wash out
    exactly the signal being asked for. Instead we widen the score-margin
    window first, then (for in-play, non-boundary states only) average over
    base codes, and only ever fall back within the same (inning, half).
    """
    ib, sb = _inning_bucket(inning), _score_bucket(score_diff)

    # Logical certainty, not an empirical estimate: if the home team's
    # half-inning just ended (outs==3) in the 9th or later while they're
    # still behind, the game is over and they lost -- by the rules of
    # baseball, not by however many (possibly sparse) historical examples
    # happen to be in the table. Bucket-averaging this case with a tied
    # score would wrongly blend a certain loss with a coin flip.
    if outs == 3 and half == 1 and ib >= MAX_INNING and score_diff < 0:
        return 0.0, float("inf"), "certain (home team trailing, game over)"

    exact = table.get((ib, half, outs, base_code, sb))
    if exact and exact[1] >= min_n:
        return exact[0], exact[1], "exact"

    # widen the score-margin window (+-1 bucket), same inning/half/outs/base
    w, n = 0.0, 0
    for (i, h, o, bc, s), (rate, cnt) in table.items():
        if (i, h, o, bc) == (ib, half, outs, base_code) and abs(s - sb) <= 1:
            w += rate * cnt
            n += cnt
    if n >= min_n:
        return w / n, n, "widened score window"

    # same inning/half/outs, widened score window, averaged over base codes
    # (meaningless no-op for the outs==3/'END' boundary state, which only
    # ever has one base code -- that's fine, it just falls through)
    w, n = 0.0, 0
    for (i, h, o, bc, s), (rate, cnt) in table.items():
        if (i, h, o) == (ib, half, outs) and abs(s - sb) <= 1:
            w += rate * cnt
            n += cnt
    if n >= min_n:
        return w / n, n, "widened score window, all base codes"

    # last resort: same inning/half/outs, ANY score margin -- still never
    # crosses into a different inning bucket.
    w, n = 0.0, 0
    for (i, h, o, bc, s), (rate, cnt) in table.items():
        if (i, h, o) == (ib, half, outs):
            w += rate * cnt
            n += cnt
    if n > 0:
        return w / n, n, "any score margin, all base codes (low confidence)"

    return 0.5, 0, "no data (default)"


def win_prob_break_even(table: dict, inning: int, half: int, outs: int,
                        base_code: str, score_diff: int, target: str,
                        min_n: int = MIN_CELL_N):
    """Break-even success rate for a steal, in win-probability terms.

    Mirrors run_expectancy.break_even_rate's structure (reward/cost around
    a "current state"), but the currency is P(win) instead of expected
    runs, and a caught-for-the-3rd-out looks up the empirical 'END' state
    (see module docstring) instead of assuming RE=0 / flipping algebraically.

    Returns (break_even, reward, cost, min_n, sources) -- min_n is the
    smallest sample size behind the three lookups this answer depends on,
    and sources names which fallback stage each one used, so callers can
    flag low-confidence answers (small min_n / coarse sources) rather than
    presenting every number with the same false precision.
    """
    cur, n_cur, src_cur = win_prob_lookup(table, inning, half, outs, base_code, score_diff, min_n)

    succ_base = _state_after_success(base_code, target)
    succ_score = score_diff + (1 if target == "H" else 0)
    wp_succ, n_succ, src_succ = win_prob_lookup(table, inning, half, outs, succ_base, succ_score, min_n)

    if outs >= 2:
        wp_caught, n_caught, src_caught = win_prob_lookup(table, inning, half, 3, "END", score_diff, min_n)
    else:
        caught_base = _state_after_caught(base_code, target)
        wp_caught, n_caught, src_caught = win_prob_lookup(
            table, inning, half, outs + 1, caught_base, score_diff, min_n)

    reward = wp_succ - cur
    cost = cur - wp_caught
    denom = reward + cost
    # Reward and cost SHOULD both be >= 0 (a successful steal can't hurt
    # your win probability, getting caught can't help it) but small-sample
    # cells can disagree by noise alone, especially since cur/wp_succ/
    # wp_caught can each land on a different fallback stage. Clip to a
    # valid probability rather than reporting a break-even below 0% or
    # above 100%, which isn't a meaningful answer either way -- treat it as
    # "always worth it" or "never worth it" rather than a precise number.
    be = min(1.0, max(0.0, cost / denom)) if denom > 0 else 1.0
    min_n_seen = min(n_cur, n_succ, n_caught)
    return be, reward, cost, min_n_seen, (src_cur, src_succ, src_caught)


def is_high_leverage(inning: int, score_diff: int, leverage_innings: int = 7,
                     leverage_margin: int = 3) -> bool:
    """Heuristic for when to swap RE24 for win probability: late innings
    (7th or later, per ROADMAP.md's "high-leverage late-game situations")
    and a game still close enough that a single run plausibly swings it."""
    return inning >= leverage_innings and abs(score_diff) <= leverage_margin


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+",
                    default=["data/retrosheet_2021", "data/retrosheet_2022",
                             "data/retrosheet_2023", "data/retrosheet_2024",
                             "data/retrosheet_2025"])
    args = ap.parse_args()

    table = build_win_prob(args.data_dirs)
    print(f"win-probability table built from {len(table)} (inning, half, outs, "
          f"base, score) cells across {', '.join(args.data_dirs)}")

    print("\nExample: runner on 1st, steal of 2nd, 2 outs, bottom of 9th or later:")
    for score_diff in (-2, -1, 0, 1, 2):
        be, reward, cost, n, sources = win_prob_break_even(table, 9, 1, 2, "1__", score_diff, "2")
        print(f"  score {score_diff:+d} (batting team perspective): "
              f"win-prob break-even = {be:.1%}  (reward {reward:+.3f}, cost {cost:+.3f}, "
              f"min_n={n})")
        print(f"      sources: cur={sources[0]}, after-success={sources[1]}, after-caught={sources[2]}")

    print("\nSame situation, but early (3rd inning) -- RE24 regime, for comparison:")
    from .run_expectancy import build_re24, break_even_rate
    re24 = build_re24(args.data_dirs)
    be, reward, cost = break_even_rate(re24, "1__", 2, "2")
    print(f"  RE24 break-even (any score, any early inning) = {be:.1%}")


if __name__ == "__main__":
    main()
