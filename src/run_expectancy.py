"""
Run expectancy (RE24) and stolen-base break-even rates -- the decision layer.

RE24 is the average number of runs a team scores from a given base-out state to
the end of the half-inning. There are 24 states: 8 base configurations x 3 out
counts. We estimate it empirically from the parsed play-by-play, then use it to
answer the real question: given a predicted success probability, is a steal
attempt worth it?

A steal is a bet:
  * success -> better base state, run expectancy rises by `reward`
  * caught  -> lose the runner and add an out, run expectancy falls by `cost`

The break-even success rate is  cost / (reward + cost).  Attempt the steal when
the model's predicted P(success) exceeds this break-even.

Built from the same post-rule-change seasons (2023-2025) as the feature table
by default -- the 2023 bigger-base/pickoff-limit rules plausibly shifted the
run-scoring environment too (more extra bases taken generally), not just
steal-specific outcomes, so mixing in pre-rule-change seasons would be the
same mistake src/features.py already avoids.

Usage:
    python -m src.run_expectancy --data-dirs data/retrosheet_2023 \
        data/retrosheet_2024 data/retrosheet_2025 --out data/sample/re24_2023_2025.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

from .retrosheet_parser import iter_plays


BASE_STATES = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]


def build_re24(data_dirs) -> dict:
    """Return {(base_code, outs): expected_runs} estimated from the data.

    Accepts either a single data directory (str) or a list of them, so the
    table can be built from multiple combined seasons.
    """
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    totals = defaultdict(float)
    counts = defaultdict(int)
    for data_dir in data_dirs:
        for rec in iter_plays(data_dir):
            if rec["outs"] > 2:
                continue
            key = (rec["base_code"], rec["outs"])
            totals[key] += rec["runs_to_end"]
            counts[key] += 1
    return {k: totals[k] / counts[k] for k in totals if counts[k] > 0}


def _cascade_free(b: dict, base: int) -> bool:
    """Free up `base` for an incoming runner. If it's already occupied --
    only possible on a double/triple steal, where more than one runner
    breaks on the same pitch -- push that runner forward one base first
    (cascading further if THAT base is also occupied). Returns True if a
    run scored because a runner got pushed off 3rd.

    Two runners can never really occupy the same base, so if `target`'s
    destination is occupied at all, the only baseball-legal explanation is
    that its occupant is simultaneously advancing on this same play too.
    """
    if base > 3 or not b.get(base):
        return False
    pushed_further_scored = _cascade_free(b, base + 1)
    if base + 1 <= 3:
        b[base + 1] = True
        b[base] = False
        return pushed_further_scored
    b[base] = False
    return True  # this runner was pushed off 3rd and scored


def _state_after_success(base_code: str, target: str) -> tuple[str, int]:
    """New base code after a successful steal to `target` (2, 3, or H), and
    how many runs scored as a side effect (the batting team's own steal-of-
    home run, plus any runner a double/triple steal cascade pushed off 3rd --
    see _cascade_free)."""
    b = {1: base_code[0] != "_", 2: base_code[1] != "_", 3: base_code[2] != "_"}
    frm, to = {"2": (1, 2), "3": (2, 3), "H": (3, 4)}[target]
    cascade_scored = _cascade_free(b, to) if to <= 3 else False
    if to <= 3:
        b[to] = True
    b[frm] = False
    runs = (1 if target == "H" else 0) + (1 if cascade_scored else 0)
    return "".join(s if b[i] else "_" for i, s in ((1, "1"), (2, "2"), (3, "3"))), runs


def _state_after_caught(base_code: str, target: str) -> str:
    """New base code after a caught stealing (runner removed)."""
    frm = {"2": 1, "3": 2, "H": 3}[target]
    b = {1: base_code[0] != "_", 2: base_code[1] != "_", 3: base_code[2] != "_"}
    b[frm] = False
    return "".join(s if b[i] else "_" for i, s in ((1, "1"), (2, "2"), (3, "3")))


def break_even_rate(re24: dict, base_code: str, outs: int, target: str):
    """Break-even success rate for a steal of `target` from this state.

    Returns (break_even, reward, cost) where reward/cost are run-expectancy
    changes. A successful steal of home also banks +1 run immediately.
    """
    cur = re24[(base_code, outs)]
    succ_state, run_bonus = _state_after_success(base_code, target)
    re_succ = run_bonus + (re24.get((succ_state, outs), 0.0)
                           if outs <= 2 else 0.0)
    if outs >= 2:  # a caught stealing for the 3rd out ends the inning (RE=0)
        re_caught = 0.0
    else:
        caught_state = _state_after_caught(base_code, target)
        re_caught = re24.get((caught_state, outs + 1), 0.0)
    reward = re_succ - cur
    cost = cur - re_caught
    denom = reward + cost
    be = cost / denom if denom > 0 else 1.0
    return be, reward, cost


def should_steal(re24: dict, base_code: str, outs: int, target: str,
                 p_success: float) -> dict:
    """Recommend whether to attempt a steal given a success probability."""
    be, reward, cost = break_even_rate(re24, base_code, outs, target)
    return {
        "break_even": be,
        "p_success": p_success,
        "attempt": p_success > be,
        "reward": reward,
        "cost": cost,
        "net_run_value": p_success * reward - (1 - p_success) * cost,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dirs", nargs="+",
                    default=["data/retrosheet_2023", "data/retrosheet_2024",
                             "data/retrosheet_2025"])
    ap.add_argument("--out", default="data/sample/re24_2023_2025.csv")
    args = ap.parse_args()

    re24 = build_re24(args.data_dirs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["base_state", "0_out", "1_out", "2_out"])
        for bc in BASE_STATES:
            w.writerow([bc] + [f"{re24.get((bc, o), float('nan')):.3f}"
                               for o in (0, 1, 2)])

    print(f"RE24 (expected runs to end of inning), {', '.join(args.data_dirs)}:")
    print(f"{'bases':>6} {'0 out':>7} {'1 out':>7} {'2 out':>7}")
    for bc in BASE_STATES:
        vals = " ".join(f"{re24.get((bc, o), float('nan')):7.3f}"
                        for o in (0, 1, 2))
        print(f"{bc:>6} {vals}")

    print("\nExample break-even success rates (steal of 2nd, runner on 1st):")
    for outs in (0, 1, 2):
        be, reward, cost = break_even_rate(re24, "1__", outs, "2")
        print(f"  {outs} out: break-even = {be:.1%}  "
              f"(reward +{reward:.3f} runs, cost -{cost:.3f} runs)")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
