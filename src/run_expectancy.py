"""
Run expectancy (RE24) and stolen-base break-even rates — the decision layer.

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

Usage:
    python -m src.run_expectancy --data-dir data/retrosheet_2023 \
        --out data/sample/re24_2023.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict

from .retrosheet_parser import iter_plays


BASE_STATES = ["___", "1__", "_2_", "__3", "12_", "1_3", "_23", "123"]


def build_re24(data_dir: str) -> dict:
    """Return {(base_code, outs): expected_runs} estimated from the data."""
    totals = defaultdict(float)
    counts = defaultdict(int)
    for rec in iter_plays(data_dir):
        if rec["outs"] > 2:
            continue
        key = (rec["base_code"], rec["outs"])
        totals[key] += rec["runs_to_end"]
        counts[key] += 1
    return {k: totals[k] / counts[k] for k in totals if counts[k] > 0}


def _state_after_success(base_code: str, target: str) -> str:
    """New base code after a successful steal to `target` (2, 3, or H)."""
    b = {1: base_code[0] != "_", 2: base_code[1] != "_", 3: base_code[2] != "_"}
    if target == "2":
        b[1], b[2] = False, True
    elif target == "3":
        b[2], b[3] = False, True
    elif target == "H":
        b[3] = False
    return "".join(s if b[i] else "_" for i, s in ((1, "1"), (2, "2"), (3, "3")))


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
    succ_state = _state_after_success(base_code, target)
    run_bonus = 1.0 if target == "H" else 0.0
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
    ap.add_argument("--data-dir", default="data/retrosheet_2023")
    ap.add_argument("--out", default="data/sample/re24_2023.csv")
    args = ap.parse_args()

    re24 = build_re24(args.data_dir)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["base_state", "0_out", "1_out", "2_out"])
        for bc in BASE_STATES:
            w.writerow([bc] + [f"{re24.get((bc, o), float('nan')):.3f}"
                               for o in (0, 1, 2)])

    print("RE24 (expected runs to end of inning), 2023:")
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
