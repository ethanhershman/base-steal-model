"""
Generate golden cross-language fixtures: the core check that the Go port
of the decision layer (backend/internal/decision) produces IDENTICAL
results to the Python original.

Calls load_tables() + load_model(model_kind="logistic") -- the SAME kind of
artifacts src/export_web_data.py wrote into backend/data/app.db (both are
deterministic given the same train split -- sklearn's default lbfgs solver
has no randomness -- so this can't silently drift from a re-fit model).

For each of a representative set of situations, records predict_steal_decision's
exact output alongside the input. backend/internal/decision/golden_test.go
loads this file and asserts the Go computation matches within a small
float tolerance.

    python -m src.export_golden_fixtures
"""
from __future__ import annotations

import argparse
import json

from .predict import load_tables, load_model, predict_steal_decision
from .run_expectancy import BASE_STATES

# Which steal targets make physical sense from each base state (you can
# only steal a base with a runner behind it to send, and "___"/bases-empty
# never has a real steal attempt).
VALID_TARGETS = {
    "1__": ["2"],
    "_2_": ["3"],
    "__3": ["H"],
    "12_": ["2", "3"],
    "1_3": ["2", "H"],
    "_23": ["3", "H"],
    "123": ["2", "3", "H"],
}

# Base states where the requested target's destination is already occupied
# -- these exercise run_expectancy.py's double/triple-steal cascade fix.
CASCADE_TARGETS = {("12_", "2"), ("_23", "3"), ("123", "2")}

PLAYER_STATS = dict(
    runner_bats_lhb=False, pitcher_throws_lhp=True,
    runner_prior_sr=0.80, runner_prior_att=30,
    pitcher_prior_sr_allowed=0.75, catcher_prior_cs_rate=0.25,
    runner_sprint_speed=29.0, runner_age=27.0, catcher_pop_time=1.95,
)


def _situation(*, base_code, target, outs, inning=3, half=0, score_diff=0, **overrides):
    kwargs = dict(
        inning=inning, half=half, outs=outs, base_code=base_code,
        score_diff=score_diff, target=target, balls=1, strikes=1,
        is_double_steal=(base_code, target) in CASCADE_TARGETS,
        **PLAYER_STATS,
    )
    kwargs.update(overrides)
    return kwargs


def build_situations() -> list[dict]:
    situations = []

    # Systematic sweep: every physically valid (base_code, target) pair x
    # every out count, in a neutral early-game (non-high-leverage) spot --
    # exercises the RE24 layer, including all 3 double/triple-steal
    # cascade cases.
    for base_code in BASE_STATES:
        if base_code == "___":
            continue
        for target in VALID_TARGETS[base_code]:
            for outs in (0, 1, 2):
                situations.append(_situation(base_code=base_code, target=target, outs=outs))

    # High-leverage boundary: inning 6 (RE24) vs. 7 (win probability), same
    # situation otherwise.
    for inning in (6, 7):
        situations.append(_situation(
            base_code="1__", target="2", outs=2, inning=inning, half=1, score_diff=-1))

    # Score-margin boundary of is_high_leverage (leverage_margin=3): 3 (high
    # leverage) vs. 4 (not), both signs, late inning.
    for score_diff in (3, -3, 4, -4):
        situations.append(_situation(
            base_code="1__", target="2", outs=1, inning=8, half=1, score_diff=score_diff))

    # Certain-loss case: home team trailing, could end the game -- forces
    # win_prob_lookup's hardcoded short-circuit for the caught-stealing
    # lookup (outs+1=3, "END" state). This is also literally the README's
    # own headline example (down 1, bottom 9th, 2 outs).
    situations.append(_situation(
        base_code="1__", target="2", outs=2, inning=9, half=1, score_diff=-1))
    situations.append(_situation(
        base_code="1__", target="2", outs=2, inning=9, half=1, score_diff=0))
    situations.append(_situation(
        base_code="1__", target="2", outs=1, inning=9, half=1, score_diff=1))

    # Missing-Statcast-stat fallback path (median-fill + _missing flags).
    situations.append(_situation(
        base_code="1__", target="2", outs=1,
        runner_sprint_speed=None, runner_age=None, catcher_pop_time=None))

    return situations


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/sample/features_2023_2025.csv")
    ap.add_argument("--out", default="backend/internal/decision/testdata/golden_fixtures.json")
    args = ap.parse_args()

    import os

    print("Building RE24 + win-probability tables...")
    tables = load_tables()
    print("Fitting logistic regression model...")
    model, medians = load_model(features_path=args.features, model_kind="logistic")

    situations = build_situations()
    fixtures = []
    for s in situations:
        result = predict_steal_decision(tables, model, medians, **s)
        fixtures.append({"input": s, "expected": result})

    # One more fixture exercising the genuine RE24-coverage-gap fallback
    # (predict.py's "WP (RE24 had no data)" path) -- every real (base_code,
    # outs) combo has data at MLB volume, so this simulates the gap the
    # same way tests/test_predict.py does, rather than hoping for a rare
    # real one.
    gapped_tables = dict(tables, re24={})
    gap_situation = _situation(base_code="1__", target="2", outs=1)
    gap_result = predict_steal_decision(gapped_tables, model, medians, **gap_situation)
    fixtures.append({"input": gap_situation, "expected": gap_result, "note": "re24_gap"})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(fixtures, fh, indent=2, default=str)

    print(f"wrote {len(fixtures)} fixtures to {args.out}")


if __name__ == "__main__":
    main()
