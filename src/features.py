"""
Turn the raw steal-attempt table into a model-ready feature matrix.

Key principle: NO LEAKAGE. A runner's "success rate" feature is computed only
from that runner's *prior* attempts (a running total that excludes the current
attempt), so the model never peeks at the outcome it is trying to predict.

Statcast skill tables (sprint speed, pop time), if present, can be joined here.

    python -m src.features --steals data/sample/steals_2023.csv \
        --out data/sample/features_2023.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def _balls_strikes(count: str):
    if len(count) == 2 and count.isdigit():
        return int(count[0]), int(count[1])
    return "", ""


def build_features(steals_path: str):
    with open(steals_path, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # running (leakage-safe) tallies computed in chronological file order
    r_att = defaultdict(int)      # runner attempts so far
    r_succ = defaultdict(int)     # runner successes so far
    p_att = defaultdict(int)      # pitcher steal attempts allowed so far
    p_succ = defaultdict(int)     # of which successful (steals allowed)
    c_att = defaultdict(int)      # catcher attempts faced so far
    c_caught = defaultdict(int)   # of which caught

    out = []
    for r in rows:
        runner, pitcher, catcher = r["runner_id"], r["pitcher_id"], r["catcher_id"]
        balls, strikes = _balls_strikes(r["count"])
        success = int(r["success"])

        # prior rates (default to league-ish priors when no history yet)
        def rate(succ, att, prior):
            return (succ + prior * 5) / (att + 5)  # 5-attempt shrinkage to prior

        feat = {
            "runner_id": runner,
            "pitcher_id": pitcher,
            "catcher_id": catcher,
            "target_base": r["target_base"],
            "steal_of_second": int(r["target_base"] == "2"),
            "inning": r["inning"],
            "late_inning": int(int(r["inning"]) >= 7),
            "outs": r["outs"],
            "balls": balls,
            "strikes": strikes,
            "score_diff": r["score_diff"],
            "close_game": int(abs(int(r["score_diff"])) <= 1),
            "runner_bats_lhb": int(r["runner_bats"] == "L"),
            # leakage-safe prior skill estimates
            "runner_prior_sr": round(rate(r_succ[runner], r_att[runner], 0.75), 4),
            "runner_prior_att": r_att[runner],
            "pitcher_prior_sr_allowed": round(rate(p_succ[pitcher], p_att[pitcher], 0.75), 4),
            "catcher_prior_cs_rate": round(rate(c_caught[catcher], c_att[catcher], 0.20), 4),
            # label
            "success": success,
        }
        out.append(feat)

        # update running tallies AFTER emitting the row
        r_att[runner] += 1
        r_succ[runner] += success
        p_att[pitcher] += 1
        p_succ[pitcher] += success
        c_att[catcher] += 1
        c_caught[catcher] += (1 - success)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steals", default="data/sample/steals_2023.csv")
    ap.add_argument("--out", default="data/sample/features_2023.csv")
    args = ap.parse_args()

    feats = build_features(args.steals)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feats[0].keys()))
        w.writeheader()
        w.writerows(feats)
    print(f"wrote {len(feats)} feature rows -> {args.out}")


if __name__ == "__main__":
    main()
