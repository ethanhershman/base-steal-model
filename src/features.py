"""
Leakage-safe, multi-season feature table for the steal-success model.

Combines the parsed steal-attempt files for the given `--seasons` into one
chronologically-ordered table, then:

  * computes running runner/pitcher/catcher success-rate priors from
    PRIOR attempts only (across season boundaries, not just within one
    season), so the model never peeks at the outcome it's predicting.
  * joins Statcast skill data (runner sprint speed, catcher pop time) for
    the matching season, via the Retrosheet<->MLBAM id crosswalk built by
    `src/id_crosswalk.py`. Rows for players with no Statcast row get a
    missing-value flag rather than a guessed number.

Default seasons are 2023-2025 only. 2021-2022 are excluded: MLB's 2023
rule changes (bigger bases, limited pickoff attempts) measurably changed
stolen-base success rates (see notebooks/eda.ipynb, section 10), so
pre-2023 attempts aren't drawn from the same underlying distribution the
model needs to predict. steals_2021.csv / steals_2022.csv are still parsed
and kept on disk for that historical comparison, just not fed into
training.

    python -m src.features --out data/sample/features_2023_2025.csv
"""
from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict


def load_crosswalk(path: str) -> dict:
    retro_to_mlbam = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            retro_to_mlbam[row["key_retro"]] = int(row["key_mlbam"])
    return retro_to_mlbam


def load_skill_table(path: str, id_col: str, value_col: str) -> dict:
    if not os.path.exists(path):
        return {}
    table = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            v = row.get(value_col)
            if v:
                table[int(row[id_col])] = float(v)
    return table


def _balls_strikes(count: str):
    if len(count) == 2 and count.isdigit():
        return int(count[0]), int(count[1])
    return "", ""


def _season_of(path: str) -> str:
    return os.path.basename(path).split("_")[-1].replace(".csv", "")


def build_features(steals_paths: list, statcast_dir: str) -> list:
    crosswalk = load_crosswalk(os.path.join(statcast_dir, "id_crosswalk.csv"))

    rows = []
    for path in steals_paths:
        season = _season_of(path)
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                r["season"] = season
                rows.append(r)

    # Chronological order across seasons, so running priors never leak from
    # a later attempt (or a later season) into an earlier one.
    rows.sort(key=lambda r: (r["date"], r["game_id"], int(r["inning"]),
                             int(r["outs"])))

    sprint_tables, pop_tables = {}, {}

    def sprint_for(season):
        if season not in sprint_tables:
            path = os.path.join(statcast_dir, f"sprint_speed_{season}.csv")
            sprint_tables[season] = load_skill_table(path, "player_id", "sprint_speed")
        return sprint_tables[season]

    def pop_for(season):
        if season not in pop_tables:
            path = os.path.join(statcast_dir, f"catcher_poptime_{season}.csv")
            pop_tables[season] = load_skill_table(path, "entity_id", "pop_2b_sba")
        return pop_tables[season]

    # running (leakage-safe) tallies computed in chronological order
    r_att, r_succ = defaultdict(int), defaultdict(int)
    p_att, p_succ = defaultdict(int), defaultdict(int)
    c_att, c_caught = defaultdict(int), defaultdict(int)

    def rate(succ, att, prior):
        return (succ + prior * 5) / (att + 5)  # 5-attempt shrinkage to prior

    out = []
    for r in rows:
        runner, pitcher, catcher = r["runner_id"], r["pitcher_id"], r["catcher_id"]
        season = r["season"]
        balls, strikes = _balls_strikes(r["count"])
        success = int(r["success"])

        runner_mlbam = crosswalk.get(runner)
        catcher_mlbam = crosswalk.get(catcher)
        sprint_speed = sprint_for(season).get(runner_mlbam) if runner_mlbam else None
        pop_time = pop_for(season).get(catcher_mlbam) if catcher_mlbam else None

        feat = {
            "season": season,
            "date": r["date"],
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
            "pitcher_throws_lhp": int(r["pitcher_throws"] == "L"),
            # leakage-safe prior skill estimates (carry across seasons)
            "runner_prior_sr": round(rate(r_succ[runner], r_att[runner], 0.75), 4),
            "runner_prior_att": r_att[runner],
            "pitcher_prior_sr_allowed": round(rate(p_succ[pitcher], p_att[pitcher], 0.75), 4),
            "catcher_prior_cs_rate": round(rate(c_caught[catcher], c_att[catcher], 0.20), 4),
            # Statcast join (season-matched, missing-flagged rather than guessed)
            "runner_sprint_speed": sprint_speed if sprint_speed is not None else "",
            "runner_sprint_speed_missing": int(sprint_speed is None),
            "catcher_pop_time": pop_time if pop_time is not None else "",
            "catcher_pop_time_missing": int(pop_time is None),
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
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025],
                    help="post rule-change seasons to include (2021-2022 predate "
                         "the bigger-base/pickoff-limit rules and are excluded "
                         "by default)")
    ap.add_argument("--steals-dir", default="data/sample")
    ap.add_argument("--statcast-dir", default="data/statcast")
    ap.add_argument("--out", default="data/sample/features_2023_2025.csv")
    args = ap.parse_args()

    paths = [os.path.join(args.steals_dir, f"steals_{y}.csv") for y in args.seasons]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing parsed season file(s): {missing}")
    feats = build_features(paths, args.statcast_dir)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feats[0].keys()))
        w.writeheader()
        w.writerows(feats)

    n_missing_sprint = sum(f["runner_sprint_speed_missing"] for f in feats)
    n_missing_pop = sum(f["catcher_pop_time_missing"] for f in feats)
    print(f"wrote {len(feats)} feature rows from {len(paths)} seasons -> {args.out}")
    print(f"missing sprint speed: {n_missing_sprint}/{len(feats)}  "
          f"missing pop time: {n_missing_pop}/{len(feats)}")


if __name__ == "__main__":
    main()
