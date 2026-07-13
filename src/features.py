"""
Leakage-safe, multi-season feature table for the steal-success model.

Combines the parsed steal-attempt files for the given `--seasons` into one
chronologically-ordered table, then:

  * computes running runner/pitcher/catcher success-rate priors from
    PRIOR attempts only (across season boundaries, not just within one
    season), so the model never peeks at the outcome it's predicting.
  * merges in every plate appearance from `battinglines_<year>.csv`
    (written by `retrosheet_parser.py --batting-out`) to compute a
    leakage-safe running AVG/OBP/SLG/HR% for the batter at the plate
    during each steal attempt -- NOT season-level stats, which would leak
    a batter's September numbers into an April prediction. Steal rows and
    batting rows are interleaved using an exact per-game play sequence
    number (`play_seq`, from the parser) rather than (inning, outs) alone,
    since out count can repeat many times within one half-inning across
    different plate appearances and isn't precise enough on its own.
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


def _load_batting_rows(paths: list) -> list:
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def build_features(steals_paths: list, statcast_dir: str,
                   batting_paths: list = None) -> list:
    crosswalk = load_crosswalk(os.path.join(statcast_dir, "id_crosswalk.csv"))

    steal_rows = []
    for path in steals_paths:
        season = _season_of(path)
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                r["season"] = season
                steal_rows.append(r)

    batting_rows = _load_batting_rows(batting_paths or [])

    # Merge into one chronological walk. Steal rows are listed BEFORE
    # batting rows in this pre-sort order so that, when both share the
    # exact same (date, game_id, play_seq) -- i.e. they're the SAME play,
    # like "K+SB2" (a strikeout and a steal attempt together) -- Python's
    # stable sort keeps the steal row first: we read the batter's prior
    # stats (not yet including this at-bat's own outcome) before applying
    # this play's own batting-outcome update.
    combined = [("steal", r) for r in steal_rows] + [("bat", r) for r in batting_rows]
    combined.sort(key=lambda item: (item[1]["date"], item[1]["game_id"],
                                    int(item[1]["play_seq"])))

    sprint_tables, age_tables, pop_tables = {}, {}, {}

    def sprint_for(season):
        if season not in sprint_tables:
            path = os.path.join(statcast_dir, f"sprint_speed_{season}.csv")
            sprint_tables[season] = load_skill_table(path, "player_id", "sprint_speed")
        return sprint_tables[season]

    def age_for(season):
        if season not in age_tables:
            path = os.path.join(statcast_dir, f"sprint_speed_{season}.csv")
            age_tables[season] = load_skill_table(path, "player_id", "age")
        return age_tables[season]

    def pop_for(season):
        if season not in pop_tables:
            path = os.path.join(statcast_dir, f"catcher_poptime_{season}.csv")
            pop_tables[season] = load_skill_table(path, "entity_id", "pop_2b_sba")
        return pop_tables[season]

    # running (leakage-safe) tallies computed in chronological order
    r_att, r_succ = defaultdict(int), defaultdict(int)
    p_att, p_succ = defaultdict(int), defaultdict(int)
    c_att, c_caught = defaultdict(int), defaultdict(int)
    # batter tallies: at-bats, hits, walks, hit-by-pitch, sac flies, total
    # bases, home runs, plate appearances (for HR%'s denominator)
    b_ab, b_hit = defaultdict(int), defaultdict(int)
    b_bb, b_hbp, b_sf = defaultdict(int), defaultdict(int), defaultdict(int)
    b_bases, b_hr, b_pa = defaultdict(int), defaultdict(int), defaultdict(int)

    def rate(succ, att, prior):
        return (succ + prior * 5) / (att + 5)  # 5-attempt shrinkage to prior

    # Shrinkage toward modern-era league-average rates; batting stats need
    # a bigger sample than steal attempts to stabilize, so a larger
    # shrinkage constant than the "+5" above.
    BAT_SHRINK = 50

    def batter_rate(num, denom, prior):
        return (num + prior * BAT_SHRINK) / (denom + BAT_SHRINK)

    out = []
    for kind, r in combined:
        if kind == "bat":
            batter = r["batter_id"]
            ab, hit, bases = int(r["ab"] == "True"), int(r["hit"] == "True"), int(r["bases"])
            bb, hbp, sf = int(r["bb"] == "True"), int(r["hbp"] == "True"), int(r["sf"] == "True")
            b_ab[batter] += ab
            b_hit[batter] += hit
            b_bb[batter] += bb
            b_hbp[batter] += hbp
            b_sf[batter] += sf
            b_bases[batter] += bases
            b_hr[batter] += int(hit and bases == 4)
            b_pa[batter] += ab + bb + hbp + sf
            continue

        runner, pitcher, catcher = r["runner_id"], r["pitcher_id"], r["catcher_id"]
        batter = r["batter_id"]
        season = r["season"]
        balls, strikes = _balls_strikes(r["count"])
        success = int(r["success"])

        runner_mlbam = crosswalk.get(runner)
        catcher_mlbam = crosswalk.get(catcher)
        sprint_speed = sprint_for(season).get(runner_mlbam) if runner_mlbam else None
        runner_age = age_for(season).get(runner_mlbam) if runner_mlbam else None
        pop_time = pop_for(season).get(catcher_mlbam) if catcher_mlbam else None

        batter_pa = b_ab[batter] + b_bb[batter] + b_hbp[batter] + b_sf[batter]

        feat = {
            "season": season,
            "date": r["date"],
            "runner_id": runner,
            "pitcher_id": pitcher,
            "catcher_id": catcher,
            "batter_id": batter,
            "target_base": r["target_base"],
            # steal_of_second is the implicit baseline; third/home get their
            # own dummies since their success rates are wildly different
            # (2nd ~79%, 3rd ~82%, home ~42% -- see notebooks/eda.ipynb).
            "steal_of_third": int(r["target_base"] == "3"),
            "steal_of_home": int(r["target_base"] == "H"),
            "inning": r["inning"],
            "late_inning": int(int(r["inning"]) >= 7),
            "outs": r["outs"],
            "balls": balls,
            "strikes": strikes,
            "score_diff": r["score_diff"],
            "close_game": int(abs(int(r["score_diff"])) <= 1),
            "runner_bats_lhb": int(r["runner_bats"] == "L"),
            "pitcher_throws_lhp": int(r["pitcher_throws"] == "L"),
            # >1 runner moving on the same pitch -- essentially guaranteed
            # safe in the historical data (see notebooks/eda.ipynb, section
            # 10.5): 100% success across 2023-2025, since the defense can
            # only really contest one of the runners.
            "is_double_steal": int(r["double_steal"]),
            # A runner ALSO on 3rd during a steal of 2nd (single-runner, not
            # a double steal) -- catchers are reluctant to risk a wild throw
            # letting that run score, so they often concede the steal.
            # 76.7% -> 91.5% success in the data, holds (and strengthens)
            # across every out count (see notebooks/eda.ipynb, section 10.6).
            # Only meaningful for target_base "2"/"3" -- for "H" the runner
            # on 3rd IS the one stealing, not another runner.
            "runner_on_third": int(bool(r["on_3b"]) and r["target_base"] != "H"),
            # leakage-safe prior skill estimates (carry across seasons)
            "runner_prior_sr": round(rate(r_succ[runner], r_att[runner], 0.75), 4),
            "runner_prior_att": r_att[runner],
            "pitcher_prior_sr_allowed": round(rate(p_succ[pitcher], p_att[pitcher], 0.75), 4),
            "catcher_prior_cs_rate": round(rate(c_caught[catcher], c_att[catcher], 0.20), 4),
            # leakage-safe running batter offense (the person AT THE PLATE
            # during the steal, not the runner) -- season-level stats would
            # leak a batter's September numbers into an April prediction,
            # so this only ever reflects plate appearances strictly before
            # this one, shrunk toward modern-era league-average rates.
            "batter_prior_avg": round(batter_rate(b_hit[batter], b_ab[batter], 0.248), 4),
            "batter_prior_obp": round(batter_rate(
                b_hit[batter] + b_bb[batter] + b_hbp[batter],
                b_ab[batter] + b_bb[batter] + b_hbp[batter] + b_sf[batter], 0.317), 4),
            "batter_prior_slg": round(batter_rate(b_bases[batter], b_ab[batter], 0.410), 4),
            "batter_prior_hr_pct": round(batter_rate(b_hr[batter], batter_pa, 0.030), 4),
            "batter_prior_pa": batter_pa,
            # Statcast join (season-matched, missing-flagged rather than guessed)
            "runner_sprint_speed": sprint_speed if sprint_speed is not None else "",
            "runner_sprint_speed_missing": int(sprint_speed is None),
            "runner_age": runner_age if runner_age is not None else "",
            "runner_age_missing": int(runner_age is None),
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

    steals_paths = [os.path.join(args.steals_dir, f"steals_{y}.csv") for y in args.seasons]
    missing = [p for p in steals_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing parsed season file(s): {missing}")
    batting_paths = [os.path.join(args.steals_dir, f"battinglines_{y}.csv") for y in args.seasons]

    feats = build_features(steals_paths, args.statcast_dir, batting_paths)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(feats[0].keys()))
        w.writeheader()
        w.writerows(feats)

    n_missing_sprint = sum(f["runner_sprint_speed_missing"] for f in feats)
    n_missing_pop = sum(f["catcher_pop_time_missing"] for f in feats)
    n_missing_batting = sum(f["batter_prior_pa"] == 0 for f in feats)
    print(f"wrote {len(feats)} feature rows from {len(steals_paths)} seasons -> {args.out}")
    print(f"missing sprint speed: {n_missing_sprint}/{len(feats)}  "
          f"missing pop time: {n_missing_pop}/{len(feats)}")
    print(f"zero prior batting PA (cold start): {n_missing_batting}/{len(feats)}")


if __name__ == "__main__":
    main()
