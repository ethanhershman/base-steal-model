"""
Full backtest of the decision layer -- ROADMAP.md's "Step 5": run EVERY
held-out test-set steal attempt through the decision layer and total up
the run/win value the model's GO/HOLD calls would have produced, against
what actually happened.

src/demo_decision.py already walks the model + break-even comparison over
a handful of real held-out attempts, but only prints ~12 of them and says
outright that it "is illustrative, not a backtest." The one thing this
module adds is aggregation over the FULL test set, using the fact that
every row in it already IS a real historical attempt (some team actually
sent the runner):

  * "what actually happened" is just the sum of every attempt's realized
    reward (safe) or -cost (caught) -- the real, historical policy.
  * "what the model would have done" sums the SAME realized reward/cost,
    but only over rows the model would have called GO. A HOLD call is
    scored as exactly 0 -- not a guess, but the break-even framework's own
    zero reference point (break-even is defined as the probability where
    EV(steal) == EV(hold) == 0 relative value; see run_expectancy.py /
    win_probability.py).

This can't be collapsed into one number: RE24 rows are denominated in
runs, win-probability rows in win probability -- different currencies for
different situations (early/low-leverage vs. late/close), so the two are
reported separately, never summed together.

Running EVERY row (not a hand-picked sample) surfaced a real bug the first
time this was written: double/triple steals, where the target base is
already occupied by another runner breaking on the same pitch, produced
nonsensical rewards/costs (break-even above 100%, negative rewards on
successful steals) because run_expectancy._state_after_success only
tracked one runner's movement. Fixed at the source (see that function's
_cascade_free helper) rather than worked around here.

Honest limits on what this can and can't show (same ones demo_decision.py
already calls out): every row here is an attempt that happened, so we can
directly check whether the model's GO calls would have paid off and
whether its HOLD calls would have avoided real caught-stealings. We can't
see the flip side -- situations where nobody attempted a steal at all, so
whether the model would have (correctly or incorrectly) sent a runner who
in reality stayed put is fundamentally unobservable from this data.

    python -m src.backtest
    python -m src.backtest --model logistic --out data/sample/backtest_2025.csv
"""
from __future__ import annotations

import argparse

from .run_expectancy import build_re24, break_even_rate
from .win_probability import (
    build_win_prob, win_prob_break_even, is_high_leverage, MIN_CELL_N,
    LEGACY_SEASONS, MODERN_SEASONS, POST_RULE_CHANGE_SEASONS, _season_dirs,
)
from .train import NUMERIC, fit_logistic, fit_xgboost

CURRENCY = {"RE24": "runs"}  # anything not "RE24" is win-probability-denominated


def decide(re24, wp_table, wp_hold_table, *, inning, half, outs, base_code,
          score_diff, target, p_model):
    """One situation's break-even + GO/HOLD call, reused by both the
    aggregate backtest below and predict.py's per-request path -- same
    RE24-first-else-win-probability logic (see predict.predict_steal_decision
    for why: high-leverage late/close games swap to win probability, and any
    (base_code, outs) RE24 has no data for falls through to win probability
    too rather than raising). Double/triple steals (the target base already
    occupied by another runner breaking on the same pitch) are handled by
    run_expectancy._state_after_success's cascade, not specially here.
    """
    high_leverage = is_high_leverage(inning, score_diff)
    min_n = None
    if not high_leverage and (base_code, outs) in re24:
        be, reward, cost = break_even_rate(re24, base_code, outs, target)
        layer = "RE24"
    else:
        be, reward, cost, min_n, _ = win_prob_break_even(
            wp_table, wp_hold_table, inning, half, outs, base_code, score_diff, target)
        layer = "WP" if high_leverage else "WP (RE24 had no data)"
    return {
        "layer": layer, "break_even": be, "reward": reward, "cost": cost,
        "decision": "GO" if p_model > be else "HOLD", "min_n": min_n,
    }


def run_backtest(test_df, model, re24, wp_table, wp_hold_table):
    """Walk every row of test_df (the FULL held-out set, not a sample --
    the whole point of a real backtest) through decide(...) and attach each
    row's break-even, GO/HOLD call, and realized value (reward if the real
    attempt succeeded, -cost if it was caught).

    Returns a pandas.DataFrame, one row per attempt, ready for summarize()
    or for saving to CSV for further inspection.
    """
    import pandas as pd

    X_te = test_df[NUMERIC].fillna(0.0)
    p = model.predict_proba(X_te)[:, 1]

    records = []
    for (_, row), p_model in zip(test_df.iterrows(), p):
        d = decide(re24, wp_table, wp_hold_table,
                  inning=int(row["inning"]), half=int(row["half"]), outs=int(row["outs"]),
                  base_code=row["base_code"], score_diff=int(row["score_diff"]),
                  target=row["target_base"], p_model=float(p_model))
        actual_value = d["reward"] if row["success"] == 1 else -d["cost"]
        records.append({
            "date": row["date"], "inning": row["inning"], "half": row["half"],
            "outs": row["outs"], "base_code": row["base_code"],
            "score_diff": row["score_diff"], "target": row["target_base"],
            "p_model": p_model, "success": int(row["success"]),
            "layer": d["layer"], "break_even": d["break_even"],
            "decision": d["decision"], "actual_value": actual_value,
            "min_n": d["min_n"],
        })
    return pd.DataFrame.from_records(records)


def summarize(results) -> None:
    """Print the aggregate comparison: real historical policy vs. the
    model's selective policy, split by currency (RE24 runs vs. win
    probability -- see module docstring for why these can't be combined).
    """
    print(f"Backtest over {len(results)} real held-out steal attempts "
         f"({results['date'].min()} to {results['date'].max()})")
    print("\nattempts by layer:")
    print(results["layer"].value_counts().to_string())

    groups = [
        ("RE24 (early/low-leverage game states)", results["layer"] == "RE24", "runs"),
        ("Win probability (high-leverage, or RE24 had no data)", results["layer"] != "RE24", "win-prob"),
    ]
    for name, mask, unit in groups:
        g = results[mask]
        if g.empty:
            continue
        n = len(g)
        go, hold = g[g["decision"] == "GO"], g[g["decision"] == "HOLD"]
        baseline_total = g["actual_value"].sum()
        model_total = go["actual_value"].sum()

        print(f"\n{name} -- {n} real attempts")
        print(f"  model calls GO on {len(go)} ({len(go)/n:.0%}), HOLD on {len(hold)} ({len(hold)/n:.0%})")
        if len(go):
            print(f"  actual success rate on GO calls  : {go['success'].mean():.1%}")
        if len(hold):
            print(f"  actual success rate on HOLD calls: {hold['success'].mean():.1%}")
            avoided, missed = (hold["success"] == 0).sum(), (hold["success"] == 1).sum()
            print(f"  of the {len(hold)} attempts the model would have held: "
                 f"{avoided} were actually caught (avoided), "
                 f"{missed} were actually safe (missed opportunity)")
        print(f"  total realized value, ACTUAL historical policy (every real attempt): "
             f"{baseline_total:+.2f} {unit}  ({baseline_total/n:+.4f}/attempt)")
        print(f"  total realized value, MODEL policy (HOLD scored as 0)        : "
             f"{model_total:+.2f} {unit}  ({model_total/n:+.4f}/attempt)")
        print(f"  model vs. actual, per real attempt: {(model_total/n) - (baseline_total/n):+.4f} {unit}/attempt")

        low_conf = g["min_n"].apply(lambda x: x is not None and x < MIN_CELL_N).sum()
        if low_conf:
            print(f"  ({low_conf}/{n} rows here rest on a thin win-probability sample, "
                 f"n<{MIN_CELL_N} -- treat with caution)")

    print("\nWhat this does and doesn't show: every row above is an attempt that "
         "really happened, so the GO-call success rate and the caught-stealings "
         "avoided by HOLD calls are real, checkable outcomes. What it can't show "
         "is the reverse case -- situations where no one attempted a steal at "
         "all, so whether the model would have (correctly or not) sent a "
         "runner who in reality held. That side is unobservable from this data.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/sample/features_2023_2025.csv")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--model", choices=["logistic", "xgboost"], default="xgboost")
    ap.add_argument("--re24-data-dirs", nargs="+", default=_season_dirs(POST_RULE_CHANGE_SEASONS))
    ap.add_argument("--wp-data-dirs", nargs="+", default=_season_dirs(POST_RULE_CHANGE_SEASONS))
    ap.add_argument("--wp-hold-data-dirs", nargs="+",
                    default=_season_dirs(LEGACY_SEASONS) + _season_dirs(MODERN_SEASONS))
    ap.add_argument("--wp-legacy-dirs", nargs="+", default=_season_dirs(LEGACY_SEASONS))
    ap.add_argument("--out", default=None, help="optional path to write the full per-attempt CSV")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.features)
    for col in ("runner_sprint_speed", "runner_age", "catcher_pop_time"):
        df[col] = df[col].fillna(df[col].median())

    split_idx = int(len(df) * (1 - args.test_frac))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:].copy()
    X_tr, y_tr = train[NUMERIC].fillna(0.0), train["success"].astype(int)

    print(f"Fitting {args.model} on {len(train)} train rows "
         f"({train['date'].min()} to {train['date'].max()})...")
    model = fit_logistic(X_tr, y_tr) if args.model == "logistic" else fit_xgboost(X_tr, y_tr)

    print(f"Building RE24 from {', '.join(args.re24_data_dirs)}...")
    re24 = build_re24(args.re24_data_dirs)
    print(f"Building win-probability table from {', '.join(args.wp_data_dirs)}...")
    wp_table = build_win_prob(args.wp_data_dirs)
    print(f"Building hold-only baseline from {', '.join(args.wp_hold_data_dirs)}...")
    wp_hold_table = build_win_prob(args.wp_hold_data_dirs, hold_only=True, legacy_dirs=args.wp_legacy_dirs)

    print(f"\nRunning the backtest over the full held-out test set "
         f"({test['date'].min()} to {test['date'].max()}, {len(test)} attempts)...\n")
    results = run_backtest(test, model, re24, wp_table, wp_hold_table)

    if args.out:
        results.to_csv(args.out, index=False)
        print(f"wrote {args.out}\n")

    summarize(results)


if __name__ == "__main__":
    main()
