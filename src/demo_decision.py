"""
End-to-end demo of the decision layer: the REAL trained success-probability
model (src/train.py) feeding real predicted probabilities into the RE24
break-even comparison (src/run_expectancy.py) to get a steal / don't-steal
recommendation.

Fits on the same train split src/train.py uses (earliest 80% of dates), then
walks a handful of real HELD-OUT test-set attempts spanning different
situations -- so every recommendation shown here is the model's honest,
out-of-sample call, and we can see whether "the model said GO" lines up with
what actually happened.

    python -m src.demo_decision
    python -m src.demo_decision --model logistic
"""
from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/sample/features_2023_2025.csv")
    ap.add_argument("--data-dirs", nargs="+",
                    default=["data/retrosheet_2023", "data/retrosheet_2024",
                             "data/retrosheet_2025"])
    ap.add_argument("--model", choices=["logistic", "xgboost"], default="xgboost")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("-n", type=int, default=12, help="how many test-set attempts to show")
    args = ap.parse_args()

    import pandas as pd

    from .train import NUMERIC, fit_logistic, fit_xgboost
    from .run_expectancy import build_re24, should_steal

    df = pd.read_csv(args.features)
    for col in ("runner_sprint_speed", "runner_age", "catcher_pop_time"):
        df[col] = df[col].fillna(df[col].median())

    split_idx = int(len(df) * (1 - args.test_frac))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:].copy()
    X_tr, y_tr = train[NUMERIC].fillna(0.0), train["success"].astype(int)
    X_te = test[NUMERIC].fillna(0.0)

    print(f"Fitting {args.model} on {len(train)} train rows "
          f"({train['date'].min()} to {train['date'].max()})...")
    model = fit_logistic(X_tr, y_tr) if args.model == "logistic" else fit_xgboost(X_tr, y_tr)
    test["p_model"] = model.predict_proba(X_te)[:, 1]

    print(f"Building RE24 from {', '.join(args.data_dirs)}...")
    re24 = build_re24(args.data_dirs)

    print(f"\nSteal-decision demo -- real model, real held-out attempts "
          f"({test['date'].min()} to {test['date'].max()})\n")

    # A spread of situations: some the model liked, some it didn't, some
    # double steals, some with a runner also on 3rd -- not cherry-picked for
    # correctness, just for variety. Sorted by date for a natural walk.
    sample = test.sample(n=min(args.n, len(test)), random_state=7).sort_values("date")

    correct = 0
    net_value_actual = 0.0
    for _, row in sample.iterrows():
        bc, outs, target = row["base_code"], int(row["outs"]), row["target_base"]
        if (bc, outs) not in re24:
            continue  # exotic base state with too few historical plays to estimate
        d = should_steal(re24, bc, outs, target, row["p_model"])
        verdict = "GO" if d["attempt"] else "HOLD"
        actual = "SAFE" if row["success"] == 1 else "CAUGHT"
        agree = (d["attempt"] and row["success"] == 1) or (not d["attempt"] and row["success"] == 0)
        correct += agree
        # net run value the recommendation would have banked THIS TIME, using
        # the actual outcome rather than the probability (a real backtest
        # would aggregate this properly -- see the note below).
        realized = d["reward"] if row["success"] == 1 else -d["cost"]
        net_value_actual += realized if d["attempt"] else 0.0

        print(f"  {row['date']}  bases {bc}  {outs} out  steal->{target}  "
              f"P(model)={row['p_model']:.0%}  break-even={d['break_even']:.0%}  "
              f"=>  {verdict:4s}   actual: {actual:6s}  "
              f"{'(agrees)' if agree else '(would have been wrong here)'}")

    print(f"\n  {correct}/{len(sample)} recommendations matched the actual outcome "
          f"(GO+safe or HOLD+caught) in this sample.")
    print("  This is illustrative, not a backtest -- a real backtest needs to "
          "run the FULL held-out set and compare against what the actual "
          "in-game decision was (every row here IS an attempt that "
          "happened, so we can't see what a HOLD call would have avoided; "
          "we can only see whether GO calls would have paid off).")


if __name__ == "__main__":
    main()
