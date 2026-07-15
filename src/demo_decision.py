"""
End-to-end demo of the decision layer: the REAL trained success-probability
model (src/train.py) feeding real predicted probabilities into a break-even
comparison to get a steal / don't-steal recommendation.

Uses RE24 (src/run_expectancy.py) for most of the game, but swaps to win
probability (src/win_probability.py) for high-leverage late-game situations
(7th inning or later, score within 3 runs -- see
win_probability.is_high_leverage) -- RE24 optimizes for total runs, which is
the wrong goal once the game is close to over: a run to tie is worth far more
than a run in a blowout, and a caught stealing that ends a trailing team's
last at-bat is a certain loss, not "0 more runs this inning."

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
    from .win_probability import build_win_prob, win_prob_break_even, is_high_leverage, MIN_CELL_N

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

    print(f"Building RE24 + win-probability tables from {', '.join(args.data_dirs)}...")
    re24 = build_re24(args.data_dirs)
    wp_table = build_win_prob(args.data_dirs)

    print(f"\nSteal-decision demo -- real model, real held-out attempts "
          f"({test['date'].min()} to {test['date'].max()})\n")

    # Bias the sample toward high-leverage situations too (rare in a plain
    # random sample -- most innings are 1-6), so the win-probability path
    # actually gets exercised here, not just the RE24 path.
    test["leverage"] = test.apply(
        lambda r: is_high_leverage(int(r["inning"]), int(r["score_diff"])), axis=1)
    hi = test[test["leverage"]].sample(n=min(args.n // 2, test["leverage"].sum()), random_state=7)
    lo = test[~test["leverage"]].sample(n=args.n - len(hi), random_state=7)
    sample = pd.concat([hi, lo]).sort_values("date")

    correct = 0
    for _, row in sample.iterrows():
        bc, outs, target = row["base_code"], int(row["outs"]), row["target_base"]
        inning, half, score_diff = int(row["inning"]), int(row["half"]), int(row["score_diff"])
        p = row["p_model"]

        if is_high_leverage(inning, score_diff):
            be, reward, cost, n, _ = win_prob_break_even(wp_table, inning, half, outs, bc, score_diff, target)
            layer = f"WP(n={n}{'*' if n < MIN_CELL_N else ''})"
        else:
            if (bc, outs) not in re24:
                continue  # exotic base state with too few historical plays to estimate
            d = should_steal(re24, bc, outs, target, p)
            be, reward, cost = d["break_even"], d["reward"], d["cost"]
            layer = "RE24"

        attempt = p > be
        verdict = "GO" if attempt else "HOLD"
        actual = "SAFE" if row["success"] == 1 else "CAUGHT"
        agree = (attempt and row["success"] == 1) or (not attempt and row["success"] == 0)
        correct += agree

        print(f"  {row['date']}  inning {inning} ({'bot' if half else 'top'})  "
              f"score {score_diff:+d}  bases {bc}  {outs} out  steal->{target}  "
              f"[{layer}]  P(model)={p:.0%}  break-even={be:.0%}  "
              f"=>  {verdict:4s}   actual: {actual:6s}  "
              f"{'(agrees)' if agree else '(would have been wrong here)'}")
    print("\n  (* = win-probability answer backed by a small sample -- low confidence)")

    print(f"\n  {correct}/{len(sample)} recommendations matched the actual outcome "
          f"(GO+safe or HOLD+caught) in this sample.")
    print("  This is illustrative, not a backtest -- a real backtest needs to "
          "run the FULL held-out set and compare against what the actual "
          "in-game decision was (every row here IS an attempt that "
          "happened, so we can't see what a HOLD call would have avoided; "
          "we can only see whether GO calls would have paid off).")


if __name__ == "__main__":
    main()
