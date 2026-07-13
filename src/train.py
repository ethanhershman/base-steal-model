"""
Baseline steal-success probability model.

Logistic regression: interpretable, naturally outputs a probability, and a
good gut-check that the model is learning real baseball (faster runners and
lefty pitchers should visibly move the odds) before reaching for gradient
boosting. Evaluated with the metrics that matter for a decision model: log
loss, Brier score, AUC, and a calibration table.

Split TEMPORALLY (train on earlier seasons, test on the held-out latest
season) rather than randomly -- a random split would let the model learn
runner/pitcher/catcher priors from games that happened after the ones it's
being tested on, which isn't available in real use. See ROADMAP.md,
"Validate across time".

    python -m src.train --features data/sample/features_2023_2025.csv
"""
from __future__ import annotations

import argparse


NUMERIC = [
    "steal_of_second", "late_inning", "outs", "balls", "strikes",
    "score_diff", "close_game", "runner_bats_lhb", "pitcher_throws_lhp",
    "runner_prior_sr", "runner_prior_att",
    "pitcher_prior_sr_allowed", "catcher_prior_cs_rate",
    "runner_sprint_speed", "runner_sprint_speed_missing",
    "catcher_pop_time", "catcher_pop_time_missing",
]


def calibration_table(y_true, p, n_bins=10):
    import pandas as pd

    df = pd.DataFrame({"y": y_true.values, "p": p})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/sample/features_2023_2025.csv")
    ap.add_argument("--test-season", type=int, default=2025,
                    help="held-out season; everything earlier is training data")
    args = ap.parse_args()

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

    df = pd.read_csv(args.features)

    # Sprint speed / pop time are ~99.8% present; median-fill the rare
    # misses so a handful of rows don't get a nonsensical 0 ft/s runner.
    for col in ("runner_sprint_speed", "catcher_pop_time"):
        df[col] = df[col].fillna(df[col].median())

    train = df[df["season"] < args.test_season]
    test = df[df["season"] == args.test_season]
    if len(test) == 0:
        raise SystemExit(f"no rows for test season {args.test_season} in {args.features}")

    X_tr, y_tr = train[NUMERIC].fillna(0.0), train["success"].astype(int)
    X_te, y_te = test[NUMERIC].fillna(0.0), test["success"].astype(int)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]

    print(f"Baseline logistic regression (train seasons < {args.test_season}, "
          f"test season {args.test_season})")
    print(f"  train rows: {len(X_tr)}   test rows: {len(X_te)}")
    print(f"  log loss : {log_loss(y_te, p):.4f}")
    print(f"  brier    : {brier_score_loss(y_te, p):.4f}")
    print(f"  auc      : {roc_auc_score(y_te, p):.4f}")
    base_rate = y_te.mean()
    print(f"  base rate: {base_rate:.4f} (predict-the-mean brier = "
          f"{base_rate * (1 - base_rate):.4f})")

    print("\n  coefficient sanity check (sign should make baseball sense):")
    for name, coef in sorted(zip(NUMERIC, model.coef_[0]),
                             key=lambda kv: -abs(kv[1])):
        print(f"    {name:28s} {coef:+.3f}")

    print("\n  calibration (predicted vs. actual success rate by decile):")
    print(calibration_table(y_te, p).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
