"""
Steal-success probability models: logistic regression baseline + XGBoost.

Logistic regression is interpretable and a good gut-check that the model is
learning real baseball (faster runners and lefty pitchers should visibly
move the odds). Gradient boosting (XGBoost) almost always beats it on
tabular data like this and captures interactions a linear model can't --
e.g. a fast runner against a slow lefty with a weak-armed catcher compounds
in a way logistic regression misses. Per ROADMAP.md, XGBoost becomes the
production model once it's shown to beat the baseline on log loss/Brier.

Both are evaluated identically: log loss, Brier score, AUC, and a
calibration table.

Split TEMPORALLY (train on the earlier X% of dates, test on the later Y%)
rather than randomly -- a random split would let the model train on rows
that happened, in real life, after some of the rows it's being tested on.
features.py already writes rows in chronological order (date, game_id,
inning, outs), so a date-based cutoff is just "first N% of rows train,
rest test" -- no future data ever ends up in training. See ROADMAP.md,
"Validate across time".

    python -m src.train --features data/sample/features_2023_2025.csv
"""
from __future__ import annotations

import argparse


NUMERIC = [
    "steal_of_third", "steal_of_home", "is_double_steal", "runner_on_third",
    "late_inning", "outs", "balls", "strikes",
    "score_diff", "close_game", "runner_bats_lhb", "pitcher_throws_lhp",
    "runner_prior_sr", "runner_prior_att",
    "pitcher_prior_sr_allowed", "catcher_prior_cs_rate",
    "runner_sprint_speed", "runner_sprint_speed_missing",
    "runner_age", "runner_age_missing",
    "catcher_pop_time", "catcher_pop_time_missing",
]


def calibration_table(y_true, p, n_bins=10):
    import pandas as pd

    df = pd.DataFrame({"y": y_true.values, "p": p})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"),
    )


def fit_logistic(X_tr, y_tr):
    from sklearn.linear_model import LogisticRegression

    # is_double_steal is ~100% success with zero counterexamples in the
    # training data (a near-perfectly-separating feature), which makes the
    # optimizer take much longer to settle than a normal feature would --
    # bump max_iter rather than let it silently stop early.
    model = LogisticRegression(max_iter=5000)
    model.fit(X_tr, y_tr)
    return model


def fit_xgboost(X_tr, y_tr, val_frac=0.15):
    from xgboost import XGBClassifier

    # Carve a validation slice off the END of the (already chronological)
    # training set to FIND how many boosting rounds is right, without
    # overfitting the noisy per-attempt outcome. But then refit on the FULL
    # training set for that many rounds -- using early stopping's model
    # directly throws away 15% of training data for no benefit once the
    # round count is already chosen, and on this data that gap was large
    # enough to flip XGBoost from beating logistic regression to losing to
    # it (log loss 0.4869 held-out-fit vs 0.4847 refit-on-full, at the same
    # early-stopped round count).
    val_idx = int(len(X_tr) * (1 - val_frac))
    X_fit, X_val = X_tr.iloc[:val_idx], X_tr.iloc[val_idx:]
    y_fit, y_val = y_tr.iloc[:val_idx], y_tr.iloc[val_idx:]

    params = dict(
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    probe = XGBClassifier(n_estimators=300, eval_metric="logloss",
                          early_stopping_rounds=30, **params)
    probe.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    best_n = probe.best_iteration
    print(f"  (early stopping on a validation slice picked {best_n} rounds; "
          f"refitting on the full training set with that round count)")

    model = XGBClassifier(n_estimators=best_n, **params)
    model.fit(X_tr, y_tr)
    return model


def feature_ranking(model, model_name):
    if model_name == "logistic":
        return sorted(zip(NUMERIC, model.coef_[0]), key=lambda kv: -abs(kv[1]))
    return sorted(zip(NUMERIC, model.feature_importances_), key=lambda kv: -kv[1])


def confusion_at(y_true, p, threshold):
    from sklearn.metrics import confusion_matrix

    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "specificity": specificity}


def print_diagnostics(test_df, y_te, p):
    import pandas as pd

    base_rate = y_te.mean()
    print("\n  confusion matrix at two thresholds:")
    print("  ('success' is the positive class; steal success is a high "
          "base-rate event, so 0.5 is not a meaningful cutoff here.)")
    for label, thresh in (("0.5 (standard)", 0.5), (f"{base_rate:.3f} (test base rate)", base_rate)):
        c = confusion_at(y_te, p, thresh)
        print(f"    threshold {label}:")
        print(f"      TP={c['tp']:4d}  FP={c['fp']:4d}  TN={c['tn']:4d}  FN={c['fn']:4d}  "
              f"precision={c['precision']:.3f}  recall={c['recall']:.3f}  "
              f"specificity={c['specificity']:.3f}")

    scored = test_df.copy()
    scored["p"] = p
    id_cols = ["runner_id", "pitcher_id", "catcher_id"]
    skill_cols = ["runner_sprint_speed", "catcher_pop_time",
                 "runner_prior_sr", "pitcher_prior_sr_allowed"]
    cols = id_cols + ["p", "success"] + skill_cols

    print("\n  most confident WRONG -- predicted high success, actually CAUGHT:")
    print(scored[scored["success"] == 0].sort_values("p", ascending=False)
          .head(5)[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n  most confident WRONG -- predicted low success, actually SAFE:")
    print(scored[scored["success"] == 1].sort_values("p")
          .head(5)[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print("\n  reading: these misses tend to have GOOD skill numbers (fast "
          "runner, high prior success rate) and still get caught, or "
          "mediocre numbers and still succeed -- i.e. the model is picking "
          "up real signal, but the actual outcome is decided by things this "
          "feature set can't see (exact lead/jump, pitch type, throw "
          "accuracy on that specific play). See ROADMAP.md, honest limitations.")


def evaluate(name, model, X_te, y_te, test_df=None, diagnostics=False):
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

    p = model.predict_proba(X_te)[:, 1]
    metrics = {
        "log_loss": log_loss(y_te, p),
        "brier": brier_score_loss(y_te, p),
        "auc": roc_auc_score(y_te, p),
    }

    print(f"\n{name}")
    print(f"  log loss : {metrics['log_loss']:.4f}")
    print(f"  brier    : {metrics['brier']:.4f}")
    print(f"  auc      : {metrics['auc']:.4f}")

    label = "coefficients" if name.startswith("Logistic") else "feature importance (gain)"
    print(f"\n  {label}, ranked:")
    for feat, val in feature_ranking(model, "logistic" if name.startswith("Logistic") else "xgboost"):
        print(f"    {feat:28s} {val:+.4f}" if name.startswith("Logistic") else f"    {feat:28s} {val:.4f}")

    print("\n  calibration (predicted vs. actual success rate by decile):")
    print(calibration_table(y_te, p).to_string(float_format=lambda v: f"{v:.3f}"))

    if diagnostics:
        print_diagnostics(test_df, y_te, p)

    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/sample/features_2023_2025.csv")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="fraction of the chronologically LATEST rows held "
                         "out as the test set")
    ap.add_argument("--model", choices=["logistic", "xgboost", "both"],
                    default="both")
    ap.add_argument("--diagnostics", action="store_true",
                    help="print a confusion matrix and the most confident "
                         "wrong predictions, to see WHERE the model is off "
                         "rather than just how often")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_csv(args.features)

    # Sprint speed / age / pop time are ~99.8% present; median-fill the rare
    # misses so a handful of rows don't get a nonsensical 0 ft/s or age-0 runner.
    for col in ("runner_sprint_speed", "runner_age", "catcher_pop_time"):
        df[col] = df[col].fillna(df[col].median())

    # Rows are already chronological (see module docstring), so a date-based
    # split is just "first N% train, last N% test" -- no re-sorting needed,
    # and no future data ever ends up in the training set.
    split_idx = int(len(df) * (1 - args.test_frac))
    train, test = df.iloc[:split_idx], df.iloc[split_idx:]

    X_tr, y_tr = train[NUMERIC].fillna(0.0), train["success"].astype(int)
    X_te, y_te = test[NUMERIC].fillna(0.0), test["success"].astype(int)

    print(f"Date-based split (test = last {args.test_frac:.0%} of rows)")
    print(f"  train dates: {train['date'].min()} to {train['date'].max()}")
    print(f"  test dates : {test['date'].min()} to {test['date'].max()}")
    print(f"  train rows: {len(X_tr)}   test rows: {len(X_te)}")
    base_rate = y_te.mean()
    print(f"  base rate: {base_rate:.4f} (predict-the-mean brier = "
          f"{base_rate * (1 - base_rate):.4f})")

    results = {}
    if args.model in ("logistic", "both"):
        model = fit_logistic(X_tr, y_tr)
        results["Logistic regression"] = evaluate(
            "Logistic regression", model, X_te, y_te, test, args.diagnostics)
    if args.model in ("xgboost", "both"):
        model = fit_xgboost(X_tr, y_tr)
        results["XGBoost"] = evaluate(
            "XGBoost", model, X_te, y_te, test, args.diagnostics)

    if len(results) > 1:
        print("\nComparison (lower log loss/brier is better, higher AUC is better):")
        print(f"  {'model':22s} {'log loss':>10s} {'brier':>10s} {'auc':>10s}")
        for name, m in results.items():
            print(f"  {name:22s} {m['log_loss']:10.4f} {m['brier']:10.4f} {m['auc']:10.4f}")
        winner = min(results, key=lambda k: results[k]["log_loss"])
        print(f"\n  best on log loss: {winner}")


if __name__ == "__main__":
    main()
