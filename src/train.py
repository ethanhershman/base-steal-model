"""
Baseline steal-success probability model.

Starts with logistic regression (interpretable, naturally calibrated) and
evaluates with the metrics that matter for a decision model: log loss, Brier
score, AUC, and a calibration check. Swap in gradient boosting (XGBoost /
LightGBM) once this baseline is beaten — the roadmap explains why.

    pip install scikit-learn pandas
    python -m src.train --features data/sample/features_2023.csv
"""
from __future__ import annotations

import argparse


NUMERIC = [
    "steal_of_second", "late_inning", "outs", "balls", "strikes",
    "score_diff", "close_game", "runner_bats_lhb",
    "runner_prior_sr", "runner_prior_att",
    "pitcher_prior_sr_allowed", "catcher_prior_cs_rate",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/sample/features_2023.csv")
    args = ap.parse_args()

    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score

    df = pd.read_csv(args.features)
    X = df[NUMERIC].fillna(0.0)
    y = df["success"].astype(int)

    # NOTE: this random split is for a quick sanity check only. For real
    # evaluation, split by SEASON (train earlier years, test later ones) so the
    # model can't learn from the future. See ROADMAP.md, "Validate across time".
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]

    print("Baseline logistic regression (single-season sanity check)")
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


if __name__ == "__main__":
    main()
