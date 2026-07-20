"""
Build backend/data/app.db -- the SQLite database the Go web app's backend
reads at startup. This is the ONE place Python's decision-layer artifacts
(RE24, win-probability tables, the fitted model) and a new player search
index get frozen into a portable, language-agnostic format.

Reuses src/predict.py's existing load_tables()/load_model() almost
verbatim -- those functions already build exactly the artifacts a caller
needs once at process startup, which is exactly what this script needs
too. The one difference: this script explicitly requests the LOGISTIC
model (model_kind="logistic"), not load_model's XGBoost default -- the web
app reimplements logistic regression natively in Go (a plain dot product +
sigmoid), so XGBoost is never invoked here.

backend/sql/schema.sql is the single source of truth for the database
shape -- this script executes that exact file rather than duplicating
CREATE TABLE statements in Python.

    python -m src.export_web_data
    python -m src.export_web_data --out backend/data/app.db --features data/sample/features_2023_2025.csv
"""
from __future__ import annotations

import argparse
import os
import sqlite3

from .predict import load_tables, load_model
from .retrosheet_parser import load_rosters
from .win_probability import LEGACY_SEASONS, MODERN_SEASONS, POST_RULE_CHANGE_SEASONS

ALL_SEASONS = sorted(set(LEGACY_SEASONS) | set(MODERN_SEASONS) | set(POST_RULE_CHANGE_SEASONS))

ROLE_SPECS = {
    "runners": dict(
        id_col="runner_id", table="runners",
        columns={
            "bats_lhb": "runner_bats_lhb",
            "prior_sr": "runner_prior_sr",
            "prior_att": "runner_prior_att",
            "sprint_speed": "runner_sprint_speed",
            "sprint_speed_missing": "runner_sprint_speed_missing",
            "age": "runner_age",
            "age_missing": "runner_age_missing",
        },
    ),
    "pitchers": dict(
        id_col="pitcher_id", table="pitchers",
        columns={
            "throws_lhp": "pitcher_throws_lhp",
            "prior_sr_allowed": "pitcher_prior_sr_allowed",
        },
    ),
    "catchers": dict(
        id_col="catcher_id", table="catchers",
        columns={
            "prior_cs_rate": "catcher_prior_cs_rate",
            "pop_time": "catcher_pop_time",
            "pop_time_missing": "catcher_pop_time_missing",
        },
    ),
}


def build_schema(conn: sqlite3.Connection, schema_path: str = "backend/sql/schema.sql") -> None:
    with open(schema_path) as fh:
        conn.executescript(fh.read())


def export_re24(conn: sqlite3.Connection, re24: dict) -> int:
    rows = [(base_code, outs, runs) for (base_code, outs), runs in re24.items()]
    conn.executemany(
        "INSERT INTO re24_cells (base_code, outs, expected_runs) VALUES (?, ?, ?)", rows)
    return len(rows)


def export_win_prob(conn: sqlite3.Connection, table: dict, hold_table: dict) -> tuple[int, int]:
    def rows_for(kind, t):
        return [(kind, ib, half, outs, bc, sb, rate, n)
                for (ib, half, outs, bc, sb), (rate, n) in t.items()]

    after_rows = rows_for("after", table)
    hold_rows = rows_for("hold", hold_table)
    conn.executemany(
        "INSERT INTO win_prob_cells "
        "(table_kind, inning_bucket, half, outs, base_code, score_bucket, win_rate, n) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        after_rows + hold_rows,
    )
    return len(after_rows), len(hold_rows)


def export_model(conn: sqlite3.Connection, model, medians: dict, meta: dict) -> None:
    from .train import NUMERIC

    coef_rows = [(name, i, float(coef))
                for i, (name, coef) in enumerate(zip(NUMERIC, model.coef_[0]))]
    conn.executemany(
        "INSERT INTO model_coefficients (feature_name, sort_order, coefficient) VALUES (?, ?, ?)",
        coef_rows,
    )
    conn.execute(
        "INSERT INTO model_meta "
        "(id, intercept, median_runner_sprint_speed, median_runner_age, "
        " median_catcher_pop_time, trained_at, train_rows) "
        "VALUES (1, ?, ?, ?, ?, ?, ?)",
        (float(model.intercept_[0]), float(medians["runner_sprint_speed"]),
         float(medians["runner_age"]), float(medians["catcher_pop_time"]),
         meta["trained_at"], meta["train_rows"]),
    )


def build_name_index(base_dir: str = "data", seasons=ALL_SEASONS) -> dict:
    """id -> {'full_name': 'First Last', 'team': 'ATL'}. Walks season dirs
    NEWEST FIRST, only filling ids not already resolved, so a player who
    last appeared in an older season still gets a name even if absent from
    the newest roster.
    """
    index: dict = {}
    for year in sorted(seasons, reverse=True):
        data_dir = os.path.join(base_dir, f"retrosheet_{year}")
        if not os.path.isdir(data_dir):
            continue
        for player_id, info in load_rosters(data_dir).items():
            if player_id in index:
                continue
            first, last = info.get("first", ""), info.get("last", "")
            full_name = f"{first} {last}".strip() or player_id
            index[player_id] = {"full_name": full_name, "team": info.get("team")}
    return index


def export_players(conn: sqlite3.Connection, features_path: str, base_dir: str = "data") -> dict:
    """For each role, take every player's LAST chronological row (features
    CSV is already sorted -- see features.py's module docstring -- so
    .groupby(id_col).tail(1) is the latest-in-time snapshot without
    re-sorting), join onto build_name_index() for a real name, and insert.

    Returns {"runners": n, "pitchers": n, "catchers": n, "unresolved_names": n}.
    """
    import pandas as pd

    df = pd.read_csv(features_path)
    names = build_name_index(base_dir)
    unresolved = 0
    counts = {}

    for role, spec in ROLE_SPECS.items():
        id_col = spec["id_col"]
        # df is already chronologically sorted (date, game_id, inning, outs --
        # see features.py's module docstring); re-sorting here would risk an
        # unstable sort reordering same-date ties, so just .tail(1) directly.
        latest = df.groupby(id_col, as_index=False).tail(1)
        rows = []
        for _, r in latest.iterrows():
            player_id = r[id_col]
            info = names.get(player_id)
            if info is None:
                unresolved += 1
                full_name, team = player_id, None
            else:
                full_name, team = info["full_name"], info["team"]

            values = []
            for db_col, csv_col in spec["columns"].items():
                v = r[csv_col]
                if db_col.endswith("_missing") or db_col in ("bats_lhb", "throws_lhp", "prior_att"):
                    v = int(v)
                elif pd.isna(v):
                    v = None
                else:
                    v = float(v)
                values.append(v)

            rows.append((player_id, full_name, full_name.lower(), team, r["date"], *values))

        col_names = ["player_id", "full_name", "search_name", "last_team", "last_seen_date"] \
            + list(spec["columns"].keys())
        placeholders = ", ".join("?" * len(col_names))
        conn.executemany(
            f"INSERT INTO {spec['table']} ({', '.join(col_names)}) VALUES ({placeholders})",
            rows,
        )
        counts[role] = len(rows)

    counts["unresolved_names"] = unresolved
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="backend/data/app.db")
    ap.add_argument("--schema", default="backend/sql/schema.sql")
    ap.add_argument("--features", default="data/sample/features_2023_2025.csv")
    ap.add_argument("--base-dir", default="data")
    args = ap.parse_args()

    import datetime
    import pandas as pd

    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    conn = sqlite3.connect(args.out)
    build_schema(conn, args.schema)

    print("Building RE24 + win-probability tables...")
    tables = load_tables(base_dir=args.base_dir)
    n_re24 = export_re24(conn, tables["re24"])
    n_after, n_hold = export_win_prob(conn, tables["wp_table"], tables["wp_hold_table"])
    print(f"  re24_cells: {n_re24}  win_prob_cells: {n_after} after + {n_hold} hold")

    print("Fitting logistic regression model...")
    model, medians = load_model(features_path=args.features, model_kind="logistic")
    train_rows = int(len(pd.read_csv(args.features)) * 0.8)
    export_model(conn, model, medians, {
        "trained_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "train_rows": train_rows,
    })
    print(f"  model_coefficients: 20 rows, trained on {train_rows} rows")

    print("Building player search index (this reads all 13 seasons' rosters)...")
    counts = export_players(conn, args.features, args.base_dir)
    print(f"  runners: {counts['runners']}  pitchers: {counts['pitchers']}  "
         f"catchers: {counts['catchers']}  (unresolved names: {counts['unresolved_names']})")

    conn.commit()
    conn.close()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
