-- Single source of truth for the web app's SQLite database. sqlc.yaml points
-- at this file directly, and src/export_web_data.py executes it verbatim
-- (sqlite3.executescript) rather than duplicating DDL in Python.
--
-- This database holds ONLY precomputed lookup/reference data the decision
-- layer needs -- no prediction history, no user accounts, no saved
-- scenarios (see the plan: /Users/colin/.claude/plans/go-with-chi-and-soft-grove.md).

CREATE TABLE re24_cells (
    base_code     TEXT    NOT NULL,   -- run_expectancy.BASE_STATES entry, e.g. '1__'
    outs          INTEGER NOT NULL,   -- 0, 1, or 2
    expected_runs REAL    NOT NULL,
    PRIMARY KEY (base_code, outs)
) WITHOUT ROWID;

-- Represents BOTH win_probability.py's `table` (after-success/after-caught,
-- post-rule-change seasons only) and `hold_table` (hold-only baseline, 13
-- seasons). table_kind keeps one schema/query/Go-struct instead of
-- duplicating both; the Go backend loads each kind into a SEPARATELY NAMED
-- map at startup and never merges them -- that separation, not this schema,
-- is what enforces "never blend these two" (see win_probability.py's module
-- docstring on why the season ranges are non-interchangeable).
CREATE TABLE win_prob_cells (
    table_kind    TEXT    NOT NULL CHECK (table_kind IN ('after', 'hold')),
    inning_bucket INTEGER NOT NULL,   -- _inning_bucket: min(inning, 9)
    half          INTEGER NOT NULL CHECK (half IN (0, 1)),
    outs          INTEGER NOT NULL CHECK (outs BETWEEN 0 AND 3),  -- 3 = 'END' state
    base_code     TEXT    NOT NULL,   -- BASE_STATES entry, or 'END' when outs = 3
    score_bucket  INTEGER NOT NULL,   -- _score_bucket: clamp(score_diff, -4, 4)
    win_rate      REAL    NOT NULL,
    n             INTEGER NOT NULL,
    PRIMARY KEY (table_kind, inning_bucket, half, outs, base_code, score_bucket)
) WITHOUT ROWID;

-- Logistic regression, ported verbatim. sort_order = the feature's index in
-- train.NUMERIC -- SQLite gives no row-order guarantee on a plain SELECT *,
-- so the Go side does ORDER BY sort_order rather than hardcoding the
-- feature list a second time.
CREATE TABLE model_coefficients (
    feature_name TEXT    PRIMARY KEY,
    sort_order   INTEGER NOT NULL UNIQUE,
    coefficient  REAL    NOT NULL
);

CREATE TABLE model_meta (
    id                         INTEGER PRIMARY KEY CHECK (id = 1),  -- singleton row
    intercept                  REAL    NOT NULL,
    median_runner_sprint_speed REAL    NOT NULL,
    median_runner_age          REAL    NOT NULL,
    median_catcher_pop_time    REAL    NOT NULL,
    trained_at                 TEXT    NOT NULL,
    train_rows                 INTEGER NOT NULL
);

-- One table per role (runner/pitcher/catcher), since each role only feeds a
-- different subset of train.NUMERIC's features -- these map directly to the
-- three player-search dropdowns the frontend needs.
CREATE TABLE runners (
    player_id            TEXT    PRIMARY KEY,   -- Retrosheet id, e.g. 'acunr001'
    full_name             TEXT    NOT NULL,
    search_name            TEXT    NOT NULL,     -- lower(full_name), for LIKE search
    last_team               TEXT,
    bats_lhb                 INTEGER NOT NULL,   -- runner_bats_lhb, 0/1
    prior_sr                  REAL    NOT NULL,   -- runner_prior_sr, latest snapshot
    prior_att                  INTEGER NOT NULL,   -- runner_prior_att
    sprint_speed                REAL,             -- NULL, not 0, when unknown
    sprint_speed_missing         INTEGER NOT NULL,
    age                           REAL,
    age_missing                    INTEGER NOT NULL,
    last_seen_date                  TEXT    NOT NULL
);
CREATE INDEX idx_runners_search_name ON runners(search_name);

CREATE TABLE pitchers (
    player_id        TEXT    PRIMARY KEY,
    full_name         TEXT    NOT NULL,
    search_name        TEXT    NOT NULL,
    last_team           TEXT,
    throws_lhp            INTEGER NOT NULL,   -- pitcher_throws_lhp
    prior_sr_allowed        REAL    NOT NULL,
    last_seen_date            TEXT    NOT NULL
);
CREATE INDEX idx_pitchers_search_name ON pitchers(search_name);

CREATE TABLE catchers (
    player_id       TEXT    PRIMARY KEY,
    full_name        TEXT    NOT NULL,
    search_name        TEXT    NOT NULL,
    last_team           TEXT,
    prior_cs_rate         REAL    NOT NULL,
    pop_time                REAL,
    pop_time_missing         INTEGER NOT NULL,
    last_seen_date             TEXT    NOT NULL
);
CREATE INDEX idx_catchers_search_name ON catchers(search_name);
