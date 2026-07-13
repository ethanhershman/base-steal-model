# Base-Steal Decision Model

Predict whether a base runner should attempt a steal in a given situation.

Two connected pieces (see [`ROADMAP.md`](ROADMAP.md) for the full design):

1. **Success-probability model** — given the runner, pitcher, catcher, and game
   situation, predict P(the steal succeeds).
2. **Decision layer** — use run expectancy (RE24) to compute the break-even
   success rate for that situation, and recommend a steal only when the
   predicted probability clears it.

This repo has the data foundation done — five parsed seasons (2021-2025),
Statcast skill data joined on via an id crosswalk, and a leakage-safe
combined feature table — plus two trained success-probability models
(logistic regression baseline + XGBoost), both temporally validated. The
decision layer (see [`ROADMAP.md`](ROADMAP.md)) comes next.

**2021-2022 are parsed and kept for comparison, but excluded from the
feature table / model by default.** MLB's 2023 rule changes (bigger bases,
limited pickoff attempts) measurably shifted stolen-base success rates —
see `notebooks/eda.ipynb`, section 10 — so pre-2023 attempts aren't drawn
from the same distribution the model needs to predict.

## Quick start

```bash
pip install -r requirements.txt
# If you have multiple Python installs (e.g. Anaconda + a separate
# framework Python), install into whichever one your Jupyter kernel
# actually uses -- `jupyter kernelspec list` shows its path -- otherwise
# notebook cells that import sklearn/xgboost will fail with a
# ModuleNotFoundError even though `pip install` succeeded elsewhere.

# 1. Parse steal attempts from the bundled Retrosheet data (2021-2025;
#    2023 ships in the repo, fetch the others first if you don't have them)
python -m src.fetch_retrosheet --seasons 2021 2022 2024 2025 --dest data
for y in 2021 2022 2023 2024 2025; do
  python -m src.retrosheet_parser --data-dir data/retrosheet_$y \
      --out data/sample/steals_$y.csv
done

# 2. Pull Statcast skill tables + build the Retrosheet<->MLBAM id crosswalk
for y in 2021 2022 2023 2024 2025; do
  python -m src.statcast_pull --season $y --out data/statcast
done
python -m src.id_crosswalk --out data/statcast/id_crosswalk.csv

# 3. Build the leakage-safe, Statcast-joined feature table (defaults to
#    the post-rule-change seasons, 2023-2025)
python -m src.features --out data/sample/features_2023_2025.csv

# 4. Train + evaluate both models (date-based split: train on the earliest
#    dates, test on the latest --test-frac of rows). --model logistic or
#    --model xgboost to run just one.
python -m src.train --features data/sample/features_2023_2025.csv --test-frac 0.2

# Validate the parser against known facts
python -m pytest tests/ -q
```

See `notebooks/eda.ipynb` for the full validation trail: parser sanity
checks, per-season leaderboards, the Statcast join, the feature table's
leakage checks, and model diagnostics (confusion matrices, calibration,
most-confident-wrong predictions).

## What's here

| File | Purpose |
|------|---------|
| `src/retrosheet_parser.py` | Parses Retrosheet event files into one row per steal attempt, tracking base/out/score state. **Tested & validated.** |
| `src/fetch_retrosheet.py` | Downloads more seasons from the Retrosheet GitHub mirror. |
| `src/statcast_pull.py` | Pulls Statcast skill data (sprint speed, pop time) per season. |
| `src/id_crosswalk.py` | Builds the Retrosheet id <-> MLBAM id crosswalk (via Chadwick register) needed to join Statcast onto Retrosheet rows. |
| `src/features.py` | Combines the post-rule-change seasons (default 2023-2025) into one leakage-safe, Statcast-joined feature table (running runner/pitcher/catcher priors from prior attempts only). |
| `src/train.py` | Trains + evaluates the logistic-regression baseline and an XGBoost model (`--model logistic\|xgboost\|both`), date-based split (train on the earliest dates, test on the latest `--test-frac`) so no future data ever leaks into training. |
| `notebooks/eda.ipynb` | Exploratory checks + validation for every step above. |
| `tests/` | Regression tests (leaderboard, success rate). |
| `data/retrosheet_2023/` | Bundled raw 2023 event + roster files. |
| `data/sample/` | Generated sample outputs (steal tables + combined feature table). |

## Data sources

Retrosheet (complete play-by-play; parsed by `retrosheet_parser.py`) gives the
situation and the steal outcomes. Statcast (`src/statcast_pull.py`) adds the
tracking skills that matter most — runner sprint speed, catcher pop time — and
is the biggest lever for improving the eventual model. See `ROADMAP.md`.

## Validation

Checked in `notebooks/eda.ipynb` against known facts (all pass):
- Every steal attempt's runner is resolved (0 unresolved in 2022-2025; 4 of
  2,972 in 2021 — the same documented, benign edge case).
- League steal-success rate ≈ 75-76% in 2021-2022, ≈ 78-80% in 2023-2025 —
  the data shows the real 2023 pickoff-limit/bigger-base rule change. This
  is exactly why 2021-2022 are excluded from the feature table by default.
- SB leaderboards match reality (Acuña Jr. #1 in 2023 with 70+, De La Cruz
  #1 in 2024 with 67).
- Statcast join lands on the correct players (known burners top the
  sprint-speed join; <0.2% of rows are missing a Statcast match).
- The combined feature table is leakage-safe: every runner's first-ever
  attempt has zero prior attempts, and running tallies match file order
  exactly.
- Steal success rate against LHP (75.4%) is meaningfully lower than RHP
  (78.9%) — the known "lefties hold runners better" effect shows up in
  the data.
- **~4% of steal plays (7% of attempt-rows) are true simultaneous double
  steals** (two runners breaking on the same pitch, encoded by Retrosheet
  as e.g. `SB3;SB2` on one play line) — **100% of them succeed**, across
  all three seasons. An earlier project note claimed this was ~0.6%; that
  was wrong (it conflated unrelated sequential steals that happen to share
  an out count). Flagged directly as the `is_double_steal` feature.
- Stealing home (42% success) is far harder than 2nd (79%) or 3rd (82%) —
  previously invisible to the model, which only distinguished "2nd" from
  "everything else." Now split into explicit `steal_of_third`/
  `steal_of_home` features.
- **A runner ALSO on 3rd during a steal of 2nd changes the odds a lot:**
  76.7% → 91.5% success. Catchers are reluctant to risk a bad throw to 2nd
  letting that run score, so they often concede the steal. The effect
  holds — and strengthens — at every out count (2 outs: 78.1% → 92.6%),
  so it isn't confounded with something else. Runner-on-1st during a steal
  of 3rd, by contrast, has no meaningful effect (81.6% vs. 81.9%). Only
  the 3rd-base case is added, as `runner_on_third`.

## Features

26 columns total; `season`/`date`/`runner_id`/`pitcher_id`/`catcher_id`/
`target_base` are identifiers, the other 20 (`NUMERIC` in `src/train.py`)
go into the model:

| feature | notes |
|---|---|
| `is_double_steal` | 2+ runners on the same pitch; ~100% success |
| `steal_of_third`, `steal_of_home` | 2nd base is the implicit baseline |
| `runner_on_third` | another runner on 3rd during a steal of 2nd/3rd; 76.7% → 91.5% success |
| `late_inning`, `inning` | success rises in innings 7+ |
| `outs`, `balls`, `strikes` | count/out-state at the pitch |
| `score_diff`, `close_game` | blowouts (either direction) → easier; close/tied → harder |
| `runner_bats_lhb`, `pitcher_throws_lhp` | lefty pitchers hold runners better |
| `runner_prior_sr`, `runner_prior_att` | leakage-safe running success rate + sample size |
| `pitcher_prior_sr_allowed` | leakage-safe running rate of steals allowed |
| `catcher_prior_cs_rate` | the catcher's own caught-stealing rate — leakage-safe |
| `runner_sprint_speed` (+missing flag) | Statcast, season-matched |
| `runner_age` (+missing flag) | Statcast, season-matched |
| `catcher_pop_time` (+missing flag) | Statcast, season-matched |

Considered and deliberately left out: pitcher tempo/time-to-plate and
pickoff-attempt rate aren't cleanly available from public Statcast
leaderboards without a much deeper pitch-level pull — see ROADMAP.md.
Runner-on-1st during a steal of 3rd, and any base-occupancy effect on
steals of home, were checked and found to have no meaningful/reliable
signal (see `notebooks/eda.ipynb`, section 10.6) — left out.

## Models

Date-based split — trained on 2023/03/30-2025/06/01, tested on the
chronologically last 20% of rows (2025/06/01-2025/11/01)
(`python -m src.train`):

| model | log loss | brier | AUC |
|---|---|---|---|
| Logistic regression | **0.4858** | **0.1600** | **0.6787** |
| XGBoost (early-stopped) | 0.4869 | 0.1603 | 0.6666 |

- Adding `is_double_steal`, `steal_of_third`/`steal_of_home`, `runner_age`,
  and `runner_on_third` took AUC from ~0.60 to **~0.67-0.68** and log loss
  from ~0.52 to **~0.486** — a real, meaningful jump, not the modest one we
  saw from Statcast alone. `is_double_steal`, `steal_of_home`, and
  `runner_on_third` are XGBoost's top-3 features by importance.
- XGBoost and logistic land within noise of each other; neither dominates.
- Calibration tracks the diagonal closely across deciles for both models.
- Logistic coefficient signs all match baseball intuition: `is_double_steal`
  and `runner_on_third` strongly positive, `steal_of_home` strongly
  negative, higher catcher pop time (slower catcher) raises success odds,
  LHP and higher catcher caught-stealing rate lower it, higher
  runner/pitcher prior success rate raises it.

### Where the model is still wrong (`python -m src.train --diagnostics`)

A standard 0.5 probability threshold is still not that meaningful — steal
success is a ~78% base-rate event, so most predicted probabilities cluster
well above 0.5. At the test base rate (~0.78) threshold, precision and
specificity are meaningfully better than before these features (precision
~0.85-0.87, specificity ~0.65-0.71, vs. ~0.81-0.85/~0.51-0.62 previously);
recall is somewhat lower since predictions are now more spread out and
selective — see the confusion-matrix tables in `notebooks/eda.ipynb`,
section 12, for exact current numbers.

For the remaining misses (normal, single-runner attempts with no other
base traffic), the pattern is unchanged: false positives (predicted high,
actually caught) are runners with *great* stats who still got thrown out;
false negatives are mediocre runners who still made it. Season-level
aggregate skill stats can't see the thing that actually decides most of
these attempts: exact lead distance, jump timing, pitch type/location, and
throw accuracy on that specific play. None of that is in the public data
(see "Known limitations" below), so this is closer to today's ceiling for
this feature set than a sign of a broken model.

## Known limitations

- No lead distance / jump data exists publicly — sprint speed is the proxy.
- Pitcher tempo, pitch type/location, and pickoff-attempt rate aren't
  cleanly available from public Statcast leaderboards.
- Late-inning decisions should eventually use win probability, not run
  expectancy (see `ROADMAP.md`, "one important upgrade").
