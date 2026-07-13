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
combined feature table — plus a first baseline model (logistic regression,
temporally validated). The decision layer (see [`ROADMAP.md`](ROADMAP.md))
and a stronger model (gradient boosting) come next.

**2021-2022 are parsed and kept for comparison, but excluded from the
feature table / model by default.** MLB's 2023 rule changes (bigger bases,
limited pickoff attempts) measurably shifted stolen-base success rates —
see `notebooks/eda.ipynb`, section 10 — so pre-2023 attempts aren't drawn
from the same distribution the model needs to predict.

## Quick start

```bash
pip install -r requirements.txt

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

# 4. Train + evaluate the logistic regression baseline (temporal split:
#    train on seasons before --test-season, test on that season)
python -m src.train --features data/sample/features_2023_2025.csv --test-season 2025

# Validate the parser against known facts
python -m pytest tests/ -q
```

See `notebooks/eda.ipynb` for the full validation trail: parser sanity
checks, per-season leaderboards, the Statcast join, and the feature table's
leakage checks.

## What's here

| File | Purpose |
|------|---------|
| `src/retrosheet_parser.py` | Parses Retrosheet event files into one row per steal attempt, tracking base/out/score state. **Tested & validated.** |
| `src/fetch_retrosheet.py` | Downloads more seasons from the Retrosheet GitHub mirror. |
| `src/statcast_pull.py` | Pulls Statcast skill data (sprint speed, pop time) per season. |
| `src/id_crosswalk.py` | Builds the Retrosheet id <-> MLBAM id crosswalk (via Chadwick register) needed to join Statcast onto Retrosheet rows. |
| `src/features.py` | Combines the post-rule-change seasons (default 2023-2025) into one leakage-safe, Statcast-joined feature table (running runner/pitcher/catcher priors from prior attempts only). |
| `src/train.py` | Baseline logistic-regression success-probability model, temporally split (train on earlier seasons, test on the held-out latest one). |
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

## Baseline model

Logistic regression, trained on 2023-2024 and tested on the held-out 2025
season (`python -m src.train`):

- **AUC 0.597** (up from ~0.58 with Retrosheet-only features) — a real but
  modest lift; gradient boosting + feature interactions is the next lever,
  not more raw features.
- Calibration tracks the diagonal closely across deciles.
- Every coefficient sign matches baseball intuition: higher catcher pop
  time (slower catcher) raises success odds, LHP and higher catcher
  caught-stealing rate lower them, higher runner/pitcher prior success
  rate raises them.

## Known limitations

- ~0.6% of attempts sit in same-play multi-steal snapshots (documented, benign).
- No lead distance / jump data exists publicly — sprint speed is the proxy.
- Late-inning decisions should eventually use win probability, not run
  expectancy (see `ROADMAP.md`, "one important upgrade").
