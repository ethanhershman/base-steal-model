# Base-Steal Decision Model

Predict whether a base runner should attempt a steal in a given situation.

Two connected pieces (see [`ROADMAP.md`](ROADMAP.md) for the full design):

1. **Success-probability model** — given the runner, pitcher, catcher, and game
   situation, predict P(the steal succeeds).
2. **Decision layer** — use run expectancy (RE24) to compute the break-even
   success rate for that situation, and recommend a steal only when the
   predicted probability clears it.

This repo already **works end to end on real 2023 MLB play-by-play.** It ships
with one parsed season so you can run everything immediately, then scale up.

## Quick start

```bash
pip install -r requirements.txt          # only needed for modeling / Statcast

# 1. Parse steal attempts from the bundled 2023 Retrosheet data
python -m src.retrosheet_parser --data-dir data/retrosheet_2023 \
    --out data/sample/steals_2023.csv

# 2. Build the run-expectancy table (the decision layer's backbone)
python -m src.run_expectancy --data-dir data/retrosheet_2023 \
    --out data/sample/re24_2023.csv

# 3. Engineer features (leakage-safe) and train a baseline model
python -m src.features --steals data/sample/steals_2023.csv \
    --out data/sample/features_2023.csv
python -m src.train --features data/sample/features_2023.csv

# 4. See the decision layer in action
python -m src.demo_decision

# Validate everything against known 2023 facts
python -m pytest tests/ -q
```

## What's here

| File | Purpose |
|------|---------|
| `src/retrosheet_parser.py` | Parses Retrosheet event files into one row per steal attempt, tracking base/out/score state. **Tested & validated.** |
| `src/run_expectancy.py` | Builds the RE24 table and computes steal break-even rates. |
| `src/features.py` | Leakage-safe feature engineering (prior runner/pitcher/catcher rates, situation). |
| `src/train.py` | Baseline logistic-regression model with the right evaluation metrics. |
| `src/demo_decision.py` | End-to-end steal / hold recommendations. |
| `src/statcast_pull.py` | Pulls Statcast skill data (sprint speed, pop time) — **run on your own machine.** |
| `src/fetch_retrosheet.py` | Downloads more seasons from the Retrosheet GitHub mirror. |
| `tests/` | Regression tests (leaderboard, success rate, RE24 anchors). |
| `data/retrosheet_2023/` | Bundled raw 2023 event + roster files. |
| `data/sample/` | Generated sample outputs. |

## Data sources

Retrosheet (complete play-by-play; **used here**) gives the situation and the
steal outcomes. Statcast (`src/statcast_pull.py`) adds the tracking skills that
matter most — runner sprint speed, catcher pop time, pitcher tempo — and is the
biggest lever for improving the model. See `ROADMAP.md`.

> Heads up: Statcast (baseballsavant.mlb.com) is firewalled inside the sandbox
> this was scaffolded in, so the model here trains on Retrosheet situation only
> (hence a modest baseline AUC). On your machine, run `src/statcast_pull.py` and
> join those skills onto the steal table — that's the intended next step.

## Validation

The parser is checked against known 2023 facts (all pass):
- Every steal attempt's runner is resolved (0 unresolved of 4,439).
- League steal-success rate ≈ 80%.
- SB leaderboard matches reality (Acuña Jr. #1 with 70+).
- RE24 table matches textbook values (empty/0-out ≈ 0.51, loaded/0-out ≈ 2.23).

## Known limitations

- ~0.6% of attempts sit in same-play multi-steal snapshots (documented, benign).
- Retrosheet has no lead distance / jump / sprint speed — add via Statcast.
- Late-inning decisions should eventually use win probability, not run
  expectancy (see `ROADMAP.md`, "one important upgrade").
