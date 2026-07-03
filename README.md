# Base-Steal Decision Model

Predict whether a base runner should attempt a steal in a given situation.

Two connected pieces (see [`ROADMAP.md`](ROADMAP.md) for the full design):

1. **Success-probability model** — given the runner, pitcher, catcher, and game
   situation, predict P(the steal succeeds).
2. **Decision layer** — use run expectancy (RE24) to compute the break-even
   success rate for that situation, and recommend a steal only when the
   predicted probability clears it.

This repo is currently focused on **step 1: getting clean, validated data.**
It ships with one parsed season (2023) so you can run the parser immediately,
then scale up to more seasons and join in Statcast skill data. The
modeling and decision-layer pieces (see [`ROADMAP.md`](ROADMAP.md)) come
after the data is trusted.

## Quick start

```bash
pip install -r requirements.txt          # only needed for Statcast / notebook

# 1. Parse steal attempts from the bundled 2023 Retrosheet data
python -m src.retrosheet_parser --data-dir data/retrosheet_2023 \
    --out data/sample/steals_2023.csv

# 2. (optional) Pull Statcast skill tables + build the Retrosheet<->MLBAM
#    id crosswalk, needed to eventually join sprint speed / pop time on
python -m src.statcast_pull --season 2023 --out data/statcast
python -m src.id_crosswalk --out data/statcast/id_crosswalk.csv

# Validate the parser against known 2023 facts
python -m pytest tests/ -q
```

See `notebooks/eda.ipynb` for exploratory checks on the parsed data
(e.g. stolen-base leaderboard, success rates, base-state distribution).

## What's here

| File | Purpose |
|------|---------|
| `src/retrosheet_parser.py` | Parses Retrosheet event files into one row per steal attempt, tracking base/out/score state. **Tested & validated.** |
| `src/statcast_pull.py` | Pulls Statcast skill data (sprint speed, pop time) — **run on your own machine.** |
| `src/id_crosswalk.py` | Builds the Retrosheet id <-> MLBAM id crosswalk (via Chadwick register) needed to join Statcast onto Retrosheet rows. |
| `src/fetch_retrosheet.py` | Downloads more seasons from the Retrosheet GitHub mirror. |
| `notebooks/eda.ipynb` | Exploratory checks on the parsed steal-attempt data. |
| `tests/` | Regression tests (leaderboard, success rate). |
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

## Known limitations

- ~0.6% of attempts sit in same-play multi-steal snapshots (documented, benign).
- Retrosheet has no lead distance / jump / sprint speed — add via Statcast.
- Late-inning decisions should eventually use win probability, not run
  expectancy (see `ROADMAP.md`, "one important upgrade").
