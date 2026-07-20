# Base-Steal Decision Model

Predict whether a base runner should attempt a steal in a given situation.

Two connected pieces (see [`ROADMAP.md`](ROADMAP.md) for the full design):

1. **Success-probability model** — given the runner, pitcher, catcher, and game
   situation, predict P(the steal succeeds).
2. **Decision layer** — use run expectancy (RE24) to compute the break-even
   success rate for that situation, and recommend a steal only when the
   predicted probability clears it.

Both pieces are now in place: five parsed seasons (2021-2025), a leakage-safe
Statcast-joined feature table, two trained success-probability models
(logistic regression baseline + XGBoost), and a real decision layer
(`src/run_expectancy.py` + `src/win_probability.py` + `src/demo_decision.py`)
that combines the trained model's predicted probability with a situational
break-even rate to make a GO/HOLD call — RE24 for most of the game, win
probability for high-leverage late/close situations, where RE24's
run-based math badly understates the cost of a caught stealing that ends a
trailing team's last chance. `src/backtest.py` (`ROADMAP.md`, "Step 5") backtests
the full system against history — see "Backtest" below.

**`src/predict.py` is the integration point for anything built on top of
this** (e.g. a website) — see "For website/app integration" below.

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
#    2023 ships in the repo, fetch the others first if you don't have them).
#    --batting-out also writes every plate appearance for leakage-safe
#    running batter stats (AVG/OBP/SLG/HR%) -- only needed for 2023-2025,
#    since 2021-2022 aren't used in the feature table anyway.
python -m src.fetch_retrosheet --seasons 2021 2022 2024 2025 --dest data
for y in 2021 2022; do
  python -m src.retrosheet_parser --data-dir data/retrosheet_$y \
      --out data/sample/steals_$y.csv
done
for y in 2023 2024 2025; do
  python -m src.retrosheet_parser --data-dir data/retrosheet_$y \
      --out data/sample/steals_$y.csv \
      --batting-out data/sample/battinglines_$y.csv
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

# 5. Fetch 2013-2020 too -- only needed for the win-probability table's
#    hold-only baseline (win_probability.py reads raw event files directly,
#    not through the parser/feature-table pipeline above), which safely
#    extends further back than the steal-outcome model can.
python -m src.fetch_retrosheet --seasons 2013 2014 2015 2016 2017 2018 2019 2020 --dest data

# 6. Build the RE24 + win-probability tables, then see the decision layer
#    make real GO/HOLD calls on real held-out attempts using the actual
#    trained model (not hardcoded example probabilities) -- RE24 for most
#    situations, win probability for high-leverage late/close ones.
python -m src.run_expectancy --out data/sample/re24_2023_2025.csv
python -m src.win_probability
python -m src.demo_decision

# 7. Try the single integration function a website backend would call --
#    same tables/model as above, unified behind one call per situation
#    (see "For website/app integration" below).
python -m src.predict

# Validate the parser + decision layer against known facts
python -m pytest tests/ -q
```

See `notebooks/eda.ipynb` for the full validation trail: parser sanity
checks, per-season leaderboards, the Statcast join, the feature table's
leakage checks, model diagnostics (confusion matrices, calibration,
most-confident-wrong predictions), and the decision layer (section 13:
the full RE24 table with sample sizes per cell, break-even rates across
all 24 valid steal situations, and the real model's GO/HOLD calls on 25
real held-out attempts; section 14: the win-probability upgrade for
late/close games, with the exact reward/cost/break-even arithmetic shown
step by step for the down-1-vs-tied bottom-of-the-9th comparison, a
3-season-vs-5-season sensitivity check, a direct test of whether the
"current state" baseline is biased by teams' own steal decisions, and
(section 14.7) the era-consistency checks behind extending the hold-only
baseline to 2013-2025).

## What's here

| File | Purpose |
|------|---------|
| `src/retrosheet_parser.py` | Parses Retrosheet event files into one row per steal attempt, tracking base/out/score state. `--batting-out` also classifies every plate appearance (AVG/OBP/SLG inputs). **Tested & validated.** |
| `src/fetch_retrosheet.py` | Downloads more seasons from the Retrosheet GitHub mirror. |
| `src/statcast_pull.py` | Pulls Statcast skill data (sprint speed, pop time) per season. |
| `src/id_crosswalk.py` | Builds the Retrosheet id <-> MLBAM id crosswalk (via Chadwick register) needed to join Statcast onto Retrosheet rows. |
| `src/features.py` | Combines the post-rule-change seasons (default 2023-2025) into one leakage-safe, Statcast-joined feature table (running runner/pitcher/catcher/batter priors from prior attempts only). |
| `src/train.py` | Trains + evaluates the logistic-regression baseline and an XGBoost model (`--model logistic\|xgboost\|both`), date-based split (train on the earliest dates, test on the latest `--test-frac`) so no future data ever leaks into training. |
| `src/run_expectancy.py` | Builds the RE24 table from Retrosheet play-by-play and computes situational steal break-even rates (`cost / (reward + cost)`). |
| `src/win_probability.py` | Empirical win-probability table (inning/half/outs/bases/score margin) and break-even math for high-leverage late/close situations. Two season ranges: 2013-2025 for the hold-only baseline (checked era-consistent), 2021-2025 for after-success/after-caught (checked NOT era-consistent). |
| `src/demo_decision.py` | Fits the real trained model, then makes GO/HOLD calls on real held-out attempts by comparing its predicted probability against the RE24 (or win-probability, if high-leverage) break-even for that exact situation. |
| `src/predict.py` | **The integration point for a website/app.** `predict_steal_decision(...)` — one function call, one row of output (current win probability, win probability if the steal succeeds, break-even, model's predicted success probability, GO/HOLD). See "For website/app integration" below. |
| `src/backtest.py` | Full backtest (ROADMAP.md, "Step 5") — runs every held-out test-set attempt (not a sample) through the decision layer and totals the model's GO-only policy's realized run/win value against what actually happened. See "Backtest" below. |
| `notebooks/eda.ipynb` | Exploratory checks + validation for the data/features/model (sections 1-12) and the decision layer (sections 13-14.8). |
| `notebooks/predict_demo.ipynb` | Runnable walkthrough of `src/predict.py`'s API — start here to see it in action before wiring it into a backend. |
| `tests/` | Regression tests (leaderboard, success rate, RE24 anchors, win-probability sanity checks, `predict.py`'s unified interface). |
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

31 columns total in `data/sample/features_2023_2025.csv`;
`season`/`date`/`runner_id`/`pitcher_id`/`catcher_id`/`batter_id`/
`target_base` are identifiers. 20 of the remaining 24 (`NUMERIC` in
`src/train.py`) go into the trained model:

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

**In the feature table but NOT in the trained model:**
`batter_prior_avg`/`obp`/`slg`/`hr_pct`/`pa` — the batter at the plate's
leakage-safe running offensive stats (see below).

Considered and deliberately left out: pitcher tempo/time-to-plate and
pickoff-attempt rate aren't cleanly available from public Statcast
leaderboards without a much deeper pitch-level pull — see ROADMAP.md.
Runner-on-1st during a steal of 3rd, and any base-occupancy effect on
steals of home, were checked and found to have no meaningful/reliable
signal (see `notebooks/eda.ipynb`, section 10.6) — left out. Day/night has
no effect at all (78.98% vs. 78.99%). In-game attempt count against the
same pitcher/catcher and a leakage-safe park success rate both showed real
univariate effects, but an ablation showed neither improves the full
model (log loss/AUC flat-to-worse in every configuration) — implemented,
tested, and reverted; see `notebooks/eda.ipynb`, section 10.7.

**Batter offense (AVG/OBP/SLG/HR%)** — computed properly this time: leakage-
safe running stats from every plate appearance (`retrosheet_parser.py
--batting-out`, cross-validated against `pybaseball`'s Baseball-Reference
data — 450 players, max discrepancy 0.014 AVG points, mean 0.0012), merged
with steal-attempt rows using an exact per-game play sequence number
(`play_seq`) rather than out-count alone. An earlier session found a real
univariate correlation using non-leakage-safe season stats; the proper
leakage-safe version doesn't survive an ablation against the full model
(log loss/AUC flat within noise in every configuration — likely because
`runner_sprint_speed`/`runner_prior_sr` already absorb most of this
signal, since better athletes tend to be both better hitters and better
runners). Kept in the feature table for future use, excluded from
training. See `notebooks/eda.ipynb`, sections 10.8-10.9.

## Models

Date-based split — trained on 2023/03/30-2025/06/01, tested on the
chronologically last 20% of rows (2025/06/01-2025/11/01)
(`python -m src.train`):

| model | log loss | brier | AUC |
|---|---|---|---|
| Logistic regression | 0.4857 | 0.1600 | **0.6789** |
| XGBoost | **0.4838** | **0.1592** | 0.6762 |

- Adding `is_double_steal`, `steal_of_third`/`steal_of_home`, `runner_age`,
  and `runner_on_third` took AUC from ~0.60 to **~0.68** and log loss
  from ~0.52 to **~0.484** — a real, meaningful jump, not the modest one we
  saw from Statcast alone. `is_double_steal`, `steal_of_home`, and
  `runner_on_third` are XGBoost's top-3 features by importance. The
  `play_seq`-based exact chronological ordering (added for the batter-stats
  work, see Features above) also improved precision slightly on top of that
  (XGBoost log loss 0.4847 → 0.4838).
- XGBoost and logistic land within noise of each other; neither dominates.
  (`fit_xgboost` uses a validation slice only to pick the right number of
  boosting rounds via early stopping, then refits on the full training set
  at that round count — using the early-stopped model directly was
  throwing away 15% of training data and had flipped XGBoost from beating
  logistic to losing to it.)
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

## Decision layer

`src/run_expectancy.py` builds the RE24 table (expected runs to the end of
the inning, by base state x outs) from the same 2023-2025 Retrosheet
play-by-play the model trains on — mixing in 2021-2022 would repeat the same
mistake `src/features.py` already avoids, since the run-scoring environment
plausibly shifted with the 2023 rules too. Values match textbook anchors
(bases empty/0-out ≈ 0.50, loaded/0-out ≈ 2.3) and the break-even rate for a
steal of 2nd lands at 71-74% depending on outs, matching ROADMAP.md's
expected 70-75% range — but it swings as high as ~90%+ when there's a
runner also on 3rd (getting caught costs a lot more when it also risks that
runner) and is lower with 2 outs (getting caught just ends the inning either
way, so there's less left to lose).

### Win probability for late/close games

RE24 optimizes for total runs, which is the wrong goal once a game is close
to over — a run to tie is worth far more than a run in a blowout, and "getting
caught just ends the inning" badly understates the cost when that inning is
a trailing team's last chance. `src/win_probability.py` builds an empirical
win-probability table the same way (real historical plays, not a formula),
but keyed on inning/half/outs/bases/score margin, with `won` = whether the
batting team went on to win the whole game. `src/demo_decision.py` swaps to
this table automatically for high-leverage situations (7th inning or later,
score within 3 runs — `is_high_leverage()`).

Four things worth knowing about how it's built:
- **A caught stealing that makes the 3rd out isn't priced via RE24-style
  algebra** ("flip to the opponent's perspective"). Instead the table
  separately tracks every real historical moment a half-inning actually
  ended, at that exact score/inning — so a trailing team's last-out loss
  falls out of the data directly, including one case handled as a logical
  certainty rather than an empirical estimate: if the home team's half-inning
  ends in the 9th or later while they're still behind, the game is over
  (P(win)=0%, by the rules of baseball, not a small sample).
- **Every answer reports the sample size behind it**, with a fallback chain
  that always preserves the exact inning/half (never blending a 9th-inning
  question with 3rd-inning data) before widening anything else. Extreme
  corners (e.g. a big lead late with runners on) are honestly sparse even
  with this much history and get flagged low-confidence rather than
  presented with false precision.
- **The "current state" baseline excludes historical steal attempts.** The
  break-even question is "steal now vs. hold now," so the pre-decision
  baseline needs to specifically represent "hold" — not an average across
  every real historical instance of that state, which would blend in the
  ~11% of the time a steal actually was attempted next (that answers "what
  usually happens here," a different question). `build_win_prob(...,
  hold_only=True)` filters those out, using a flag `iter_plays_for_win_prob`
  tags on every play.
- **Two DIFFERENT season ranges feed two DIFFERENT tables, and the split is
  load-bearing, not stylistic.** `win_prob_break_even` takes both a `table`
  (for after-success/after-caught) and a `hold_table` (for the current-state
  baseline) as separate arguments — see below for why they can't share a
  season range.

**Concretely, down 1 with 2 outs in the bottom of the 9th** (runner on 1st,
steal of 2nd): win-probability break-even is **46.5%** — well below RE24's
~71-74% for the same base-out situation, since you have much less to
protect when you're already unlikely to win. Tied in that same spot is
still below RE24's baseline (**64.8%**), just by a smaller margin — a
caught stealing there doesn't lose the game (it just sends it to extras),
so the bar drops, but not nearly as much as when you're trailing outright.
Leading (up 1 or more) flips the picture entirely: break-even hits 100% —
protect the lead, don't risk it. None of this is hardcoded — it's what the
data says once "value" means win probability instead of runs.

**Four real corrections happened getting here, each surfaced by a direct
question rather than assumed away** (see `notebooks/eda.ipynb`, sections
14.5-14.8 for the full before/after comparisons):
1. *Sample size.* The first version (3 seasons, unconditional baseline) gave
   45.6% (down 1) / 90.2% (tied). Neither cell was ever flagged
   low-confidence (both cleared `n>=20`), but adding 2021-2022 moved both by
   double digits. `n>=20` was enough to avoid nonsensical results, not
   enough for a *stable* one.
2. *Baseline definition.* Even with 5 seasons, using the unconditional
   "current state" value (blending in real steal attempts) gave 53.7% /
   61.4% — still measurably off from the correct hold-only answer, especially
   for the tied case. Validated the fix against an independent,
   from-scratch computation (manually filtering to "no steal attempted
   next" from the raw play-by-play) — matched closely.
3. *How far back is safe to extend.* Pulled 2013-2020 (eight more seasons)
   to shrink the sparse cells further, restricted to `hold_only=True` so
   the 2023 rule change couldn't leak in directly. Checked whether that
   restriction was *sufficient*, not just applied it: the hold-only
   baseline turned out to be genuinely era-consistent (2013-2020 vs.
   2021-2025 differ by 0.04 percentage points for the down-1 case) — safe
   to extend. But the after-success value is NOT era-consistent (11.2% vs.
   13.9% for the same state) — because once a runner reaches 2nd, whether
   *they* subsequently advance further is itself shaped by the bigger-base
   rule (more generally: taking extra bases on ANY play, not just literal
   steals, got easier in 2023 — confirmed by checking two completely
   steal-unrelated states, which showed only noise-level era gaps by
   comparison), a channel `hold_only` doesn't reach.
4. *A real bug this uncovered.* Asked directly whether "one era was ruining
   the actual probability" — checking that question caught a genuine bug:
   `table`'s default season range was `MODERN_SEASONS` (2021-2025), reused
   from the hold-only baseline's range, which **still includes the two
   pre-rule-change seasons** step 3 had just proven that table is sensitive
   to. Fixed by adding a third, distinct constant —
   `POST_RULE_CHANGE_SEASONS` (2023-2025) — and using it explicitly for
   `table`'s default, rather than reusing a "modern" constant that was
   designed around a *different* rule boundary (2020's automatic
   extra-innings runner, not 2023's bigger bases). This changed the
   after-success value for the down-1 case from 13.9% to the correct 15.3%,
   and moved the final break-even numbers from 51.4%/75.3% to 46.5%/64.8%.

The qualitative story (trailing lowers the bar well below RE24's baseline,
tied lowers it too but by much less, leading raises it to "never risk it")
held up across every version — that's the part to trust throughout. But the
exact decimals needed all four fixes to settle where they should, and step
4 in particular is a reminder that "we checked X is safe" doesn't
automatically mean every piece of code that touches related data actually
uses that checked-safe range — worth an explicit regression test
(`tests/test_win_probability.py`) rather than trusting it stays fixed.

`src/demo_decision.py` fits the real model (not hardcoded example
probabilities) on the same train split `src/train.py` uses, then walks real
**held-out** test-set attempts — biased to include some high-leverage
situations too, since a plain random sample is mostly innings 1-6 — comparing
each one's predicted probability against its own situational break-even
(RE24 or win probability, whichever applies) for a GO/HOLD call, and shows
what actually happened next to it. This is illustrative, not a backtest:
every row shown is an attempt that happened, so we can see whether GO calls
would have paid off, but not what a HOLD call would have avoided.

## Backtest

`src/backtest.py` is the real backtest (ROADMAP.md, "Step 5"): every row of
the **full** held-out test set (not a hand-picked sample) gets a GO/HOLD
call, and the model's GO-only policy's realized value (HOLD scored as 0 —
the break-even framework's own zero point) is totaled up against what
actually happened, separately for RE24 (runs) and win-probability
(high-leverage) situations, since the two aren't the same currency.

```bash
python -m src.backtest --out data/sample/backtest_2025.csv
```

Result, XGBoost, 2,714 held-out attempts (2025/06/01-2025/11/01):

| layer | attempts | GO success rate | HOLD success rate | actual policy | model policy |
|---|---|---|---|---|---|
| RE24 (runs) | 2,039 | 81.1% | 69.5% | +0.0076/attempt | **+0.0309/attempt** |
| Win probability (late/close) | 675 | 82.2% | 80.7% | +0.0069/attempt | **+0.0211/attempt** |

The model's calls would have added real value over the real historical
policy in both regimes, mainly by avoiding the lowest-probability attempts
(of the 819 RE24 situations it would have held, 250 were actually caught
vs. 569 missed opportunities).

**Running every row instead of a sample surfaced a real bug**: double/triple
steals (the target base already occupied by another runner breaking on the
same pitch — e.g. runner on 1st+2nd, trailing runner steals 2nd while the
lead runner simultaneously steals 3rd) were producing nonsensical
break-evens (above 100%, negative rewards on successful steals), because
`run_expectancy._state_after_success` only tracked one runner's movement and
silently overwrote the other. Fixed with a cascade (`_cascade_free`): a
runner arriving at an occupied base pushes the existing occupant forward one
base first (recursively, if that base is occupied too), scoring a run if
someone gets pushed off 3rd. `win_probability.py` and `predict.py` both
consume the same fixed function, so the fix applies everywhere a double
steal's post-state matters, not just the backtest.

What this can and can't show: every row is an attempt that really happened,
so the GO-call success rate and the caught-stealings avoided by HOLD calls
are real, checkable outcomes. It can't show the reverse case — situations
where no one attempted a steal at all, so whether the model would have
(correctly or not) sent a runner who in reality held stays unobservable
from this data.

## For website/app integration

`src/predict.py` is the one module a website backend needs — it wraps
everything above (the trained model, RE24, win probability, the
RE24-vs-win-probability switch) behind a single function call, so nothing
above this section needs to be reimplemented or re-derived to build on top
of it. Start with `notebooks/predict_demo.ipynb` to see it run before
wiring it into a backend.

```python
from src.predict import load_tables, load_model, predict_steal_decision

# Build once, at process/server startup -- both read a decade+ of
# historical data and are too slow to redo per web request.
tables = load_tables()          # RE24 + win-probability tables
model, medians = load_model()   # trained XGBoost model + fill-in values
                                 # for missing runner/catcher inputs

# Call once per request/situation.
result = predict_steal_decision(
    tables, model, medians,
    # game state
    inning=9, half=1,           # half: 0 = top (visiting team batting), 1 = bottom (home team batting)
    outs=2,
    base_code="1__",            # 8-char base state, e.g. "1__" = runner on 1st only,
                                 # "_23" = runners on 2nd and 3rd -- see src/run_expectancy.BASE_STATES
    score_diff=-1,               # batting team's score minus the other team's, right now
    target="2",                 # which base the runner is stealing: "2", "3", or "H"
    # player inputs -- all optional; omit/None falls back to a sensible default
    runner_sprint_speed=29.5,   # Statcast ft/s
    runner_age=27,
    catcher_pop_time=1.95,      # Statcast seconds, catcher's throw-to-base time
    runner_prior_sr=0.82,       # runner's stolen-base success rate so far
    runner_prior_att=25,        # runner's stolen-base attempts so far
    pitcher_prior_sr_allowed=0.78,
    catcher_prior_cs_rate=0.28,
    runner_bats_lhb=False,
    pitcher_throws_lhp=False,
    balls=1, strikes=1,
    is_double_steal=False,
)
```

`result` is a plain dict, ready to hand straight to a frontend:

| key | meaning |
|---|---|
| `win_prob_current` | win probability right now, before the steal |
| `win_prob_if_success` | win probability if the steal succeeds |
| `break_even` | minimum success rate needed to justify attempting — same 0-1 scale as `p_success` regardless of which layer (RE24 or win probability) produced it |
| `p_success` | the trained model's predicted success probability for this exact situation |
| `decision` | `"GO"` if `p_success > break_even`, else `"HOLD"` |
| `layer` | which engine actually decided: `"RE24"`, `"WP"` (win probability, high-leverage), or `"WP (RE24 had no data)"` (see below) |
| `min_n` / `low_confidence` | sample size behind the answer and whether it's thin (`< 20`) — worth showing as a caveat in the UI rather than a bare number |

For showing several situations in one table (e.g. a "what if" comparison
across counts, or a full lineup), `predict_steal_decisions_table` takes a
list of the same kwargs and returns a single `pandas.DataFrame` with
`decision` as the last column.

**Two things this handles automatically, without extra code on the caller's
side:**
- **Missing player inputs.** A website user won't always know a specific
  runner's exact sprint speed, or the app may not have looked it up yet.
  Leave that field `None` (or omit it) and it falls back to the training
  set's median value — never a crash, never a nonsensical zero.
- **Rare base/out combinations RE24 has no historical data for.** Rather
  than raising an error, `predict_steal_decision` falls through to the
  win-probability path (which degrades gracefully on its own — see
  "Win probability for late/close games" above) and flags the result as
  `"WP (RE24 had no data)"` so the caller can see that happened.

## Known limitations

- No lead distance / jump data exists publicly — sprint speed is the proxy.
- Pitcher tempo, pitch type/location, and pickoff-attempt rate aren't
  cleanly available from public Statcast leaderboards.
- Late-inning decisions should eventually use win probability, not run
  expectancy (see `ROADMAP.md`, "one important upgrade").
