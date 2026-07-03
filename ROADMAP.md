# Stolen Base Decision Model — Project Roadmap

*A build plan for Ethan & Colin*

## What we're building

Two connected pieces:

1. **A success-probability model.** Given a specific game situation — who's running, who's pitching, who's catching, the inning, the score, the count, and so on — it outputs a single number: the probability that the runner would successfully steal the base.

2. **A decision layer.** It takes that probability and answers the real question: *is attempting the steal actually worth it?* A 60% chance of stealing is a great idea with two outs in a tie game and a terrible idea when you're the tying run in the 9th. The decision layer uses **run expectancy** to figure out the break-even point for each situation and compares your predicted probability against it.

The first piece is a machine-learning problem. The second is mostly a math/lookup problem layered on top. Keeping them separate is deliberate — it means you can improve the "how likely" engine and the "is it worth it" engine independently, and you can always see *why* the model recommended what it did.

---

## Step 0: Understanding the data sources (the part that was unclear)

To train a model that predicts steal success, we need historical examples: thousands of real steal attempts with their outcomes (safe or out) and the surrounding context. There are three main places that data lives. Here's what each one actually is, in plain terms.

**Retrosheet** is a decades-long volunteer project that has recorded *every play of every MLB game* going back over a century. It's free. For any play it tells you the base-out state (who's on which base, how many outs), the inning, the score, the batter, pitcher, and the result — including every stolen base and every runner thrown out. What it does **not** have is any "tracking" data: no running speeds, no throw times. It's the *what happened* dataset.

**Statcast** is MLB's camera/radar tracking system, live since 2015. It records the physical stuff: how fast a runner sprints, how hard a pitcher throws, and (through separate leaderboards) catcher "pop time" — how quickly a catcher gets the ball to second base — and pitcher "tempo," how long a pitcher takes to deliver. This is the *how good is each player at the specific skills that matter for stealing* dataset. The tradeoff is it only goes back to 2015, and some of the most steal-specific numbers (a runner's exact lead distance and "jump" on a given play) aren't cleanly available through the public feed.

**FanGraphs** is a stats site that publishes useful pre-computed summaries — like each runner's speed score and each player's baserunning value. Handy for sanity checks and as backup features, not essential.

All three are reachable from Python. The library **`pybaseball`** wraps Statcast and Retrosheet so you can pull data in a few lines of code without scraping websites by hand.

### Recommendation

**Use Statcast as the backbone, enriched with Retrosheet, focused on 2015–present.**

The reason: stealing is *specifically* about runner speed vs. pitcher-hold-time vs. catcher arm. Statcast is the only source that measures those three skills directly, and they're almost certainly your most predictive features. Retrosheet then fills in clean situational context and gives you a much larger, well-structured set of labeled steal attempts. Limiting to the Statcast era (2015+) costs you deep history but keeps every row comparable and every player measurable. That's the right trade for a first working model — and roughly 10 seasons is plenty of data.

---

## Step 1: Build the dataset

The goal of this phase is a single clean table where **each row is one "steal opportunity"** and the columns are everything we knew *before* the pitch, plus the outcome.

Defining "steal opportunity" matters. The cleanest starting definition: a runner on first (or second) base, with the next base open, at the moment of a pitch. Most of these opportunities end in *no attempt*, some end in a *successful steal*, and some in a *caught stealing*. For the first model you'll likely focus on the attempts (safe vs. out) to predict success probability. Later you can model the "will they even go?" question too.

Watch out for a subtle trap here called **data leakage**: when you attach a runner's season stolen-base success rate as a feature, make sure it's computed from data that *excludes* the play you're predicting (ideally use the *prior* season, or a running total up to that date). Otherwise the model gets to peek at the answer and will look brilliant in testing and fail in real life.

---

## Step 2: Features — what the model gets to look at

Grouped by where each comes from.

**The runner**

- Sprint speed (Statcast) — the single best raw indicator
- Career and prior-season stolen-base success rate, plus number of attempts (so the model knows how trustworthy that rate is)
- Age

**The pitcher**

- Time to the plate / tempo from the stretch — slow deliveries get stolen on
- Handedness — lefties face first base and hold runners at first much better
- Career rate of steals allowed and pickoffs

**The catcher**

- Pop time (Statcast) and arm strength
- Career caught-stealing percentage

**The situation**

- Which base is being stolen (second is far easier than third)
- The count (runners often go on counts favorable to them)
- Number of outs
- Inning and score differential (this mostly feeds the *decision* layer, but can also affect whether pitchers/catchers are focused on holding the runner)

Start with this set. Don't over-engineer before you have a baseline working.

---

## Step 3: The success-probability model

**Baseline first: logistic regression.** It's simple, fast, naturally outputs a probability between 0 and 1, and — crucially — is *interpretable*. You'll be able to read off roughly how much each factor moves the odds, which is a great gut-check that the model learned real baseball ("faster runners and slower pitchers steal more") rather than noise.

**Then: gradient boosting (XGBoost or LightGBM).** These almost always beat logistic regression on this kind of tabular data and automatically capture interactions — e.g. a fast runner against a slow lefty with a weak-armed catcher compounds in a way a simple linear model misses. This becomes your production model once it's beating the baseline.

**Calibration is the thing to obsess over.** For this project, being *well-calibrated* matters more than raw accuracy: when the model says 70%, runners should actually be safe about 70% of the time. That's the whole premise of the decision layer. So you'll evaluate primarily with **log loss** and **Brier score** (both reward honest probabilities), check **AUC** for ranking ability, and — most importantly — draw a **calibration curve** (predicted probability on one axis, actual success rate on the other; you want it hugging the diagonal). If the raw model is miscalibrated, apply Platt scaling or isotonic regression to fix it.

**Validate across time, not randomly.** Train on earlier seasons, test on later ones (e.g. train 2015–2022, test 2023–2024). A random split lets the model "learn" from the future, which you won't have in real use.

---

## Step 4: The decision layer — "is it worth it?"

This is where run expectancy comes in, and it's the part that turns a probability into a recommendation.

**Run expectancy (RE24)** is a table with 24 rows — one for each combination of base state (8 possibilities) and number of outs (0, 1, 2). Each cell says: *on average, how many more runs does a team score for the rest of this inning starting from this state?* You can compute this table directly from Retrosheet. For example, a runner on first with one out is worth some expected number of runs; move him to second and that number goes up; get him thrown out and it drops.

A stolen base attempt is a bet between two of those cells:

- **If safe:** you move to a better base state → run expectancy goes *up* by some amount (call it the *reward*).
- **If caught:** you lose the runner *and* add an out → run expectancy goes *down* by some amount (call it the *cost*). The cost is almost always bigger than the reward.

From those two numbers you get the **break-even success rate** — the probability at which attempting is exactly neutral:

> break-even % = cost ÷ (reward + cost)

The decision rule is then simply: **attempt the steal if the model's predicted success probability is greater than the break-even rate for that exact situation.** Because reward and cost change with every base-out state, the break-even bar is different in every situation — typically somewhere around 70–75%, but higher when an out is especially costly and lower when the extra base is especially valuable.

**One important upgrade for late, close games.** Run expectancy optimizes for *total runs*, which is the right goal in the early innings. But in the 8th or 9th of a one-run game, you care about *winning*, not scoring the most runs — and there, runs aren't worth a linear amount (one run to tie is worth far more than a fourth run in a blowout). For those spots you'd swap the run-expectancy table for a **win-probability** table and compute the break-even the same way. A clean plan: ship the run-expectancy version first, then layer win-probability on top for high-leverage late-game situations.

---

## Step 5: Evaluate the whole system

Two different questions, two different tests.

*Is the probability model good?* → log loss, Brier score, AUC, and the calibration curve, all on the held-out later seasons.

*Is the decision layer good?* → backtest it. Run every real historical steal attempt through the system and compare the total run/win value your recommendations would have produced against what actually happened. If your model's "green light" attempts collectively gained more expected runs than the league's real decisions, the system is adding value. This is also the most fun output to show people.

---

## Suggested phased plan

**Phase 1 — Get the data flowing.** Install `pybaseball`, pull a single season of Statcast, and successfully extract every steal attempt with its outcome and context. Success = one clean table you can eyeball.

**Phase 2 — Baseline model.** Add features, train logistic regression, check that its coefficients make baseball sense and that it's roughly calibrated. Success = a probability you trust for obvious cases.

**Phase 3 — Real model.** Add all seasons, train gradient boosting, calibrate it, validate across time. Success = beats the baseline on log loss and Brier score.

**Phase 4 — Decision layer.** Build the run-expectancy table from Retrosheet, compute break-even rates, wire up the "steal / don't steal" recommendation. Success = sensible recommendations on hand-checked scenarios.

**Phase 5 — Backtest & polish.** Evaluate the full system against history, add the win-probability upgrade for late innings, and build a simple interface where you type in a situation and get a recommendation.

---

## Tech stack

Python throughout. `pybaseball` for data, `pandas` for wrangling, `scikit-learn` for the baseline and calibration tools, `xgboost` or `lightgbm` for the main model, `matplotlib` for the calibration plots. A Jupyter notebook is the natural home for Phases 1–3; you can graduate to plain scripts once things stabilize.

## Honest limitations to keep in mind

- **Lead and jump aren't cleanly available.** A runner's exact lead and reaction time are huge in reality but not in the public data, so the model can't see them directly. Sprint speed is a decent proxy.
- **Pitcher intent is invisible.** Whether a pitcher was actively holding a runner or pitching out isn't recorded, which adds irreducible noise.
- **The data reflects decisions managers already made.** Runners mostly go when conditions favor them, so observed success rates are biased upward relative to a random runner going in a random spot. The backtest partly accounts for this, but it's worth remembering.

None of these sink the project — they just set a realistic ceiling on accuracy and are good to name up front.

---

*Next step whenever you're ready: I can start on Phase 1 and write the actual data-pull code so you and Colin have a real table to look at.*
