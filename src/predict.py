"""
Single entry point for the website handoff: turn game-state + player inputs
into one decision row that carries every number the UI should show, not
just the final GO/HOLD verdict -- current win probability, win probability
if the steal succeeds, the break-even success rate the model has to clear,
the model's own predicted success probability, and the decision itself.

This doesn't reimplement any decision math -- it wires together the two
existing layers (src/run_expectancy.py's RE24 for most of the game,
src/win_probability.py's win-probability break-even for high-leverage
late/close situations -- see is_high_leverage) into one row, the same
hybrid src/demo_decision.py already uses. Two things this adds on top:

  * `win_prob_current` / `win_prob_if_success` are always computed and
    shown, even when the DECISION itself is made by RE24. Early in the
    game, win probability is noisier and RE24 is the right optimization
    target (see win_probability.py's module docstring), but it's still
    informative to a user watching the count tick toward "break-even."
  * If RE24 has no data at all for the requested (base_code, outs) --
    should_steal()/break_even_rate() would raise a bare KeyError in that
    case -- this falls through to the win-probability path instead, which
    already has a graduated fallback chain (win_prob_lookup) ending in a
    flagged "no data (default)" rather than an exception. A website
    shouldn't 500 because a user picked an exotic base-out combo.

Both `break_even` (whichever layer produced it) and `p_success` (the
model's prediction) are probabilities in the same units -- "success rate
needed" vs. "predicted success rate" -- directly comparable regardless of
which layer answered, so the final `decision` column is just p_success >
break_even either way.

Usage as a library (what a website backend would do -- build the tables
and fit the model ONCE at process startup, then call predict_steal_decision
per request):

    from src.predict import load_tables, load_model, predict_steal_decision

    tables = load_tables()
    model, medians = load_model()
    row = predict_steal_decision(
        tables, model, medians,
        inning=9, half=1, outs=2, base_code="1__", score_diff=-1, target="2",
        runner_sprint_speed=29.5, catcher_pop_time=1.9, runner_prior_sr=0.82,
    )

CLI demo, walking a handful of built-in example situations into one table:
    python -m src.predict
"""
from __future__ import annotations

from .run_expectancy import (
    build_re24, break_even_rate, _state_after_success, BASE_STATES,
)
from .win_probability import (
    build_win_prob, win_prob_lookup, win_prob_break_even, is_high_leverage,
    LEGACY_SEASONS, MODERN_SEASONS, POST_RULE_CHANGE_SEASONS, _season_dirs,
)
from .train import NUMERIC, fit_logistic, fit_xgboost


def load_tables() -> dict:
    """Build every table predict_steal_decision needs, once. Season ranges
    match src/demo_decision.py exactly -- see win_probability.py's module
    docstring for why the after-success/after-caught table and the
    hold-only baseline deliberately use different, non-interchangeable
    season ranges.
    """
    re24 = build_re24(_season_dirs(POST_RULE_CHANGE_SEASONS))
    wp_table = build_win_prob(_season_dirs(POST_RULE_CHANGE_SEASONS))
    wp_hold_table = build_win_prob(
        _season_dirs(LEGACY_SEASONS) + _season_dirs(MODERN_SEASONS),
        hold_only=True, legacy_dirs=_season_dirs(LEGACY_SEASONS),
    )
    return {"re24": re24, "wp_table": wp_table, "wp_hold_table": wp_hold_table}


def load_model(features_path="data/sample/features_2023_2025.csv",
               model_kind="xgboost", test_frac=0.2):
    """Fit on the same train split src/train.py/demo_decision.py use
    (earliest 1-test_frac of dates). Returns (model, medians) -- medians
    are the training-set medians for the three Statcast join columns, so a
    caller with a missing runner_sprint_speed/runner_age/catcher_pop_time
    (e.g. an unknown or minor-league player) gets the same median-fill
    behavior train.py applies, instead of a nonsensical 0.
    """
    import pandas as pd

    df = pd.read_csv(features_path)
    medians = {col: df[col].median() for col in
              ("runner_sprint_speed", "runner_age", "catcher_pop_time")}
    for col, med in medians.items():
        df[col] = df[col].fillna(med)

    split_idx = int(len(df) * (1 - test_frac))
    train = df.iloc[:split_idx]
    X_tr, y_tr = train[NUMERIC].fillna(0.0), train["success"].astype(int)
    model = fit_logistic(X_tr, y_tr) if model_kind == "logistic" else fit_xgboost(X_tr, y_tr)
    return model, medians


def _build_feature_row(medians, *, outs, balls, strikes, score_diff, target,
                       runner_on_third, is_double_steal, inning,
                       runner_bats_lhb, pitcher_throws_lhp,
                       runner_prior_sr, runner_prior_att,
                       pitcher_prior_sr_allowed, catcher_prior_cs_rate,
                       runner_sprint_speed, runner_age, catcher_pop_time) -> dict:
    """Derive the model's NUMERIC feature row the same way src/features.py
    does from raw play-by-play, but from direct game-state/player inputs
    instead. Any of the three Statcast fields left as None gets the
    training-set median (see load_model) with its _missing flag set,
    exactly matching how features.py handles an unmatched Statcast join.
    """
    return {
        "steal_of_third": int(target == "3"),
        "steal_of_home": int(target == "H"),
        "is_double_steal": int(is_double_steal),
        "runner_on_third": int(runner_on_third and target != "H"),
        "late_inning": int(inning >= 7),
        "outs": outs,
        "balls": balls,
        "strikes": strikes,
        "score_diff": score_diff,
        "close_game": int(abs(score_diff) <= 1),
        "runner_bats_lhb": int(runner_bats_lhb),
        "pitcher_throws_lhp": int(pitcher_throws_lhp),
        "runner_prior_sr": runner_prior_sr,
        "runner_prior_att": runner_prior_att,
        "pitcher_prior_sr_allowed": pitcher_prior_sr_allowed,
        "catcher_prior_cs_rate": catcher_prior_cs_rate,
        "runner_sprint_speed": runner_sprint_speed if runner_sprint_speed is not None else medians["runner_sprint_speed"],
        "runner_sprint_speed_missing": int(runner_sprint_speed is None),
        "runner_age": runner_age if runner_age is not None else medians["runner_age"],
        "runner_age_missing": int(runner_age is None),
        "catcher_pop_time": catcher_pop_time if catcher_pop_time is not None else medians["catcher_pop_time"],
        "catcher_pop_time_missing": int(catcher_pop_time is None),
    }


def predict_steal_decision(
    tables: dict, model, medians: dict, *,
    inning: int, half: int, outs: int, base_code: str, score_diff: int, target: str,
    balls: int = 0, strikes: int = 0, is_double_steal: bool = False,
    runner_bats_lhb: bool = False, pitcher_throws_lhp: bool = False,
    runner_prior_sr: float = 0.0, runner_prior_att: int = 0,
    pitcher_prior_sr_allowed: float = 0.0, catcher_prior_cs_rate: float = 0.0,
    runner_sprint_speed: float | None = None, runner_age: float | None = None,
    catcher_pop_time: float | None = None,
) -> dict:
    """One steal decision, every number the UI needs in one dict.

    base_code: 8-char state string like run_expectancy.BASE_STATES
    ("1__", "_23", ...). target: "2", "3", or "H".
    """
    re24, wp_table, wp_hold_table = tables["re24"], tables["wp_table"], tables["wp_hold_table"]
    runner_on_third = base_code[2] != "_"

    row = _build_feature_row(
        medians, outs=outs, balls=balls, strikes=strikes, score_diff=score_diff,
        target=target, runner_on_third=runner_on_third, is_double_steal=is_double_steal,
        inning=inning, runner_bats_lhb=runner_bats_lhb, pitcher_throws_lhp=pitcher_throws_lhp,
        runner_prior_sr=runner_prior_sr, runner_prior_att=runner_prior_att,
        pitcher_prior_sr_allowed=pitcher_prior_sr_allowed, catcher_prior_cs_rate=catcher_prior_cs_rate,
        runner_sprint_speed=runner_sprint_speed, runner_age=runner_age, catcher_pop_time=catcher_pop_time,
    )
    import pandas as pd
    X = pd.DataFrame([row])[NUMERIC]
    p_success = float(model.predict_proba(X)[:, 1][0])

    # Always compute win probability for display, regardless of which layer
    # ends up making the decision -- see module docstring.
    succ_base = _state_after_success(base_code, target)
    succ_score = score_diff + (1 if target == "H" else 0)
    wp_current, n_cur, src_cur = win_prob_lookup(wp_hold_table, inning, half, outs, base_code, score_diff)
    wp_if_success, n_succ, src_succ = win_prob_lookup(wp_table, inning, half, outs, succ_base, succ_score)

    high_leverage = is_high_leverage(inning, score_diff)
    layer = "WP"
    min_n = min(n_cur, n_succ)
    sources = (src_cur, src_succ)
    if not high_leverage and (base_code, outs) in re24:
        be, reward, cost = break_even_rate(re24, base_code, outs, target)
        layer = "RE24"
    else:
        # High-leverage situation, OR RE24 simply has no data for this
        # (base_code, outs) combo (a genuine coverage gap rather than the
        # usual high-leverage swap) -- either way, win-probability's
        # fallback chain degrades gracefully instead of raising.
        be, reward, cost, min_n, sources = win_prob_break_even(
            wp_table, wp_hold_table, inning, half, outs, base_code, score_diff, target)
        if not high_leverage:
            layer = "WP (RE24 had no data)"

    return {
        "inning": inning, "half": half, "outs": outs, "base_code": base_code,
        "score_diff": score_diff, "target": target,
        "win_prob_current": wp_current,
        "win_prob_if_success": wp_if_success,
        "break_even": be,
        "p_success": p_success,
        "decision": "GO" if p_success > be else "HOLD",
        "layer": layer,
        "min_n": min_n,
        "low_confidence": min_n < 20,
        "sources": sources,
    }


def predict_steal_decisions_table(tables: dict, model, medians: dict, situations: list):
    """Batch version: a list of predict_steal_decision(...)-style kwarg
    dicts in, one pandas DataFrame out -- one row per situation, decision
    as the final column, ready to hand a website table straight to a
    frontend.
    """
    import pandas as pd

    rows = [predict_steal_decision(tables, model, medians, **s) for s in situations]
    cols = ["inning", "half", "outs", "base_code", "score_diff", "target",
           "win_prob_current", "win_prob_if_success", "break_even", "p_success",
           "decision", "layer", "min_n", "low_confidence"]
    return pd.DataFrame(rows)[cols]


def main():
    print("Building RE24 + win-probability tables...")
    tables = load_tables()
    print("Fitting model...")
    model, medians = load_model()

    situations = [
        dict(inning=3, half=0, outs=1, base_code="1__", score_diff=0, target="2",
             runner_sprint_speed=29.0, catcher_pop_time=2.0, runner_prior_sr=0.78, runner_prior_att=40),
        dict(inning=9, half=1, outs=2, base_code="1__", score_diff=-1, target="2",
             runner_sprint_speed=29.5, catcher_pop_time=1.95, runner_prior_sr=0.82, runner_prior_att=25),
        dict(inning=9, half=1, outs=2, base_code="1__", score_diff=0, target="2",
             runner_sprint_speed=29.5, catcher_pop_time=1.95, runner_prior_sr=0.82, runner_prior_att=25),
        dict(inning=8, half=1, outs=1, base_code="1__", score_diff=1, target="2",
             runner_sprint_speed=27.0, catcher_pop_time=2.0, runner_prior_sr=0.70, runner_prior_att=15),
        dict(inning=5, half=0, outs=0, base_code="_2_", score_diff=2, target="3",
             runner_sprint_speed=30.0, catcher_pop_time=1.9, runner_prior_sr=0.85, runner_prior_att=20),
    ]
    df = predict_steal_decisions_table(tables, model, medians, situations)
    print()
    print(df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
