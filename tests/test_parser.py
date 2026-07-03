"""Regression tests validating the Retrosheet parser against known 2023 facts.

Run:  python -m pytest tests/ -q      (from the repo root)
"""
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrosheet_parser import parse_season  # noqa: E402
from src.run_expectancy import build_re24        # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "retrosheet_2023")


def test_every_runner_resolved():
    rows, diag, _ = parse_season(DATA)
    assert diag["missing_runner"] == 0, "some steal runners were unresolved"
    assert len(rows) > 4000  # 2023 was a high-steal season (rule changes)


def test_success_rate_reasonable():
    rows, _, _ = parse_season(DATA)
    sb = sum(r["success"] for r in rows)
    rate = sb / len(rows)
    assert 0.78 <= rate <= 0.82, f"league SB success rate off: {rate:.3f}"


def test_leaderboard_matches_reality():
    rows, _, _ = parse_season(DATA)
    sb = Counter(r["runner_id"] for r in rows if r["success"])
    leader, count = sb.most_common(1)[0]
    # Ronald Acuna Jr. (acunr001) led MLB in 2023 with ~70+ steals.
    assert leader == "acunr001"
    assert count >= 70


def test_re24_matches_textbook():
    re24 = build_re24(DATA)
    # Classic run-expectancy anchors (tolerances are generous).
    assert abs(re24[("___", 0)] - 0.51) < 0.06   # bases empty, 0 out
    assert abs(re24[("123", 0)] - 2.28) < 0.15   # bases loaded, 0 out
    # Run expectancy must fall as outs increase from the same base state.
    for bc in ("___", "1__", "123"):
        assert re24[(bc, 0)] > re24[(bc, 1)] > re24[(bc, 2)]
