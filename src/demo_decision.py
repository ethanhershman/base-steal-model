"""
End-to-end demo of the decision layer: combine a predicted success probability
with the run-expectancy break-even to get a steal / don't-steal recommendation.

    python -m src.demo_decision
"""
from __future__ import annotations

from .run_expectancy import build_re24, should_steal


def main():
    re24 = build_re24("data/retrosheet_2023")

    print("Steal-decision demo (RE24 from 2023)\n")
    scenarios = [
        # base_code, outs, target, predicted P(success)
        ("1__", 1, "2", 0.80),   # good runner, steal of 2nd
        ("1__", 1, "2", 0.65),   # marginal runner
        ("1__", 2, "2", 0.70),   # 2 outs lowers the bar
        ("_2_", 0, "3", 0.75),   # steal of 3rd, no outs (costly if caught)
        ("__3", 1, "H", 0.55),   # steal of home
    ]
    for bc, outs, tgt, p in scenarios:
        d = should_steal(re24, bc, outs, tgt, p)
        verdict = "GO" if d["attempt"] else "HOLD"
        print(f"  bases {bc}  {outs} out  steal->{tgt}  "
              f"P(success)={p:.0%}  break-even={d['break_even']:.0%}  "
              f"net={d['net_run_value']:+.3f} runs  =>  {verdict}")


if __name__ == "__main__":
    main()
