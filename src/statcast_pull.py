"""
Statcast data pull via pybaseball — RUN THIS ON YOUR OWN MACHINE.

Statcast (baseballsavant.mlb.com) is blocked inside the sandbox this repo was
scaffolded in, but it works fine from a normal machine. Statcast gives you the
tracking features that matter most for steals and that Retrosheet lacks:

  * runner sprint speed        (statcast_sprint_speed leaderboard)
  * catcher pop time / arm     (statcast_catcher_poptime leaderboard)
  * pitcher tempo / handedness (per-pitch data + player info)

This module pulls those season-level skill leaderboards so they can be joined
onto the Retrosheet steal-attempt table by player. Extend as needed.

Setup:
    pip install pybaseball pandas
    python -m src.statcast_pull --season 2023 --out data/statcast
"""
from __future__ import annotations

import argparse
import os


def pull_skill_tables(season: int, out_dir: str) -> None:
    # Imported lazily so the rest of the repo runs without pybaseball installed.
    from pybaseball import (
        statcast_sprint_speed,
        statcast_catcher_poptime,
    )

    os.makedirs(out_dir, exist_ok=True)

    # Runner speed: min_opp=1 keeps nearly everyone who ran.
    sprint = statcast_sprint_speed(season, min_opp=1)
    sprint.to_csv(os.path.join(out_dir, f"sprint_speed_{season}.csv"),
                  index=False)
    print(f"sprint speed: {len(sprint)} players")

    # Catcher pop time / arm strength on steal attempts of 2nd.
    pop = statcast_catcher_poptime(season, min_2b_att=1)
    pop.to_csv(os.path.join(out_dir, f"catcher_poptime_{season}.csv"),
               index=False)
    print(f"catcher pop time: {len(pop)} catchers")

    print(f"wrote Statcast skill tables to {out_dir}")
    print("Next: join these onto data/sample/steals_<season>.csv by player. "
          "Note Statcast uses MLBAM ids; Retrosheet uses its own ids — use "
          "pybaseball.playerid_lookup or the Chadwick register to bridge them.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--out", default="data/statcast")
    args = ap.parse_args()
    pull_skill_tables(args.season, args.out)


if __name__ == "__main__":
    main()
