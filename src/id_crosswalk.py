"""
Build a Retrosheet <-> MLBAM player-ID crosswalk via the Chadwick register.

Retrosheet IDs look like "acunr001"; Statcast (and everything pybaseball
pulls) keys on numeric MLBAM IDs like 660670. The Chadwick Bureau register
carries both, so we pull it once and cache a small two-column CSV that
`features.py` can join against.

    python -m src.id_crosswalk --out data/statcast/id_crosswalk.csv
"""
from __future__ import annotations

import argparse
import os


def build_crosswalk():
    from pybaseball import chadwick_register

    reg = chadwick_register()
    reg = reg[reg["key_retro"].notna() & reg["key_mlbam"].notna()]
    reg = reg[["key_retro", "key_mlbam"]].drop_duplicates("key_retro")
    reg["key_mlbam"] = reg["key_mlbam"].astype(int)
    return reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/statcast/id_crosswalk.csv")
    args = ap.parse_args()

    reg = build_crosswalk()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    reg.to_csv(args.out, index=False)
    print(f"wrote {len(reg)} retro<->mlbam id mappings -> {args.out}")


if __name__ == "__main__":
    main()
