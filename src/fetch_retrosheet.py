"""
Fetch additional seasons of Retrosheet event files from the GitHub mirror.

The repo ships with 2023 already parsed. To train on more history, pull more
seasons. This uses a sparse git checkout so you only download the seasons you
want, not the entire (large) Retrosheet repository.

    python -m src.fetch_retrosheet --seasons 2021 2022 2024 --dest data

Each season lands in data/retrosheet_<year>/ ready for the parser:

    python -m src.retrosheet_parser --data-dir data/retrosheet_2022 \
        --out data/sample/steals_2022.csv
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile

REPO = "https://github.com/chadwickbureau/retrosheet.git"


def fetch(seasons, dest):
    tmp = tempfile.mkdtemp(prefix="retrosheet_")
    try:
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout",
             "--depth", "1", REPO, tmp],
            check=True,
        )
        paths = [f"seasons/{y}" for y in seasons]
        subprocess.run(["git", "-C", tmp, "checkout", "HEAD", "--", *paths],
                       check=True)
        for y in seasons:
            src = os.path.join(tmp, "seasons", str(y))
            dst = os.path.join(dest, f"retrosheet_{y}")
            os.makedirs(dst, exist_ok=True)
            for name in os.listdir(src):
                if name.split(".")[-1] in ("EVN", "EVA", "EVE", "ROS") \
                        or name.endswith(("EVN", "EVA", "EVE", "ROS")):
                    shutil.copy(os.path.join(src, name), dst)
            print(f"season {y}: files -> {dst}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--dest", default="data")
    args = ap.parse_args()
    fetch(args.seasons, args.dest)


if __name__ == "__main__":
    main()
