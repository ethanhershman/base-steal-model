"""
Retrosheet event-file parser -> stolen-base attempt dataset.

Retrosheet distributes complete MLB play-by-play as ".EVN"/".EVA" event files
(one per home team per season). Each play is one line:

    play,inning,half,batter_id,count,pitches,event

where `half` is 0 (visitor batting) or 1 (home batting) and `event` is the
Retrosheet event notation (e.g. "S8", "63", "K+SB2", "CS2(26)").

This module walks every play in order, maintaining base/out/score state, and
emits one row per *stolen-base attempt* (SB or CS/POCS) with the surrounding
game context and the identities of the runner, pitcher, and catcher.

It is intentionally dependency-light (standard library only) so it runs
anywhere. Season SB/CS totals are validated against known values in
`tests/test_parser_totals.py`.

Usage:
    python -m src.retrosheet_parser --data-dir data/retrosheet_2023 \
        --out data/sample/steals_2023.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Roster handling (for batter/runner handedness)
# ---------------------------------------------------------------------------
def load_rosters(data_dir: str) -> dict:
    """Map player_id -> {'bats': L/R/B, 'throws': L/R} from .ROS files.

    Roster line format: id,last,first,bats,throws,team,pos
    """
    players = {}
    for path in glob.glob(os.path.join(data_dir, "*.ROS")):
        with open(path, newline="") as fh:
            for row in csv.reader(fh):
                if len(row) >= 5 and row[0]:
                    players[row[0]] = {"bats": row[3], "throws": row[4]}
    return players


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------
@dataclass
class GameState:
    game_id: str = ""
    date: str = ""
    home: str = ""
    visitor: str = ""
    park: str = ""
    # bases[1|2|3] = runner id or None
    bases: dict = field(default_factory=lambda: {1: None, 2: None, 3: None})
    outs: int = 0
    score: list = field(default_factory=lambda: [0, 0])  # [visitor, home]
    inning: int = 0
    half: int = -1  # 0 top / 1 bottom
    # defensive personnel per team: pitcher[team], catcher[team]
    pitcher: dict = field(default_factory=lambda: {0: None, 1: None})
    catcher: dict = field(default_factory=lambda: {0: None, 1: None})
    # extra-innings "ghost runner" placement waiting for the next half-inning
    pending_radj: list = field(default_factory=list)
    # monotonically increasing per-game play counter, for an unambiguous
    # chronological sort key -- (inning, outs) alone can repeat multiple
    # times within one half-inning (any play before the first out shares
    # outs=0, for instance), which isn't precise enough to interleave
    # per-plate-appearance batting rows with steal-attempt rows correctly.
    play_seq: int = 0

    def new_half_inning(self, inning: int, half: int) -> None:
        if inning != self.inning or half != self.half:
            self.bases = {1: None, 2: None, 3: None}
            self.outs = 0
            self.inning = inning
            self.half = half


# ---------------------------------------------------------------------------
# Steal detection
# ---------------------------------------------------------------------------
STEAL_RE = re.compile(r"(SB|CS|POCS)([23H])")
# base a runner steals *from* given the target base of the steal
FROM_BASE = {"2": 1, "3": 2, "H": 3}


def _strip_markers(event: str) -> str:
    # Remove uncertainty / exceptional-play markers that don't affect state.
    return event.replace("!", "").replace("#", "").replace("?", "")


def find_steal_events(event: str):
    """Return list of (kind, target_base) for steal/CS tokens in an event.

    kind is 'SB' (success) or 'CS' (caught, includes POCS pickoff-caught).
    Excludes plain pickoffs (PO1/PO2/PO3) where the runner was not stealing.
    """
    out = []
    # Avoid matching the "CS" inside "POCS" twice: findall on the combined RE
    for m in STEAL_RE.finditer(event):
        kind = "SB" if m.group(1) == "SB" else "CS"
        out.append((kind, m.group(2)))
    return out


# ---------------------------------------------------------------------------
# Base-state updater
# ---------------------------------------------------------------------------
BATTER_HIT = {"S": 1, "D": 2, "T": 3}


def _apply_explicit_advances(adv: str, gs: GameState, snapshot: dict,
                             batter: str) -> bool:
    """Apply the runner-advance segment (after the '.') e.g. '2-H;1-3;BX2'.

    Runner sources are resolved against `snapshot` (base state at the START of
    the play) so that placing the batter earlier can't corrupt a runner's
    identity. Returns True if the batter's own advance (B-x/BXx) was specified.
    """
    batter_moved = False
    if not adv:
        return batter_moved
    for piece in adv.split(";"):
        piece = piece.strip()
        m = re.match(r"([B123])([-X])([123H])", piece)
        if not m:
            continue
        src, kind, dest = m.group(1), m.group(2), m.group(3)
        # An error in the parenthetical negates an out ('X' -> safe).
        safe_on_error = kind == "X" and bool(re.search(r"\([0-9]*E", piece))
        runner = batter if src == "B" else snapshot.get(int(src))
        if src == "B":
            batter_moved = True
        else:
            gs.bases[int(src)] = None  # vacate the original base
        if kind == "X" and not safe_on_error:
            if runner is not None:
                gs.outs += 1
        else:
            if dest == "H":
                if runner is not None:
                    gs.score[gs.half] += 1
            else:
                gs.bases[int(dest)] = runner
    return batter_moved


def update_state(event: str, gs: GameState, batter: str) -> None:
    """Mutate game state (bases/outs/score) for a single play's event.

    Order of operations matters. We snapshot base occupancy first, resolve all
    runner movements/outs against that snapshot, and only THEN place the batter.
    """
    event = _strip_markers(event).strip()
    if event in ("NP", ""):  # no play (substitution placeholder)
        return

    basic, _, adv = event.partition(".")
    primary = basic.split("/")[0]
    tokens = re.split(r"[+;]", primary)

    snapshot = dict(gs.bases)   # base state before the play
    batter_dest = None          # 1/2/3/'H'/'out'/None

    for tok in tokens:
        # ---- stolen bases / caught stealing / pickoffs (runner moves) ----
        sm = re.match(r"(SB|CS|POCS|PO)([123H])", tok)
        if sm:
            act, base = sm.group(1), sm.group(2)
            errored = bool(re.search(r"\([0-9]*E", tok))
            frm = FROM_BASE.get(base)
            if act == "SB":
                runner = snapshot.get(frm)
                gs.bases[frm] = None
                if base == "H":
                    if runner is not None:
                        gs.score[gs.half] += 1
                else:
                    gs.bases[int(base)] = runner
            elif act in ("CS", "POCS"):
                runner = snapshot.get(frm)
                gs.bases[frm] = None
                if errored:  # safe on error, advances
                    if base == "H":
                        if runner is not None:
                            gs.score[gs.half] += 1
                    else:
                        gs.bases[int(base)] = runner
                elif runner is not None:
                    gs.outs += 1
            elif act == "PO":  # pickoff (not a steal): runner out unless errored
                if base in ("1", "2", "3"):
                    b = int(base)
                    if not errored and snapshot.get(b) is not None:
                        gs.bases[b] = None
                        gs.outs += 1
            continue

        # ---- batter outcomes (record destination; place batter last) ----
        if tok and tok[0] in BATTER_HIT:
            batter_dest = BATTER_HIT[tok[0]]
            continue
        if tok.startswith("HP"):
            batter_dest = 1
            continue
        if tok.startswith("HR") or (tok.startswith("H") and not tok.startswith("HP")):
            batter_dest = "H"
            continue
        if tok.startswith(("IW", "I", "W")) and not tok.startswith("WP"):
            batter_dest = 1
            continue
        if tok.startswith("E") and not tok.startswith("FLE"):
            batter_dest = 1  # reached on error
            continue
        if tok.startswith("FC"):
            batter_dest = 1  # fielder's choice: batter safe, runner out in adv
            continue
        if tok.startswith(("DGR", "GR")):
            batter_dest = 2
            continue
        if tok.startswith("K"):
            if not re.search(r"\bB-", adv):
                batter_dest = "out"
            continue

        # ---- fielded outs with possible force-outs, e.g. 63, 54(1)3, 8 ----
        if tok and tok[0].isdigit():
            forced = re.findall(r"\((\d)\)", tok)  # runners forced out
            for f in forced:
                b = int(f)
                if b in (1, 2, 3) and snapshot.get(b) is not None:
                    gs.bases[b] = None
                    gs.outs += 1
            if not forced:
                batter_dest = "out"
            elif re.search(r"\)\d", tok):  # trailing fielders => batter also out
                batter_dest = "out"
            continue

    # explicit advances (runner sources resolved against the snapshot)
    batter_moved = _apply_explicit_advances(adv, gs, snapshot, batter)

    # place the batter last, after all runner movement is resolved
    if batter_moved:
        pass  # batter's fate given explicitly in the advance field
    elif isinstance(batter_dest, int):
        gs.bases[batter_dest] = batter
    elif batter_dest == "H":
        gs.score[gs.half] += 1
    elif batter_dest == "out":
        gs.outs += 1


def classify_batter_outcome(event: str):
    """Classify a play's *batter* outcome for cumulative batting stats.

    Returns None if this play doesn't resolve the batter's plate appearance
    (a bare baserunning play like a steal/pickoff with no batter token, or
    'NP'). Otherwise returns a dict with the fields needed for AVG/OBP/SLG/
    HR%: ab, hit, bases (total bases if a hit), bb, hbp, sf (sacrifice fly).

    Mirrors the same token classification as `update_state` (kept in sync
    deliberately -- these read the same Retrosheet event grammar) but
    doesn't touch game state, and additionally checks for the '/SF' and
    '/SH' modifiers (stripped off before `update_state` ever sees them)
    since sacrifices are excluded from the AB/OBP-denominator counts that
    matter for batting stats but not for state tracking.
    """
    event = _strip_markers(event).strip()
    if event in ("NP", ""):
        return None

    basic, _, adv = event.partition(".")
    modifiers = basic.split("/")[1:]
    primary = basic.split("/")[0]
    is_sac_fly = any(m.startswith("SF") for m in modifiers)
    is_sac_bunt = any(m.startswith("SH") for m in modifiers)
    tokens = re.split(r"[+;]", primary)

    batter_dest = None
    for tok in tokens:
        if re.match(r"(SB|CS|POCS|PO)[123H]", tok):
            continue
        if tok and tok[0] in BATTER_HIT:
            batter_dest = BATTER_HIT[tok[0]]
            continue
        if tok.startswith("HP"):
            batter_dest = "hbp"
            continue
        if tok.startswith("HR") or (tok.startswith("H") and not tok.startswith("HP")):
            batter_dest = "hr"
            continue
        if tok.startswith(("IW", "I", "W")) and not tok.startswith("WP"):
            batter_dest = "bb"
            continue
        if tok.startswith("E") and not tok.startswith("FLE"):
            batter_dest = "reached"
            continue
        if tok.startswith("FC"):
            batter_dest = "reached"
            continue
        if tok.startswith(("DGR", "GR")):
            batter_dest = 2
            continue
        if tok.startswith("K"):
            if not re.search(r"\bB-", adv):
                batter_dest = "out"
            continue
        if tok and tok[0].isdigit():
            forced = re.findall(r"\((\d)\)", tok)
            if not forced or re.search(r"\)\d", tok):
                batter_dest = "out"
            continue

    if batter_dest is None:
        # No token classified the batter -- either a bare baserunning play
        # (e.g. a plain SB with nothing else), or the batter reached via an
        # explicit advance with no distinguishing primary token. Treat the
        # latter (an explicit 'B-'/'BX' advance) as a non-hit at-bat.
        if re.search(r"\bB[-X]", adv):
            return {"ab": True, "hit": False, "bases": 0, "bb": False, "hbp": False, "sf": False}
        return None

    if isinstance(batter_dest, int):
        return {"ab": True, "hit": True, "bases": batter_dest, "bb": False, "hbp": False, "sf": False}
    if batter_dest == "hr":
        return {"ab": True, "hit": True, "bases": 4, "bb": False, "hbp": False, "sf": False}
    if batter_dest == "hbp":
        return {"ab": False, "hit": False, "bases": 0, "bb": False, "hbp": True, "sf": False}
    if batter_dest == "bb":
        return {"ab": False, "hit": False, "bases": 0, "bb": True, "hbp": False, "sf": False}
    if batter_dest == "reached":
        return {"ab": True, "hit": False, "bases": 0, "bb": False, "hbp": False, "sf": False}
    if batter_dest == "out":
        if is_sac_fly:
            return {"ab": False, "hit": False, "bases": 0, "bb": False, "hbp": False, "sf": True}
        if is_sac_bunt:
            return None  # sac bunt: excluded from AB and OBP-denominator entirely
        return {"ab": True, "hit": False, "bases": 0, "bb": False, "hbp": False, "sf": False}
    return None


# ---------------------------------------------------------------------------
# Main parse loop
# ---------------------------------------------------------------------------
STEAL_ROW_FIELDS = [
    "game_id", "date", "park", "inning", "half", "outs", "play_seq",
    "target_base", "runner_id", "runner_bats",
    "pitcher_id", "pitcher_throws", "catcher_id", "batter_id", "count",
    "score_bat", "score_def", "score_diff",
    "on_1b", "on_2b", "on_3b", "success", "double_steal",
]

BATTING_ROW_FIELDS = [
    "game_id", "date", "inning", "half", "outs", "play_seq", "batter_id",
    "ab", "hit", "bases", "bb", "hbp", "sf",
]


def parse_file(path: str, players: dict, rows: list, diag: dict,
              batting_rows: list = None) -> None:
    gs = GameState()
    with open(path, newline="") as fh:
        reader = csv.reader(fh)
        for rec in reader:
            if not rec:
                continue
            tag = rec[0]

            if tag == "id":
                gs = GameState(game_id=rec[1])
            elif tag == "info":
                key = rec[1] if len(rec) > 1 else ""
                val = rec[2] if len(rec) > 2 else ""
                if key == "hometeam":
                    gs.home = val
                elif key == "visteam":
                    gs.visitor = val
                elif key == "date":
                    gs.date = val
                elif key == "site":
                    gs.park = val
            elif tag in ("start", "sub"):
                # start/sub,pid,name,team,batorder,fieldpos
                pid, team, pos = rec[1], int(rec[3]), int(rec[5])
                if pos == 1:
                    gs.pitcher[team] = pid
                elif pos == 2:
                    gs.catcher[team] = pid
            elif tag == "radj":
                # extra-innings automatic runner: radj,player_id,base
                gs.pending_radj.append((rec[1], int(rec[2])))
            elif tag == "play":
                gs.play_seq += 1
                inning, half, batter = int(rec[1]), int(rec[2]), rec[3]
                count = rec[4]
                event = rec[6] if len(rec) > 6 else ""
                gs.new_half_inning(inning, half)
                # apply any pending ghost-runner placement for this half-inning
                if gs.pending_radj:
                    for pid, base in gs.pending_radj:
                        gs.bases[base] = pid
                    gs.pending_radj = []

                defense = 1 - half  # defensive team index
                # --- record any steal attempts BEFORE mutating state ---
                steal_events = find_steal_events(_strip_markers(event))
                # A double (or triple) steal is >1 runner moving on the SAME
                # play (one pitch) -- Retrosheet encodes that as multiple
                # SB/CS tokens on a single play line, e.g. "SB3;SB2". These
                # behave very differently from a normal single-runner steal
                # (the defense can typically only contest one runner), so
                # flag it rather than let it look like an ordinary attempt.
                is_double = len(steal_events) > 1
                for kind, base in steal_events:
                    frm = FROM_BASE[base]
                    runner = gs.bases.get(frm)
                    diag["attempts"] += 1
                    if runner is None:
                        diag["missing_runner"] += 1
                    rows.append({
                        "game_id": gs.game_id,
                        "date": gs.date,
                        "park": gs.park,
                        "inning": inning,
                        "half": half,
                        "outs": gs.outs,
                        "play_seq": gs.play_seq,
                        "target_base": base,
                        "runner_id": runner,
                        "runner_bats": players.get(runner, {}).get("bats", ""),
                        "pitcher_id": gs.pitcher[defense],
                        "pitcher_throws": players.get(gs.pitcher[defense], {}).get("throws", ""),
                        "catcher_id": gs.catcher[defense],
                        "batter_id": batter,
                        "count": count,
                        "score_bat": gs.score[half],
                        "score_def": gs.score[defense],
                        "score_diff": gs.score[half] - gs.score[defense],
                        "on_1b": gs.bases[1],
                        "on_2b": gs.bases[2],
                        "on_3b": gs.bases[3],
                        "success": 1 if kind == "SB" else 0,
                        "double_steal": int(is_double),
                    })

                # --- batter outcome, for leakage-safe running batting stats
                # (AVG/OBP/SLG) elsewhere -- emitted BEFORE update_state so a
                # play that's both a steal attempt and the batter's own
                # plate-appearance-ending event (e.g. "K+SB2") keeps the
                # correct order: read priors first, this outcome updates them
                # only afterward (see src/features.py).
                if batting_rows is not None:
                    outcome = classify_batter_outcome(event)
                    if outcome is not None:
                        batting_rows.append({
                            "game_id": gs.game_id,
                            "date": gs.date,
                            "inning": inning,
                            "half": half,
                            "outs": gs.outs,
                            "play_seq": gs.play_seq,
                            "batter_id": batter,
                            **outcome,
                        })

                update_state(event, gs, batter)


def base_code(bases: dict) -> str:
    """3-char occupancy code, e.g. '1_3' for runners on first and third."""
    return "".join(str(b) if bases[b] else "_" for b in (1, 2, 3))


def iter_plays(data_dir: str):
    """Yield one record per play with the base-out state BEFORE the play and
    the number of runs the batting team scores from that state to the end of
    the half-inning (the raw material for a RE24 run-expectancy table).

    Yields dicts: {base_code, outs, runs_to_end, inning, half, game_id}.
    """
    files = sorted(glob.glob(os.path.join(data_dir, "*.EV*")))
    for path in files:
        gs = GameState()
        buffer = []          # (base_code, outs, runs_before) for current half
        half_key = None
        start_runs = 0

        def flush(final_runs):
            for bc, o, runs_before in buffer:
                yield_rec = {
                    "base_code": bc,
                    "outs": o,
                    "runs_to_end": final_runs - runs_before,
                }
                recs.append(yield_rec)

        recs = []
        with open(path, newline="") as fh:
            for rec in csv.reader(fh):
                if not rec:
                    continue
                tag = rec[0]
                if tag == "id":
                    if buffer:
                        flush(gs.score[gs.half])
                        buffer = []
                    gs = GameState(game_id=rec[1])
                    half_key = None
                elif tag == "radj":
                    gs.pending_radj.append((rec[1], int(rec[2])))
                elif tag in ("start", "sub"):
                    pass
                elif tag == "play":
                    inning, half, batter = int(rec[1]), int(rec[2]), rec[3]
                    event = rec[6] if len(rec) > 6 else ""
                    key = (inning, half)
                    if key != half_key:
                        if buffer:
                            flush(gs.score[half_key[1]] if half_key else 0)
                            buffer = []
                        half_key = key
                    gs.new_half_inning(inning, half)
                    if gs.pending_radj:
                        for pid, base in gs.pending_radj:
                            gs.bases[base] = pid
                        gs.pending_radj = []
                    if event.strip() not in ("NP", ""):
                        buffer.append((base_code(gs.bases), gs.outs,
                                       gs.score[half]))
                    update_state(event, gs, batter)
        if buffer:
            flush(gs.score[gs.half])
        for r in recs:
            r["game_id"] = gs.game_id
            yield r


def parse_season(data_dir: str, collect_batting: bool = False):
    players = load_rosters(data_dir)
    rows: list = []
    batting_rows = [] if collect_batting else None
    diag = {"attempts": 0, "missing_runner": 0, "impossible_state": 0}
    files = sorted(glob.glob(os.path.join(data_dir, "*.EV*")))
    for path in files:
        parse_file(path, players, rows, diag, batting_rows)
    return rows, diag, len(files), batting_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/retrosheet_2023")
    ap.add_argument("--out", default="data/sample/steals_2023.csv")
    ap.add_argument("--batting-out", default=None,
                    help="also write every plate appearance's batting "
                         "outcome (for leakage-safe AVG/OBP/SLG/HR%% in "
                         "src/features.py) to this path, e.g. "
                         "data/sample/battinglines_2023.csv")
    args = ap.parse_args()

    rows, diag, nfiles, batting_rows = parse_season(
        args.data_dir, collect_batting=bool(args.batting_out))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=STEAL_ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)

    sb = sum(r["success"] for r in rows)
    cs = len(rows) - sb
    print(f"parsed {nfiles} event files")
    print(f"steal attempts: {len(rows)}  (SB={sb}, CS={cs})")
    print(f"success rate: {sb / len(rows):.1%}" if rows else "no rows")
    print(f"runner-id resolved on {len(rows) - diag['missing_runner']}/"
          f"{len(rows)} attempts "
          f"({diag['missing_runner']} unresolved base-state cases)")
    print(f"wrote {args.out}")

    if args.batting_out:
        os.makedirs(os.path.dirname(args.batting_out), exist_ok=True)
        with open(args.batting_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=BATTING_ROW_FIELDS)
            w.writeheader()
            w.writerows(batting_rows)
        print(f"wrote {len(batting_rows)} plate-appearance rows -> {args.batting_out}")


if __name__ == "__main__":
    main()
