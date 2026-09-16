"""End-game scoring auditor for Agricola (Family Game traces).

Recomputes every scoring category the rulebook pins down exactly from the
final state — including pasture detection by fence-enclosure flood fill —
and compares against the engine's reported scores. Improvement card points
and craft-building endgame bonuses are printed on cards, so they are only
bounded, not recomputed; that bound is declared as a schema gap.

Trust in this checker comes from tools/mutate_agricola.py, not from who
wrote it: it must stay silent on the real engine and catch planted scoring
bugs it was never shown.
"""
NAME = "scoring"
RULE = ("Final scores equal the official category table: fields, pastures, "
        "grain, vegetables, animals, unused spaces, fenced stables, rooms, "
        "family, begging, plus card points within printed bounds.")

ROWS, COLS = 3, 5

FIELD_PTS = {0: -1, 1: -1, 2: 1, 3: 2, 4: 3}          # 5+ -> 4
PASTURE_PTS = {0: -1, 1: 1, 2: 2, 3: 3}               # 4+ -> 4
GRAIN_PTS = [(0, -1), (3, 1), (5, 2), (7, 3)]         # 8+ -> 4
VEG_PTS = [(0, -1), (1, 1), (2, 2), (3, 3)]           # 4+ -> 4
SHEEP_PTS = [(0, -1), (3, 1), (5, 2), (7, 3)]         # 8+ -> 4
BOAR_PTS = [(0, -1), (2, 1), (4, 2), (6, 3)]          # 7+ -> 4
CATTLE_PTS = [(0, -1), (1, 1), (3, 2), (5, 3)]        # 6+ -> 4


def _table(n, breaks):
    """breaks = [(hi, pts), ...]: n <= hi scores pts; past the last -> 4."""
    for hi, pts in breaks:
        if n <= hi:
            return pts
    return 4


def _edges_of(r, c):
    return {("H", r, c), ("H", r + 1, c), ("V", r, c), ("V", r, c + 1)}


def _shared_edge(a, b):
    (r1, c1), (r2, c2) = a, b
    if r1 == r2:
        return ("V", r1, max(c1, c2))
    return ("H", max(r1, r2), c1)


def _pastures(fences, blocked):
    """Maximal regions of non-blocked cells fully enclosed by fences.
    An unfenced edge to the board border or to a blocked (room/field) cell
    breaks enclosure — tile edges do not count as fences."""
    cells = [(r, c) for r in range(ROWS) for c in range(COLS)
             if (r, c) not in blocked]
    seen, out = set(), []
    for start in cells:
        if start in seen:
            continue
        region, leak, stack = set(), False, [start]
        seen.add(start)
        while stack:
            r, c = stack.pop()
            region.add((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                edge = ("H", max(r, nr), c) if dr else ("V", r, max(c, nc))
                if edge in fences:
                    continue
                if not (0 <= nr < ROWS and 0 <= nc < COLS) or (nr, nc) in blocked:
                    leak = True
                    continue
                if (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if not leak:
            out.append(region)
    return out


def check(trace):
    msgs = ["SCHEMA GAP: improvement-card point values and craft-building "
            "endgame bonuses are printed on cards, not in the excerpts; each "
            "player's card contribution is only bounded to [1x, 7x] the "
            "number of owned majors (0 if none), not recomputed."]
    final = trace["final"]
    scores = [s[0] for s in trace["scores"]]
    summary = trace.get("summary", {})
    if summary.get("score") and summary["score"] != scores:
        msgs.append(f"summary score list {summary['score']} disagrees with "
                    f"scores {scores}")
    for p, pl in enumerate(final["players"]):
        fences = {tuple(e) for e in pl["fences"]}
        rooms = {tuple(x) for x in pl["rooms"]}
        fields = [tuple(f) for f in pl["fields"]]
        field_cells = {(f[0], f[1]) for f in fields}
        stables = {tuple(s) for s in pl["stables"]}
        pastures = _pastures(fences, rooms | field_cells)
        pcells = set().union(*pastures) if pastures else set()
        fenced_stables = len(stables & pcells)
        used = rooms | field_cells | pcells | stables
        unused = ROWS * COLS - len(used)
        grain = pl["goods"][4] + sum(f[3] for f in fields if f[2] == "grain")
        veg = pl["goods"][5] + sum(f[3] for f in fields if f[2] == "vegetable")
        sheep, boar, cattle = pl["animals"]
        house_pts = {0: 0, 1: 1, 2: 2}[pl["house"]] * len(rooms)

        pinned = (
            _table(len(fields), FIELD_PTS_LIST) +
            (PASTURE_PTS.get(len(pastures), 4)) +
            _table(grain, GRAIN_PTS) + _table(veg, VEG_PTS) +
            _table(sheep, SHEEP_PTS) + _table(boar, BOAR_PTS) +
            _table(cattle, CATTLE_PTS) +
            (-1) * unused + fenced_stables + house_pts +
            3 * pl["family"] + (-3) * pl["begging"]
        )
        n_maj = len(pl.get("majors", []))
        rem = scores[p] - pinned
        lo, hi = (n_maj, 7 * n_maj) if n_maj else (0, 0)
        if not (lo <= rem <= hi):
            msgs.append(
                f"P{p}: reported {scores[p]}, pinned categories total {pinned} "
                f"(fields {len(fields)}, pastures {len(pastures)}, grain {grain}, "
                f"veg {veg}, animals {sheep}/{boar}/{cattle}, unused {unused}, "
                f"fenced stables {fenced_stables}, house {house_pts}, family "
                f"{pl['family']}x3, begging {pl['begging']}x-3); card remainder "
                f"{rem} outside [{lo},{hi}] for {n_maj} majors")
    return msgs


FIELD_PTS_LIST = [(1, -1), (2, 1), (3, 2), (4, 3)]     # 5+ -> 4
