"""Agricola (Lookout Games, revised edition) -- Family Game variant.

Deterministic simulation engine implementing the ``Game`` protocol from
engine.py.  No occupation or minor improvement cards; everything else --
major improvements, harvests, animal breeding, and the full end-game
scoring -- applies.  Supports 2, 3 and 4 players.

Farmyard: a 3x5 grid, rows 0..2 top to bottom, columns 0..4 left to
right.  The two starting wood rooms occupy (1, 0) and (2, 0).  Fence
edges are named ("H", r, c) for the horizontal edge ABOVE cell (r, c)
(r may equal 3 for the bottom border) and ("V", r, c) for the vertical
edge LEFT of cell (r, c) (c may equal 5 for the right border).

Actions (tuples; first element names the kind):
  ("place", space_index)              -- work phase: put a person on a space
  ("pass",)                           -- work phase: only if no placement is legal
  ("plow", r, c)                      -- plow a field on cell (r, c)
  ("sow", r, c, "grain"|"vegetable")  -- sow an empty field
  ("bake",)                           -- bake 1 grain at the best available rate
  ("room", r, c) / ("stable", r, c)   -- Farm Expansion builds
  ("fence", cells)                    -- enclose one connected region (cells is a
                                         sorted tuple of (r, c)); pays 1 wood per
                                         newly built fence segment
  ("major", i) / ("upgrade", i)       -- build major improvement i / upgrade a
                                         Fireplace into Cooking Hearth i
  ("take", resource_name)             -- "Side Job": 1 building resource of choice
  ("convert", goods_name)             -- feeding: turn one good into food
  ("feed",)                           -- feeding: pay food, beg for the shortfall
  ("cook", t) / ("release", t)        -- animal overflow: cook/discard 1 animal of
                                         type t (0 sheep, 1 boar, 2 cattle)
  ("done",)                           -- finish the current multi-step action
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from typing import Any, Optional

# ---------------------------------------------------------------- constants

ROWS, COLS = 3, 5
CELLS = tuple((r, c) for r in range(ROWS) for c in range(COLS))
_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

WOOD, CLAY, REED, STONE, GRAIN, VEG = range(6)
GOOD_NAMES = ("wood", "clay", "reed", "stone", "grain", "vegetable")
SHEEP, BOAR, CATTLE = range(3)
ANIMAL_NAMES = ("sheep", "boar", "cattle")

HARVEST_ROUNDS = (4, 7, 9, 11, 13, 14)
LAST_ROUND = 14
MAX_FAMILY = 5
MAX_STABLES = 4
MAX_FENCES = 15
START_ROOMS = ((1, 0), (2, 0))
HOUSE_NAMES = ("wood", "clay", "stone")

# ------------------------------------------------------- major improvements


@dataclass(frozen=True)
class Major:
    name: str
    points: int
    cost: tuple  # (wood, clay, reed, stone)
    cook: Optional[tuple]  # food per (sheep, boar, cattle, vegetable)
    bake: Optional[tuple]  # (food per grain, max grain per Bake action or None)


MAJORS: tuple = (
    Major("Fireplace (2 clay)", 1, (0, 2, 0, 0), (2, 2, 3, 2), (2, None)),
    Major("Fireplace (3 clay)", 1, (0, 3, 0, 0), (2, 2, 3, 2), (2, None)),
    Major("Cooking Hearth (4 clay)", 1, (0, 4, 0, 0), (2, 3, 4, 3), (3, None)),
    Major("Cooking Hearth (5 clay)", 1, (0, 5, 0, 0), (2, 3, 4, 3), (3, None)),
    Major("Well", 4, (1, 0, 0, 3), None, None),
    Major("Clay Oven", 2, (0, 3, 0, 1), None, (5, 1)),
    Major("Stone Oven", 3, (0, 1, 0, 3), None, (4, 2)),
    Major("Joinery", 2, (2, 0, 0, 2), None, None),
    Major("Pottery", 2, (0, 2, 0, 2), None, None),
    Major("Basketmaker's Workshop", 2, (0, 0, 2, 2), None, None),
)
FIREPLACES = (0, 1)
HEARTHS = (2, 3)
WELL, CLAY_OVEN, STONE_OVEN, JOINERY, POTTERY, BASKET = 4, 5, 6, 7, 8, 9
# craft buildings: goods index -> (improvement, food per unit, craft_used slot)
CRAFT = {WOOD: (JOINERY, 2, 0), CLAY: (POTTERY, 2, 1), REED: (BASKET, 3, 2)}
# scoring bonus thresholds, checked top down: (at least, bonus points)
CRAFT_BONUS = {
    JOINERY: (WOOD, ((7, 3), (5, 2), (3, 1))),
    POTTERY: (CLAY, ((7, 3), (5, 2), (3, 1))),
    BASKET: (REED, ((6, 3), (4, 2), (2, 1))),
}

# ------------------------------------------------------------ action spaces
# RULING: the Family Game covers the "Lessons" action space(s) with the
# "Side Job" tile (components list); the appendix describing that tile is not
# part of the provided rules text.  We implement Side Job as "take 1 building
# resource of your choice" and, in 3-/4-player games, remove the second
# Lessons space outright (there is only one Side Job tile).
# RULING: goods amounts for spaces the rulebook text does not spell out are
# taken from the published boards: Day Laborer 2 food; Resource Market
# 1 reed + 1 stone + 1 food; Copse 1 wood, Grove 2 wood; Hollow 1 clay in the
# 3-player game and 2 clay in the 4-player game.  Forest 3 wood, Clay Pit
# 1 clay, Reed Bank 1 reed and Fishing 1 food are as illustrated/stated.


def _permanent_spaces(n: int) -> tuple:
    base = ["farm_expansion", "meeting_place", "grain_seeds", "farmland",
            "side_job", "day_laborer", "forest", "clay_pit", "reed_bank",
            "fishing"]
    if n == 3:
        base += ["grove", "hollow3", "resource_market"]
    elif n == 4:
        base += ["copse", "grove", "hollow4", "traveling_players",
                 "resource_market"]
    return tuple(base)


# stage cards: shuffled within each stage; stage 1 covers rounds 1-4, stage 2
# rounds 5-7, stage 3 rounds 8-9, stage 4 rounds 10-11, stage 5 rounds 12-13,
# stage 6 round 14.
STAGES = (
    ("sheep_market", "grain_utilization", "major_improvement", "fencing"),
    ("basic_wish_for_children", "house_redevelopment", "western_quarry"),
    ("pig_market", "vegetable_seeds"),
    ("cattle_market", "eastern_quarry"),
    ("urgent_wish_for_children", "cultivation"),
    ("farm_redevelopment",),
)

# accumulation spaces: id -> (kind, amount added every preparation phase)
ACC = {
    "forest": ("wood", 3), "clay_pit": ("clay", 1), "reed_bank": ("reed", 1),
    "fishing": ("food", 1), "copse": ("wood", 1), "grove": ("wood", 2),
    "hollow3": ("clay", 1), "hollow4": ("clay", 2),
    "traveling_players": ("food", 1),
    "sheep_market": ("sheep", 1), "pig_market": ("boar", 1),
    "cattle_market": ("cattle", 1),
    "western_quarry": ("stone", 1), "eastern_quarry": ("stone", 1),
}

SPACE_NAMES = {
    "farm_expansion": "Farm Expansion", "meeting_place": "Meeting Place",
    "grain_seeds": "Grain Seeds", "farmland": "Farmland",
    "side_job": "Side Job", "day_laborer": "Day Laborer", "forest": "Forest",
    "clay_pit": "Clay Pit", "reed_bank": "Reed Bank", "fishing": "Fishing",
    "grove": "Grove", "hollow3": "Hollow", "hollow4": "Hollow",
    "copse": "Copse", "traveling_players": "Traveling Players",
    "resource_market": "Resource Market",
    "sheep_market": "Sheep Market", "grain_utilization": "Grain Utilization",
    "major_improvement": "Major Improvement", "fencing": "Fencing",
    "basic_wish_for_children": "Basic Wish for Children",
    "house_redevelopment": "House Redevelopment",
    "western_quarry": "Western Quarry", "pig_market": "Pig Market",
    "vegetable_seeds": "Vegetable Seeds", "cattle_market": "Cattle Market",
    "eastern_quarry": "Eastern Quarry",
    "urgent_wish_for_children": "Urgent Wish for Children",
    "cultivation": "Cultivation", "farm_redevelopment": "Farm Redevelopment",
}

# --------------------------------------------------------------- geometry


def _cell_edges(r: int, c: int) -> tuple:
    return (("H", r, c), ("H", r + 1, c), ("V", r, c), ("V", r, c + 1))


def _edge_between(a, b):
    (r1, c1), (r2, c2) = sorted((a, b))
    if r1 == r2:
        return ("V", r1, max(c1, c2))
    return ("H", max(r1, r2), c1)


def _edge_cells(edge) -> tuple:
    kind, r, c = edge
    cells = []
    if kind == "H":
        if r - 1 >= 0:
            cells.append((r - 1, c))
        if r < ROWS:
            cells.append((r, c))
    else:
        if c - 1 >= 0:
            cells.append((r, c - 1))
        if c < COLS:
            cells.append((r, c))
    return tuple(cells)


@lru_cache(maxsize=8192)
def _pastures(fences: tuple) -> tuple:
    """All fully enclosed regions ('pastures') as frozensets of cells."""
    f = frozenset(fences)
    reach = set()
    dq = deque()
    for c in range(COLS):
        if ("H", 0, c) not in f:
            dq.append((0, c))
        if ("H", ROWS, c) not in f:
            dq.append((ROWS - 1, c))
    for r in range(ROWS):
        if ("V", r, 0) not in f:
            dq.append((r, 0))
        if ("V", r, COLS) not in f:
            dq.append((r, COLS - 1))
    while dq:
        cell = dq.popleft()
        if cell in reach:
            continue
        reach.add(cell)
        r, c = cell
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if 0 <= nb[0] < ROWS and 0 <= nb[1] < COLS and nb not in reach \
                    and _edge_between(cell, nb) not in f:
                dq.append(nb)
    enclosed = {cell for cell in CELLS if cell not in reach}
    comps, seen = [], set()
    for cell in sorted(enclosed):
        if cell in seen:
            continue
        comp, dq2 = set(), deque([cell])
        while dq2:
            cur = dq2.popleft()
            if cur in comp:
                continue
            comp.add(cur)
            seen.add(cur)
            r, c = cur
            for dr, dc in _DIRS:
                nb = (r + dr, c + dc)
                if nb in enclosed and nb not in comp \
                        and _edge_between(cur, nb) not in f:
                    dq2.append(nb)
        comps.append(frozenset(comp))
    return tuple(comps)


def _connected(cells) -> bool:
    cells = set(cells)
    if not cells:
        return True
    dq = deque([next(iter(sorted(cells)))])
    seen = set()
    while dq:
        cur = dq.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        r, c = cur
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb in cells:
                dq.append(nb)
    return seen == cells


# ---------------------------------------------------------------- states


@dataclass(frozen=True)
class Player:
    food: int = 0
    goods: tuple = (0, 0, 0, 0, 0, 0)          # wood clay reed stone grain veg
    animals: tuple = (0, 0, 0)                 # sheep boar cattle
    begging: int = 0
    family: int = 2
    born: int = 0                              # newborns of the current round
    to_place: int = 2                          # people still to place
    passed: int = 0                            # forced passes this round
    rooms: tuple = START_ROOMS
    house: int = 0                             # 0 wood, 1 clay, 2 stone
    fields: tuple = ()                         # ((r, c, crop, count), ...)
    stables: tuple = ()                        # ((r, c), ...)
    fences: tuple = ()                         # sorted edge tuples
    majors: tuple = ()                         # indices into MAJORS
    craft_used: tuple = (False, False, False)  # joinery/pottery/basket


@dataclass(frozen=True)
class State:
    n: int
    round: int
    phase: str
    current: int
    start: int
    players: tuple
    spaces: tuple          # permanent spaces + all 14 stage cards (reveal order)
    n_perm: int
    piles: tuple           # goods sitting on accumulation spaces
    occupied: tuple        # player index or -1, per space
    major_owner: tuple     # len 10, player index or -1
    well: tuple            # len 15; food waiting on round space r at index r
    well_owner: int = -1
    pend: tuple = ()
    terminal: bool = False


# ------------------------------------------------------- player board logic


def _field_cells(pl):
    return {(r, c) for (r, c, _, _) in pl.fields}


def _pasture_cells(pl):
    cells = set()
    for comp in _pastures(pl.fences):
        cells |= comp
    return cells


def _used_cells(pl):
    return set(pl.rooms) | _field_cells(pl) | set(pl.stables) | _pasture_cells(pl)


def _tile_cells(pl):
    return set(pl.rooms) | _field_cells(pl)


def _plow_targets(pl):
    empty = [c for c in CELLS if c not in _used_cells(pl)]
    fields = _field_cells(pl)
    if not fields:
        return sorted(empty)
    out = []
    for (r, c) in empty:
        if any((r + dr, c + dc) in fields for dr, dc in _DIRS):
            out.append((r, c))
    return sorted(out)


def _room_cost(pl):
    res = (WOOD, CLAY, STONE)[pl.house]
    return res, 5, 2  # resource, amount, reed


def _can_afford_room(pl):
    res, amt, reed = _room_cost(pl)
    return pl.goods[res] >= amt and pl.goods[REED] >= reed


def _room_targets(pl):
    rooms = set(pl.rooms)
    empty = [c for c in CELLS if c not in _used_cells(pl)]
    out = []
    for (r, c) in empty:
        if any((r + dr, c + dc) in rooms for dr, dc in _DIRS):
            out.append((r, c))
    return sorted(out)


def _stable_targets(pl):
    blocked = _tile_cells(pl) | set(pl.stables)
    return sorted(c for c in CELLS if c not in blocked)


def _can_build_stable(pl):
    return (pl.goods[WOOD] >= 2 and len(pl.stables) < MAX_STABLES
            and bool(_stable_targets(pl)))


def _can_renovate(pl):
    if pl.house >= 2 or pl.goods[REED] < 1:
        return False
    res = CLAY if pl.house == 0 else STONE
    return pl.goods[res] >= len(pl.rooms)


def _sow_options(pl):
    out = []
    for (r, c, crop, count) in pl.fields:
        if count == 0:
            if pl.goods[GRAIN] > 0:
                out.append(("sow", r, c, "grain"))
            if pl.goods[VEG] > 0:
                out.append(("sow", r, c, "vegetable"))
    return sorted(out)


def _has_bake(pl):
    return any(MAJORS[i].bake is not None for i in pl.majors)


def _best_bake(pl, clay_left, stone_left):
    """(food, which) for baking one grain now; which in {'clay','stone','flat'}."""
    best, which = 0, None
    for i in pl.majors:
        b = MAJORS[i].bake
        if b is None:
            continue
        food, cap = b
        if i == CLAY_OVEN:
            if clay_left <= 0:
                continue
        elif i == STONE_OVEN:
            if stone_left <= 0:
                continue
        if food > best:
            best = food
            which = "clay" if i == CLAY_OVEN else ("stone" if i == STONE_OVEN
                                                   else "flat")
    return best, which


def _cook_value(pl, t):
    v = 0
    for i in pl.majors:
        ck = MAJORS[i].cook
        if ck is not None:
            v = max(v, ck[t])
    return v


def _veg_value(pl):
    v = 1
    for i in pl.majors:
        ck = MAJORS[i].cook
        if ck is not None:
            v = max(v, ck[3])
    return v


def _pasture_info(pl):
    stables = set(pl.stables)
    out = []
    for comp in _pastures(pl.fences):
        s = len(comp & stables)
        out.append((comp, 2 * len(comp) * (2 ** s)))
    return out


def _animals_fit(pl, animals=None) -> bool:
    counts = tuple(animals if animals is not None else pl.animals)
    if sum(counts) == 0:
        return True
    info = _pasture_info(pl)
    caps = [cap for _, cap in info]
    pcells = set()
    for comp, _ in info:
        pcells |= comp
    flex = 1 + sum(1 for s in pl.stables if s not in pcells)  # pet + stables
    for assign in product((0, 1, 2), repeat=len(caps)):
        by_type = [0, 0, 0]
        for lab, cap in zip(assign, caps):
            by_type[lab] += cap
        short = sum(max(0, counts[t] - by_type[t]) for t in range(3))
        if short <= flex:
            return True
    return False


# minimum possible perimeter of a k-cell region (used only to prune search)
_MINP = {1: 4, 2: 6, 3: 8, 4: 8, 5: 10, 6: 10, 7: 12, 8: 12, 9: 12,
         10: 14, 11: 14, 12: 14, 13: 16, 14: 16, 15: 16}


def _region_new_edges(cells, fences):
    missing = []
    for (r, c) in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb in cells:
                continue
            if 0 <= nb[0] < ROWS and 0 <= nb[1] < COLS:
                e = _edge_between((r, c), nb)
            else:
                if dr == -1:
                    e = ("H", r, c)
                elif dr == 1:
                    e = ("H", r + 1, c)
                elif dc == -1:
                    e = ("V", r, c)
                else:
                    e = ("V", r, c + 1)
            if e not in fences:
                missing.append(e)
    return missing


def _valid_fence_layout(fences: tuple, tiles) -> bool:
    pas = _pastures(fences)
    if fences and not pas:
        return False
    enclosed = set()
    for comp in pas:
        enclosed |= comp
    if enclosed & tiles:          # a room or field ended up inside a pasture
        return False
    for e in fences:              # no dangling fences
        if not any(cell in enclosed for cell in _edge_cells(e)):
            return False
    if len(pas) > 1:              # pastures must be adjacent to pastures
        adj = {i: set() for i in range(len(pas))}
        for i in range(len(pas)):
            for j in range(i + 1, len(pas)):
                if any((r + dr, c + dc) in pas[j]
                       for (r, c) in pas[i] for dr, dc in _DIRS):
                    adj[i].add(j)
                    adj[j].add(i)
        seen, dq = set(), deque([0])
        while dq:
            i = dq.popleft()
            if i in seen:
                continue
            seen.add(i)
            dq.extend(adj[i])
        if len(seen) != len(pas):
            return False
    return True


def _connected_subsets(cells, kmax):
    cells = sorted(cells)
    idx = {c: i for i, c in enumerate(cells)}
    cellset = set(cells)
    out = []
    for i, start in enumerate(cells):
        base = frozenset([start])
        seen = {base}
        stack = [base]
        while stack:
            cur = stack.pop()
            out.append(cur)
            if len(cur) >= kmax:
                continue
            for (r, c) in cur:
                for dr, dc in _DIRS:
                    nb = (r + dr, c + dc)
                    if nb in cellset and nb not in cur and idx[nb] > i:
                        nxt = cur | {nb}
                        if nxt not in seen:
                            seen.add(nxt)
                            stack.append(nxt)
    return out


def _fence_options(pl, first_only=False):
    """Legal ("fence", cells) purchases: enclose one connected region."""
    budget = min(pl.goods[WOOD], MAX_FENCES - len(pl.fences))
    if budget < 1:
        return []
    f = set(pl.fences)
    tiles = _tile_cells(pl)
    elig = [c for c in CELLS if c not in tiles]
    kmax = 0
    for k in range(1, 16):
        if _MINP[k] - len(f) <= budget:
            kmax = k
    if kmax == 0:
        return []
    opts, seen_edge_sets = [], set()
    for cells in _connected_subsets(elig, kmax):
        new_edges = _region_new_edges(cells, f)
        cost = len(new_edges)
        if cost < 1 or cost > budget:
            continue
        key = frozenset(new_edges)
        if key in seen_edge_sets:
            continue
        newf = tuple(sorted(f | key))
        if len(newf) > MAX_FENCES:
            continue
        if not _valid_fence_layout(newf, tiles):
            continue
        seen_edge_sets.add(key)
        opts.append(("fence", tuple(sorted(cells))))
        if first_only:
            return opts
    return sorted(opts)


# -------------------------------------------------------------- scoring


def _bracket(v, table, floor=-1):
    for lo, pts in table:
        if v >= lo:
            return pts
    return floor


FIELD_PTS = ((5, 4), (4, 3), (3, 2), (2, 1))
PASTURE_PTS = ((4, 4), (3, 3), (2, 2), (1, 1))
GRAIN_PTS = ((8, 4), (6, 3), (4, 2), (1, 1))
VEG_PTS = ((4, 4), (3, 3), (2, 2), (1, 1))
SHEEP_PTS = ((8, 4), (6, 3), (4, 2), (1, 1))
BOAR_PTS = ((7, 4), (5, 3), (3, 2), (1, 1))
CATTLE_PTS = ((6, 4), (4, 3), (2, 2), (1, 1))


def _score_breakdown(pl):
    fields_n = len(pl.fields)
    pas = _pastures(pl.fences)
    pcells = set()
    for comp in pas:
        pcells |= comp
    grain_total = pl.goods[GRAIN] + sum(cnt for (_, _, crop, cnt) in pl.fields
                                        if crop == "grain")
    veg_total = pl.goods[VEG] + sum(cnt for (_, _, crop, cnt) in pl.fields
                                    if crop == "vegetable")
    used = _used_cells(pl)
    unused = len(CELLS) - len(used)
    fenced_stables = sum(1 for s in pl.stables if s in pcells)
    room_pts = len(pl.rooms) * (0, 1, 2)[pl.house]
    imp_pts = sum(MAJORS[i].points for i in pl.majors)
    bonus = 0
    for i in pl.majors:
        if i in CRAFT_BONUS:
            gi, table = CRAFT_BONUS[i]
            bonus += _bracket(pl.goods[gi], table, 0)
    b = {
        "fields": _bracket(fields_n, FIELD_PTS),
        "pastures": _bracket(len(pas), PASTURE_PTS),
        "grain": _bracket(grain_total, GRAIN_PTS),
        "vegetables": _bracket(veg_total, VEG_PTS),
        "sheep": _bracket(pl.animals[SHEEP], SHEEP_PTS),
        "wild boar": _bracket(pl.animals[BOAR], BOAR_PTS),
        "cattle": _bracket(pl.animals[CATTLE], CATTLE_PTS),
        "unused spaces": -unused,
        "fenced stables": fenced_stables,
        "rooms": room_pts,
        "family": 3 * pl.family,
        "begging": -3 * pl.begging,
        "improvements": imp_pts,
        "bonus points": bonus,
    }
    return b


# ------------------------------------------------------------------- game


class Agricola:
    name = "agricola-family"

    # ----------------------------------------------------------- protocol

    def initial_state(self, n_players: int, seed: int) -> State:
        assert 2 <= n_players <= 4, "Family Game engine supports 2-4 players"
        rng = random.Random(seed)
        start = rng.randrange(n_players)
        reveal = []
        for stage in STAGES:
            ids = list(stage)
            rng.shuffle(ids)
            reveal.extend(ids)
        perm = _permanent_spaces(n_players)
        spaces = perm + tuple(reveal)
        players = tuple(
            Player(food=(2 if p == start else 3)) for p in range(n_players))
        piles = [0] * len(spaces)
        st = State(n=n_players, round=1, phase="main", current=start,
                   start=start, players=players, spaces=spaces,
                   n_perm=len(perm), piles=tuple(piles),
                   occupied=tuple([-1] * len(spaces)),
                   major_owner=tuple([-1] * len(MAJORS)),
                   well=tuple([0] * 15))
        return self._prepare(st)

    def current_player(self, state: State) -> int:
        return state.current

    def phase(self, state: State) -> str:
        return state.phase

    def is_terminal(self, state: State) -> bool:
        return state.terminal

    def observation(self, state: State, player: int) -> Any:
        return state

    def features(self, state: State, player: int) -> tuple:
        return ()

    def scores(self, state: State):
        return [(sum(_score_breakdown(pl).values()),) for pl in state.players]

    # ------------------------------------------------------ legal actions

    def legal_actions(self, st: State):
        if st.terminal:
            return []
        p = st.current
        pl = st.players[p]
        ph = st.phase
        if ph == "main":
            acts = []
            for i in range(st.n_perm + st.round):
                if st.occupied[i] != -1:
                    continue
                if self._usable(st, p, i):
                    acts.append(("place", i))
            if not acts:
                # RULING: the rulebook does not cover a player who cannot use
                # any free action space; that person forfeits this placement.
                acts = [("pass",)]
            return acts
        if ph == "plow":
            return [("plow", r, c) for (r, c) in _plow_targets(pl)]
        if ph == "side_job":
            return [("take", g) for g in ("wood", "clay", "reed", "stone")]
        if ph == "farm_expansion":
            acts = []
            if _can_afford_room(pl):
                acts += [("room", r, c) for (r, c) in _room_targets(pl)]
            if _can_build_stable(pl):
                acts += [("stable", r, c) for (r, c) in _stable_targets(pl)]
            if st.pend[0]:
                acts.append(("done",))
            return acts
        if ph == "sow_bake":
            did, cl, sl = st.pend
            acts = list(_sow_options(pl))
            if pl.goods[GRAIN] > 0 and _best_bake(pl, cl, sl)[0] > 0:
                acts.append(("bake",))
            if did:
                acts.append(("done",))
            return acts
        if ph == "plow_sow":
            did, plowed = st.pend
            acts = []
            if not plowed:
                acts += [("plow", r, c) for (r, c) in _plow_targets(pl)]
            acts += _sow_options(pl)
            if did:
                acts.append(("done",))
            return acts
        if ph == "fencing":
            did, required = st.pend
            acts = _fence_options(pl)
            if did or not required:
                acts.append(("done",))
            return acts
        if ph == "improvement":
            (required,) = st.pend
            acts = self._improvement_options(st, p)
            if not required:
                acts.append(("done",))
            return acts
        if ph == "bake":
            cl, sl = st.pend
            acts = []
            if pl.goods[GRAIN] > 0 and _best_bake(pl, cl, sl)[0] > 0:
                acts.append(("bake",))
            acts.append(("done",))
            return acts
        if ph == "adjust":
            acts = []
            for t in range(3):
                if pl.animals[t] > 0:
                    if _cook_value(pl, t) > 0:
                        acts.append(("cook", t))
                    acts.append(("release", t))
            return acts
        if ph == "feeding":
            acts = self._feed_converts(pl)
            acts.append(("feed",))
            return acts
        raise AssertionError(f"unknown phase {ph!r}")

    def _usable(self, st: State, p: int, i: int) -> bool:
        sid = st.spaces[i]
        pl = st.players[p]
        if sid in ACC:
            return st.piles[i] > 0
        if sid in ("day_laborer", "grain_seeds", "vegetable_seeds",
                   "resource_market", "side_job"):
            return True
        if sid == "meeting_place":
            # RULING: taking the starting player token is always a legal
            # action, even for the player who already holds it.
            return True
        if sid == "farmland":
            return bool(_plow_targets(pl))
        if sid == "grain_utilization":
            return bool(_sow_options(pl)) or (pl.goods[GRAIN] > 0
                                              and _has_bake(pl))
        if sid == "cultivation":
            return bool(_plow_targets(pl)) or bool(_sow_options(pl))
        if sid == "farm_expansion":
            return ((_can_afford_room(pl) and bool(_room_targets(pl)))
                    or _can_build_stable(pl))
        if sid == "fencing":
            return bool(_fence_options(pl, first_only=True))
        if sid == "major_improvement":
            return bool(self._improvement_options(st, p))
        if sid in ("house_redevelopment", "farm_redevelopment"):
            return _can_renovate(pl)
        if sid == "basic_wish_for_children":
            return pl.family < MAX_FAMILY and pl.family < len(pl.rooms)
        if sid == "urgent_wish_for_children":
            return pl.family < MAX_FAMILY
        raise AssertionError(f"unknown space {sid!r}")

    def _improvement_options(self, st: State, p: int):
        pl = st.players[p]
        opts = []
        for i, m in enumerate(MAJORS):
            if st.major_owner[i] == -1 and all(
                    pl.goods[k] >= m.cost[k] for k in range(4)):
                opts.append(("major", i))
        if any(st.major_owner[fp] == p for fp in FIREPLACES):
            for h in HEARTHS:
                if st.major_owner[h] == -1:
                    opts.append(("upgrade", h))
        return opts

    def _feed_converts(self, pl):
        acts = []
        if pl.goods[GRAIN] > 0:
            acts.append(("convert", "grain"))
        if pl.goods[VEG] > 0:
            acts.append(("convert", "vegetable"))
        for t in range(3):
            if pl.animals[t] > 0 and _cook_value(pl, t) > 0:
                acts.append(("convert", ANIMAL_NAMES[t]))
        for gi, (imp, val, slot) in CRAFT.items():
            if imp in pl.majors and not pl.craft_used[slot] \
                    and pl.goods[gi] > 0:
                acts.append(("convert", GOOD_NAMES[gi]))
        return acts

    # --------------------------------------------------------------- apply

    def apply(self, st: State, action) -> State:
        st = self._apply_one(st, action)
        # Resolve forced steps automatically: whenever a sub-phase leaves the
        # deciding player exactly one legal action, take it.
        for _ in range(2000):
            if st.terminal or st.phase == "main":
                break
            acts = self.legal_actions(st)
            if len(acts) != 1:
                break
            st = self._apply_one(st, acts[0])
        return st

    def _apply_one(self, st: State, a) -> State:
        assert not st.terminal, "game is over"
        p = st.current
        pl = st.players[p]
        ph = st.phase
        kind = a[0]

        if ph == "main":
            if kind == "pass":
                st = self._setp(st, p, replace(pl, to_place=pl.to_place - 1,
                                               passed=pl.passed + 1))
                return self._end_turn(st)
            assert kind == "place"
            i = a[1]
            assert st.occupied[i] == -1 and i < st.n_perm + st.round
            occ = list(st.occupied)
            occ[i] = p
            st = replace(st, occupied=tuple(occ))
            st = self._setp(st, p, replace(pl, to_place=pl.to_place - 1))
            return self._resolve_space(st, p, i)

        if ph == "plow":
            assert kind == "plow"
            st = self._do_plow(st, p, a[1], a[2])
            return self._end_turn(st)

        if ph == "side_job":
            assert kind == "take"
            gi = GOOD_NAMES.index(a[1])
            assert gi < 4
            st = self._add_goods(st, p, gi, 1)
            return self._end_turn(st)

        if ph == "farm_expansion":
            if kind == "done":
                assert st.pend[0]
                return self._end_turn(st)
            if kind == "room":
                _, r, c = a
                assert (r, c) in _room_targets(pl) and _can_afford_room(pl)
                res, amt, reed = _room_cost(pl)
                goods = list(pl.goods)
                goods[res] -= amt
                goods[REED] -= reed
                st = self._setp(st, p, replace(
                    pl, goods=tuple(goods),
                    rooms=tuple(sorted(pl.rooms + ((r, c),)))))
                return replace(st, pend=(1,))
            assert kind == "stable"
            _, r, c = a
            assert (r, c) in _stable_targets(pl) and _can_build_stable(pl)
            goods = list(pl.goods)
            goods[WOOD] -= 2
            st = self._setp(st, p, replace(
                pl, goods=tuple(goods),
                stables=tuple(sorted(pl.stables + ((r, c),)))))
            return replace(st, pend=(1,))

        if ph == "sow_bake":
            did, cl, sl = st.pend
            if kind == "done":
                assert did
                return self._end_turn(st)
            if kind == "sow":
                st = self._do_sow(st, p, a[1], a[2], a[3])
                return replace(st, pend=(1, cl, sl))
            assert kind == "bake"
            st, cl, sl = self._do_bake(st, p, cl, sl)
            return replace(st, pend=(1, cl, sl))

        if ph == "plow_sow":
            did, plowed = st.pend
            if kind == "done":
                assert did
                return self._end_turn(st)
            if kind == "plow":
                assert not plowed
                st = self._do_plow(st, p, a[1], a[2])
                return replace(st, pend=(1, 1))
            assert kind == "sow"
            st = self._do_sow(st, p, a[1], a[2], a[3])
            return replace(st, pend=(1, plowed))

        if ph == "fencing":
            did, required = st.pend
            if kind == "done":
                assert did or not required
                return self._maybe_adjust(st, p)
            assert kind == "fence"
            cells = a[1]
            f = set(pl.fences)
            new_edges = _region_new_edges(set(cells), f)
            cost = len(new_edges)
            assert 1 <= cost <= pl.goods[WOOD]
            newf = tuple(sorted(f | set(new_edges)))
            assert len(newf) <= MAX_FENCES
            assert _valid_fence_layout(newf, _tile_cells(pl))
            goods = list(pl.goods)
            goods[WOOD] -= cost
            st = self._setp(st, p, replace(pl, goods=tuple(goods),
                                           fences=newf))
            return replace(st, pend=(1, required))

        if ph == "improvement":
            (required,) = st.pend
            if kind == "done":
                assert not required
                return self._end_turn(st)
            if kind == "major":
                i = a[1]
                m = MAJORS[i]
                assert st.major_owner[i] == -1
                assert all(pl.goods[k] >= m.cost[k] for k in range(4))
                goods = [g - c for g, c in zip(pl.goods[:4], m.cost)] \
                    + list(pl.goods[4:])
                st = self._own_major(st, p, i, tuple(goods))
                return self._after_improvement(st, p, i)
            assert kind == "upgrade"
            h = a[1]
            assert h in HEARTHS and st.major_owner[h] == -1
            owned_fp = [fp for fp in FIREPLACES if st.major_owner[fp] == p]
            assert owned_fp
            # RULING: with two Fireplaces in front of you, the lower-numbered
            # one is returned (the cards are functionally identical).
            fp = owned_fp[0]
            owner = list(st.major_owner)
            owner[fp] = -1
            pl2 = replace(pl, majors=tuple(m for m in pl.majors if m != fp))
            st = self._setp(replace(st, major_owner=tuple(owner)), p, pl2)
            st = self._own_major(st, p, h, st.players[p].goods)
            return self._after_improvement(st, p, h)

        if ph == "bake":
            cl, sl = st.pend
            if kind == "done":
                return self._end_turn(st)
            assert kind == "bake"
            st, cl, sl = self._do_bake(st, p, cl, sl)
            return replace(st, pend=(cl, sl))

        if ph == "adjust":
            t = a[1]
            assert pl.animals[t] > 0
            animals = list(pl.animals)
            animals[t] -= 1
            food = pl.food
            if kind == "cook":
                v = _cook_value(pl, t)
                assert v > 0
                food += v
            else:
                assert kind == "release"
            pl2 = replace(pl, animals=tuple(animals), food=food)
            st = self._setp(st, p, pl2)
            if _animals_fit(pl2):
                return self._end_turn(st)
            return st

        if ph == "feeding":
            if kind == "convert":
                return self._do_convert(st, p, a[1])
            assert kind == "feed"
            need = 2 * (pl.family - pl.born) + pl.born
            food = pl.food - need
            beg = 0
            if food < 0:
                beg = -food
                food = 0
            st = self._setp(st, p, replace(pl, food=food,
                                           begging=pl.begging + beg))
            if p + 1 < st.n:
                return replace(st, current=p + 1)
            st = self._breed_all(st)
            return self._next_round(st)

        raise AssertionError(f"cannot apply {a!r} in phase {ph!r}")

    # ------------------------------------------------------- space effects

    def _resolve_space(self, st: State, p: int, i: int) -> State:
        sid = st.spaces[i]
        pl = st.players[p]
        if sid in ACC:
            kindname, _ = ACC[sid]
            amt = st.piles[i]
            piles = list(st.piles)
            piles[i] = 0
            st = replace(st, piles=tuple(piles))
            pl = st.players[p]
            if kindname == "food":
                st = self._setp(st, p, replace(pl, food=pl.food + amt))
                return self._end_turn(st)
            if kindname in ANIMAL_NAMES:
                t = ANIMAL_NAMES.index(kindname)
                animals = list(pl.animals)
                animals[t] += amt
                st = self._setp(st, p, replace(pl, animals=tuple(animals)))
                return self._maybe_adjust(st, p)
            gi = GOOD_NAMES.index(kindname)
            st = self._add_goods(st, p, gi, amt)
            return self._end_turn(st)
        if sid == "day_laborer":
            st = self._setp(st, p, replace(pl, food=pl.food + 2))
            return self._end_turn(st)
        if sid == "grain_seeds":
            st = self._add_goods(st, p, GRAIN, 1)
            return self._end_turn(st)
        if sid == "vegetable_seeds":
            st = self._add_goods(st, p, VEG, 1)
            return self._end_turn(st)
        if sid == "resource_market":
            goods = list(pl.goods)
            goods[REED] += 1
            goods[STONE] += 1
            st = self._setp(st, p, replace(pl, goods=tuple(goods),
                                           food=pl.food + 1))
            return self._end_turn(st)
        if sid == "meeting_place":
            return self._end_turn(replace(st, start=p))
        if sid == "side_job":
            return replace(st, phase="side_job", pend=())
        if sid == "farmland":
            return replace(st, phase="plow", pend=())
        if sid == "farm_expansion":
            return replace(st, phase="farm_expansion", pend=(0,))
        if sid == "grain_utilization":
            cl = 1 if CLAY_OVEN in pl.majors else 0
            sl = 2 if STONE_OVEN in pl.majors else 0
            return replace(st, phase="sow_bake", pend=(0, cl, sl))
        if sid == "cultivation":
            return replace(st, phase="plow_sow", pend=(0, 0))
        if sid == "fencing":
            return replace(st, phase="fencing", pend=(0, 1))
        if sid == "major_improvement":
            return replace(st, phase="improvement", pend=(1,))
        if sid == "house_redevelopment":
            st = self._do_renovate(st, p)
            return replace(st, phase="improvement", pend=(0,))
        if sid == "farm_redevelopment":
            st = self._do_renovate(st, p)
            return replace(st, phase="fencing", pend=(0, 0))
        if sid == "basic_wish_for_children":
            assert pl.family < len(pl.rooms) and pl.family < MAX_FAMILY
            st = self._setp(st, p, replace(pl, family=pl.family + 1,
                                           born=pl.born + 1))
            return self._end_turn(st)
        if sid == "urgent_wish_for_children":
            assert pl.family < MAX_FAMILY
            st = self._setp(st, p, replace(pl, family=pl.family + 1,
                                           born=pl.born + 1))
            return self._end_turn(st)
        raise AssertionError(f"unknown space {sid!r}")

    # -------------------------------------------------------- sub-effects

    def _setp(self, st: State, p: int, pl: Player) -> State:
        players = list(st.players)
        players[p] = pl
        return replace(st, players=tuple(players))

    def _add_goods(self, st: State, p: int, gi: int, amt: int) -> State:
        pl = st.players[p]
        goods = list(pl.goods)
        goods[gi] += amt
        return self._setp(st, p, replace(pl, goods=tuple(goods)))

    def _do_plow(self, st: State, p: int, r: int, c: int) -> State:
        pl = st.players[p]
        assert (r, c) in _plow_targets(pl)
        fields = tuple(sorted(pl.fields + ((r, c, "", 0),)))
        return self._setp(st, p, replace(pl, fields=fields))

    def _do_sow(self, st: State, p: int, r: int, c: int, crop: str) -> State:
        pl = st.players[p]
        gi = GRAIN if crop == "grain" else VEG
        assert pl.goods[gi] > 0
        fields = []
        found = False
        for (fr, fc, fcrop, cnt) in pl.fields:
            if (fr, fc) == (r, c):
                assert cnt == 0
                found = True
                fields.append((fr, fc, crop, 3 if crop == "grain" else 2))
            else:
                fields.append((fr, fc, fcrop, cnt))
        assert found
        goods = list(pl.goods)
        goods[gi] -= 1
        return self._setp(st, p, replace(pl, goods=tuple(goods),
                                         fields=tuple(sorted(fields))))

    def _do_bake(self, st: State, p: int, cl: int, sl: int):
        pl = st.players[p]
        assert pl.goods[GRAIN] > 0
        food, which = _best_bake(pl, cl, sl)
        assert food > 0
        if which == "clay":
            cl -= 1
        elif which == "stone":
            sl -= 1
        goods = list(pl.goods)
        goods[GRAIN] -= 1
        st = self._setp(st, p, replace(pl, goods=tuple(goods),
                                       food=pl.food + food))
        return st, cl, sl

    def _do_renovate(self, st: State, p: int) -> State:
        pl = st.players[p]
        assert _can_renovate(pl)
        res = CLAY if pl.house == 0 else STONE
        goods = list(pl.goods)
        goods[REED] -= 1
        goods[res] -= len(pl.rooms)
        return self._setp(st, p, replace(pl, goods=tuple(goods),
                                         house=pl.house + 1))

    def _own_major(self, st: State, p: int, i: int, goods: tuple) -> State:
        owner = list(st.major_owner)
        owner[i] = p
        st = replace(st, major_owner=tuple(owner))
        pl = st.players[p]
        st = self._setp(st, p, replace(pl, goods=goods,
                                       majors=tuple(sorted(pl.majors + (i,)))))
        if i == WELL:
            # RULING: the Well puts 1 food on each of the next 5 round spaces;
            # future round spaces are exactly the ones "not covered by an
            # action space card yet".  Rounds past 14 are lost.
            well = list(st.well)
            for r in range(st.round + 1, min(LAST_ROUND, st.round + 5) + 1):
                well[r] += 1
            st = replace(st, well=tuple(well), well_owner=p)
        return st

    def _after_improvement(self, st: State, p: int, i: int) -> State:
        # RULING: building any improvement with the bake-bread symbol
        # (Fireplace, Cooking Hearth, Clay Oven, Stone Oven) immediately
        # grants an optional "Bake Bread" action ("When you build one of
        # these ovens, you immediately get a 'Bake Bread' action").
        pl = st.players[p]
        if MAJORS[i].bake is not None and pl.goods[GRAIN] > 0:
            cl = 1 if CLAY_OVEN in pl.majors else 0
            sl = 2 if STONE_OVEN in pl.majors else 0
            return replace(st, phase="bake", pend=(cl, sl))
        return self._end_turn(st)

    def _do_convert(self, st: State, p: int, what: str) -> State:
        pl = st.players[p]
        goods = list(pl.goods)
        animals = list(pl.animals)
        craft = list(pl.craft_used)
        food = pl.food
        if what == "grain":
            assert goods[GRAIN] > 0
            goods[GRAIN] -= 1
            food += 1
        elif what == "vegetable":
            assert goods[VEG] > 0
            goods[VEG] -= 1
            food += _veg_value(pl)
        elif what in ANIMAL_NAMES:
            t = ANIMAL_NAMES.index(what)
            v = _cook_value(pl, t)
            assert animals[t] > 0 and v > 0
            animals[t] -= 1
            food += v
        else:
            gi = GOOD_NAMES.index(what)
            imp, val, slot = CRAFT[gi]
            assert imp in pl.majors and not craft[slot] and goods[gi] > 0
            goods[gi] -= 1
            craft[slot] = True
            food += val
        return self._setp(st, p, replace(pl, goods=tuple(goods),
                                         animals=tuple(animals),
                                         craft_used=tuple(craft), food=food))

    def _maybe_adjust(self, st: State, p: int) -> State:
        # RULING: animals that cannot be accommodated must be dealt with
        # immediately (cooked with a Fireplace/Cooking Hearth or released to
        # the general supply).  The optional "cook animals at any time"
        # ability is surfaced here and in the feeding phase -- the only
        # moments at which it can matter.
        if _animals_fit(st.players[p]):
            return self._end_turn(st)
        return replace(st, phase="adjust", pend=())

    # ------------------------------------------------------- flow control

    def _end_turn(self, st: State) -> State:
        for k in range(1, st.n + 1):
            q = (st.current + k) % st.n
            if st.players[q].to_place > 0:
                return replace(st, current=q, phase="main", pend=())
        return self._end_work(st)

    def _end_work(self, st: State) -> State:
        # 3. returning home is implicit; 4. harvest on the marked rounds.
        if st.round in HARVEST_ROUNDS:
            st = self._field_phase(st)
            return replace(st, phase="feeding", current=0, pend=())
        return self._next_round(st)

    def _field_phase(self, st: State) -> State:
        players = []
        for pl in st.players:
            goods = list(pl.goods)
            fields = []
            for (r, c, crop, cnt) in pl.fields:
                if cnt > 0:
                    goods[GRAIN if crop == "grain" else VEG] += 1
                    cnt -= 1
                    if cnt == 0:
                        crop = ""
                fields.append((r, c, crop, cnt))
            players.append(replace(pl, goods=tuple(goods),
                                   fields=tuple(sorted(fields)),
                                   craft_used=(False, False, False)))
        return replace(st, players=tuple(players))

    def _breed_all(self, st: State) -> State:
        # RULING: newborn animals are awarded automatically in the order
        # sheep -> wild boar -> cattle, each only while the whole herd still
        # fits on the farm.  (The rulebook lets the owner arrange animals
        # freely; this fixed order only matters in the rare case where there
        # is room for some but not all eligible newborns.)
        players = []
        for pl in st.players:
            animals = list(pl.animals)
            for t in range(3):
                if animals[t] >= 2:
                    trial = list(animals)
                    trial[t] += 1
                    if _animals_fit(pl, trial):
                        animals = trial
            players.append(replace(pl, animals=tuple(animals)))
        return replace(st, players=tuple(players))

    def _next_round(self, st: State) -> State:
        if st.round >= LAST_ROUND:
            # keep the round-14 born/passed counters so the bookkeeping
            # invariant still holds on the final state
            return replace(st, terminal=True, phase="end")
        players = tuple(replace(pl, born=0, passed=0) for pl in st.players)
        st = replace(st, players=players, round=st.round + 1,
                     occupied=tuple([-1] * len(st.spaces)))
        return self._prepare(st)

    def _prepare(self, st: State) -> State:
        # preparation phase: reveal this round's card (implicit), pay out any
        # food waiting on the round space, replenish accumulation spaces.
        players = list(st.players)
        if st.well[st.round] > 0 and st.well_owner >= 0:
            pl = players[st.well_owner]
            players[st.well_owner] = replace(pl,
                                             food=pl.food + st.well[st.round])
            well = list(st.well)
            well[st.round] = 0
            st = replace(st, well=tuple(well))
        players = [replace(pl, to_place=pl.family) for pl in players]
        piles = list(st.piles)
        for i in range(st.n_perm + st.round):
            sid = st.spaces[i]
            if sid in ACC:
                piles[i] += ACC[sid][1]
        return replace(st, players=tuple(players), piles=tuple(piles),
                       phase="main", current=st.start, pend=())

    # ---------------------------------------------------------- invariants

    def check_invariants(self, st: State) -> None:
        assert 1 <= st.round <= LAST_ROUND
        assert 0 <= st.current < st.n and 0 <= st.start < st.n
        occ_counts = [0] * st.n
        for i, o in enumerate(st.occupied):
            assert o == -1 or 0 <= o < st.n
            if o != -1:
                assert i < st.n_perm + st.round, "person on unrevealed space"
                occ_counts[o] += 1
        for i, amt in enumerate(st.piles):
            assert amt >= 0
            if amt > 0:
                assert st.spaces[i] in ACC
        for i, o in enumerate(st.major_owner):
            assert -1 <= o < st.n
        for p, pl in enumerate(st.players):
            assert pl.food >= 0 and pl.begging >= 0
            assert all(g >= 0 for g in pl.goods)
            assert all(x >= 0 for x in pl.animals)
            # NOTE: family may legally exceed the number of rooms via the
            # "Family Growth Even without Room" action; the structural cap is
            # the five people of the player's colour.
            assert 2 <= pl.family <= MAX_FAMILY
            assert 0 <= pl.born <= pl.family
            assert 0 <= pl.to_place <= pl.family
            assert pl.house in (0, 1, 2)
            assert len(pl.stables) <= MAX_STABLES
            assert len(pl.fences) <= MAX_FENCES
            rooms = set(pl.rooms)
            fields = _field_cells(pl)
            stables = set(pl.stables)
            assert len(rooms) == len(pl.rooms) >= 2
            assert len(fields) == len(pl.fields)
            assert len(stables) == len(pl.stables)
            for cell in rooms | fields | stables:
                assert cell in CELLS
            assert not rooms & fields
            assert not rooms & stables
            assert not fields & stables
            assert _connected(rooms), "rooms must be adjacent to rooms"
            assert _connected(fields), "fields must be adjacent to fields"
            for (r, c, crop, cnt) in pl.fields:
                assert cnt >= 0 and (crop != "") == (cnt > 0)
            for (kind, r, c) in pl.fences:
                if kind == "H":
                    assert 0 <= r <= ROWS and 0 <= c < COLS
                else:
                    assert 0 <= r < ROWS and 0 <= c <= COLS
            assert _valid_fence_layout(pl.fences, _tile_cells(pl))
            # RULING: all fences of a Fencing action are placed first and the
            # animals are only rearranged once the action is complete.  A
            # fence sub-step may therefore leave the herd temporarily without
            # a home (e.g. enclosing previously unfenced stables collapses
            # their individual 1-animal slots into capacity doubling of a
            # single-species pasture) -- and a *further* sub-step could even
            # restore capacity by fencing a second region.  Overflow is thus
            # tolerated while the fencing phase is still open; ("done",)
            # routes through _maybe_adjust, which forces the cook/release
            # "adjust" phase whenever no assignment of the animals to
            # (house pet, unfenced stables, single-species pastures) exists.
            if (st.current != p
                    or st.phase not in ("adjust", "fencing")):
                assert _animals_fit(pl), "animals exceed farm capacity"
            assert sorted(pl.majors) == sorted(
                i for i, o in enumerate(st.major_owner) if o == p)
            # people placed + forced passes + people still to place add up
            assert occ_counts[p] + pl.passed + pl.to_place \
                == pl.family - pl.born
        # scores must always be computable
        self.scores(st)

    # ------------------------------------------------------------- summary

    def summary(self, st: State) -> dict:
        out = {"score": [], "begging": [], "rooms": [], "family": [],
               "fields": [], "pastures": [], "sheep": [], "boar": [],
               "cattle": [], "major_improvements": []}
        for pl in st.players:
            out["score"].append(sum(_score_breakdown(pl).values()))
            out["begging"].append(pl.begging)
            out["rooms"].append(len(pl.rooms))
            out["family"].append(pl.family)
            out["fields"].append(len(pl.fields))
            out["pastures"].append(len(_pastures(pl.fences)))
            out["sheep"].append(pl.animals[SHEEP])
            out["boar"].append(pl.animals[BOAR])
            out["cattle"].append(pl.animals[CATTLE])
            out["major_improvements"].append(len(pl.majors))
        return out

    # ----------------------------------------------------------- describe

    def describe_state(self, st: State) -> str:
        bits = []
        abbrev = ("FP", "FP", "CH", "CH", "We", "CO", "SO", "Jo", "Po", "Ba")
        for p, pl in enumerate(st.players):
            g = pl.goods
            star = "*" if p == st.start else ""
            born = f"({pl.born} newborn)" if pl.born else ""
            sown = "".join(f" {cnt}{crop[0]}" for (_, _, crop, cnt) in
                           pl.fields if cnt > 0)
            imps = ("," + "+".join(abbrev[i] for i in pl.majors)
                    if pl.majors else "")
            bits.append(
                f"P{p}{star}: {pl.food}f fam{pl.family}{born} "
                f"[{HOUSE_NAMES[pl.house]}x{len(pl.rooms)} "
                f"fld{len(pl.fields)}{sown} pas{len(_pastures(pl.fences))} "
                f"stb{len(pl.stables)}{imps}] "
                f"W{g[WOOD]} C{g[CLAY]} R{g[REED]} S{g[STONE]} "
                f"G{g[GRAIN]} V{g[VEG]} "
                f"sh{pl.animals[0]} bo{pl.animals[1]} ca{pl.animals[2]} "
                f"beg{pl.begging}")
        head = "END" if st.terminal else f"R{st.round:02d} {st.phase} " \
                                         f"P{st.current} to act"
        return head + " | " + " | ".join(bits)

    def describe_action(self, st: State, a) -> str:
        p = st.current
        pl = st.players[p]
        kind = a[0]
        if kind == "place":
            sid = st.spaces[a[1]]
            extra = ""
            if sid in ACC:
                extra = f" (takes {st.piles[a[1]]} {ACC[sid][0]})"
            return f"P{p} places a person on {SPACE_NAMES[sid]}{extra}"
        if kind == "pass":
            return f"P{p} cannot place a person and forfeits one placement"
        if kind == "plow":
            return f"P{p} plows a field at {a[1:]}"
        if kind == "sow":
            n = 3 if a[3] == "grain" else 2
            return f"P{p} sows {a[3]} on field {a[1:3]} (now {n} {a[3]})"
        if kind == "bake":
            cl, sl = (st.pend[1], st.pend[2]) if st.phase == "sow_bake" \
                else st.pend
            food, _ = _best_bake(pl, cl, sl)
            return f"P{p} bakes 1 grain into {food} food"
        if kind == "room":
            res, amt, reed = _room_cost(pl)
            return (f"P{p} builds a {HOUSE_NAMES[pl.house]} room at {a[1:]} "
                    f"({amt} {GOOD_NAMES[res]} + {reed} reed)")
        if kind == "stable":
            return f"P{p} builds a stable at {a[1:]} (2 wood)"
        if kind == "fence":
            cost = len(_region_new_edges(set(a[1]), set(pl.fences)))
            return (f"P{p} fences the region {list(a[1])} "
                    f"({cost} new fences, {cost} wood)")
        if kind == "major":
            return f"P{p} builds the {MAJORS[a[1]].name}"
        if kind == "upgrade":
            return f"P{p} upgrades a Fireplace to the {MAJORS[a[1]].name}"
        if kind == "take":
            return f"P{p} takes 1 {a[1]} (Side Job)"
        if kind == "convert":
            return f"P{p} converts 1 {a[1]} into food"
        if kind == "feed":
            need = 2 * (pl.family - pl.born) + pl.born
            short = max(0, need - pl.food)
            s = f"P{p} feeds the family ({need} food needed)"
            if short:
                s += f" and begs for {short} missing food"
            return s
        if kind == "cook":
            return (f"P{p} cooks 1 {ANIMAL_NAMES[a[1]]} "
                    f"(+{_cook_value(pl, a[1])} food, no room on the farm)")
        if kind == "release":
            return f"P{p} releases 1 {ANIMAL_NAMES[a[1]]} (no room on the farm)"
        if kind == "done":
            return f"P{p} finishes the action"
        return f"P{p}: {a!r}"

    def describe_final(self, st: State) -> str:
        lines = []
        for p, pl in enumerate(st.players):
            b = _score_breakdown(pl)
            total = sum(b.values())
            parts = ", ".join(f"{k} {v:+d}" for k, v in b.items())
            lines.append(f"P{p}: {total} points ({parts})")
        return "\n".join(lines)
