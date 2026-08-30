"""Splendor card and noble data.

Loads `cards.csv` / `nobles.csv` (the real deck) from the data directory. If
they are missing, generates a SYNTHETIC deck with the same shape as the real
game (40/30/20 cards, matching cost/point distributions) so the engine still
runs.

CSV formats:
  cards.csv : id,tier,points,bonus,white,blue,green,red,black
  nobles.csv: id,points,white,blue,green,red,black
`bonus` is a color name.
"""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path

COLORS = ("white", "blue", "green", "red", "black")
N_COLORS = 5
GOLD = 5  # index of the gold/joker token
DATA_DIR = Path(__file__).parent / "data"


@dataclass(frozen=True)
class Card:
    id: int
    tier: int          # 1, 2, 3
    points: int
    bonus: int         # color index 0..4
    cost: tuple        # len 5, one per color


@dataclass(frozen=True)
class Noble:
    id: int
    points: int
    req: tuple         # len 5, bonus cards required per color


def load_cards(path: Path = DATA_DIR / "cards.csv") -> tuple[Card, ...]:
    if not path.exists():
        return synthetic_cards()
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append(Card(
                id=int(row["id"]), tier=int(row["tier"]),
                points=int(row["points"]), bonus=COLORS.index(row["bonus"]),
                cost=tuple(int(row[c]) for c in COLORS)))
    return tuple(out)


def load_nobles(path: Path = DATA_DIR / "nobles.csv") -> tuple[Noble, ...]:
    if not path.exists():
        return synthetic_nobles()
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            out.append(Noble(id=int(row["id"]), points=int(row["points"]),
                             req=tuple(int(row[c]) for c in COLORS)))
    return tuple(out)


def write_csvs(cards, nobles, data_dir: Path = DATA_DIR) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / "cards.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tier", "points", "bonus", *COLORS])
        for c in cards:
            w.writerow([c.id, c.tier, c.points, COLORS[c.bonus], *c.cost])
    with (data_dir / "nobles.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "points", *COLORS])
        for n in nobles:
            w.writerow([n.id, n.points, *n.req])


# ---------------------------------------------------------------------------
# Synthetic deck: same tier sizes and roughly the same cost/point shapes as the
# real game, but NOT the real cards.
# ---------------------------------------------------------------------------

# Cost shapes as (points, [amounts]) — amounts get assigned to colors other
# than the card's own bonus color (the real game mostly avoids self-cost).
_T1_SHAPES = [
    (0, [1, 1, 1, 1]), (0, [1, 2, 1, 1]), (0, [2, 2, 1]), (0, [2, 1]),
    (0, [3]), (0, [2, 2]), (0, [1, 1, 1, 2]), (1, [4]),
]
_T2_SHAPES = [
    (1, [3, 2, 2]), (1, [2, 3, 3]), (2, [5]), (2, [4, 2, 1]),
    (2, [5, 3]), (3, [6]),
]
_T3_SHAPES = [
    (3, [3, 3, 5, 3]), (4, [7]), (4, [3, 6, 3]), (5, [7, 3]),
]


def synthetic_cards(seed: int = 0) -> tuple[Card, ...]:
    rng = random.Random(seed)
    cards = []
    cid = 0
    for tier, shapes in ((1, _T1_SHAPES), (2, _T2_SHAPES), (3, _T3_SHAPES)):
        for bonus in range(N_COLORS):
            others = [c for c in range(N_COLORS) if c != bonus]
            for points, amounts in shapes:
                cost = [0] * N_COLORS
                cols = rng.sample(others, len(amounts))
                for col, amt in zip(cols, amounts):
                    cost[col] = amt
                cards.append(Card(cid, tier, points, bonus, tuple(cost)))
                cid += 1
    return tuple(cards)


def synthetic_nobles(seed: int = 0) -> tuple[Noble, ...]:
    rng = random.Random(seed)
    nobles = []
    for i in range(10):
        req = [0] * N_COLORS
        if i < 5:
            for c in rng.sample(range(N_COLORS), 3):
                req[c] = 3
        else:
            for c in rng.sample(range(N_COLORS), 2):
                req[c] = 4
        nobles.append(Noble(i, 3, tuple(req)))
    return tuple(nobles)
