"""Parallel simulation harness. Agents are passed as spec strings so worker
processes can rebuild them; every game gets its own seed."""
from __future__ import annotations

import csv
import json
import time
from multiprocessing import Pool

from ..agents import make_agent
from ..engine import GameRecord, play_game
from ..games import make_game

_GAME = None


def _init(game_name: str):
    global _GAME
    _GAME = make_game(game_name)


def _one(args):
    seeds, specs, rotate, debug = args
    out = []
    for idx, seed in seeds:
        order = list(range(len(specs)))
        if rotate:
            k = idx % len(specs)
            order = order[k:] + order[:k]
        agents = [make_agent(specs[j], seed * 100 + i) for i, j in enumerate(order)]
        rec = play_game(_GAME, agents, seed, debug=debug)
        rec.extra["seat_agent"] = [specs[j] for j in order]
        out.append(rec)
    return out


def simulate(game_name: str, specs: list[str], n_games: int, seed: int = 0,
             workers: int = 1, rotate: bool = False, debug: bool = False,
             chunk: int = 50) -> tuple[list[GameRecord], float]:
    jobs = [(list(enumerate(range(seed + i, seed + min(i + chunk, n_games)), i)),
             specs, rotate, debug)
            for i in range(0, n_games, chunk)]
    t0 = time.perf_counter()
    if workers <= 1:
        _init(game_name)
        results = [_one(j) for j in jobs]
    else:
        with Pool(workers, initializer=_init, initargs=(game_name,)) as pool:
            results = pool.map(_one, jobs)
    elapsed = time.perf_counter() - t0
    return [r for batch in results for r in batch], elapsed


def write_records(records: list[GameRecord], path: str) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "n_players", "n_turns", "n_actions", "winners",
                    "scores", "agents", "extra"])
        for r in records:
            w.writerow([r.seed, r.n_players, r.n_turns, r.n_actions,
                        json.dumps(r.winners), json.dumps(r.scores),
                        json.dumps(r.agents), json.dumps(r.extra)])


def read_records(path: str) -> list[GameRecord]:
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append(GameRecord(
                seed=int(row["seed"]), n_players=int(row["n_players"]),
                n_turns=int(row["n_turns"]), n_actions=int(row["n_actions"]),
                winners=json.loads(row["winners"]),
                scores=[tuple(s) for s in json.loads(row["scores"])],
                agents=json.loads(row["agents"]), extra=json.loads(row["extra"])))
    return out
