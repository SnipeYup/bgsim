"""Evolve greedy weight vectors by tournament.

Each generation: sample tables of `n_players` distinct population members,
play seeded games, fitness = win share. Keep the top half, refill with
mutated crossovers. The best vector at the end is the strongest strategy
the search could find — the seed of the "dominant strategy" report.
"""
from __future__ import annotations

import json
import random
import time
from multiprocessing import Pool

from ..engine import play_game
from ..games import make_game
from . import EXPERT_WEIGHTS, GreedyAgent

_GAME = None


def _init(game_name):
    global _GAME
    _GAME = make_game(game_name)


def _table(args):
    seed, members, weights = args
    agents = [GreedyAgent(weights[m], seed * 10 + i, name=f"pop{m}")
              for i, m in enumerate(members)]
    rec = play_game(_GAME, agents, seed)
    share = 1.0 / len(rec.winners)
    return [(members[w], share) for w in rec.winners], [(m, 1.0) for m in members]


def evolve(game_name: str = "splendor", n_players: int = 4, pop_size: int = 24,
           generations: int = 10, games_per_gen: int = 120, seed: int = 0,
           workers: int = 1, sigma: float = 0.3, seed_expert: bool = True,
           log=print) -> dict:
    rng = random.Random(seed)
    dim = len(EXPERT_WEIGHTS)
    pop = [[rng.uniform(-1, 1) * (10 if i == 0 else 1) for i in range(dim)]
           for _ in range(pop_size)]
    if seed_expert:
        pop[0] = list(EXPERT_WEIGHTS)
    history = []
    game_seed = seed * 1_000_000
    _init(game_name)
    pool = Pool(workers, initializer=_init, initargs=(game_name,)) if workers > 1 else None
    try:
        for g in range(generations):
            t0 = time.perf_counter()
            jobs = []
            for _ in range(games_per_gen):
                members = rng.sample(range(pop_size), n_players)
                jobs.append((game_seed, members, pop))
                game_seed += 1
            results = pool.map(_table, jobs) if pool else [_table(j) for j in jobs]
            wins = [0.0] * pop_size
            played = [0.0] * pop_size
            for ws, ps in results:
                for m, s in ws:
                    wins[m] += s
                for m, s in ps:
                    played[m] += s
            fitness = [wins[i] / played[i] if played[i] else 0.0 for i in range(pop_size)]
            order = sorted(range(pop_size), key=lambda i: -fitness[i])
            best = order[0]
            history.append({"gen": g, "best_fitness": fitness[best],
                            "best_weights": list(pop[best]),
                            "mean_fitness": sum(fitness) / pop_size})
            log(f"gen {g:2d}  best win rate {fitness[best]:.2f}  "
                f"mean {sum(fitness) / pop_size:.2f}  "
                f"({time.perf_counter() - t0:.1f}s)")
            survivors = [pop[i] for i in order[: pop_size // 2]]
            children = []
            while len(survivors) + len(children) < pop_size:
                a, b = rng.sample(survivors, 2)
                child = [(x if rng.random() < 0.5 else y) + rng.gauss(0, sigma)
                         for x, y in zip(a, b)]
                children.append(child)
            pop = survivors + children
    finally:
        if pool:
            pool.close()
            pool.join()
    best = history[-1]["best_weights"]
    return {"weights": best, "history": history,
            "feature_names": list(_GAME.FEATURE_NAMES) if hasattr(_GAME, "FEATURE_NAMES") else None}


def save(result: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
