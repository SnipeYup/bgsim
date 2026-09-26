"""Trained player populations.

Strength is learned, not designed: several independent evolution runs (from
different random starts) each produce a small set of strong archetypes at
equilibrium. A "population" is one run's archetypes plus their mixture. Games
for the balance analysis are played by players drawn from a population, so no
single bot's habits define the result — and the same statistic computed under
each population separately tells us whether a finding is stable.

Also measures the strength ladder: rounds-to-finish for random, novice
(score-greedy) and trained play, so every report says how good its players are.
"""
from __future__ import annotations

import random
import statistics
import time

from . import balance as _b
from .balance import evolve_vs, feature_scales, payoff_matrix, equilibrium, archetype


def train_population(game, names, n_players=2, rounds=2, seed=0, tick=None, budget_s=240, size="normal"):
    rng = random.Random(seed)
    scales = feature_scales(game, n_players, rng)
    names = (list(names) + [f"feature_{i}" for i in range(len(scales))])[:len(scales)]
    t_end = time.time() + budget_s
    _b._deadline[0] = time.process_time() + budget_s      # the search honours its CPU budget
    pop, gens, games = (16, 8, 12) if size == "normal" else (12, 6, 10)
    specs = ["scoregreedy"]
    w, fit = evolve_vs(game, specs, n_players, scales, rng, pop=pop, gens=gens, games=games, tick=tick)
    specs.append({"weights": w})
    for r in range(rounds - 1):
        if time.time() > t_end:
            break
        M = payoff_matrix(game, specs, n_players, 20, rng, tick=tick)
        mix = equilibrium(M)
        opp = [s for s, m in zip(specs, mix) if m > 0.02] or specs
        inc = max(range(len(specs)), key=lambda i: mix[i])
        w2, f2 = evolve_vs(game, opp, n_players, scales, rng, pop=pop, gens=gens, games=games,
                           seed_from=(specs[inc]["weights"] if isinstance(specs[inc], dict) else None), tick=tick)
        if f2 <= 0.55:
            break
        specs.append({"weights": w2})
    M = payoff_matrix(game, specs, n_players, 30, rng, tick=tick)
    mix = equilibrium(M)
    trained = [(s["weights"], m) for s, m in zip(specs, mix) if isinstance(s, dict) and m > 0.02]
    if not trained:   # keep the strongest evolved one regardless
        trained = [(specs[-1]["weights"], 1.0)]
    tot = sum(m for _, m in trained)
    return {"seed": seed, "n_players": n_players, "feature_names": names,
            "archetypes": [{"weights": w, "share": m / tot, "what": archetype(w, names, scales)} for w, m in trained]}


def train_populations(game, names, n_players=2, populations=3, rounds=2, tick=None, budget_s=600, size="normal"):
    out = []
    per = budget_s / populations
    for k in range(populations):
        out.append(train_population(game, names, n_players, rounds=rounds, seed=101 * (k + 1), tick=tick, budget_s=per, size=size))
    _b._deadline[0] = None
    return out


class PopulationAgent:
    """A player drawn from a population: each game picks an archetype by share."""
    def __init__(self, population, seed=0, name="trained"):
        from bgsim.agents import GreedyAgent
        rng = random.Random(seed)
        arch = population["archetypes"]
        r = rng.random(); acc = 0.0; chosen = arch[-1]
        for a in arch:
            acc += a["share"]
            if r <= acc:
                chosen = a; break
        self.inner = GreedyAgent(chosen["weights"], seed)
        self.name = name

    def act(self, game, state, player):
        return self.inner.act(game, state, player)


def strength_ladder(game, populations, n_players=2, games=12, max_actions=3000):
    """Median rounds-to-finish for random / novice / trained play, plus how
    often the trained players beat the novice — the number every report opens with."""
    from bgsim.agents import make_agent
    from bgsim.engine import play_game
    def rounds(make):
        rs, stalls = [], 0
        for seed in range(games):
            rec = play_game(game, [make(seed * 10 + i) for i in range(n_players)], seed, max_actions=max_actions)
            rs.append(rec.n_turns / n_players); stalls += bool(rec.extra.get("unfinished"))
        return statistics.median(rs), stalls
    r_rand, s_rand = rounds(lambda s: make_agent("random", s))
    r_nov, s_nov = rounds(lambda s: make_agent("scoregreedy", s))
    r_tr, s_tr = rounds(lambda s: PopulationAgent(populations[s % len(populations)], s))
    wins = 0.0
    for seed in range(games):
        seats = [(0 if (i + seed) % 2 == 0 else 1) for i in range(n_players)]
        agents = [PopulationAgent(populations[seed % len(populations)], seed * 10 + i) if si == 0 else make_agent("scoregreedy", seed * 10 + i)
                  for i, si in enumerate(seats)]
        rec = play_game(game, agents, 1000 + seed, max_actions=max_actions)
        for w in rec.winners:
            if seats[w] == 0:
                wins += 1 / len(rec.winners)
    return {"random": {"rounds": r_rand, "stalls": s_rand}, "novice": {"rounds": r_nov, "stalls": s_nov},
            "trained": {"rounds": r_tr, "stalls": s_tr}, "trained_beats_novice": wins / games, "games": games}
