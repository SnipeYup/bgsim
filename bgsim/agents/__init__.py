"""Agents. All are seeded so a (game seed, agent seed) pair replays exactly."""
from __future__ import annotations

import json
import random
from typing import Sequence

from ..engine import Game, State


class RandomAgent:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def act(self, game: Game, state: State, player: int):
        return self.rng.choice(game.legal_actions(state))


class GreedyAgent:
    """One-ply: apply each legal action, score the resulting state with a
    weighted feature sum, take the best. Forced sub-phases (discard, noble
    choice) are scored the same way, so no special casing is needed."""

    def __init__(self, weights: Sequence[float], seed: int = 0,
                 name: str = "greedy"):
        self.weights = tuple(weights)
        self.rng = random.Random(seed)
        self.name = name

    def act(self, game: Game, state: State, player: int):
        best, best_score = [], None
        for a in game.legal_actions(state):
            nxt = game.apply(state, a)
            f = game.features(nxt, player)
            score = sum(w * x for w, x in zip(self.weights, f))
            if best_score is None or score > best_score + 1e-9:
                best, best_score = [a], score
            elif abs(score - best_score) <= 1e-9:
                best.append(a)
        return self.rng.choice(best)


# Hand-written yardstick for Splendor (feature order: see Splendor.FEATURE_NAMES)
EXPERT_WEIGHTS = (10.0, 2.0, 0.3, 0.6, 0.5, 0.8, 3.0, -0.2, -1.0)


def make_agent(spec: str, seed: int = 0):
    """Build an agent from a CLI spec string.

    random | greedy | expert | greedy:[w0,w1,...] | greedy@path.json
    """
    if spec == "random":
        return RandomAgent(seed)
    if spec == "expert":
        return GreedyAgent(EXPERT_WEIGHTS, seed, name="expert")
    if spec == "greedy":
        rng = random.Random(seed)
        w = [rng.uniform(-1, 1) for _ in EXPERT_WEIGHTS]
        w[0] = abs(w[0]) * 10  # points always matter
        return GreedyAgent(w, seed, name="greedy")
    if spec.startswith("greedy:"):
        return GreedyAgent(json.loads(spec[len("greedy:"):]), seed, name="greedy")
    if spec.startswith("greedy@"):
        with open(spec[len("greedy@"):]) as f:
            return GreedyAgent(json.load(f)["weights"], seed, name="evolved")
    raise ValueError(f"unknown agent spec {spec!r}")
