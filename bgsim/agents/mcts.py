"""A generic Monte Carlo tree search player.

Needs only the Game protocol (legal_actions / apply / is_terminal / scores /
current_player): no features, no weights, no knowledge of the game. Plays far
better than one-move greedy, at a CPU cost per move. Used for "competent
play" measurements, not for bulk statistics.

Multi-player: each node credits the player who acted into it; a rollout that
does not reach the end is scored by final rank (share of first place).
"""
from __future__ import annotations

import math
import random


def _rank_credit(scores, n_players):
    """1.0 for the sole leader, shared for ties, 0 for everyone else."""
    keyed = [tuple(s) if isinstance(s, (tuple, list)) else (s,) for s in scores]
    top = max(keyed)
    leaders = [i for i, s in enumerate(keyed) if s == top]
    return [1.0 / len(leaders) if i in leaders else 0.0 for i in range(n_players)]


class _Node:
    __slots__ = ("state", "player", "children", "untried", "n", "w", "action")

    def __init__(self, state, player, actions, action=None):
        self.state, self.player, self.action = state, player, action
        self.children, self.untried = [], list(actions)
        self.n, self.w = 0, 0.0


class MCTSAgent:
    def __init__(self, iterations: int = 60, rollout_depth: int = 40, seed: int = 0, c: float = 1.2,
                 name: str = "mcts"):
        self.iterations, self.depth, self.c = iterations, rollout_depth, c
        self.rng = random.Random(seed)
        self.name = name

    def act(self, game, state, player):
        legal = game.legal_actions(state)
        if len(legal) == 1:
            return legal[0]
        n_players = len(game.scores(state))
        root = _Node(state, player, legal)
        self.rng.shuffle(root.untried)
        for _ in range(self.iterations):
            node, path = root, [root]
            # select
            while not node.untried and node.children:
                node = max(node.children, key=lambda ch: ch.w / ch.n + self.c * math.sqrt(math.log(node.n + 1) / ch.n))
                path.append(node)
            # expand
            if node.untried:
                a = node.untried.pop()
                st = game.apply(node.state, a)
                acts = [] if game.is_terminal(st) else game.legal_actions(st)
                child = _Node(st, node.player, acts, a)
                # the player who chose `a` is node's mover; the child stores who moves next
                child.player = game.current_player(st) if not game.is_terminal(st) else node.player
                node.children.append(child); path.append(child); node = child
            # rollout
            st, d = node.state, 0
            while not game.is_terminal(st) and d < self.depth:
                acts = game.legal_actions(st)
                st = game.apply(st, self.rng.choice(acts)); d += 1
            credit = _rank_credit(game.scores(st), n_players)
            # backprop: each node's value is from the viewpoint of the player who moved INTO it
            for i in range(1, len(path)):
                mover = path[i - 1].player
                path[i].n += 1; path[i].w += credit[mover]
            root.n += 1
        best = max(root.children, key=lambda ch: ch.n)
        return best.action
