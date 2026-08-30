"""Random playouts with invariants checked on every action, all player counts."""
import pytest

from bgsim.agents import GreedyAgent, RandomAgent, EXPERT_WEIGHTS
from bgsim.engine import play_game
from bgsim.games.splendor.game import Splendor


@pytest.mark.parametrize("n_players", [2, 3, 4])
def test_random_playouts_hold_invariants(n_players):
    game = Splendor()
    for seed in range(150):
        rec = play_game(game, [RandomAgent(seed * 10 + i) for i in range(n_players)],
                        seed=seed, debug=True)
        assert max(s[0] for s in rec.scores) >= 15 or rec.extra["stalemate"]


def test_greedy_playouts_hold_invariants():
    game = Splendor()
    for seed in range(20):
        agents = [GreedyAgent(EXPERT_WEIGHTS, seed * 10 + i) for i in range(4)]
        rec = play_game(game, agents, seed=seed, debug=True)
        assert max(s[0] for s in rec.scores) >= 15
