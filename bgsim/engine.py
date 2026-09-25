"""Generic game engine API.

Every game implements `Game`. States are immutable values; `apply` returns a
new state. Actions are small hashable tuples. All randomness happens in
`initial_state(seed)` (pre-shuffled decks) so play is fully deterministic.

This is the surface a rules-to-code generator will target later, so keep it
small and boring.

Transition atomicity: `apply` may auto-resolve forced bookkeeping, but any
state change that embodies a rule event (paying food, assigning penalty
markers, breeding, scoring) must occur in a transition whose phase names
that event — never fused into a transition of an unrelated phase. External
auditors read traces transition by transition and must be able to attribute
every delta to a rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Protocol, Sequence

Action = Hashable
State = Any


# Passing: a "pass"/"do nothing" action is legal ONLY when the rules explicitly
# allow voluntary passing, or when a player has no other legal action. Never
# offer it alongside real actions by default — greedy players will take it, and
# an "everyone passed -> game ends" rule will then end games instantly.

class Game(Protocol):
    name: str

    def initial_state(self, n_players: int, seed: int) -> State: ...
    def current_player(self, state: State) -> int: ...
    def phase(self, state: State) -> str: ...
    def legal_actions(self, state: State) -> list[Action]: ...
    def apply(self, state: State, action: Action) -> State: ...
    def is_terminal(self, state: State) -> bool: ...
    def scores(self, state: State) -> list[tuple]:
        """Per-player sortable score tuples, higher is better. Tiebreaks live
        inside the tuple (e.g. (points, -n_cards))."""
        ...
    def observation(self, state: State, player: int) -> Any:
        """What `player` can see. Identity for perfect-information games."""
        ...
    def check_invariants(self, state: State) -> None:
        """Raise AssertionError if the state is impossible under the rules."""
        ...
    def features(self, state: State, player: int) -> tuple[float, ...]:
        """Cheap scalar features of `state` from `player`'s viewpoint, used by
        heuristic agents. Not part of the rules; may be a stub."""
        ...


class Agent(Protocol):
    name: str

    def act(self, game: Game, state: State, player: int) -> Action: ...


def winners(game: Game, state: State) -> list[int]:
    scores = game.scores(state)
    best = max(scores)
    return [i for i, s in enumerate(scores) if s == best]


@dataclass
class GameRecord:
    seed: int
    n_players: int
    n_turns: int            # decisions made in the 'main' phase
    n_actions: int          # all decisions incl. forced sub-phases
    winners: list[int]
    scores: list[tuple]
    agents: list[str]
    extra: dict = field(default_factory=dict)


def play_game(game: Game, agents: Sequence[Agent], seed: int,
              debug: bool = False, max_actions: int = 5000) -> GameRecord:
    n = len(agents)
    state = game.initial_state(n, seed)
    n_turns = 0
    n_actions = 0
    unfinished = False
    timeline = []   # (turn, [primary score per player]) sampled each main-phase turn
    while not game.is_terminal(state):
        if n_actions >= max_actions:
            # The rules allow play to continue forever (e.g. two players taking
            # and discarding the same token). Score the game as it stands and
            # flag it; the report counts these.
            unfinished = True
            break
        p = game.current_player(state)
        if game.phase(state) == "main":
            n_turns += 1
            if n_turns % max(1, len(agents)) == 0:   # once per round
                try:
                    timeline.append((n_turns, [float(sc[0]) if isinstance(sc, (tuple, list)) else float(sc)
                                              for sc in game.scores(state)]))
                except Exception:
                    pass
        action = agents[p].act(game, state, p)
        if debug:
            legal = game.legal_actions(state)
            assert action in legal, f"illegal action {action!r} by player {p}"
        state = game.apply(state, action)
        n_actions += 1
        if debug:
            game.check_invariants(state)
    extra = game.summary(state) if hasattr(game, "summary") else {}
    extra["unfinished"] = unfinished
    if timeline:
        step = max(1, len(timeline) // 24)   # keep records small: ≤ ~24 samples
        extra["timeline"] = timeline[::step] + ([timeline[-1]] if (len(timeline) - 1) % step else [])
    return GameRecord(seed=seed, n_players=n, n_turns=n_turns,
                      n_actions=n_actions, winners=winners(game, state),
                      scores=game.scores(state),
                      agents=[a.name for a in agents], extra=extra)
