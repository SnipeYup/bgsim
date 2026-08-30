"""Inspect an engine that has no reference implementation.

    python -m bgsim.inspect bgsim.games.agricola.game:Agricola --players 2 --games 30 --log 1

Plays seeded games with invariants checked on every action, prints a
narrated log of the first `--log` games using the engine's optional
`describe_state(state)` / `describe_action(state, action)` hooks, then a
histogram of action kinds and score statistics. This is the designer-facing
verification surface: read the logs like a rulebook lawyer, and check the
numbers against what real games of this design produce.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from statistics import mean, median

from .agents import make_agent
from .games import make_game


def narrate(game, state, action, player) -> str:
    da = getattr(game, "describe_action", None)
    txt = da(state, action) if da else repr(action)
    return f"  P{player}: {txt}"


def run(game, agents, seed, log: bool, max_actions: int = 20000):
    state = game.initial_state(len(agents), seed)
    kinds = Counter()
    lines = []
    last_phase_desc = None
    for step in range(max_actions):
        if game.is_terminal(state):
            break
        p = game.current_player(state)
        ds = getattr(game, "describe_state", None)
        if log and ds:
            desc = ds(state)
            if desc != last_phase_desc:
                lines.append(desc)
                last_phase_desc = desc
        action = agents[p].act(game, state, p)
        legal = game.legal_actions(state)
        assert action in legal, f"agent returned illegal action {action!r}"
        kinds[action[0] if isinstance(action, tuple) else str(action)] += 1
        if log:
            lines.append(narrate(game, state, action, p))
        state = game.apply(state, action)
        game.check_invariants(state)
    else:
        lines.append(f"  [unfinished: {max_actions} actions without reaching the end; "
                     f"the rules allow indefinite play here]")
        kinds["__unfinished__"] += 1
    return state, kinds, lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("game", help="registered name or module:Class")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--agents", default="scoregreedy", help="spec, or comma list per seat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=int, default=1, help="narrate this many games in full")
    a = ap.parse_args(argv)

    game = make_game(a.game)
    specs = a.agents.split(",")
    if len(specs) == 1:
        specs = specs * a.players
    all_kinds = Counter()
    scores, lengths, summaries = [], [], []
    for g in range(a.games):
        seed = a.seed + g
        agents = [make_agent(s, seed * 100 + i) for i, s in enumerate(specs)]
        try:
            state, kinds, lines = run(game, agents, seed, log=g < a.log)
        except AssertionError as e:
            print(f"\nINVARIANT/LEGALITY FAILURE in game {g} (seed {seed}): {e}")
            sys.exit(1)
        if g < a.log:
            print(f"\n===== game {g} (seed {seed}) =====")
            print("\n".join(lines))
            sf = getattr(game, "describe_final", None)
            if sf:
                print(sf(state))
        all_kinds.update(kinds)
        sc = game.scores(state)
        scores.append(sc)
        lengths.append(sum(kinds.values()))
        if hasattr(game, "summary"):
            summaries.append(game.summary(state))

    print(f"\n===== {a.games} games, {a.players} players, agents {specs} =====")
    print("invariants held on every action of every game")
    unfinished = all_kinds.pop("__unfinished__", 0)
    if unfinished:
        print(f"UNFINISHED GAMES: {unfinished} of {a.games} hit the action cap — "
              f"the rules permit stalling loops with these agents")
    print(f"decisions per game: mean {mean(lengths):.0f}, min {min(lengths)}, max {max(lengths)}")
    print("action kinds:", ", ".join(f"{k} {v}" for k, v in all_kinds.most_common()))
    firsts = [s[0] for sc in scores for s in sc]
    print(f"score (first element of score tuple): mean {mean(firsts):.1f}, "
          f"median {median(firsts):.1f}, min {min(firsts)}, max {max(firsts)}")
    win = Counter()
    for sc in scores:
        best = max(sc)
        ws = [i for i, x in enumerate(sc) if x == best]
        for w in ws:
            win[w] += 1 / len(ws)
    print("wins by seat:", ", ".join(f"seat {i}: {win[i] / a.games:.0%}" for i in range(a.players)))
    if summaries and isinstance(summaries[0], dict):
        keys = [k for k, v in summaries[0].items()
                if isinstance(v, list) and v and isinstance(v[0], (int, float))]
        for k in keys:
            vals = [x for s in summaries for x in s[k]]
            print(f"{k}: mean {mean(vals):.2f}, min {min(vals)}, max {max(vals)}")


if __name__ == "__main__":
    main()
