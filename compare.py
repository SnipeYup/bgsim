"""Compare a candidate Splendor engine against the reference, state by state.

Usage:
    python -m bgsim.compare games.splendor_gen.game:Splendor --games 10000

The candidate must implement the bgsim engine API and load the same card CSVs.
For each seed, both engines play the same random action sequence. At every
step the candidate's `legal_actions` (as a set) must equal the reference's,
and after applying the shared action the scores / phase / current player
must agree. The first disagreement is printed with the full reference state,
which is the feedback to hand back to the code generator.

Exit code 0 = engines agree on every state of every game.
"""
from __future__ import annotations

import argparse
import importlib
import random
import sys
from dataclasses import asdict, is_dataclass

from .games.splendor.game import Splendor


def load(spec: str):
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls or "Splendor")()


def describe(state) -> str:
    return repr(asdict(state) if is_dataclass(state) else state)


def compare(ref, cand, n_games: int, n_players: int, seed: int, max_actions: int = 5000):
    for g in range(n_games):
        game_seed = seed + g
        rng = random.Random(game_seed)
        s_ref = ref.initial_state(n_players, game_seed)
        s_cand = cand.initial_state(n_players, game_seed)
        for step in range(max_actions):
            done_ref, done_cand = ref.is_terminal(s_ref), cand.is_terminal(s_cand)
            if done_ref != done_cand:
                return fail(g, step, "is_terminal", done_ref, done_cand, s_ref)
            if done_ref:
                if ref.scores(s_ref) != cand.scores(s_cand):
                    return fail(g, step, "final scores", ref.scores(s_ref),
                                cand.scores(s_cand), s_ref)
                break
            for name, fn in (("current_player", "current_player"), ("phase", "phase")):
                a, b = getattr(ref, fn)(s_ref), getattr(cand, fn)(s_cand)
                if a != b:
                    return fail(g, step, name, a, b, s_ref)
            la_ref = ref.legal_actions(s_ref)
            la_cand = cand.legal_actions(s_cand)
            if set(la_ref) != set(la_cand):
                missing = sorted(set(la_ref) - set(la_cand), key=repr)
                extra = sorted(set(la_cand) - set(la_ref), key=repr)
                return fail(g, step, "legal_actions",
                            f"candidate is missing {missing[:10]}",
                            f"candidate wrongly allows {extra[:10]}", s_ref)
            action = rng.choice(sorted(la_ref, key=repr))
            s_ref = ref.apply(s_ref, action)
            s_cand = cand.apply(s_cand, action)
            if ref.scores(s_ref) != cand.scores(s_cand):
                return fail(g, step, f"scores after {action!r}", ref.scores(s_ref),
                            cand.scores(s_cand), s_ref)
        else:
            return fail(g, max_actions, "game length", "terminated", "still running", s_ref)
        if (g + 1) % 500 == 0:
            print(f"  {g + 1}/{n_games} games agree", file=sys.stderr)
    return True


def fail(game, step, what, expected, got, state) -> bool:
    print(f"DIVERGENCE in game {game}, step {step}: {what}")
    print(f"  reference: {expected}")
    print(f"  candidate: {got}")
    print("  reference state at this point:")
    print("  " + describe(state))
    return False


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="module:Class implementing the engine API")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--players", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    ok = True
    for n in ([a.players] if a.players else [2, 3, 4]):
        print(f"comparing {a.games} games at {n} players...", file=sys.stderr)
        ok = compare(Splendor(), load(a.candidate), a.games, n, a.seed) and ok
        if not ok:
            break
    print("OK: engines agree" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
