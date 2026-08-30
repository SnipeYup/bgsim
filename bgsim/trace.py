"""Game traces: the raw material for independent verification.

A trace is a JSON-serialisable record of one full game: every state, every
action, and the final scoring. Checkers (see verify.py) re-derive individual
rules from the rulebook and audit traces without ever seeing engine code.

    python -m bgsim.trace bgsim.games.agricola.game:Agricola --players 2 --seed 0 --schema

prints the trace schema and a few sample steps — that text goes into the
checker-generation prompt so a fresh session knows the data shape.
"""
from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from .agents import make_agent
from .games import make_game


def _plain(x):
    if is_dataclass(x):
        return _plain(asdict(x))
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def record_trace(game, agent_specs, n_players: int, seed: int,
                 max_actions: int = 20000) -> dict:
    agents = [make_agent(s, seed * 100 + i) for i, s in enumerate(agent_specs)]
    state = game.initial_state(n_players, seed)
    steps = []
    for i in range(max_actions):
        if game.is_terminal(state):
            break
        p = game.current_player(state)
        action = agents[p].act(game, state, p)
        nxt = game.apply(state, action)
        steps.append({
            "i": i, "phase": game.phase(state), "player": p,
            "action": _plain(action), "before": _plain(state),
            "after": _plain(nxt),
        })
        state = nxt
    return {
        "meta": {"game": game.name, "n_players": n_players, "seed": seed,
                 "agents": agent_specs, "finished": game.is_terminal(state)},
        "steps": steps,
        "final": _plain(state),
        "scores": _plain(game.scores(state)),
        "summary": _plain(game.summary(state)) if hasattr(game, "summary") else {},
    }


def save_trace(trace: dict, path: str | Path) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(trace, f)


def load_trace(path: str | Path) -> dict:
    with gzip.open(path, "rt") as f:
        return json.load(f)


# ---------------------------------------------------------------- schema
def _schema(x, depth=0):
    if isinstance(x, dict):
        return {k: _schema(v, depth + 1) for k, v in x.items()}
    if isinstance(x, list):
        if not x:
            return ["<empty list>"]
        return [_schema(x[0], depth + 1), f"... x{len(x)}"] if len(x) > 1 else [_schema(x[0], depth + 1)]
    return type(x).__name__


def print_schema(trace: dict) -> str:
    out = ["TRACE FORMAT", "============",
           "A trace is a JSON object: {meta, steps, final, scores, summary}.",
           "steps is a list; each step is {i, phase, player, action, before, after}",
           "where before/after are full game states. Schema of one state:", "",
           json.dumps(_schema(trace["steps"][0]["before"]), indent=1), "",
           "meta: " + json.dumps(trace["meta"]),
           "scores: " + json.dumps(trace["scores"]),
           "summary: " + json.dumps(_schema(trace["summary"]), indent=1), "",
           "SAMPLE STEPS (action + state deltas are what checkers audit)", ""]
    for s in trace["steps"][:2] + trace["steps"][-2:]:
        out.append(json.dumps({"i": s["i"], "phase": s["phase"],
                               "player": s["player"], "action": s["action"]}))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agents", default="scoregreedy")
    ap.add_argument("--out", default=None, help="write trace to this .json.gz")
    ap.add_argument("--schema", action="store_true", help="print schema + samples")
    a = ap.parse_args(argv)
    game = make_game(a.game)
    specs = a.agents.split(",")
    if len(specs) == 1:
        specs = specs * a.players
    tr = record_trace(game, specs, a.players, a.seed)
    if a.out:
        save_trace(tr, a.out)
        print(f"wrote {a.out} ({len(tr['steps'])} steps)")
    if a.schema or not a.out:
        print(print_schema(tr))


if __name__ == "__main__":
    main()
