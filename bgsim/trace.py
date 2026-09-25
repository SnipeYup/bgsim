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
        rec = {
            "i": i, "phase": game.phase(state), "player": p,
            "action": _plain(action), "before": _plain(state),
            "after": _plain(nxt),
        }
        da = getattr(game, "describe_action", None)
        if da:
            try:
                rec["describe"] = str(da(state, action))[:200]
            except TypeError:
                try:
                    rec["describe"] = str(da(action))[:200]
                except Exception:
                    pass
            except Exception:
                pass
        steps.append(rec)
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
           "ACTION ENCODINGS — one worked example per kind, with exactly what changed.",
           "Read the encoding from these examples, never guess the meaning of a position", ""]
    def _nums(x):
        if isinstance(x, bool):
            return []
        if isinstance(x, (int, float)):
            return [x]
        if isinstance(x, (list, tuple)):
            return [n for y in x for n in _nums(y)]
        return []
    # one example per action kind; prefer an example whose numbers are all
    # different, so a reader cannot confuse which position means what
    best = {}
    for st in trace["steps"]:
        kind = st["action"][0] if isinstance(st["action"], (list, tuple)) and st["action"] else str(st["action"])
        nums = _nums(st["action"])
        distinct = len(nums) == len(set(nums))
        if kind not in best or (distinct and not best[kind][0]):
            best[kind] = (distinct, st)
    for kind, (_, st) in best.items():
        out.append(json.dumps({"i": st["i"], "phase": st["phase"], "player": st["player"], "action": st["action"]}))
        if st.get("describe"):
            out.append("   engine says: " + st["describe"])
        out.append("   changed: " + state_diff(st["before"], st["after"]))
        out.append("")
    return "\n".join(out)


def state_diff(before, after, path="", limit=40) -> str:
    """Compact list of what differs between two states, with lengths for
    lists — enough to see 'decks[2]: 16 -> 15' without dumping the state."""
    diffs = []

    def walk(a, b, p):
        if len(diffs) >= limit:
            return
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                walk(a.get(k), b.get(k), f"{p}.{k}" if p else str(k))
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append(f"{p}: {len(a)} items -> {len(b)} items")
            elif a != b and all(isinstance(x, (int, float)) for x in a + b):
                diffs.append(f"{p}: {a} -> {b}")
            else:
                for idx, (x, y) in enumerate(zip(a, b)):
                    if x != y:
                        walk(x, y, f"{p}[{idx}]")
        elif a != b:
            diffs.append(f"{p}: {a!r} -> {b!r}")
    walk(before, after, path)
    return "; ".join(diffs) if diffs else "(nothing)"


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
