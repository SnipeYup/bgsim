"""Run generated engines in a child process with a time budget.

A generated engine can loop forever inside apply() or legal_actions(); nothing
in the same process can interrupt that. So every place the app *executes* an
engine goes through run(): the work happens in a spawned process, and if it
does not finish in time the process is killed and an EngineHung error is
raised whose message the repair loop can hand back to the model.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path


class EngineHung(RuntimeError):
    pass


def load_engine(path: str):
    p = Path(path)
    spec = importlib.util.spec_from_file_location(f"gen_{p.parent.name}_{p.stem}", p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    for obj in vars(mod).values():
        if isinstance(obj, type) and hasattr(obj, "initial_state"):
            return obj()
    raise RuntimeError("generated file has no engine class")


# ----------------------------- tasks (run in the child) ----------------------

def validate(path: str, players=(2, 3, 4), seeds: int = 12):
    """Random games with invariants on. Returns None if clean, else traceback text."""
    from bgsim.agents import make_agent
    from bgsim.engine import play_game
    try:
        game = load_engine(path)
        for n in players:
            for seed in range(seeds):
                play_game(game, [make_agent("random", seed * 10 + i) for i in range(n)], seed, debug=True)
    except Exception:
        return traceback.format_exc()
    return None


def play(path: str, specs: list[str], n_games: int, rotate: bool, seed0: int = 0, debug: bool = False):
    from bgsim.agents import make_agent
    from bgsim.engine import play_game
    game = load_engine(path)
    out = []
    for seed in range(seed0, seed0 + n_games):
        order = list(range(len(specs)))
        if rotate:
            k = seed % len(specs)
            order = order[k:] + order[:k]
        agents = [make_agent(specs[j], seed * 100 + i) for i, j in enumerate(order)]
        rec = play_game(game, agents, seed, debug=debug)
        rec.extra["seat_agent"] = [specs[j] for j in order]
        out.append(rec)
    return out


def audit(path: str, checker_dir: str, n_games: int, specs_by_seed):
    from bgsim.trace import record_trace
    from bgsim.verify import load_checkers
    game = load_engine(path)
    checkers = load_checkers(checker_dir)
    names = [getattr(c, "NAME", c.__checker_file__) for c in checkers]
    results = {n: {"rule": getattr(c, "RULE", ""), "gaps": [], "findings": []}
               for n, c in zip(names, checkers)}
    games_hit = {n: set() for n in names}
    for seed in range(n_games):
        specs = specs_by_seed[seed % len(specs_by_seed)]
        tr = record_trace(game, specs, len(specs), 500 + seed)
        for name, c in zip(names, checkers):
            try:
                msgs = c.check(tr) or []
            except Exception as e:
                msgs = [f"CHECKER CRASHED: {type(e).__name__}: {e}"]
            for m in msgs:
                bucket = "gaps" if m.startswith("SCHEMA GAP") else "findings"
                if bucket == "findings":
                    games_hit[name].add(seed)
                if m not in results[name][bucket]:
                    results[name][bucket].append(m if bucket == "gaps" else f"game {seed}: {m}")
    return results, {n: len(s) for n, s in games_hit.items()}


def schema(path: str):
    from bgsim.trace import record_trace, print_schema
    return print_schema(record_trace(load_engine(path), ["random", "random"], 2, 0))


def quality(path: str, n_games: int = 300):
    from web.quality import quality_report
    return quality_report(load_engine(path), n_games=n_games)


def second(path_a: str, path_b: str, n_games: int = 150):
    from web.quality import second_opinion
    return second_opinion(load_engine(path_a), load_engine(path_b), n_games=n_games)


TASKS = {"validate": validate, "play": play, "audit": audit, "schema": schema,
         "quality": quality, "second": second}


# ------------------------------ entry point -----------------------------------

def run(name: str, timeout: float = 300, **kwargs):
    """Run a task in a fresh interpreter (python -m web.sandbox). Raises
    EngineHung on timeout, RuntimeError with the child's traceback on failure."""
    import os, pickle, subprocess, tempfile
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        inp, outp = Path(td) / "in.pkl", Path(td) / "out.pkl"
        inp.write_bytes(pickle.dumps((name, kwargs)))
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""))
        try:
            proc = subprocess.run([sys.executable, "-m", "web.sandbox", str(inp), str(outp)],
                                  cwd=str(root), env=env, timeout=timeout,
                                  capture_output=True, text=True)
        except subprocess.TimeoutExpired:
            raise EngineHung(
                f"the engine did not finish '{name}' within {int(timeout)} seconds. This almost always "
                "means a rule loops forever (a phase that never ends, a forced action that never "
                "resolves, or no end condition is reachable). Add termination guarantees.")
        if not outp.exists():
            raise RuntimeError(f"engine process failed:\n{proc.stderr[-3000:]}")
        status, payload = pickle.loads(outp.read_bytes())
    if status == "err":
        raise RuntimeError(payload)
    return payload


if __name__ == "__main__":
    import pickle
    _name, _kwargs = pickle.loads(Path(sys.argv[1]).read_bytes())
    try:
        _res = ("ok", TASKS[_name](**_kwargs))
    except BaseException:
        _res = ("err", traceback.format_exc())
    Path(sys.argv[2]).write_bytes(pickle.dumps(_res))
