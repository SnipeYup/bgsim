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
        k = 0
        for n in players:
            for seed in range(seeds):
                _tick(k, len(players) * seeds); k += 1
                play_game(game, [make_agent("random", seed * 10 + i) for i in range(n)], seed, debug=True)
    except Exception:
        return traceback.format_exc()
    return None


_PROGRESS = None   # set by the child entry point


def _tick(done: int, total: int) -> None:
    if _PROGRESS:
        try:
            Path(_PROGRESS).write_text(f"{done}/{total}")
        except Exception:
            pass


def play(path: str, specs: list[str], n_games: int, rotate: bool, seed0: int = 0, debug: bool = False):
    from bgsim.agents import make_agent
    from bgsim.engine import play_game
    game = load_engine(path)
    out = []
    for k, seed in enumerate(range(seed0, seed0 + n_games)):
        _tick(k, n_games)
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
        _tick(seed, n_games)
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


def signature(path: str, n_games: int = 12, players=(2, 3)):
    """A fingerprint of observable behaviour: per seeded game, the number of
    turns and the final scores. Two engines with the same fingerprint play the
    same games — a 'fix' that leaves it unchanged changed nothing that matters."""
    from bgsim.agents import make_agent
    from bgsim.engine import play_game
    game = load_engine(path)
    out = []
    for n in players:
        for seed in range(n_games):
            rec = play_game(game, [make_agent("random", seed * 10 + i) for i in range(n)], seed)
            out.append((n, seed, rec.n_turns, [tuple(sc) if isinstance(sc, (tuple, list)) else sc for sc in rec.scores]))
    return out


def _desc(da, st, a):
    """describe_action(state, action) or describe_action(action) — engines vary."""
    if not da:
        return str(a)
    try:
        return str(da(st, a))
    except TypeError:
        return str(da(a))


def stuck_tail(path: str, specs: list[str], seed: int, rotate: bool, n_tail: int = 10, max_steps: int = 4000):
    """Replay one game and return the last steps in the engine's own words,
    plus the current player's legal actions — to show WHY a game never ends."""
    import collections
    from bgsim.agents import make_agent
    game = load_engine(path)
    order = list(range(len(specs)))
    if rotate:
        k = seed % len(specs); order = order[k:] + order[:k]
    agents = [make_agent(specs[j], seed * 100 + i) for i, j in enumerate(order)]
    st = game.initial_state(len(specs), seed); i = 0
    tail = collections.deque(maxlen=n_tail)
    ds = getattr(game, "describe_state", None); da = getattr(game, "describe_action", None)
    while not game.is_terminal(st) and i < max_steps:
        p = game.current_player(st); a = agents[p].act(game, st, p)
        tail.append(f"step {i} · {game.phase(st)} · player {p + 1}: " + _desc(da, st, a))
        st = game.apply(st, a); i += 1
    legal = game.legal_actions(st)
    return {"ended": game.is_terminal(st), "steps": i, "tail": list(tail),
            "state": (ds(st) if ds else "")[:600],
            "legal": [_desc(da, st, a) for a in legal[:8]], "n_legal": len(legal)}


def find_failure(path: str, players=(2, 3, 4), seeds: int = 400):
    """Play random games with invariants on until one fails; return its
    traceback (with the seed and player count) or None."""
    from bgsim.agents import make_agent
    from bgsim.engine import play_game
    game = load_engine(path)
    k = 0
    for seed in range(seeds):
        for n in players:
            _tick(k, seeds * len(players)); k += 1
            try:
                play_game(game, [make_agent("random", seed * 10 + i) for i in range(n)], seed, debug=True)
            except Exception:
                return f"(random game, {n} players, seed {seed})\n" + traceback.format_exc()
    return None


def schema(path: str):
    from bgsim.trace import record_trace, print_schema
    return print_schema(record_trace(load_engine(path), ["random", "random"], 2, 0))


def quality(path: str, n_games: int = 300):
    from web.quality import quality_report
    return quality_report(load_engine(path), n_games=n_games)


def second(path_a: str, path_b: str, n_games: int = 150):
    from web.quality import second_opinion
    return second_opinion(load_engine(path_a), load_engine(path_b), n_games=n_games)


TASKS = {"validate": validate, "play": play, "audit": audit, "schema": schema, "signature": signature, "stuck_tail": stuck_tail, "find_failure": find_failure,
         "quality": quality, "second": second}


# ------------------------------ entry point -----------------------------------

def run(name: str, timeout: float = 300, stall: float = 240, on_progress=None, hard_max: float = 4 * 3600, **kwargs):
    """Run a task in a fresh interpreter (python -m web.sandbox).

    The watchdog is progress-based: the child reports each game it finishes,
    and it is killed only if NOTHING has finished for `stall` seconds (a
    genuine infinite loop), however long the whole job takes. `timeout` only
    applies until the first game finishes (a task that never starts), and
    `hard_max` is a safety ceiling. Raises EngineHung on a stall, RuntimeError
    with the child's traceback on failure."""
    import os, pickle, subprocess, tempfile, time as _t
    root = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as td:
        inp, outp, prog = Path(td) / "in.pkl", Path(td) / "out.pkl", Path(td) / "progress"
        inp.write_bytes(pickle.dumps((name, kwargs)))
        env = dict(os.environ, PYTHONPATH=str(root) + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   BGSIM_PROGRESS=str(prog))
        proc = subprocess.Popen([sys.executable, "-m", "web.sandbox", str(inp), str(outp)],
                                cwd=str(root), env=env, stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE, text=True)
        t0 = _t.time(); last_change, last_seen = t0, ""
        while proc.poll() is None:
            _t.sleep(1)
            now = _t.time()
            try:
                cur = prog.read_text() if prog.exists() else ""
            except Exception:
                cur = last_seen
            if cur != last_seen:
                last_seen, last_change = cur, now
                if on_progress:
                    try:
                        on_progress(cur)
                    except Exception:
                        pass
            started = bool(last_seen)
            limit = stall if started else timeout
            if now - last_change > limit or now - t0 > hard_max:
                proc.kill(); proc.wait(5)
                what = (f"no game finished for {int(limit)} seconds (progress stuck at {last_seen or 'nothing'})"
                        if now - t0 <= hard_max else f"exceeded the {int(hard_max / 3600)}-hour ceiling")
                raise EngineHung(
                    f"the engine stalled during '{name}': {what}. This almost always means a rule loops "
                    "forever (a phase that never ends, a forced action that never resolves, or no end "
                    "condition is reachable). Add termination guarantees.")
        stderr = proc.stderr.read() if proc.stderr else ""
        if not outp.exists():
            raise RuntimeError(f"engine process failed:\n{stderr[-3000:]}")
        status, payload = pickle.loads(outp.read_bytes())
    if status == "err":
        raise RuntimeError(payload)
    return payload


def run_parallel(name: str, n_games: int, workers: int | None = None, on_progress=None, **kwargs):
    """Split a per-game task across processes; results are concatenated in seed order."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    workers = max(1, workers or min(8, (os.cpu_count() or 2) - 1))
    if n_games < 2 * workers:
        return run(name, n_games=n_games, on_progress=on_progress, **kwargs)
    chunks = []
    base = kwargs.pop("seed0", 0)
    per = n_games // workers
    for w in range(workers):
        n = per + (1 if w < n_games % workers else 0)
        chunks.append((base + sum(c[1] for c in chunks), n))
    done = {}

    def prog(idx):
        def cb(txt):
            try:
                d, t = txt.split("/"); done[idx] = int(d)
            except Exception:
                return
            if on_progress:
                on_progress(f"{sum(done.values())}/{n_games}")
        return cb
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(run, name, seed0=s0, n_games=n, on_progress=prog(i), **kwargs)
                for i, (s0, n) in enumerate(chunks)]
        out = []
        for f in futs:
            out += f.result()
    return out


if __name__ == "__main__":
    import pickle, os
    _PROGRESS = os.environ.get("BGSIM_PROGRESS")
    _name, _kwargs = pickle.loads(Path(sys.argv[1]).read_bytes())
    try:
        _res = ("ok", TASKS[_name](**_kwargs))
    except BaseException:
        _res = ("err", traceback.format_exc())
    Path(sys.argv[2]).write_bytes(pickle.dumps(_res))
