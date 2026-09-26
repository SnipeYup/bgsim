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
        agents = [_agent_for(specs[j], seed * 100 + i) for i, j in enumerate(order)]
        rec = play_game(game, agents, seed, debug=debug)
        rec.extra["seat_agent"] = [(specs[j] if isinstance(specs[j], str) else specs[j].get("name", "strategy")) for j in order]
        pops = [sp.get("population", {}).get("seed") for sp in specs if isinstance(sp, dict) and "population" in sp]
        if pops:
            rec.extra["population"] = pops[0]
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


def narrate(path: str, specs: list[str], seed: int, rotate: bool, max_steps: int = 400):
    """One full game in the engine's own words, for the report's sample games."""
    from bgsim.agents import make_agent
    game = load_engine(path)
    order = list(range(len(specs)))
    if rotate:
        k = seed % len(specs); order = order[k:] + order[:k]
    agents = [_agent_for(specs[j], seed * 100 + i) for i, j in enumerate(order)]
    st = game.initial_state(len(specs), seed); i = 0; lines = []
    ds = getattr(game, "describe_state", None); da = getattr(game, "describe_action", None)
    while not game.is_terminal(st) and i < max_steps:
        p = game.current_player(st); a = agents[p].act(game, st, p)
        lines.append(f"{i + 1}. Player {p + 1}: {_desc(da, st, a)}")
        st = game.apply(st, a); i += 1
    scores = game.scores(st)
    return {"seed": seed, "ended": game.is_terminal(st), "steps": i, "lines": lines,
            "final": ds(st)[:500] if ds else "", "scores": [list(s) if isinstance(s, (tuple, list)) else s for s in scores]}


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


def feature_names(path: str):
    game = load_engine(path)
    names = list(getattr(game, "FEATURE_NAMES", ()) or ())
    st = game.initial_state(2, 0)
    n = len(game.features(st, 0))
    if len(names) != n:
        names = (names + [f"feature_{i}" for i in range(n)])[:n]
    return names


def _agent_for(spec, seed):
    """spec: a make_agent string, {"weights": [...]} or {"population": {...}}."""
    from bgsim.agents import make_agent, GreedyAgent
    if isinstance(spec, dict) and "population" in spec:
        from bgsim.analysis.players import PopulationAgent
        return PopulationAgent(spec["population"], seed, name=spec.get("name", "trained"))
    if isinstance(spec, dict):
        return GreedyAgent(spec["weights"], seed, name=spec.get("name", "strategy"))
    return make_agent(spec, seed)


def matrix(path: str, strategies: list, n_players: int, games_per_pair: int):
    """Round-robin: each unordered pair plays games_per_pair games with the two
    strategies alternating seats (for 3-4 players, A B A B… rotated). Returns
    out[i][j] = share of games in the (i, j) pairing won by i (ties split)."""
    from bgsim.engine import play_game
    game = load_engine(path)
    k = len(strategies)
    out = [[None] * k for _ in range(k)]
    pairs = [(i, jdx) for i in range(k) for jdx in range(i + 1, k)]
    total = len(pairs) * games_per_pair; done = 0
    for i, jdx in pairs:
        wi = wj = 0.0
        for g in range(games_per_pair):
            _tick(done, total); done += 1
            seats = [(i if (s + g) % 2 == 0 else jdx) for s in range(n_players)]
            agents = [_agent_for(strategies[si], g * 100 + s) for s, si in enumerate(seats)]
            rec = play_game(game, agents, 7000 + i * 1000 + jdx * 100 + g)
            for w in rec.winners:
                if seats[w] == i:
                    wi += 1 / len(rec.winners)
                else:
                    wj += 1 / len(rec.winners)
        out[i][jdx] = wi / games_per_pair
        out[jdx][i] = wj / games_per_pair
    return out


def counter(path: str, target: dict, n_players: int = 2, pop_size: int = 12, generations: int = 6,
            games_per_eval: int = 20, seed: int = 0):
    """Evolve a strategy whose only fitness is beating `target`."""
    import random
    from bgsim.engine import play_game
    from bgsim.agents import GreedyAgent
    game = load_engine(path)
    tw = target["weights"]; dim = len(tw)
    rng = random.Random(seed)
    pop = [[w + rng.gauss(0, 0.35) for w in tw] for _ in range(pop_size // 2)]
    pop += [[rng.uniform(-1, 1) * (10 if i == 0 else 1) for i in range(dim)] for _ in range(pop_size - len(pop))]

    def h2h(cand, games, base):
        w = 0.0
        for g in range(games):
            seats = [(0 if (s + g) % 2 == 0 else 1) for s in range(n_players)]
            agents = [GreedyAgent(cand if si == 0 else tw, g * 10 + s) for s, si in enumerate(seats)]
            rec = play_game(game, agents, base + g)
            for win in rec.winners:
                if seats[win] == 0:
                    w += 1 / len(rec.winners)
        return w / games
    total = generations * pop_size; done = 0; hist = []; gs = 90_000
    for gen in range(generations):
        fit = []
        for cand in pop:
            _tick(done, total); done += 1
            fit.append(h2h(cand, games_per_eval, gs)); gs += games_per_eval
        order = sorted(range(pop_size), key=lambda i: -fit[i])
        hist.append({"gen": gen, "best": fit[order[0]]})
        surv = [pop[i] for i in order[: pop_size // 2]]
        kids = []
        while len(surv) + len(kids) < pop_size:
            a, b = rng.sample(surv, 2)
            kids.append([(x if rng.random() < 0.5 else y) + rng.gauss(0, 0.35) for x, y in zip(a, b)])
        pop = surv + kids
    best = pop[0]
    confirm = h2h(best, 120, 990_000)
    return {"weights": [round(x, 3) for x in best], "winrate_vs_target": round(confirm, 3), "history": hist}


def balance(path: str, n_players: int = 2, rounds: int = 4, seed: int = 0, budget_s: int = 900):
    from bgsim.analysis.balance import balance_scan
    game = load_engine(path)
    names = list(getattr(game, "FEATURE_NAMES", ()) or ())
    done = [0]
    def tick():
        done[0] += 1; _tick(done[0], 400)
    return balance_scan(game, names, n_players=n_players, rounds=rounds, seed=seed, tick=tick, budget_s=budget_s)


def competent(path: str, n_players: int, n_games: int, iterations: int = 60, seed0: int = 0):
    """Games between MCTS players, for seat/lock-in figures under competent play."""
    from bgsim.agents.mcts import MCTSAgent
    from bgsim.engine import play_game
    game = load_engine(path)
    out = []
    for k, seed in enumerate(range(seed0, seed0 + n_games)):
        _tick(k, n_games)
        agents = [MCTSAgent(iterations=iterations, seed=seed * 10 + i) for i in range(n_players)]
        rec = play_game(game, agents, seed)
        rec.extra["seat_agent"] = [f"mcts{iterations}"] * n_players
        out.append(rec)
    return out


def train(path: str, n_players: int = 2, populations: int = 3, rounds: int = 2, budget_s: int = 600, size: str = "normal"):
    from bgsim.analysis.players import train_populations, strength_ladder
    game = load_engine(path)
    names = list(getattr(game, "FEATURE_NAMES", ()) or ())
    done = [0]
    def tick():
        done[0] += 1; _tick(done[0], 300)
    pops = train_populations(game, names, n_players, populations=populations, rounds=rounds, tick=tick, budget_s=budget_s, size=size)
    ladder = strength_ladder(game, pops, n_players)
    return {"populations": pops, "ladder": ladder}


def schema(path: str):
    from bgsim.trace import record_trace, print_schema
    return print_schema(record_trace(load_engine(path), ["random", "random"], 2, 0))


def quality(path: str, n_games: int = 300):
    from web.quality import quality_report
    return quality_report(load_engine(path), n_games=n_games)


def second(path_a: str, path_b: str, n_games: int = 150):
    from web.quality import second_opinion
    return second_opinion(load_engine(path_a), load_engine(path_b), n_games=n_games)


TASKS = {"validate": validate, "play": play, "audit": audit, "schema": schema, "signature": signature, "stuck_tail": stuck_tail, "find_failure": find_failure, "feature_names": feature_names, "matrix": matrix, "counter": counter, "balance": balance, "competent": competent, "train": train, "narrate": narrate,
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
            # the ceiling only fires when the child is also NOT progressing: a
            # laptop that slept for hours and woke up mid-run must not be killed
            over_ceiling = now - t0 > hard_max and now - last_change > 30
            if now - last_change > limit or over_ceiling:
                proc.kill(); proc.wait(5)
                what = (f"no game finished for {int(limit)} seconds (progress stuck at {last_seen or 'nothing'})"
                        if not over_ceiling else f"exceeded the {int(hard_max / 3600)}-hour ceiling without finishing")
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
