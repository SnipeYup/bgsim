"""Run independent rule-checkers over game traces.

A checker is a Python file with:

    NAME = "feeding"                     # short id
    RULE = "one-paragraph restatement"   # the rule it re-derives, for humans

    def check(trace: dict) -> list[str]:
        # audit the whole trace; return one string per violation, "" clean
        ...

Checkers are generated in fresh sessions from single rulebook sections and
must never import or read the engine. Their power is independence: an engine
bug and a checker bug are unlikely to agree, so any disagreement is either a
code bug or an ambiguity in the rulebook itself — both worth surfacing.

    python -m bgsim.verify bgsim.games.agricola.game:Agricola \
        --checkers checkers/agricola --games 30 --players 2

Exit code is nonzero if any checker reports anything.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

from .games import make_game
from .trace import record_trace


def load_checkers(directory: str | Path) -> list:
    mods = []
    for path in sorted(Path(directory).glob("*.py")):
        if path.name.startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"checker_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "check"):
            print(f"warning: {path.name} has no check(); skipped", file=sys.stderr)
            continue
        mod.__checker_file__ = path.name
        mods.append(mod)
    return mods


def run(game_spec: str, checkers_dir: str, n_games: int, n_players: int,
        agents: str, seed: int, fail_fast: bool = False, max_report: int = 8):
    game = make_game(game_spec)
    checkers = load_checkers(checkers_dir)
    if not checkers:
        print(f"no checkers found in {checkers_dir}", file=sys.stderr)
        return 2
    specs = agents.split(",")
    if len(specs) == 1:
        specs = specs * n_players
    findings: dict[str, list[tuple[int, str]]] = {getattr(c, "NAME", c.__checker_file__): [] for c in checkers}
    t0 = time.perf_counter()
    for g in range(n_games):
        tr = record_trace(game, specs, n_players, seed + g)
        for c in checkers:
            name = getattr(c, "NAME", c.__checker_file__)
            try:
                out = c.check(tr) or []
            except Exception as e:  # checker crash is itself a finding
                out = [f"CHECKER CRASHED: {type(e).__name__}: {e}"]
            for msg in out:
                findings[name].append((seed + g, msg))
            if out and fail_fast:
                break
        if fail_fast and any(findings.values()):
            break
    elapsed = time.perf_counter() - t0
    total = sum(len(v) for v in findings.values())
    print(f"{n_games} games x {len(checkers)} checkers in {elapsed:.1f}s — "
          f"{total} finding(s)")
    for name, items in findings.items():
        status = "clean" if not items else f"{len(items)} finding(s)"
        print(f"\n[{name}] {status}")
        for game_seed, msg in items[:max_report]:
            print(f"  seed {game_seed}: {msg}")
        if len(items) > max_report:
            print(f"  ... {len(items) - max_report} more")
    return 1 if total else 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--checkers", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--agents", default="scoregreedy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fail-fast", action="store_true")
    a = ap.parse_args(argv)
    sys.exit(run(a.game, a.checkers, a.games, a.players, a.agents, a.seed,
                 a.fail_fast))


if __name__ == "__main__":
    main()
