"""bgsim command line.

  python -m bgsim run --game splendor --agents expert,expert,random,random --games 1000 --out out.csv
  python -m bgsim report out.csv
  python -m bgsim evolve --generations 10 --out best.json
  python -m bgsim stress --games 20000        # random play with invariants on
"""
from __future__ import annotations

import argparse
import os
import sys

from .agents import evolve as ev
from .analysis import report
from .games import make_game
from .sim import read_records, simulate, write_records


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bgsim")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="simulate games")
    r.add_argument("--game", default="splendor")
    r.add_argument("--agents", default="expert,expert,expert,expert",
                   help="comma-separated: random | greedy | expert | greedy:[...] | greedy@file.json")
    r.add_argument("--games", type=int, default=1000)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    r.add_argument("--rotate", action="store_true", help="rotate agents through seats")
    r.add_argument("--debug", action="store_true", help="check invariants every action")
    r.add_argument("--out", default="results.csv")
    r.add_argument("--no-report", action="store_true")

    p = sub.add_parser("report", help="analyze a results file")
    p.add_argument("path")
    p.add_argument("--game", default="splendor")
    p.add_argument("--top", type=int, default=10)

    e = sub.add_parser("evolve", help="search for strong strategies")
    e.add_argument("--game", default="splendor")
    e.add_argument("--players", type=int, default=4)
    e.add_argument("--pop", type=int, default=24)
    e.add_argument("--generations", type=int, default=10)
    e.add_argument("--games-per-gen", type=int, default=120)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    e.add_argument("--out", default="best.json")

    s = sub.add_parser("stress", help="random playouts with invariants checked")
    s.add_argument("--game", default="splendor")
    s.add_argument("--games", type=int, default=10000)
    s.add_argument("--players", type=int, default=4)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--workers", type=int, default=os.cpu_count() or 1)

    a = ap.parse_args(argv)

    if a.cmd == "run":
        specs = a.agents.split(",")
        recs, secs = simulate(a.game, specs, a.games, a.seed, a.workers,
                              rotate=a.rotate, debug=a.debug)
        write_records(recs, a.out)
        print(f"{len(recs)} games in {secs:.1f}s "
              f"({len(recs) / secs:.0f} games/s, {a.workers} workers) -> {a.out}",
              file=sys.stderr)
        if not a.no_report:
            print(report(recs, make_game(a.game)))
    elif a.cmd == "report":
        print(report(read_records(a.path), make_game(a.game), top=a.top))
    elif a.cmd == "evolve":
        res = ev.evolve(a.game, a.players, a.pop, a.generations, a.games_per_gen,
                        a.seed, a.workers)
        ev.save(res, a.out)
        names = res.get("feature_names") or [f"w{i}" for i in range(len(res["weights"]))]
        print("best weights:")
        for n, w in zip(names, res["weights"]):
            print(f"  {n:18s} {w:+.2f}")
        print(f"-> {a.out}  (use with --agents greedy@{a.out})")
    elif a.cmd == "stress":
        specs = ["random"] * a.players
        recs, secs = simulate(a.game, specs, a.games, a.seed, a.workers, debug=True)
        print(f"OK: {len(recs)} random games, invariants held on every action "
              f"({secs:.1f}s)")


if __name__ == "__main__":
    main()
