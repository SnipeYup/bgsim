"""Mutation test: prove the checker set catches broken engines.

Builds five deliberately broken copies of the Agricola engine, runs every
checker in checkers/agricola over each, and reports catches. A checker set
is only trustworthy when it is silent on the real engine AND loud on these.

    python tools/mutate_agricola.py
"""
import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MUTANTS = {
    "m1_cheap_adults": ("need = 2 * (pl.family - pl.born) + pl.born",
                        "need = 1 * (pl.family - pl.born) + pl.born", 1),
    "m2_no_begging": ("begging=pl.begging + beg", "begging=pl.begging + 0", None),
    "m3_lonely_breeding": ("if animals[t] >= 2:", "if animals[t] >= 1:", None),
    "m4_hungry_newborns": ("need = 2 * (pl.family - pl.born) + pl.born",
                           "need = 2 * (pl.family - pl.born) + 2 * pl.born", 1),
    "m5_cheap_begging": ('"begging": -3 * pl.begging', '"begging": -1 * pl.begging', None),
}


def main(games: int = 6) -> int:
    src = (ROOT / "bgsim/games/agricola/game.py").read_text()
    mdir = ROOT / "mutants"
    for name, (old, new, count) in MUTANTS.items():
        assert old in src, f"pattern for {name} not found; update MUTANTS"
        text = (src[:src.index(old)] + new + src[src.index(old) + len(old):]
                ) if count == 1 else src.replace(old, new)
        d = mdir / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "__init__.py").write_text("")
        (d / "game.py").write_text(text)
    (mdir / "__init__.py").write_text("")

    from bgsim.trace import record_trace
    from bgsim.verify import load_checkers
    checkers = load_checkers(ROOT / "checkers/agricola")
    missed = []
    for name in MUTANTS:
        for mod in [m for m in sys.modules if m.startswith("mutants.")]:
            del sys.modules[mod]
        G = importlib.import_module(f"mutants.{name}.game").Agricola
        g = G()
        real = []
        for seed in range(games):
            try:
                tr = record_trace(g, ["scoregreedy", "scoregreedy"], 2, seed)
                for c in checkers:
                    real += [f"[{c.NAME}] {m}" for m in (c.check(tr) or [])
                             if not m.startswith("SCHEMA GAP")]
            except AssertionError as e:
                real.append(f"[invariant] {e}")
        tag = f"{len(real)} findings" if real else "NOT CAUGHT"
        print(f"{name}: {tag}" + (f" | {real[0][:110]}" if real else ""))
        if not real:
            missed.append(name)
    if missed:
        print(f"\nuncaught: {missed} — expected for bugs outside the current "
              f"checker set's sections (m5 needs the scoring checker)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
