"""Automatic balance analysis over the strategy space — no designer input,
no language model. Needs only the engine's features()/FEATURE_NAMES.

1. evolve:       find strong strategies from random starts
2. double oracle: repeatedly add the best response to the current
                  equilibrium mixture, until nothing beats it
3. equilibrium:   fictitious play over the payoff matrix -> which strategies
                  survive, with what share
4. archetypes:    name each strategy from its (scale-normalised) weights
5. sensitivity:   regress win rate on weights over random strategies -> which
                  levers in the game matter, and in which direction
"""
from __future__ import annotations

import random
import statistics
import time

SEARCH_STEPS = 600      # a search game that hasn't ended by then is scored by rank
_deadline = [None]      # wall-clock budget for the whole scan


def _timeup():
    return _deadline[0] is not None and time.process_time() > _deadline[0]


# --------------------------------------------------------------- helpers

def feature_scales(game, n_players: int, rng, games: int = 6):
    """Std of each feature over states seen in random play, so weights can be
    compared across features of different magnitude."""
    from bgsim.agents import make_agent
    cols = None
    for seed in range(games):
        agents = [make_agent("random", seed * 10 + i) for i in range(n_players)]
        st = game.initial_state(n_players, seed)
        k = 0
        while not game.is_terminal(st) and k < 400:
            p = game.current_player(st)
            f = game.features(st, p)
            if cols is None:
                cols = [[] for _ in f]
            for i, x in enumerate(f):
                cols[i].append(float(x))
            st = game.apply(st, agents[p].act(game, st, p)); k += 1
    return [max(1e-6, statistics.pstdev(c)) if len(c) > 1 else 1.0 for c in (cols or [])]


def play_vs(game, cand, opp_specs, n_players, games, seed0, rng):
    """Win share of `cand` (weights) vs opponents drawn from opp_specs."""
    from bgsim.agents import GreedyAgent, make_agent
    from bgsim.engine import play_game
    w = 0.0
    for g in range(games):
        me = g % n_players
        agents = []
        for s in range(n_players):
            if s == me:
                agents.append(GreedyAgent(cand, seed0 + g * 10 + s))
            else:
                spec = rng.choice(opp_specs)
                agents.append(GreedyAgent(spec["weights"], seed0 + g * 10 + s) if isinstance(spec, dict)
                              else make_agent(spec, seed0 + g * 10 + s))
        rec = play_game(game, agents, seed0 + g, max_actions=SEARCH_STEPS)
        if me in rec.winners:
            w += 1.0 / len(rec.winners)
        if _timeup():
            return w / (g + 1)
    return w / games


def evolve_vs(game, opp_specs, n_players, scales, rng, pop=10, gens=4, games=10, seed_from=None, tick=None):
    """Evolve weights maximising win rate vs the opponent pool."""
    dim = len(scales)
    def rand():
        w = [rng.gauss(0, 1) / scales[i] for i in range(dim)]
        w[0] = abs(w[0]) * 3
        return w
    P = [rand() for _ in range(pop)]
    if seed_from:
        P[0] = list(seed_from)
    base = rng.randrange(10**6)
    best = None
    for g in range(gens):
        fit = []
        for c in P:
            fit.append(play_vs(game, c, opp_specs, n_players, games, base, rng)); base += games
            if tick:
                tick()
        if _timeup() and best is not None:
            break
        order = sorted(range(pop), key=lambda i: -fit[i])
        best = (P[order[0]], fit[order[0]])
        surv = [P[i] for i in order[: pop // 2]]
        kids = []
        while len(surv) + len(kids) < pop:
            a, b = rng.sample(surv, 2)
            kids.append([(x if rng.random() < 0.5 else y) + rng.gauss(0, 0.3) / scales[i] for i, (x, y) in enumerate(zip(a, b))])
        P = surv + kids
    return best


def payoff_matrix(game, specs, n_players, games, rng, tick=None):
    """M[i][j] = win share of i in i-vs-j pairings (ties split)."""
    from bgsim.engine import play_game
    from bgsim.agents import GreedyAgent, make_agent
    k = len(specs)
    def ag(spec, seed):
        return GreedyAgent(spec["weights"], seed) if isinstance(spec, dict) else make_agent(spec, seed)
    M = [[0.5] * k for _ in range(k)]
    for i in range(k):
        for j in range(i + 1, k):
            wi = 0.0
            for g in range(games):
                seats = [(i if (s + g) % 2 == 0 else j) for s in range(n_players)]
                agents = [ag(specs[si], 5000 + g * 10 + s) for s, si in enumerate(seats)]
                rec = play_game(game, agents, 40000 + i * 1000 + j * 50 + g, max_actions=SEARCH_STEPS)
                for w in rec.winners:
                    wi += (1.0 / len(rec.winners)) if seats[w] == i else 0.0
                if _timeup():
                    games = g + 1
                    break
            M[i][j] = wi / games; M[j][i] = 1 - wi / games
            if tick:
                tick()
    return M


def equilibrium(M, iters=3000):
    """Fictitious play on the symmetric win-rate game: mixture over strategies."""
    k = len(M)
    counts = [1.0] * k
    for _ in range(iters):
        tot = sum(counts)
        mix = [c / tot for c in counts]
        payoff = [sum(M[i][j] * mix[j] for j in range(k)) for i in range(k)]
        counts[max(range(k), key=lambda i: payoff[i])] += 1
    tot = sum(counts)
    return [c / tot for c in counts]


def archetype(weights, names, scales):
    """Human words for a weight vector: what it chases, what it avoids."""
    eff = [w * s for w, s in zip(weights, scales)]
    order = sorted(range(len(eff)), key=lambda i: -abs(eff[i]))
    mag = max(1e-9, abs(eff[order[0]]))
    chase = [names[i] for i in order if eff[i] > 0.25 * mag][:3]
    avoid = [names[i] for i in order if eff[i] < -0.25 * mag][:2]
    s = "prioritises " + ", ".join(chase) if chase else "no strong priorities"
    if avoid:
        s += "; avoids " + ", ".join(avoid)
    return s


def sensitivity(game, opp_specs, n_players, scales, names, rng, samples=120, games=6, tick=None):
    """Least-squares regression of win rate on scale-normalised weights."""
    dim = len(scales)
    X, y = [], []
    base = 70000
    for _ in range(samples):
        z = [rng.gauss(0, 1) for _ in range(dim)]
        w = [z[i] / scales[i] for i in range(dim)]
        X.append(z); y.append(play_vs(game, w, opp_specs, n_players, games, base, rng)); base += games
        if tick:
            tick()
    # normal equations with ridge, pure python
    n = dim + 1
    A = [[0.0] * n for _ in range(n)]; b = [0.0] * n
    for xi, yi in zip(X, y):
        row = [1.0] + xi
        for r in range(n):
            b[r] += row[r] * yi
            for c in range(n):
                A[r][c] += row[r] * row[c]
    for r in range(1, n):
        A[r][r] += 1.0
    # gaussian elimination
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(A[r][col]))
        A[col], A[piv] = A[piv], A[col]; b[col], b[piv] = b[piv], b[col]
        if abs(A[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                f = A[r][col] / A[col][col]
                for c in range(col, n):
                    A[r][c] -= f * A[col][c]
                b[r] -= f * b[col]
    coef = [b[i] / A[i][i] if abs(A[i][i]) > 1e-12 else 0.0 for i in range(n)]
    return sorted([(names[i], coef[i + 1]) for i in range(dim)], key=lambda t: -abs(t[1]))


# ------------------------------------------------------------------ scan

def balance_scan(game, names, n_players=2, rounds=4, seed=0, tick=None, budget_s=2400):
    rng = random.Random(seed)
    _deadline[0] = time.process_time() + budget_s   # a safety cap only; the plan is `rounds` rounds
    t_start = time.time()
    planned_rounds = rounds
    scales = feature_scales(game, n_players, rng)
    names = (list(names) + [f"feature_{i}" for i in range(len(scales))])[:len(scales)]
    specs = ["scoregreedy"]; labels = ["score-greedy (baseline)"]; trail = []
    # 1) strong strategy from nothing
    best, fit = evolve_vs(game, specs, n_players, scales, rng, pop=12, gens=6, games=12, tick=tick)
    specs.append({"weights": best}); labels.append("evolved #1")
    trail.append(f"evolved #1 beats the baseline {fit:.0%} of the time")
    rounds_done = 0
    cut_short = False
    # 2) double oracle
    for r in range(rounds):
        if _timeup():
            cut_short = True
            trail.append("stopped early by the safety cap")
            break
        rounds_done = r + 1
        M = payoff_matrix(game, specs, n_players, 24, rng, tick=tick)
        mix = equilibrium(M)
        opp = [s for s, m in zip(specs, mix) if m > 0.02] or specs
        incumbent = max(range(len(specs)), key=lambda i: mix[i])
        seed_w = specs[incumbent]["weights"] if isinstance(specs[incumbent], dict) else None
        br, brfit = evolve_vs(game, opp, n_players, scales, rng, pop=12, gens=6, games=12,
                              seed_from=seed_w, tick=tick)
        trail.append(f"round {r + 1}: best response to the current mix wins {brfit:.0%}")
        if brfit <= 0.55:
            trail.append("nothing beats the current mix by much — the meta-game has settled")
            break
    if _timeup() and rounds_done < planned_rounds:
        cut_short = True
        specs.append({"weights": br}); labels.append(f"evolved #{len(specs) - 1}")
    M = payoff_matrix(game, specs, n_players, 40, rng, tick=tick)
    mix = equilibrium(M)
    # 3) archetypes
    arche = [archetype(s["weights"], names, scales) if isinstance(s, dict) else "takes whatever raises its score most right now" for s in specs]
    # 4) sensitivity
    sens = sensitivity(game, [s for s, m in zip(specs, mix) if m > 0.02] or specs, n_players, scales, names, rng, tick=tick)
    # 5) findings
    findings = []
    top = max(range(len(mix)), key=lambda i: mix[i])
    if mix[top] >= 0.85:
        if rounds_done >= 2:
            findings.append(f"**Dominant strategy**: at equilibrium, {mix[top]:.0%} of play is '{labels[top]}' ({arche[top]}). "
                            f"{rounds_done} rounds of best-response search found nothing that beats it — the game rewards "
                            "one way of playing.")
        else:
            findings.append(f"**One strategy on top so far**: '{labels[top]}' ({arche[top]}) beat everything in "
                            f"{rounds_done} round(s) of search — too little to call it dominant. Rerun or search deeper.")
    elif sum(1 for m in mix if m > 0.1) >= 3:
        findings.append("**Healthy rock-paper-scissors**: three or more strategies share the equilibrium, "
                        "each beaten by another — no single best way to play.")
    else:
        findings.append("**Two-way meta-game**: play settles between two strategies; check the matrix for how close they are.")
    for i, m in enumerate(mix):
        if m < 0.02 and isinstance(specs[i], dict):
            findings.append(f"'{labels[i]}' ({arche[i]}) drops out of the equilibrium — a way of playing that never pays.")
    strong = [(n, c) for n, c in sens if c > 0.008][:3]; weak = [(n, c) for n, c in sens if c < -0.008][:3]
    if strong:
        findings.append("**Levers that win**: valuing " + ", ".join(f"{n} (+{c * 100:.1f} pts of win rate per step)" for n, c in strong))
    if weak:
        findings.append("**Levers that lose**: valuing " + ", ".join(f"{n} ({c * 100:.1f} pts per step)" for n, c in weak))
    if not strong and not weak:
        findings.append("No single feature moves the win rate much on its own — outcomes depend on combinations, or the features miss what matters.")
    findings.append("Caveat: this searches one-move-lookahead strategies over the engine's features. A 'dominant' result "
                    "means nothing in that space beats it; deeper play (see the competent-play check) can differ.")
    completeness = 1.0 if not cut_short else min(1.0, (rounds_done + 1) / (planned_rounds + 1))
    if cut_short:
        findings.insert(0, f"**Search {completeness:.0%} complete** — it was stopped by the safety cap after "
                           f"{rounds_done} of {planned_rounds} rounds. Read the findings below as provisional; rerun to finish.")
    return {"players": n_players, "labels": labels, "archetypes": arche, "matrix": M, "equilibrium": mix,
            "seconds": round(time.time() - t_start), "completeness": completeness,
            "rounds_done": rounds_done, "rounds_planned": planned_rounds,
            "sensitivity": sens, "trail": trail, "findings": findings, "feature_names": names,
            "strategies": [s if isinstance(s, dict) else {"spec": s} for s in specs]}
