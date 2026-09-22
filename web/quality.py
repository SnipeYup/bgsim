"""Reference-free engine correctness checks.

Nothing here knows the game, and nothing here uses balance as evidence:
whether a seat wins more is a *finding* for the balance report, never a
test of the engine. Correctness evidence comes from universal properties
(termination, determinism, invariants) and from agreement between two
independently generated engines (second_opinion).
"""
from __future__ import annotations

import math
from collections import Counter

from bgsim.agents import make_agent
from bgsim.engine import play_game


def _chi2_p(counts, expected):
    """Chi-square goodness-of-fit p-value (survival function), no scipy."""
    k = len(counts)
    stat = sum((c - expected) ** 2 / expected for c in counts)
    df = k - 1
    # Wilson-Hilferty approximation of the chi-square survival function
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / math.sqrt(2))


def quality_report(game, n_games: int = 300, player_counts=(2, 4)) -> list[dict]:
    """Returns a list of {check, status, detail, question} dicts.
    status is 'pass', 'flag', or 'info'."""
    out = []
    for n in player_counts:
        seat_wins = [0.0] * n
        unfinished = 0
        lengths, winner_pts, all_pts = [], [], []
        try:
            for seed in range(n_games):
                agents = [make_agent("random", seed * 10 + i) for i in range(n)]
                rec = play_game(game, agents, seed, debug=True)
                for w in rec.winners:
                    seat_wins[w] += 1 / len(rec.winners)
                unfinished += 1 if rec.extra.get("unfinished") else 0
                lengths.append(rec.n_turns / n)
                sc = [s[0] for s in rec.scores]
                all_pts += sc
                winner_pts.append(max(sc))
        except AssertionError as e:
            out.append({"check": f"invariants ({n}p)", "status": "flag",
                        "detail": f"engine's own invariant failed: {e}",
                        "question": "The engine reached a state it considers impossible."})
            continue

        # --- termination
        if unfinished:
            out.append({"check": f"termination ({n}p)", "status": "flag" if unfinished > n_games * 0.02 else "info",
                        "detail": f"{unfinished} of {n_games} games hit the action cap",
                        "question": "Random play can loop forever. Is there a rule that forces the "
                                    "game toward an ending (round limit, depleting supply)? If yes, "
                                    "the engine may be missing it; if no, this is a property of "
                                    "your design worth knowing."})
        else:
            out.append({"check": f"termination ({n}p)", "status": "pass",
                        "detail": f"all {n_games} games ended", "question": ""})

        # --- endgame shape (info for the designer to eyeball against the rules)
        lengths_s = sorted(lengths)
        out.append({"check": f"game shape ({n}p)", "status": "info",
                    "detail": (f"rounds: median {lengths_s[len(lengths_s) // 2]:.0f} "
                               f"(min {lengths_s[0]:.0f}, max {lengths_s[-1]:.0f}); "
                               f"winner score: mean {sum(winner_pts) / len(winner_pts):.1f}, "
                               f"min {min(winner_pts)}, max {max(winner_pts)}; "
                               f"all players: mean {sum(all_pts) / len(all_pts):.1f}"),
                    "question": "Do these match what the rules imply — e.g. a fixed round count, "
                                "or a target score that winners should reach but not far exceed?"})

    # --- determinism
    n = player_counts[0]
    a1 = [make_agent("random", i) for i in range(n)]
    a2 = [make_agent("random", i) for i in range(n)]
    r1, r2 = play_game(game, a1, 4242), play_game(game, a2, 4242)
    same = (r1.scores == r2.scores and r1.n_actions == r2.n_actions)
    out.append({"check": "determinism", "status": "pass" if same else "flag",
                "detail": "same seed replays identically" if same else
                          "same seed produced a different game",
                "question": "" if same else "The engine uses randomness outside initial_state; "
                                           "results won't be reproducible."})
    return out


def _dist(game, n_games, n):
    """Agent-free fingerprint: per-seat win rates, round lengths, scores."""
    seat = [0.0] * n; lengths = []; winners = []; allpts = []; unfinished = 0
    for seed in range(n_games):
        agents = [make_agent("random", 7000 + seed * 10 + i) for i in range(n)]
        rec = play_game(game, agents, 7000 + seed)
        for w in rec.winners:
            seat[w] += 1 / len(rec.winners)
        lengths.append(rec.n_turns / n)
        sc = [x[0] for x in rec.scores]
        winners.append(max(sc)); allpts += sc
        unfinished += 1 if rec.extra.get("unfinished") else 0
    return {"seat": [w / n_games for w in seat], "lengths": sorted(lengths),
            "winners": sorted(winners), "allpts": sorted(allpts), "unfinished": unfinished}


def _ks(a, b):
    """Kolmogorov-Smirnov statistic between two sorted samples (0 = identical)."""
    import bisect
    pts = sorted(set(a) | set(b)); best = 0.0
    for x in pts:
        fa = bisect.bisect_right(a, x) / len(a); fb = bisect.bisect_right(b, x) / len(b)
        best = max(best, abs(fa - fb))
    return best


def second_opinion(engine_a, engine_b, n_games: int = 150, player_counts=(2, 4)) -> list[dict]:
    """Compare two independently generated engines on agent-free statistics.
    Agreement is evidence both implement the same rules; a seat asymmetry
    present in BOTH is evidence it is a real property of the game."""
    out = []
    for n in player_counts:
        A, B = _dist(engine_a, n_games, n), _dist(engine_b, n_games, n)
        crit = 1.36 * math.sqrt(2 / n_games)  # KS 5% critical value, equal samples
        for key, label in (("lengths", "game length"), ("winners", "winning score"),
                           ("allpts", "score distribution")):
            d = _ks(A[key], B[key])
            out.append({"check": f"{label} ({n}p)", "status": "pass" if d < crit else "flag",
                        "detail": f"engines differ by KS={d:.2f} (agreement threshold {crit:.2f})",
                        "question": "" if d < crit else
                        "The two independent engines produce different games here — one of "
                        "them reads a rule differently. Compare their behaviour around this "
                        "statistic (e.g. how the game ends, how points accrue)."})
        sa = " / ".join(f"{x:.0%}" for x in A["seat"]); sb = " / ".join(f"{x:.0%}" for x in B["seat"])
        maxdiff = max(abs(x - y) for x, y in zip(A["seat"], B["seat"]))
        agree = maxdiff < 2.5 * math.sqrt(0.25 * 0.75 / n_games) * 1.5
        out.append({"check": f"seat effects ({n}p)", "status": "info" if agree else "flag",
                    "detail": f"engine A {sa} · engine B {sb}",
                    "question": ("Both engines agree on seat win rates — any imbalance here is a "
                                 "property of the game, not an engine bug." if agree else
                                 "The engines disagree on which seat is favored: one mishandles "
                                 "turn order or the end of the game.")})
        if A["unfinished"] != B["unfinished"]:
            out.append({"check": f"termination ({n}p)", "status": "info",
                        "detail": f"unfinished games: A {A['unfinished']}, B {B['unfinished']}",
                        "question": "One engine stalls where the other ends; check any rule that "
                                    "forces progress."})
    return out
