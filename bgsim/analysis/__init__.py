"""Turn game records into a markdown balance report."""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, median

from ..engine import GameRecord


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def report(records: list[GameRecord], game=None, top: int = 10) -> str:
    n = len(records)
    if n == 0:
        return "no games"
    n_players = records[0].n_players
    lines = [f"# Balance report — {n} games, {n_players} players", ""]

    # --- game length
    rounds = [r.n_turns / r.n_players for r in records]
    lines += ["## Game length",
              f"- rounds per game: mean {mean(rounds):.1f}, median {median(rounds):.1f}, "
              f"min {min(rounds):.0f}, max {max(rounds):.0f}",
              f"- ties: {sum(1 for r in records if len(r.winners) > 1)}",
              f"- unfinished (hit the action cap; rules allow indefinite play): "
              f"{sum(1 for r in records if r.extra.get('unfinished'))}", ""]

    # --- stability: the same headline numbers under each trained population
    lines += stability(records)

    # --- lead lock-in (a finding, not a fault: some games are meant to snowball)
    lines += lead_lockin(records)

    # --- seat advantage
    seat_wins = [0.0] * n_players
    for r in records:
        for w in r.winners:
            seat_wins[w] += 1 / len(r.winners)
    lines += ["## Win rate by seat", "| seat | win rate |", "|---|---|"]
    lines += [f"| {i} | {_pct(seat_wins[i] / n)} |" for i in range(n_players)]
    lines.append("")

    # --- agent strength
    agent_wins, agent_played = Counter(), Counter()
    for r in records:
        names = r.extra.get("seat_agent", r.agents)
        for i, a in enumerate(names):
            agent_played[a] += 1
            if i in r.winners:
                agent_wins[a] += 1 / len(r.winners)
    lines += ["## Win rate by agent", "| agent | seats played | win rate |", "|---|---|---|"]
    for a in agent_played:
        lines.append(f"| {a} | {agent_played[a]} | {_pct(agent_wins[a] / agent_played[a])} |")
    lines.append("")

    # --- scores
    pts = [s[0] for r in records for s in r.scores]
    win_pts = [r.scores[w][0] for r in records for w in r.winners]
    lines += ["## Scores",
              f"- winner points: mean {mean(win_pts):.1f}",
              f"- all players: mean {mean(pts):.1f}, min {min(pts)}, max {max(pts)}", ""]

    # --- end reasons
    reasons = Counter(str(r.extra.get("end_reason", "")) for r in records if r.extra.get("end_reason"))
    if reasons:
        lines += ["## How games ended"] + [f"- {k}: {_pct(v / n)}" for k, v in reasons.most_common()] + [""]

    # --- components (generic): {"components": {set: [[labels] per player]}}
    comps = {}
    for r in records:
        c = r.extra.get("components")
        if not isinstance(c, dict):
            continue
        for set_name, per_player in c.items():
            if not isinstance(per_player, (list, tuple)):
                continue
            tot, winr = comps.setdefault(set_name, (Counter(), Counter()))
            for seat, labels in enumerate(per_player):
                if not isinstance(labels, (list, tuple)):
                    continue
                for lab in labels:
                    if not isinstance(lab, (str, int)):
                        continue
                    tot[lab] += 1
                    if seat in r.winners:
                        winr[lab] += 1
    for set_name, (tot, winr) in comps.items():
        if not tot:
            continue
        base = sum(1 / max(1, r.n_players) * 0 + (len(r.winners) / r.n_players) for r in records) / n
        rows = []
        for lab, cnt in tot.most_common():
            if cnt < max(10, n // 50):
                continue
            rows.append((lab, cnt, winr[lab] / cnt))
        rows.sort(key=lambda x: -x[2])
        lines += [f"## {set_name} — who ends up holding them, and does it win",
                  f"Baseline: a piece held by a random player is in a winning hand about {_pct(base)} of the time.",
                  "", "| piece | games held | holder wins |", "|---|---|---|"]
        lines += [f"| {lab} | {cnt} | {_pct(w)} |" for lab, cnt, w in rows[:top]]
        strong = [lab for lab, cnt, w in rows if w >= base + 0.2 and cnt >= n // 10]
        weak = [lab for lab, cnt, w in rows if w <= base - 0.2 and cnt >= n // 10]
        if strong:
            lines.append(f"- **finding**: holders win far more often than baseline: {', '.join(strong[:6])}")
        if weak:
            lines.append(f"- **finding**: holders win far less often than baseline: {', '.join(weak[:6])}")
        lines.append("")

    # --- cards (Splendor-specific legacy summaries)
    bought, bought_by_winner = Counter(), Counter()
    for r in records:
        purchased = r.extra.get("purchased")
        if not purchased:
            continue
        if not isinstance(purchased, (list, tuple)):
            continue
        for seat, cards in enumerate(purchased):
            if not isinstance(cards, (list, tuple)):   # a count, not a list of ids: nothing per-card to say
                continue
            for c in cards:
                if isinstance(c, (list, dict)):
                    continue
                bought[c] += 1
                if seat in r.winners:
                    bought_by_winner[c] += 1
    if bought:
        lines += ["## Cards", ""]
        by_id = getattr(game, "card_by_id", {}) if game else {}

        def describe(cid):
            c = by_id.get(cid)
            if not c:
                return str(cid)
            from ..games.splendor.data import COLORS
            letters = {"white": "W", "blue": "U", "green": "G", "red": "R", "black": "K"}
            cost = "/".join(f"{amt}{letters[COLORS[i]]}" for i, amt in enumerate(c.cost) if amt)
            return f"#{cid} T{c.tier} {c.points}pt {COLORS[c.bonus]} ({cost})"

        if by_id:
            for tier in (1, 2, 3):
                ids = [cid for cid, c in by_id.items() if c.tier == tier]
                rate = sum(bought[cid] for cid in ids) / (n * len(ids))
                lines.append(f"- tier {tier}: each card bought in {_pct(rate)} of games on average")
            lines.append("")
        all_ids = list(by_id) if by_id else list(bought)
        ranked = sorted(all_ids, key=lambda c: -bought[c])
        lines += [f"### Most bought (top {top})", "| card | bought/game | bought by winner |", "|---|---|---|"]
        for cid in ranked[:top]:
            share = bought_by_winner[cid] / bought[cid] if bought[cid] else 0
            lines.append(f"| {describe(cid)} | {bought[cid] / n:.2f} | {_pct(share)} |")
        lines += ["", f"### Least bought (bottom {top})", "| card | bought/game | bought by winner |", "|---|---|---|"]
        for cid in ranked[-top:]:
            share = bought_by_winner[cid] / bought[cid] if bought[cid] else 0
            lines.append(f"| {describe(cid)} | {bought[cid] / n:.2f} | {_pct(share)} |")
        # win correlation: cards that, when bought, are disproportionately held by winners
        base = 1 / n_players
        strong = sorted((c for c in all_ids if bought[c] >= max(20, n * 0.05)),
                        key=lambda c: -(bought_by_winner[c] / bought[c]))
        lines += ["", f"### Strongest win association (min {max(20, int(n * 0.05))} purchases; baseline {_pct(base)})",
                  "| card | bought/game | bought by winner |", "|---|---|---|"]
        for cid in strong[:top]:
            lines.append(f"| {describe(cid)} | {bought[cid] / n:.2f} | {_pct(bought_by_winner[cid] / bought[cid])} |")
        lines.append("")

    # --- nobles
    nobles = Counter()
    nobles_winner = Counter()
    games_with_noble = 0
    for r in records:
        nb = r.extra.get("nobles")
        if not isinstance(nb, (list, tuple)):
            continue
        if any(nb):
            games_with_noble += 1
        for seat, ids in enumerate(nb):
            if isinstance(ids, (int, float)):        # a count per player: still counts as "a noble visited"
                continue
            if not isinstance(ids, (list, tuple)):
                continue
            for nid in ids:
                if isinstance(nid, (list, dict)):
                    continue
                nobles[nid] += 1
                if seat in r.winners:
                    nobles_winner[nid] += 1
    if nobles:
        lines += ["## Nobles",
                  f"- games with at least one noble claimed: {_pct(games_with_noble / n)}",
                  f"- nobles claimed per game: {sum(nobles.values()) / n:.2f}",
                  f"- claimed by eventual winner: {_pct(sum(nobles_winner.values()) / max(1, sum(nobles.values())))}", ""]
    return "\n".join(lines)


def _leader_at(timeline, turn):
    """Leader (or None on a tie) using the last sample at or before `turn`."""
    best = None
    for t, sc in timeline:
        if t <= turn:
            best = sc
        else:
            break
    if best is None:
        return None, None
    top = max(best)
    leaders = [i for i, v in enumerate(best) if v == top]
    ranked = sorted(best, reverse=True)
    gap = ranked[0] - ranked[1] if len(ranked) > 1 else 0.0
    return (leaders[0] if len(leaders) == 1 else None), gap


def lead_lockin(records, fractions=(0.25, 0.4, 0.5, 0.6, 0.75, 0.9)) -> list[str]:
    tl = [r for r in records if r.extra.get("timeline") and len(r.winners) == 1]
    if len(tl) < 20:
        return []
    rows = []
    for f in fractions:
        hits, tot, gaps = 0, 0, []
        for r in tl:
            leader, gap = _leader_at(r.extra["timeline"], f * r.n_turns)
            if leader is None:
                continue
            tot += 1
            hits += leader in r.winners
            final = max(sc[0] if isinstance(sc, (tuple, list)) else sc for sc in r.scores) or 1
            gaps.append(gap / final)
        if tot:
            rows.append((f, hits / tot, mean(gaps), tot))
    if not rows:
        return []
    half = next((p for f, p, _, _ in rows if abs(f - 0.5) < 1e-9), None)
    early = next((p for f, p, _, _ in rows if abs(f - 0.25) < 1e-9), None)
    out = ["## Lead lock-in",
           "How often the player leading at a given point goes on to win. Reading it: "
           "a curve that climbs gradually means games stay contested; one that is already "
           "high early means leads snowball. Neither is a fault by itself — some designs "
           "want a runaway leader — but it should be a choice.",
           "", "| point in game | leader goes on to win | lead size (share of winning score) | games |",
           "|---|---|---|---|"]
    out += [f"| {int(f * 100)}% | {_pct(p)} | {g:.0%} | {t} |" for f, p, g, t in rows]
    notes = []
    if half is not None:
        notes.append(f"- comeback rate: in {_pct(1 - half)} of games the mid-game leader did NOT win")
    if early is not None and early >= 0.8:
        notes.append("- **finding**: the leader at the quarter mark wins "
                     f"{_pct(early)} of the time — the outcome is largely settled early")
    elif half is not None and half >= 0.85:
        notes.append("- **finding**: the mid-game leader wins "
                     f"{_pct(half)} of the time — the second half rarely changes the result")
    notes.append("- caveat: simple agents rarely execute comebacks, so these numbers run higher "
                 "than a human table; compare between games or versions rather than reading them "
                 "as absolutes")
    return out + [""] + notes + [""]


def _headline(recs):
    n = len(recs)
    if not n:
        return None
    n_players = recs[0].n_players
    seat0 = sum(1 / len(r.winners) for r in recs if 0 in r.winners) / n
    rounds = mean(r.n_turns / n_players for r in recs)
    half = []
    for r in recs:
        tl = r.extra.get("timeline")
        if tl and len(r.winners) == 1:
            leader, _ = _leader_at(tl, 0.5 * r.n_turns)
            if leader is not None:
                half.append(leader in r.winners)
    lock = (sum(half) / len(half)) if half else None
    return {"seat0": seat0, "rounds": rounds, "lock50": lock, "n": n}


def stability(records) -> list[str]:
    """If games were played by several independent trained populations, compute
    the headline numbers per population. Agreement earns the number; a spread
    flags it as a possible play-style artefact rather than a fact about the game."""
    by_pop = {}
    for r in records:
        tag = r.extra.get("population")
        if tag is not None:
            by_pop.setdefault(tag, []).append(r)
    if len(by_pop) < 2:
        return []
    rows = {k: _headline(v) for k, v in by_pop.items()}
    out = ["## Are these numbers stable across player populations?",
           "The same games were played by independently trained player populations. Numbers the "
           "populations agree on are properties of the game; ones they disagree on may be habits of a "
           "particular set of players.", "",
           "| population | games | seat 1 wins | rounds | mid-game leader wins |", "|---|---|---|---|---|"]
    for k, h in sorted(rows.items()):
        out.append(f"| {k} | {h['n']} | {_pct(h['seat0'])} | {h['rounds']:.1f} | {_pct(h['lock50']) if h['lock50'] is not None else '—'} |")
    def spread(key):
        vals = [h[key] for h in rows.values() if h.get(key) is not None]
        return (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    flags = []
    if spread("seat0") > 0.08:
        flags.append("- **unstable**: the seat-1 win rate differs by more than 8 points between populations — treat the seat effect as unproven")
    else:
        flags.append("- seat effect: stable across populations")
    if spread("lock50") > 0.12:
        flags.append("- **unstable**: lead lock-in differs by more than 12 points between populations — it depends on how these players play, not only on the rules")
    else:
        flags.append("- lead lock-in: stable across populations")
    rel = spread("rounds") / max(1.0, mean(h["rounds"] for h in rows.values()))
    flags.append("- game length: " + ("**unstable** (differs by more than 15%)" if rel > 0.15 else "stable"))
    return out + [""] + flags + [""]
