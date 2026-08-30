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

    # --- cards
    bought, bought_by_winner = Counter(), Counter()
    for r in records:
        purchased = r.extra.get("purchased")
        if not purchased:
            continue
        for seat, cards in enumerate(purchased):
            for c in cards:
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
        if nb is None:
            continue
        if any(nb):
            games_with_noble += 1
        for seat, ids in enumerate(nb):
            for nid in ids:
                nobles[nid] += 1
                if seat in r.winners:
                    nobles_winner[nid] += 1
    if nobles or games_with_noble:
        lines += ["## Nobles",
                  f"- games with at least one noble claimed: {_pct(games_with_noble / n)}",
                  f"- nobles claimed per game: {sum(nobles.values()) / n:.2f}",
                  f"- claimed by eventual winner: {_pct(sum(nobles_winner.values()) / max(1, sum(nobles.values())))}", ""]
    return "\n".join(lines)
