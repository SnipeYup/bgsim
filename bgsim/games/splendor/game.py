"""Splendor implemented against the bgsim engine API.

Rules covered: take 3 different / take 2 same (pile >= 4), reserve (face-up
or blind from deck, +1 gold if available, max 3), buy (market or reserved,
bonuses discount, gold as wildcard), 10-token hand limit with a discard phase,
nobles (auto-award if one eligible, choice phase if several), end-of-round
finish at 15 points, tiebreak fewest development cards.

Simplifications, flagged as config so they can be revisited:
  - Payment always spends colored tokens first and gold only for shortfall.
  - `allow_fewer_than_three`: taking 1 or 2 different gems is always legal
    (as on Board Game Arena). Set False for strict "only when forced".
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from itertools import combinations

from .data import (COLORS, GOLD, N_COLORS, Card, Noble, load_cards,
                   load_nobles)

TOKENS_PER_PLAYER_COUNT = {2: 4, 3: 5, 4: 7}
GOLD_TOKENS = 5
MARKET_SIZE = 4
MAX_RESERVED = 3
HAND_LIMIT = 10
WIN_POINTS = 15


@dataclass(frozen=True)
class PlayerState:
    tokens: tuple          # len 6, index 5 = gold
    bonuses: tuple         # len 5
    points: int
    reserved: tuple        # card ids
    purchased: tuple       # card ids
    nobles: tuple          # noble ids


@dataclass(frozen=True)
class SplendorState:
    n_players: int
    current: int
    phase: str             # 'main' | 'discard' | 'noble'
    bank: tuple            # len 6
    market: tuple          # 3 tuples of card ids (face-up, per tier)
    decks: tuple           # 3 tuples of card ids; top of deck is LAST element
    nobles: tuple          # noble ids still available
    players: tuple         # PlayerState per player
    final_round: bool
    done: bool
    turn: int              # main actions taken so far
    pass_streak: int = 0   # consecutive forced passes; a full round ends the game


class Splendor:
    name = "splendor"

    def __init__(self, cards: tuple[Card, ...] | None = None,
                 nobles: tuple[Noble, ...] | None = None,
                 allow_fewer_than_three: bool = True):
        self.cards = cards or load_cards()
        self.nobles = nobles or load_nobles()
        self.card_by_id = {c.id: c for c in self.cards}
        self.noble_by_id = {n.id: n for n in self.nobles}
        self.allow_fewer_than_three = allow_fewer_than_three
        self._take3 = [tuple(c) for k in (3, 2, 1)
                       for c in combinations(range(N_COLORS), k)]

    # ------------------------------------------------------------------ setup
    def initial_state(self, n_players: int, seed: int) -> SplendorState:
        if n_players not in TOKENS_PER_PLAYER_COUNT:
            raise ValueError("Splendor supports 2-4 players")
        rng = random.Random(seed)
        decks, market = [], []
        for tier in (1, 2, 3):
            ids = [c.id for c in self.cards if c.tier == tier]
            rng.shuffle(ids)
            face_up = tuple(ids[-MARKET_SIZE:])
            decks.append(tuple(ids[:-MARKET_SIZE]))
            market.append(face_up)
        noble_ids = [n.id for n in self.nobles]
        rng.shuffle(noble_ids)
        n_tok = TOKENS_PER_PLAYER_COUNT[n_players]
        bank = (n_tok,) * N_COLORS + (GOLD_TOKENS,)
        empty = PlayerState(tokens=(0,) * 6, bonuses=(0,) * N_COLORS, points=0,
                            reserved=(), purchased=(), nobles=())
        return SplendorState(
            n_players=n_players, current=0, phase="main", bank=bank,
            market=tuple(market), decks=tuple(decks),
            nobles=tuple(noble_ids[:n_players + 1]),
            players=(empty,) * n_players, final_round=False, done=False, turn=0)

    # ---------------------------------------------------------------- queries
    def current_player(self, s: SplendorState) -> int:
        return s.current

    def phase(self, s: SplendorState) -> str:
        return s.phase

    def is_terminal(self, s: SplendorState) -> bool:
        return s.done

    def observation(self, s: SplendorState, player: int):
        return s  # perfect information (deck order is hidden but never queried)

    def scores(self, s: SplendorState) -> list[tuple]:
        return [(p.points, -len(p.purchased)) for p in s.players]

    def summary(self, s: SplendorState) -> dict:
        return {
            "points": [p.points for p in s.players],
            "n_cards": [len(p.purchased) for p in s.players],
            "purchased": [list(p.purchased) for p in s.players],
            "nobles": [list(p.nobles) for p in s.players],
            "reserved_left": [len(p.reserved) for p in s.players],
            "stalemate": s.pass_streak >= s.n_players,
        }

    # --------------------------------------------------------------- helpers
    def shortfall(self, ps: PlayerState, card: Card) -> int:
        """Gold needed to buy `card` after bonuses and colored tokens."""
        need = 0
        for c in range(N_COLORS):
            owed = card.cost[c] - ps.bonuses[c]
            if owed > ps.tokens[c]:
                need += owed - ps.tokens[c]
        return need

    def can_afford(self, ps: PlayerState, card: Card) -> bool:
        return self.shortfall(ps, card) <= ps.tokens[GOLD]

    def _eligible_nobles(self, s: SplendorState, ps: PlayerState) -> list[int]:
        out = []
        for nid in s.nobles:
            req = self.noble_by_id[nid].req
            if all(ps.bonuses[c] >= req[c] for c in range(N_COLORS)):
                out.append(nid)
        return out

    # ---------------------------------------------------------- legal actions
    def legal_actions(self, s: SplendorState) -> list:
        if s.done:
            return []
        ps = s.players[s.current]
        if s.phase == "discard":
            return [("discard", c) for c in range(6) if ps.tokens[c] > 0]
        if s.phase == "noble":
            return [("noble", nid) for nid in self._eligible_nobles(s, ps)]

        acts = []
        nonempty = [c for c in range(N_COLORS) if s.bank[c] > 0]
        n_avail = len(nonempty)
        for combo in self._take3:
            if all(s.bank[c] > 0 for c in combo):
                if len(combo) == 3 or self.allow_fewer_than_three \
                        or len(combo) == n_avail:
                    acts.append(("take3", combo))
        for c in range(N_COLORS):
            if s.bank[c] >= 4:
                acts.append(("take2", c))
        if len(ps.reserved) < MAX_RESERVED:
            for t in range(3):
                for i in range(len(s.market[t])):
                    acts.append(("reserve", t, i))
                if s.decks[t]:
                    acts.append(("reserve_deck", t))
        for t in range(3):
            for i, cid in enumerate(s.market[t]):
                if self.can_afford(ps, self.card_by_id[cid]):
                    acts.append(("buy", t, i))
        for i, cid in enumerate(ps.reserved):
            if self.can_afford(ps, self.card_by_id[cid]):
                acts.append(("buy_reserved", i))
        if not acts:
            acts.append(("pass",))
        return acts

    # ------------------------------------------------------------------ apply
    def apply(self, s: SplendorState, action) -> SplendorState:
        kind = action[0]
        p = s.current
        ps = s.players[p]

        if kind == "discard":
            c = action[1]
            assert s.phase == "discard" and ps.tokens[c] > 0
            ps = replace(ps, tokens=_dec(ps.tokens, c))
            s = _set_player(s, p, ps, bank=_inc(s.bank, c))
            if sum(ps.tokens) > HAND_LIMIT:
                return s
            return self._after_main(s)

        if kind == "noble":
            nid = action[1]
            assert s.phase == "noble" and nid in self._eligible_nobles(s, ps)
            s = self._award_noble(s, p, nid)
            return self._end_turn(s)

        assert s.phase == "main", f"{kind} not legal in phase {s.phase}"
        bank = s.bank
        market, decks = s.market, s.decks

        if kind == "take3":
            for c in action[1]:
                assert bank[c] > 0
                bank = _dec(bank, c)
                ps = replace(ps, tokens=_inc(ps.tokens, c))
        elif kind == "take2":
            c = action[1]
            assert bank[c] >= 4
            bank = _add(bank, c, -2)
            ps = replace(ps, tokens=_add(ps.tokens, c, 2))
        elif kind in ("reserve", "reserve_deck"):
            assert len(ps.reserved) < MAX_RESERVED
            t = action[1]
            if kind == "reserve":
                cid = market[t][action[2]]
                market, decks = _take_from_market(market, decks, t, action[2])
            else:
                assert decks[t]
                cid = decks[t][-1]
                decks = _replace_tier(decks, t, decks[t][:-1])
            ps = replace(ps, reserved=ps.reserved + (cid,))
            if bank[GOLD] > 0:
                bank = _dec(bank, GOLD)
                ps = replace(ps, tokens=_inc(ps.tokens, GOLD))
        elif kind in ("buy", "buy_reserved"):
            if kind == "buy":
                t, i = action[1], action[2]
                cid = market[t][i]
                market, decks = _take_from_market(market, decks, t, i)
            else:
                i = action[1]
                cid = ps.reserved[i]
                ps = replace(ps, reserved=ps.reserved[:i] + ps.reserved[i + 1:])
            card = self.card_by_id[cid]
            assert self.can_afford(ps, card)
            tokens = list(ps.tokens)
            bank_l = list(bank)
            for c in range(N_COLORS):
                owed = max(0, card.cost[c] - ps.bonuses[c])
                pay = min(owed, tokens[c])
                tokens[c] -= pay
                bank_l[c] += pay
                gold_needed = owed - pay
                tokens[GOLD] -= gold_needed
                bank_l[GOLD] += gold_needed
            assert tokens[GOLD] >= 0
            ps = replace(ps, tokens=tuple(tokens),
                         bonuses=_inc(ps.bonuses, card.bonus),
                         points=ps.points + card.points,
                         purchased=ps.purchased + (cid,))
            bank = tuple(bank_l)
        elif kind == "pass":
            pass
        else:
            raise ValueError(f"unknown action {action!r}")

        streak = s.pass_streak + 1 if kind == "pass" else 0
        s = replace(s, bank=bank, market=market, decks=decks, turn=s.turn + 1,
                    pass_streak=streak)
        s = _set_player(s, p, ps)
        if sum(ps.tokens) > HAND_LIMIT:
            return replace(s, phase="discard")
        return self._after_main(s)

    # ------------------------------------------------------- turn resolution
    def _after_main(self, s: SplendorState) -> SplendorState:
        """Token limit satisfied; resolve nobles then end the turn."""
        p = s.current
        eligible = self._eligible_nobles(s, s.players[p])
        if len(eligible) == 1:
            s = self._award_noble(s, p, eligible[0])
        elif len(eligible) > 1:
            return replace(s, phase="noble")
        return self._end_turn(s)

    def _award_noble(self, s, p, nid) -> SplendorState:
        ps = s.players[p]
        noble = self.noble_by_id[nid]
        ps = replace(ps, points=ps.points + noble.points, nobles=ps.nobles + (nid,))
        return _set_player(s, p, ps, nobles=tuple(n for n in s.nobles if n != nid))

    def _end_turn(self, s: SplendorState) -> SplendorState:
        final = s.final_round or s.players[s.current].points >= WIN_POINTS
        nxt = (s.current + 1) % s.n_players
        # Stalemate rule (engine-level, not in the rulebook): if every player
        # in turn had no legal action, the game ends and is scored as it stands.
        done = (final and nxt == 0) or s.pass_streak >= s.n_players
        return replace(s, phase="main", current=nxt, final_round=final, done=done)

    # ------------------------------------------------------------ invariants
    def check_invariants(self, s: SplendorState) -> None:
        n_tok = TOKENS_PER_PLAYER_COUNT[s.n_players]
        for c in range(6):
            total = s.bank[c] + sum(p.tokens[c] for p in s.players)
            expected = GOLD_TOKENS if c == GOLD else n_tok
            assert total == expected, f"token conservation broken for {c}"
            assert s.bank[c] >= 0
        seen = []
        for t in range(3):
            assert len(s.market[t]) <= MARKET_SIZE
            assert len(s.market[t]) == MARKET_SIZE or not s.decks[t], \
                "market not refilled while deck non-empty"
            seen += list(s.market[t]) + list(s.decks[t])
        for p in s.players:
            assert len(p.reserved) <= MAX_RESERVED
            assert all(x >= 0 for x in p.tokens)
            if s.phase == "main":
                assert sum(p.tokens) <= HAND_LIMIT, "hand limit violated"
            pts = sum(self.card_by_id[c].points for c in p.purchased)
            pts += sum(self.noble_by_id[n].points for n in p.nobles)
            assert p.points == pts, "points do not match cards+nobles"
            for c in range(N_COLORS):
                assert p.bonuses[c] == sum(1 for cid in p.purchased
                                           if self.card_by_id[cid].bonus == c)
            seen += list(p.reserved) + list(p.purchased)
        assert sorted(seen) == sorted(self.card_by_id), "card conservation broken"
        all_nobles = list(s.nobles) + [n for p in s.players for n in p.nobles]
        assert len(all_nobles) == len(set(all_nobles)) == s.n_players + 1

    # -------------------------------------------------------------- features
    FEATURE_NAMES = ("points", "bonus_cards", "colored_tokens", "gold_tokens",
                     "affordable_cards", "proximity", "noble_progress",
                     "reserved", "excess_tokens")

    def features(self, s: SplendorState, player: int) -> tuple[float, ...]:
        ps = s.players[player]
        colored = sum(ps.tokens[:N_COLORS])
        affordable = 0
        proximity = 0.0
        visible = [self.card_by_id[c] for t in range(3) for c in s.market[t]]
        visible += [self.card_by_id[c] for c in ps.reserved]
        for card in visible:
            sf = self.shortfall(ps, card)
            if sf <= ps.tokens[GOLD]:
                affordable += 1
            proximity += (card.points + 1) / (1 + sf)
        noble_prog = 0.0
        for nid in s.nobles:
            req = self.noble_by_id[nid].req
            tot = sum(req)
            got = sum(min(ps.bonuses[c], req[c]) for c in range(N_COLORS))
            noble_prog = max(noble_prog, got / tot)
        return (float(ps.points), float(sum(ps.bonuses)), float(colored),
                float(ps.tokens[GOLD]), float(affordable), proximity,
                noble_prog, float(len(ps.reserved)),
                float(max(0, sum(ps.tokens) - HAND_LIMIT)))


# ------------------------------------------------------------- tuple helpers
def _inc(t, i):
    return t[:i] + (t[i] + 1,) + t[i + 1:]


def _dec(t, i):
    return t[:i] + (t[i] - 1,) + t[i + 1:]


def _add(t, i, k):
    return t[:i] + (t[i] + k,) + t[i + 1:]


def _replace_tier(tiers, t, new):
    return tiers[:t] + (new,) + tiers[t + 1:]


def _take_from_market(market, decks, t, i):
    row = market[t]
    deck = decks[t]
    if deck:
        row = row[:i] + (deck[-1],) + row[i + 1:]
        deck = deck[:-1]
    else:
        row = row[:i] + row[i + 1:]
    return _replace_tier(market, t, row), _replace_tier(decks, t, deck)


def _set_player(s, p, ps, **kw):
    players = s.players[:p] + (ps,) + s.players[p + 1:]
    return replace(s, players=players, **kw)
