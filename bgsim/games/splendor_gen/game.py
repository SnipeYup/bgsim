"""Splendor as a deterministic `Game`.

Conventions (must match the reference implementation exactly):
  colours 0..4 = white, blue, green, red, black; gold = 5.
  phases: "main", "discard" (over hand limit), "noble" (choose among >1).
  actions:
    ("take3", (c1[, c2[, c3]]))   ascending colours, non-empty piles
    ("take2", c)                  pile must hold >= 4
    ("reserve", tier, slot)       face-up card
    ("reserve_deck", tier)        blind from the top of the deck
    ("buy", tier, slot)
    ("buy_reserved", i)
    ("discard", c)                discard phase only
    ("noble", noble_id)           noble phase only
    ("pass",)                     only when nothing else is legal
"""
from __future__ import annotations

import random
from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Optional, Sequence

from ..splendor.data import COLORS, GOLD, N_COLORS, Card, Noble, load_cards, load_nobles

HAND_LIMIT = 10
MAX_RESERVED = 3
WIN_POINTS = 15
N_GOLD = 5
MARKET_SIZE = 4
BANK_PER_COLOUR = {2: 4, 3: 5, 4: 7}


@dataclass(frozen=True)
class PlayerState:
    tokens: tuple            # len 6 (incl. gold)
    bonuses: tuple           # len 5, permanent card bonuses
    cards: tuple             # purchased card ids
    reserved: tuple          # reserved card ids
    nobles: tuple            # noble ids acquired
    points: int


@dataclass(frozen=True)
class SplendorState:
    n_players: int
    current: int
    phase: str               # "main" | "discard" | "noble"
    bank: tuple              # len 6
    market: tuple            # 3 tuples of face-up card ids
    decks: tuple             # 3 tuples of card ids; top of deck is LAST
    nobles: tuple            # face-up noble ids
    players: tuple           # tuple[PlayerState]
    passes: int              # consecutive forced passes
    end_triggered: bool      # someone has reached 15
    done: bool


class Splendor:
    name = "splendor"

    def __init__(self, cards: Optional[Sequence[Card]] = None,
                 nobles: Optional[Sequence[Noble]] = None):
        self.cards = tuple(cards) if cards is not None else load_cards()
        self.nobles = tuple(nobles) if nobles is not None else load_nobles()
        self.card_by_id = {c.id: c for c in self.cards}
        self.noble_by_id = {n.id: n for n in self.nobles}

    # ------------------------------------------------------------------ setup
    def initial_state(self, n_players: int, seed: int) -> SplendorState:
        if n_players not in BANK_PER_COLOUR:
            raise ValueError("Splendor is for 2-4 players")
        rng = random.Random(seed)
        market = [None] * 3
        decks = [None] * 3
        for tier in (1, 2, 3):
            ids = [c.id for c in self.cards if c.tier == tier]
            rng.shuffle(ids)
            market[tier - 1] = tuple(ids[-MARKET_SIZE:])
            decks[tier - 1] = tuple(ids[:-MARKET_SIZE])
        noble_ids = [n.id for n in self.nobles]
        rng.shuffle(noble_ids)
        nobles = tuple(noble_ids[:n_players + 1])
        per = BANK_PER_COLOUR[n_players]
        bank = (per,) * N_COLORS + (N_GOLD,)
        empty = PlayerState(tokens=(0,) * (N_COLORS + 1), bonuses=(0,) * N_COLORS,
                            cards=(), reserved=(), nobles=(), points=0)
        return SplendorState(
            n_players=n_players, current=0, phase="main", bank=bank,
            market=tuple(market), decks=tuple(decks), nobles=nobles,
            players=(empty,) * n_players, passes=0,
            end_triggered=False, done=False)

    # --------------------------------------------------------------- queries
    def current_player(self, state: SplendorState) -> int:
        return state.current

    def phase(self, state: SplendorState) -> str:
        return state.phase

    def is_terminal(self, state: SplendorState) -> bool:
        return state.done

    def scores(self, state: SplendorState) -> list[tuple]:
        return [(p.points, -len(p.cards)) for p in state.players]

    def observation(self, state: SplendorState, player: int) -> Any:
        return state

    def features(self, state: SplendorState, player: int) -> tuple[float, ...]:
        return ()

    def summary(self, state: SplendorState) -> dict:
        return {
            "points": [p.points for p in state.players],
            "cards": [len(p.cards) for p in state.players],
            "nobles": [len(p.nobles) for p in state.players],
            "end_triggered": state.end_triggered,
        }

    # ---------------------------------------------------------------- helpers
    def _payment(self, tokens: tuple, bonuses: tuple, card: Card):
        """Return (new_tokens, paid) or None if unaffordable. Coloured tokens
        are spent first, gold only for the remaining shortfall."""
        new = list(tokens)
        paid = [0] * (N_COLORS + 1)
        gold_needed = 0
        for c in range(N_COLORS):
            need = max(0, card.cost[c] - bonuses[c])
            use = min(new[c], need)
            new[c] -= use
            paid[c] = use
            gold_needed += need - use
        if gold_needed > new[GOLD]:
            return None
        new[GOLD] -= gold_needed
        paid[GOLD] = gold_needed
        return tuple(new), tuple(paid)

    def _eligible_nobles(self, state: SplendorState, player: PlayerState) -> list[int]:
        out = []
        for nid in state.nobles:
            req = self.noble_by_id[nid].req
            if all(player.bonuses[c] >= req[c] for c in range(N_COLORS)):
                out.append(nid)
        return out

    def _remove_from_market(self, state: SplendorState, tier: int, slot: int):
        """Take the card at market[tier][slot]; refill from the deck top if
        possible, else shrink the row. Returns (card_id, market, decks)."""
        row = list(state.market[tier])
        deck = list(state.decks[tier])
        cid = row[slot]
        if deck:
            row[slot] = deck.pop()
        else:
            del row[slot]
        market = tuple(tuple(r) if i != tier else tuple(row)
                       for i, r in enumerate(state.market))
        decks = tuple(tuple(d) if i != tier else tuple(deck)
                      for i, d in enumerate(state.decks))
        return cid, market, decks

    @staticmethod
    def _set_player(state: SplendorState, idx: int, player: PlayerState) -> tuple:
        players = list(state.players)
        players[idx] = player
        return tuple(players)

    # --------------------------------------------------------- legal actions
    def legal_actions(self, state: SplendorState) -> list:
        if state.done:
            return []
        p = state.players[state.current]
        if state.phase == "discard":
            return [("discard", c) for c in range(N_COLORS + 1) if p.tokens[c] > 0]
        if state.phase == "noble":
            return [("noble", nid) for nid in self._eligible_nobles(state, p)]

        actions: list = []
        avail = [c for c in range(N_COLORS) if state.bank[c] > 0]
        for k in (1, 2, 3):
            for combo in combinations(avail, k):
                actions.append(("take3", combo))
        for c in range(N_COLORS):
            if state.bank[c] >= 4:
                actions.append(("take2", c))
        if len(p.reserved) < MAX_RESERVED:
            for tier in range(3):
                for slot in range(len(state.market[tier])):
                    actions.append(("reserve", tier, slot))
            for tier in range(3):
                if state.decks[tier]:
                    actions.append(("reserve_deck", tier))
        for tier in range(3):
            for slot, cid in enumerate(state.market[tier]):
                if self._payment(p.tokens, p.bonuses, self.card_by_id[cid]) is not None:
                    actions.append(("buy", tier, slot))
        for i, cid in enumerate(p.reserved):
            if self._payment(p.tokens, p.bonuses, self.card_by_id[cid]) is not None:
                actions.append(("buy_reserved", i))
        if not actions:
            actions.append(("pass",))
        return actions

    # ------------------------------------------------------------------ apply
    def apply(self, state: SplendorState, action) -> SplendorState:
        kind = action[0]
        idx = state.current
        p = state.players[idx]

        if state.phase == "discard":
            if kind != "discard":
                raise ValueError(f"expected discard action, got {action!r}")
            c = action[1]
            if p.tokens[c] <= 0:
                raise ValueError(f"no token of colour {c} to discard")
            tokens = list(p.tokens)
            tokens[c] -= 1
            bank = list(state.bank)
            bank[c] += 1
            state = replace(state, bank=tuple(bank),
                            players=self._set_player(state, idx, replace(p, tokens=tuple(tokens))))
            return self._after_discard(state)

        if state.phase == "noble":
            if kind != "noble":
                raise ValueError(f"expected noble action, got {action!r}")
            nid = action[1]
            if nid not in self._eligible_nobles(state, p):
                raise ValueError(f"noble {nid} not available")
            state = self._award_noble(state, idx, nid)
            return self._end_turn(state)

        # ---- main phase
        if kind == "pass":
            if len(self.legal_actions(state)) != 1:
                raise ValueError("pass is only legal when nothing else is")
            state = replace(state, passes=state.passes + 1)
            return self._end_turn(state)

        bank = list(state.bank)
        tokens = list(p.tokens)
        market, decks = state.market, state.decks
        new_p = p

        if kind == "take3":
            cols = action[1]
            if not cols or len(cols) > 3 or len(set(cols)) != len(cols) \
                    or list(cols) != sorted(cols) or any(c >= N_COLORS for c in cols):
                raise ValueError(f"bad take3 {action!r}")
            for c in cols:
                if bank[c] <= 0:
                    raise ValueError(f"colour {c} pile is empty")
                bank[c] -= 1
                tokens[c] += 1
            new_p = replace(p, tokens=tuple(tokens))

        elif kind == "take2":
            c = action[1]
            if c >= N_COLORS or bank[c] < 4:
                raise ValueError(f"bad take2 {action!r}")
            bank[c] -= 2
            tokens[c] += 2
            new_p = replace(p, tokens=tuple(tokens))

        elif kind == "reserve":
            tier, slot = action[1], action[2]
            if len(p.reserved) >= MAX_RESERVED:
                raise ValueError("reserve limit reached")
            if not (0 <= tier < 3 and 0 <= slot < len(state.market[tier])):
                raise ValueError(f"bad reserve {action!r}")
            cid, market, decks = self._remove_from_market(state, tier, slot)
            if bank[GOLD] > 0:
                bank[GOLD] -= 1
                tokens[GOLD] += 1
            new_p = replace(p, tokens=tuple(tokens), reserved=p.reserved + (cid,))

        elif kind == "reserve_deck":
            tier = action[1]
            if len(p.reserved) >= MAX_RESERVED:
                raise ValueError("reserve limit reached")
            if not (0 <= tier < 3) or not state.decks[tier]:
                raise ValueError(f"bad reserve_deck {action!r}")
            deck = list(state.decks[tier])
            cid = deck.pop()
            decks = tuple(tuple(d) if i != tier else tuple(deck)
                          for i, d in enumerate(state.decks))
            if bank[GOLD] > 0:
                bank[GOLD] -= 1
                tokens[GOLD] += 1
            new_p = replace(p, tokens=tuple(tokens), reserved=p.reserved + (cid,))

        elif kind == "buy":
            tier, slot = action[1], action[2]
            if not (0 <= tier < 3 and 0 <= slot < len(state.market[tier])):
                raise ValueError(f"bad buy {action!r}")
            card = self.card_by_id[state.market[tier][slot]]
            pay = self._payment(p.tokens, p.bonuses, card)
            if pay is None:
                raise ValueError(f"cannot afford {action!r}")
            new_tokens, paid = pay
            for c in range(N_COLORS + 1):
                bank[c] += paid[c]
            _, market, decks = self._remove_from_market(state, tier, slot)
            new_p = self._add_card(p, card, new_tokens, p.reserved)

        elif kind == "buy_reserved":
            i = action[1]
            if not (0 <= i < len(p.reserved)):
                raise ValueError(f"bad buy_reserved {action!r}")
            card = self.card_by_id[p.reserved[i]]
            pay = self._payment(p.tokens, p.bonuses, card)
            if pay is None:
                raise ValueError(f"cannot afford {action!r}")
            new_tokens, paid = pay
            for c in range(N_COLORS + 1):
                bank[c] += paid[c]
            reserved = p.reserved[:i] + p.reserved[i + 1:]
            new_p = self._add_card(p, card, new_tokens, reserved)

        else:
            raise ValueError(f"unknown action {action!r}")

        state = replace(state, bank=tuple(bank), market=market, decks=decks,
                        players=self._set_player(state, idx, new_p), passes=0)
        return self._after_action(state)

    @staticmethod
    def _add_card(p: PlayerState, card: Card, tokens: tuple, reserved: tuple) -> PlayerState:
        bonuses = list(p.bonuses)
        bonuses[card.bonus] += 1
        return replace(p, tokens=tokens, bonuses=tuple(bonuses),
                       cards=p.cards + (card.id,), reserved=reserved,
                       points=p.points + card.points)

    def _award_noble(self, state: SplendorState, idx: int, nid: int) -> SplendorState:
        p = state.players[idx]
        noble = self.noble_by_id[nid]
        new_p = replace(p, nobles=p.nobles + (nid,), points=p.points + noble.points)
        nobles = tuple(n for n in state.nobles if n != nid)
        return replace(state, nobles=nobles, players=self._set_player(state, idx, new_p))

    # --------------------------------------------------------- turn sequencing
    def _after_action(self, state: SplendorState) -> SplendorState:
        p = state.players[state.current]
        if sum(p.tokens) > HAND_LIMIT:
            return replace(state, phase="discard")
        return self._noble_check(state)

    def _after_discard(self, state: SplendorState) -> SplendorState:
        p = state.players[state.current]
        if sum(p.tokens) > HAND_LIMIT:
            return replace(state, phase="discard")
        return self._noble_check(state)

    def _noble_check(self, state: SplendorState) -> SplendorState:
        idx = state.current
        eligible = self._eligible_nobles(state, state.players[idx])
        if len(eligible) > 1:
            return replace(state, phase="noble")
        if len(eligible) == 1:
            state = self._award_noble(state, idx, eligible[0])
        return self._end_turn(state)

    def _end_turn(self, state: SplendorState) -> SplendorState:
        idx = state.current
        n = state.n_players
        end_triggered = state.end_triggered or any(
            pl.points >= WIN_POINTS for pl in state.players)
        done = (end_triggered and idx == n - 1) or state.passes >= n
        return replace(state, current=(idx + 1) % n, phase="main",
                       end_triggered=end_triggered, done=done)

    # ------------------------------------------------------------ invariants
    def check_invariants(self, state: SplendorState) -> None:
        n = state.n_players
        assert 0 <= state.current < n
        assert state.phase in ("main", "discard", "noble")

        # token conservation
        per = BANK_PER_COLOUR[n]
        for c in range(N_COLORS + 1):
            total = state.bank[c] + sum(pl.tokens[c] for pl in state.players)
            expected = N_GOLD if c == GOLD else per
            assert total == expected, f"token conservation broken for colour {c}"
            assert state.bank[c] >= 0
            assert all(pl.tokens[c] >= 0 for pl in state.players)

        # card conservation
        seen = []
        for tier in range(3):
            assert len(state.market[tier]) <= MARKET_SIZE
            if state.decks[tier]:
                assert len(state.market[tier]) == MARKET_SIZE, "market not refilled"
            seen.extend(state.market[tier])
            seen.extend(state.decks[tier])
            for cid in state.market[tier] + state.decks[tier]:
                assert self.card_by_id[cid].tier == tier + 1
        for pl in state.players:
            seen.extend(pl.cards)
            seen.extend(pl.reserved)
        assert len(seen) == len(set(seen)), "duplicate card"
        assert set(seen) == set(self.card_by_id), "card conservation broken"

        # noble conservation
        nobles = list(state.nobles)
        for pl in state.players:
            nobles.extend(pl.nobles)
        assert len(nobles) == len(set(nobles)) == n + 1, "noble conservation broken"

        # per-player limits and derived values
        for i, pl in enumerate(state.players):
            assert len(pl.reserved) <= MAX_RESERVED
            if not (state.phase == "discard" and i == state.current):
                assert sum(pl.tokens) <= HAND_LIMIT, "hand limit exceeded"
            bonuses = [0] * N_COLORS
            pts = 0
            for cid in pl.cards:
                card = self.card_by_id[cid]
                bonuses[card.bonus] += 1
                pts += card.points
            for nid in pl.nobles:
                pts += self.noble_by_id[nid].points
            assert tuple(bonuses) == pl.bonuses, "bonuses do not match cards"
            assert pts == pl.points, "points do not match cards + nobles"

        if state.phase == "noble":
            assert len(self._eligible_nobles(state, state.players[state.current])) > 1
        if state.phase == "discard":
            assert sum(state.players[state.current].tokens) > HAND_LIMIT
