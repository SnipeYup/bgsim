from dataclasses import replace

import pytest

from bgsim.engine import play_game, winners
from bgsim.agents import RandomAgent
from bgsim.games.splendor.data import GOLD, N_COLORS, Card, Noble
from bgsim.games.splendor.game import HAND_LIMIT, Splendor


@pytest.fixture
def game():
    return Splendor()


def set_player(s, p, **kw):
    ps = replace(s.players[p], **kw)
    return replace(s, players=s.players[:p] + (ps,) + s.players[p + 1:])


def actions_of(game, s, kind):
    return [a for a in game.legal_actions(s) if a[0] == kind]


def test_initial_setup(game):
    s = game.initial_state(4, 1)
    assert s.bank == (7, 7, 7, 7, 7, 5)
    assert all(len(row) == 4 for row in s.market)
    assert [len(d) for d in s.decks] == [36, 26, 16]
    assert len(s.nobles) == 5
    s2 = game.initial_state(2, 1)
    assert s2.bank[:5] == (4,) * 5 and len(s2.nobles) == 3
    game.check_invariants(s)


def test_take3_moves_tokens(game):
    s = game.initial_state(4, 1)
    s2 = game.apply(s, ("take3", (0, 1, 2)))
    assert s2.players[0].tokens == (1, 1, 1, 0, 0, 0)
    assert s2.bank == (6, 6, 6, 7, 7, 5)
    assert s2.current == 1 and s2.phase == "main"


def test_take2_requires_pile_of_four(game):
    s = game.initial_state(4, 1)
    s = replace(s, bank=(3, 4, 7, 7, 7, 5))
    take2 = actions_of(game, s, "take2")
    assert ("take2", 0) not in take2
    assert ("take2", 1) in take2


def test_take3_only_from_nonempty_piles(game):
    s = game.initial_state(4, 1)
    s = replace(s, bank=(0, 0, 0, 7, 7, 5))
    take3 = actions_of(game, s, "take3")
    assert ("take3", (3, 4)) in take3
    assert all(all(s.bank[c] > 0 for c in a[1]) for a in take3)


def test_strict_mode_forbids_voluntary_fewer(game):
    strict = Splendor(allow_fewer_than_three=False)
    s = strict.initial_state(4, 1)
    combos = [a[1] for a in actions_of(strict, s, "take3")]
    assert all(len(c) == 3 for c in combos)
    s = replace(s, bank=(0, 0, 0, 7, 7, 5))
    combos = [a[1] for a in actions_of(strict, s, "take3")]
    assert combos == [(3, 4)]


def test_reserve_gives_gold_and_caps_at_three(game):
    s = game.initial_state(4, 1)
    cid = s.market[0][0]
    s2 = game.apply(s, ("reserve", 0, 0))
    ps = s2.players[0]
    assert ps.reserved == (cid,) and ps.tokens[GOLD] == 1 and s2.bank[GOLD] == 4
    assert len(s2.market[0]) == 4 and len(s2.decks[0]) == 35  # refilled
    # no gold left in bank -> reserve gives none
    s3 = replace(s, bank=s.bank[:5] + (0,))
    s3 = game.apply(s3, ("reserve_deck", 2))
    assert s3.players[0].tokens[GOLD] == 0 and len(s3.players[0].reserved) == 1
    # three reserved -> no more reserve actions
    s4 = set_player(s, 0, reserved=(1, 2, 3))
    assert not actions_of(game, s4, "reserve") and not actions_of(game, s4, "reserve_deck")


def test_buy_pays_with_bonuses_then_gold(game):
    card = Card(id=999, tier=1, points=1, bonus=0, cost=(0, 2, 2, 0, 0))
    g = Splendor(cards=game.cards[:-1] + (card,), nobles=game.nobles)
    s = g.initial_state(4, 1)
    # put the test card face-up, and give player 0 one purchased blue card as bonus
    deck0 = s.decks[0]
    blue = next(c for c in deck0 if g.card_by_id[c].bonus == 1)
    swapped = s.market[0][0]
    deck0 = tuple(c for c in deck0 if c not in (blue, 999)) + (swapped,)
    s = replace(s, market=((999,) + s.market[0][1:],) + s.market[1:],
                decks=(deck0,) + s.decks[1:])
    # bonus 1 blue, tokens 1 blue + 1 green + 1 gold -> owes 1 blue, 2 green: pays b, g, gold
    s = set_player(s, 0, bonuses=(0, 1, 0, 0, 0), purchased=(blue,),
                   points=g.card_by_id[blue].points, tokens=(0, 1, 1, 0, 0, 1))
    s = replace(s, bank=(7, 6, 6, 7, 7, 4))
    assert ("buy", 0, 0) in g.legal_actions(s)
    s2 = g.apply(s, ("buy", 0, 0))
    ps = s2.players[0]
    assert ps.tokens == (0, 0, 0, 0, 0, 0)          # blue token, green token, gold all spent
    assert ps.points == 1 + g.card_by_id[blue].points and ps.bonuses == (1, 1, 0, 0, 0)
    assert ps.purchased == (blue, 999)
    assert s2.bank == (7, 7, 7, 7, 7, 5)
    g.check_invariants(s2)


def test_cannot_buy_unaffordable(game):
    s = game.initial_state(4, 1)
    assert not actions_of(game, s, "buy")


def test_buy_reserved_card(game):
    s = game.initial_state(4, 1)
    cid = s.market[1][0]
    card = game.card_by_id[cid]
    s = game.apply(s, ("reserve", 1, 0))          # player 0 reserves, gets 1 gold
    s = replace(s, current=0)                      # hand the turn back for the test
    s = set_player(s, 0, tokens=card.cost + (1,))
    s = replace(s, bank=tuple(b - t for b, t in zip(s.bank, card.cost + (0,))))
    assert ("buy_reserved", 0) in game.legal_actions(s)
    s2 = game.apply(s, ("buy_reserved", 0))
    assert s2.players[0].reserved == () and s2.players[0].purchased == (cid,)
    game.check_invariants(s2)


def test_hand_limit_forces_discard(game):
    s = game.initial_state(4, 1)
    s = set_player(s, 0, tokens=(2, 2, 2, 2, 0, 0))
    s = replace(s, bank=(5, 5, 5, 5, 7, 5))
    s2 = game.apply(s, ("take3", (0, 1, 2)))
    assert s2.phase == "discard" and s2.current == 0
    assert sum(s2.players[0].tokens) == HAND_LIMIT + 1
    legal = game.legal_actions(s2)
    assert all(a[0] == "discard" for a in legal)
    s3 = game.apply(s2, ("discard", 3))
    assert s3.phase == "main" and s3.current == 1
    assert sum(s3.players[0].tokens) == HAND_LIMIT
    game.check_invariants(s3)


def test_noble_auto_award_when_single(game):
    s = game.initial_state(4, 1)
    noble = game.noble_by_id[s.nobles[0]]
    bonuses = tuple(noble.req)
    # keep only that noble in play to guarantee a single eligible
    s = replace(s, nobles=(noble.id,))
    s = set_player(s, 0, bonuses=bonuses)
    s2 = game.apply(s, ("take3", (0, 1, 2)))
    assert s2.players[0].nobles == (noble.id,)
    assert s2.players[0].points == noble.points
    assert s2.nobles == () and s2.current == 1


def test_noble_choice_when_several(game):
    a = Noble(100, 3, (3, 3, 3, 0, 0))
    b = Noble(101, 3, (0, 0, 3, 3, 3))
    g = Splendor(cards=game.cards, nobles=(a, b))
    s = g.initial_state(2, 1)
    s = replace(s, nobles=(100, 101))
    s = set_player(s, 0, bonuses=(3, 3, 3, 3, 3))
    s2 = g.apply(s, ("take3", (0, 1, 2)))
    assert s2.phase == "noble" and s2.current == 0
    assert sorted(g.legal_actions(s2)) == [("noble", 100), ("noble", 101)]
    s3 = g.apply(s2, ("noble", 101))
    assert s3.players[0].nobles == (101,) and s3.nobles == (100,)
    assert s3.phase == "main" and s3.current == 1


def test_final_round_gives_equal_turns(game):
    s = game.initial_state(3, 1)
    s = set_player(s, 1, points=15)
    # player 0 acts, then player 1 (already at 15) triggers final round
    s = game.apply(s, ("take3", (0, 1, 2)))
    s = game.apply(s, ("take3", (0, 1, 2)))
    assert s.final_round and not s.done and s.current == 2
    s = game.apply(s, ("take3", (0, 1, 2)))
    assert s.done
    assert game.legal_actions(s) == []


def test_reaching_15_as_last_player_ends_immediately(game):
    s = game.initial_state(2, 1)
    s = set_player(s, 1, points=15)
    s = replace(s, current=1)
    s = game.apply(s, ("take3", (0, 1, 2)))
    assert s.done


def test_tiebreak_fewest_cards(game):
    s = game.initial_state(2, 1)
    s = set_player(s, 0, points=16, purchased=(1, 2, 3))
    s = set_player(s, 1, points=16, purchased=(4, 5))
    assert winners(game, s) == [1]


def test_full_random_game_terminates(game):
    rec = play_game(game, [RandomAgent(i) for i in range(4)], seed=7, debug=True)
    assert rec.n_turns > 0 and len(rec.winners) >= 1
    assert max(sc[0] for sc in rec.scores) >= 15


def test_stalemate_ends_after_full_round_of_passes(game):
    """Bank empty, both players at the hand limit with 3 reserves and nothing
    affordable: only 'pass' is legal. A full round of passes ends the game."""
    card = Card(id=999, tier=1, points=0, bonus=0, cost=(9, 9, 9, 9, 9))
    g = Splendor(cards=tuple(replace(c, cost=(9, 9, 9, 9, 9)) for c in game.cards),
                 nobles=game.nobles)
    s = g.initial_state(2, 1)
    s = replace(s, bank=(0, 0, 0, 0, 0, 5))
    s = set_player(s, 0, tokens=(2, 2, 2, 2, 2, 0), reserved=(1, 2, 3))
    s = set_player(s, 1, tokens=(2, 2, 2, 2, 2, 0), reserved=(4, 5, 6))
    assert g.legal_actions(s) == [("pass",)]
    s = g.apply(s, ("pass",))
    assert not s.done and s.current == 1
    s = g.apply(s, ("pass",))
    assert s.done
