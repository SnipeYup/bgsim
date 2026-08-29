# Go/no-go experiment: rulebook → engine

Question: given only the engine API and the Splendor rulebook, how many
verification rounds does a model need to produce an engine that agrees with
the hand-written reference on every state of 10,000 seeded random games?

That number, repeated across three games of different shapes, is the
feasibility test for the product.

## Procedure

1. Start a **fresh session** (new chat, or a new Claude Code session in an
   empty folder). It must not have seen this repo's `game.py`.
2. Give it exactly three things:
   - `bgsim/engine.py`
   - `bgsim/games/splendor/data.py` (card loader; it will read the CSVs)
   - the Splendor rulebook text, and the prompt below
3. It returns one file. Save it as `bgsim/games/splendor_gen/game.py`
   (create an empty `__init__.py` next to it).
4. Run the comparator:

       python -m bgsim.compare bgsim.games.splendor_gen.game:Splendor --games 10000 --players 4

   Also run with `--players 2` and `--players 3`.
5. If it prints a divergence, paste the divergence output (and nothing else)
   back to the session and ask it to fix the file. That is one round.
6. Stop when all three player counts print `OK: engines agree`.
   Record the round count below. Then run the CI gates on the generated
   engine (`pytest` with the fixture pointed at it, `stress`, strength gate).

Rules for a fair count: no hints beyond the comparator output; no reading the
reference engine; one round = one fix attempt, however large.

## Results

| game | model | rounds to agree | notes |
|---|---|---|---|
| Splendor | | | |
| (worker placement game) | | | |
| (hidden information game) | | | |

## Prompt

Paste everything from the line below to the end of the file, after the
rulebook text.

---

You are implementing a board game as a deterministic simulation engine.
Attached are `engine.py`, which defines the `Game` protocol every engine
targets, and `data.py`, which loads the card and noble data. The rulebook is
above. Write a single file, `game.py`, containing a class `Splendor` that
implements the `Game` protocol for this rulebook. Use only the Python
standard library. Import the data loader as
`from ..splendor.data import COLORS, GOLD, N_COLORS, Card, Noble, load_cards, load_nobles`
and call `load_cards()` / `load_nobles()` in `__init__` (the constructor
takes optional `cards` and `nobles` overrides). Do not hardcode any card.

Your engine will be verified by playing thousands of seeded random games in
lockstep against a reference implementation. At every state the set of legal
actions must match exactly, so the following conventions are mandatory.

Colours are indices 0..4 in the order white, blue, green, red, black; gold is
index 5. Token tuples have length 6, bonus tuples length 5.

Phases: `"main"` for a normal turn; `"discard"` while a player is over the
hand limit and must return tokens one at a time; `"noble"` when a player
qualifies for more than one noble and must choose. During `discard` and
`noble` the current player does not change. If exactly one noble is
available to a player, award it automatically with no phase.

Actions are tuples:

- `("take3", (c1, c2, c3))` — colours in ascending order. Taking 1 or 2
  different colours is also legal, as `("take3", (c1,))` / `("take3", (c1, c2))`,
  whether or not three colours are available. Only non-empty piles.
- `("take2", c)`
- `("reserve", tier_index, slot_index)` — face-up card; tier_index 0..2, slot_index 0..3
- `("reserve_deck", tier_index)` — blind from the top of that deck
- `("buy", tier_index, slot_index)`
- `("buy_reserved", i)` — index into the player's reserved tuple
- `("discard", c)` — c in 0..5, only in the discard phase
- `("noble", noble_id)` — only in the noble phase
- `("pass",)` — only when no other action is legal. If every player in turn
  is forced to pass, the game ends and is scored as it stands.

Payment: spend coloured tokens first, gold only for the remaining shortfall.

Setup from a seed, to match the reference exactly:

    rng = random.Random(seed)
    for tier in (1, 2, 3):
        ids = [c.id for c in self.cards if c.tier == tier]   # CSV order
        rng.shuffle(ids)
        market[tier-1] = tuple(ids[-4:])      # face-up, in this order
        decks[tier-1]  = tuple(ids[:-4])      # top of deck is the LAST element
    noble_ids = [n.id for n in self.nobles]
    rng.shuffle(noble_ids)
    nobles = tuple(noble_ids[:n_players + 1])

Bank: 7 of each colour for 4 players, 5 for 3, 4 for 2; 5 gold. When a
face-up card leaves the market, the top of its deck goes into the same slot;
if the deck is empty the row simply gets shorter (later slots shift down).

`scores(state)` returns one tuple per player, `(points, -number_of_purchased_cards)`.
`observation` returns the state unchanged. `features` may return an empty
tuple. `check_invariants` should assert token and card conservation, reserve
and hand limits, and that points equal card points plus noble points.

States must be immutable (frozen dataclasses with tuples); `apply` returns a
new state. Never use randomness after `initial_state`.

Return only the complete `game.py`.
