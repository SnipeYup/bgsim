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
| Splendor | Claude (fresh chat), 2026-08-28 | **0** | 431-line engine; agreed on 1,500 seeded random games at each of 2/3/4 players, first attempt. Reference test suite failed 7/21 only on internal field names (`purchased`, `final_round`) — a test-design lesson, not a rules bug. Rulebook was the clean `docs/splendor-rules.md`; treat as an easy case. |
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

---

# Test 2: Agricola (Family Game) — worker placement, no reference engine

There is no hand-written oracle for this one, so `compare.py` cannot be used.
Verification is what the product will actually do: invariant assertions,
readable game logs, and score sanity. Use the publisher's rulebook text, not
a summary written for this purpose.

## Procedure

1. Fresh session. Give it `engine.py`, the Family Game rules text from the
   official rulebook (revised edition; occupations and minor improvements are
   excluded in the Family Game), and the prompt below.
2. Save the result as `bgsim/games/agricola/game.py` with an empty `__init__.py`.
3. Run the inspector:

       python -m bgsim.inspect bgsim.games.agricola.game:Agricola --players 2 --games 30 --log 2
       python -m bgsim.inspect bgsim.games.agricola.game:Agricola --players 4 --games 30 --log 1 --agents random

   It stops at the first invariant or legality failure; otherwise it prints
   two narrated games and the statistics.
4. Read the narrated games against this checklist. Every miss is one item of
   feedback; paste the log excerpt plus the rule it breaks back to the session.
   One fix attempt = one round.
5. Stop when a full read of two 2-player and one 4-player game finds nothing,
   invariants hold over 100 games, and the statistics are plausible.

## Log-reading checklist

- Harvest happens only after rounds 4, 7, 9, 11, 13 and 14, in that order:
  field phase (1 grain/vegetable per sown field), feeding, breeding.
- Feeding: 2 food per family member, 1 for a newborn from this round. Every
  missing food is one begging marker (−3 points each).
- Breeding: each animal type with 2+ animals gains exactly 1, only if there
  is space for it.
- Accumulation spaces gain their goods every round, whether taken or not.
- Family growth requires a free room; the newborn acts this round only if
  the space allows it, otherwise next round.
- Renovation goes wood → clay → stone, one step at a time, and requires the
  costs for every room.
- New fields must be adjacent to existing fields (unless it is the first);
  new rooms adjacent to the house; sowing puts 3 grain or 2 vegetables on an
  empty field.
- Pastures: each holds 2 animals, doubled by a stable; an unfenced stable
  holds 1; the house holds 1 pet. Fences total 15 per player. Animals with
  nowhere to go must be converted or released immediately.
- Scoring per the rulebook table (fields, pastures, grain, vegetables, sheep,
  boar, cattle, unused farmyard spaces −1 each, rooms, family, begging,
  major improvement points).

## Plausibility

- Random agents should starve badly (many begging markers, low or negative
  scores). Score-greedy agents should mostly avoid begging and score roughly
  15–35, the range real Family Game play produces.
- Every game must end after round 14; no unfinished games.

## Results

| game | model | rounds | notes |
|---|---|---|---|
| Agricola Family, 2-4p | Claude (fresh chat), 2026-08-30 | **1** | 1,479-line engine. First attempt: harvests, feeding, breeding, growth, renovation, improvements and full scoring all verified correct by log audit; one invariant failure (fencing over occupied stables didn't trigger animal redistribution). Fixed in one round from a 3-line repro; 91 further games clean incl. the failing configuration class. Verification was manual expert audit — the thing the verification-compiler layer must automate. |

## Prompt

---

You are implementing a board game as a deterministic simulation engine for a
balance-testing tool. Attached is `engine.py`, which defines the `Game`
protocol every engine targets. The rulebook is above. Implement the Family
Game variant: no occupation or minor improvement cards; everything else,
including major improvements, harvests, breeding and the full end-game
scoring, applies. Support 2, 3 and 4 players.

Write a single file, `game.py`, containing a class `Agricola` implementing
the protocol, standard library only. Requirements:

- States are frozen dataclasses built from tuples; `apply` returns a new
  state. All randomness happens in `initial_state(n_players, seed)` — shuffle
  the round cards within each stage there, nowhere else.
- Actions are tuples whose first element is a string naming the kind. Break
  multi-step decisions into forced sub-phases with their own small action
  sets rather than enumerating every combination in one action. In
  particular, fencing should be built up segment by segment (or by choosing
  among candidate enclosed regions) with a `("done",)` action, and feeding
  should let a player convert goods to food one step at a time before
  begging markers are assigned. `phase(state)` names the current sub-phase;
  `current_player(state)` is whoever must decide.
- `legal_actions` must be exact: nothing illegal under the rules, nothing
  legal omitted. When a player has no choice, resolve automatically.
- `scores(state)` returns one tuple per player, `(points,)`, computed with
  the official end-game scoring applied to the current state at any time,
  so a one-ply agent can use it mid-game.
- `check_invariants(state)` asserts every structural rule you can express:
  farmyard tiles (rooms + fields + pastures + unfenced stables) never exceed
  15; fences used never exceed 15; family members never exceed rooms; each
  animal type is within capacity; no negative goods; begging markers only
  ever increase; round is 1..14 and the game ends after round 14; at most one
  family member per action space per round.
- Implement `describe_state(state)` (one line: round, phase, whose turn,
  each player's food, family, goods, animals, begging markers),
  `describe_action(state, action)` (one line, plain English), and
  `describe_final(state)` (score breakdown per player by category). These
  are read by a human verifying the rules.
- Implement `summary(state)` returning a dict of per-player lists: `score`,
  `begging`, `rooms`, `family`, `fields`, `pastures`, `sheep`, `boar`,
  `cattle`, `major_improvements`. `features` may return an empty tuple.
- Do not simplify rules for convenience. If the rulebook is ambiguous, pick
  the reading you think is right and put a comment `# RULING:` with the
  choice at that point in the code, so it can be checked.

Return only the complete `game.py`.
