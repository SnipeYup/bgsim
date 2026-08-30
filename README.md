# bgsim — board game balance simulator (MVP)

Proves the engine API on Splendor: a hand-written rules engine, random and
heuristic agents, an evolutionary strategy search, a parallel simulation
harness, and a markdown balance report. No web app, no LLM yet — this is the
oracle the rulebook→code experiment will be measured against.

Pure Python, no dependencies (pytest for tests).

## Layout

    bgsim/engine.py            generic Game / Agent API, play_game
    bgsim/games/splendor/      Splendor rules (game.py), card data (data.py, data/*.csv)
    bgsim/agents/              RandomAgent, GreedyAgent (weighted one-ply), evolve.py
    bgsim/sim/                 parallel harness, CSV records
    bgsim/analysis/            balance report
    bgsim/cli.py               command line
    tests/                     rule-by-rule tests + invariant stress tests

## Usage

    python -m pytest -q                                   # 21 tests
    python -m bgsim stress --games 10000                  # random play, invariants on every action
    python -m bgsim run --agents expert,expert,expert,expert --games 5000 --out exp.csv
    python -m bgsim run --agents expert,random,random,random --games 500 --rotate
    python -m bgsim report exp.csv --top 15
    python -m bgsim evolve --generations 20 --games-per-gen 200 --out best.json
    python -m bgsim run --agents greedy@best.json,expert,expert,expert --games 1000 --rotate

Agent specs: `random`, `expert` (hand-written weights), `greedy` (random
weights), `greedy:[w0,...]`, `greedy@file.json`. `--rotate` cycles agents
through seats so agent strength and seat advantage don't confound each other.
`--workers` defaults to all cores.

## Card data

`bgsim/games/splendor/data/cards.csv` holds the real 90-card deck, taken from
the `bouk/splendimax` repository's card list and validated against the known
deck structure: 40/30/20 cards, 8/6/4 per colour per tier, points 0-1 / 1-3 /
3-5, cost totals 3-5 / 5-8 / 7-14 in the right proportions, no duplicates,
fully colour-symmetric. `nobles.csv` holds the 10 nobles (five 4+4 pairs and
five 3+3+3 triples of cyclically adjacent colours, 3 points each). Worth a
five-minute spot-check against a physical copy before trusting a finding
about one specific card. If either CSV is deleted, a synthetic deck of the
same shape is generated instead.

    cards.csv : id,tier,points,bonus,white,blue,green,red,black
    nobles.csv: id,points,white,blue,green,red,black

## Validation status (real deck, single core)

| gate | result |
|---|---|
| rule tests | 21/21 pass |
| 7,500 random games, invariants on every action | 0 violations |
| expert greedy vs 3 random (rotated) | 99.5% win rate |
| throughput | random ~70 games/s/core, greedy ~11 games/s/core |
| evolve loop | runs; expert stays top over a 3-generation smoke test |

At ~11 greedy games/s/core, 10k four-player games is ~2 minutes on 8 cores.
The hot path is `GreedyAgent.act` applying every legal action (~60 per turn).
Optimisation options, in order: score features incrementally instead of via
full `apply`; memoise `legal_actions`; port `Splendor.apply` to Rust via PyO3.

First findings from 600 all-expert games:

- ~31 rounds per game — long; real 4-player games are shorter, which points
  at the agents (see below), not the rules.
- seat 0 wins 30.0% vs the 25% baseline — a first-player edge, as expected.
- the 3-point tier-2 "six of one colour" cards carry the strongest win
  association (bought by the eventual winner ~42-44% of the time).
- tier-3 cards are bought in only ~8% of games each. Real players buy them
  far more; one-ply greedy can't plan the 7-10 tokens of saving they need.
  This is the known ceiling of the current agents and the reason MCTS is
  next on the agent side.
- nobles are claimed in 64% of games, 74% of them by the eventual winner.

## GitHub Actions

- `CI` runs on every push: the test suite, a 3,000-game invariant stress test,
  and a strength gate that fails the build if the expert agent stops beating
  random play.
- `Simulate` is manual (Actions tab → Simulate → Run workflow): choose agents,
  game count, seat rotation, and optionally run strategy search first. The
  balance report lands in the job summary and the CSV/JSON in the artifacts.
  A 2-core runner does ~20 greedy games/s, so 5,000 games is ~4 minutes.

## Rules notes

- Payment spends coloured tokens first and gold only for the shortfall.
  Real players may spend gold to keep colours; a config flag later.
- `allow_fewer_than_three=True` lets a player take 1 or 2 different gems even
  when 3 are available (Board Game Arena behaviour). `False` = only when forced.
- Stalemate rule (engine-level, not in the rulebook): if every player in turn
  has no legal action, the game ends and is scored as it stands. Random
  2-player play hits this occasionally; sensible agents never do.
- Decks are shuffled once from the seed; play is fully deterministic.

## Engine API (what generated code will target)

    initial_state(n_players, seed) -> State
    current_player(state) -> int
    phase(state) -> str                # 'main' | game-specific forced sub-phases
    legal_actions(state) -> list[Action]
    apply(state, action) -> State      # states immutable, actions hashable tuples
    is_terminal(state) -> bool
    scores(state) -> list[tuple]       # sortable, tiebreaks inside the tuple
    observation(state, player)         # identity for perfect information
    check_invariants(state)            # raises on impossible states
    features(state, player) -> tuple   # cheap scalars for heuristic agents

## Verification without an oracle

`bgsim/trace.py` records full game traces; `bgsim/verify.py` runs independent
rule-checkers over them. Checkers are generated in fresh sessions from single
rulebook sections (protocol and prompt in `docs/VERIFIER.md`) so that engine
and checker mistakes cannot coincide — every disagreement is an engine bug, a
checker bug, or a rulebook ambiguity, and all three are worth surfacing.
`checkers/<game>/` holds the checker set for each game.

## Next: the go/no-go experiment

Give a model only this API (engine.py + this section) and the Splendor
rulebook. Have it write `games/splendor_gen/game.py`. Measure:

1. Rounds of verification until its engine agrees with this one on 10k
   seeded games (identical `legal_actions` at every state of a random playout).
2. Repeat for a worker-placement game and a hidden-information game.

Those three numbers decide whether the product exists.
