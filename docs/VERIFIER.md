# The verification compiler

The engine is generated from the rulebook. Verifying it must not depend on
an expert reading game logs — that was the manual step in the Agricola test.
Instead, the same rulebook is compiled a second time into **independent
checkers**: small programs that each re-derive ONE rule from ONE rulebook
section and audit game traces for violations.

Independence is the entire trick. The engine session and each checker
session never see each other's code and never see more of the rulebook than
they need. An engine bug and a checker bug are unlikely to coincide, so any
disagreement is one of three valuable things: an engine bug, a checker bug,
or an ambiguity in the rulebook itself. All three get surfaced to the
designer as a concrete question with numbers attached.

## Workflow

1. Generate the engine (docs/EXPERIMENT.md protocol).
2. Get the trace format:

       python -m bgsim.trace <module:Class> --players 2 --seed 0 --schema > schema.txt

3. Split the rulebook into sections that state self-contained rules
   (feeding/harvest, end-game scoring, replenishment, capacities, growth,
   action costs...). For each section, open a FRESH session and give it:
   the section text, `schema.txt`, and the prompt template below. Nothing
   else — no engine code, no other checkers, no other sections.
4. Save each result as `checkers/<game>/<name>.py` and run:

       python -m bgsim.verify <module:Class> --checkers checkers/<game> --games 30 --players 2
       python -m bgsim.verify <module:Class> --checkers checkers/<game> --games 30 --players 4

5. Triage each finding: paste it into the engine session ("a checker claims
   X at step N; here is the trace excerpt") AND reread the rulebook line.
   Fix whichever side is wrong; if both readings are defensible, that is a
   rulebook ambiguity — the designer answers it, the losing side is fixed.
6. Done when all checkers run clean over fresh seeds at 2 and 4 players.

## Recommended checker set (worker-placement games)

- `feeding.py` — harvest timing, food owed per person, begging arithmetic
- `scoring.py` — recompute every category of the final score from the final
  state and compare with `trace["scores"]`
- `replenish.py` — accumulation-space stock arithmetic round by round
- `capacity.py` — animals within (pet + stables + pastures) at every
  non-adjustment step
- `growth.py` — family size vs rooms and the without-room exception

## Prompt template for one checker

---

You are writing an independent auditor for a board game simulation. You get
one section of the rulebook (above) and the trace format (below). You have
NOT seen the simulation's code and must not assume anything about it beyond
the trace format.

Write a single Python file, standard library only, with:

    NAME = "<short id>"
    RULE = "<one-sentence restatement of the rule you are checking>"

    def check(trace: dict) -> list[str]:

`check` audits one full game trace and returns one message per violation
(empty list if clean). Re-derive the rule ONLY from the rulebook section:
recompute what should have happened and compare against what the trace
records. Every message must carry the step index and the concrete numbers,
e.g. "step 41: P1 owed 9 food (4 adults, 1 newborn), paid 7, begging rose by
1 (expected 2)". Audit every step and the final state; do not sample. Do not
use fields that are not in the schema; if the schema lacks information the
rule needs, return that as a single message starting with "SCHEMA GAP:"
instead of guessing. Be strict: tolerate nothing the section forbids, flag
nothing it allows.

Return only the complete checker file.

---
