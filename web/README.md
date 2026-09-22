# bgsim web app (session 1)

Local workbench: create a game (built-in engine, or generate one from a
pasted rulebook), run simulations, read the balance report.

## Run

    pip install fastapi uvicorn
    python web/run.py            # -> http://127.0.0.1:8000

Built-in engines (Splendor, Agricola) work with no API key. To generate an
engine from a rulebook, set a key first:

    export ANTHROPIC_API_KEY=sk-...    # the server calls the model itself
    python web/run.py

## Shape

- `web/app.py` — FastAPI: projects, /generate (model call), /simulate
  (background thread over the existing harness), /report.
- `web/llm.py` — rulebook -> engine code via the Anthropic API.
- `web/static/index.html` — single-file frontend.
- `web/run.py` — launcher (sets the path so `bgsim` imports cleanly).
- Projects persist under `data/<id>/`.

## Pipeline (session 2)

1. Rules workshop (Phase A, attended) — the model restates the rules as an
   outline with every line tagged stated/inferred/unclear, walks a narrated
   sample round flagging assumptions, reviews for gaps, and triages which
   questions the text already answers. The designer sees only the uncertain
   items: confirm, correct, or "not sure". Answers append to the rulebook as
   clarifications; re-review loops up to 5 passes; a readiness meter gates
   engine generation (override possible, with a warning).
2. Generate + validate + repair — engine from the clarified rulebook.
3. Rule checks — the rulebook is split into sections; an independent model
   writes one auditor per section (never sees engine code); auditors run over
   30 games; findings can be sent back as a repair round.
4. Second opinion — an independent second engine, compared on agent-free
   statistics.
5. Balance report — simulations with any table of agents.

`tools/fake_api.py` fakes the model API so all of this runs offline in tests.

## Phase B — unattended (session 3)

`POST /run-all` sequences: engine (if needed) → rule checkers → second
opinion → jury → balance report. The **jury** frames each auditor finding as
a concrete scenario and factual question, asks a proxy rules expert cold
(rulebook + scenario only), and routes by agreement: auditor + proxy against
the engine → automatic fix (archived, re-audited); proxy sides with the engine
→ auditor muted; rulebook silent → provisional ruling kept and a question
queued in the **inbox**. Inbox answers become clarifications; one that
overturns a provisional ruling triggers a fix and re-audit. Reports note any
provisional rulings they rest on. Every automatic decision is logged in the
jury trail.

## Not yet (next sessions)

- Generated engines are executed by the server: run locally / for yourself
  only until sandboxed.
- Exploiter and evolve in the UI; accounts, slots, payment; deployment;
  notifications when a run finishes.
