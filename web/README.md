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

## Not yet (next sessions)

- Generated engines are executed by the server: run locally / for yourself
  only until sandboxed.
- Verification (checkers + mutation harness) isn't wired into the UI yet;
  neither is the exploiter search. Deploy target still to choose.
