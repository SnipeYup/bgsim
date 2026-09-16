"""bgsim web app — session-1 backend.

Projects live in data/<id>/ (rulebook.txt, game.py, report.md, meta.json).
Engines are either built-in (splendor, agricola) or generated from the
project's rulebook by a model call (needs ANTHROPIC_API_KEY). Simulations
run as background threads using the existing harness.

Run:  uvicorn web.app:app --reload      (pip install 'bgsim[web]' deps first)

Security note: generated engines are Python executed by this server. Run it
locally or for yourself only until sandboxing lands.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from bgsim.analysis import report as make_report
from bgsim.engine import play_game
from bgsim.agents import make_agent
from bgsim.games import make_game
from bgsim.sim import simulate
from web import llm

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

BUILTIN = {
    "splendor": "splendor",
    "agricola": "bgsim.games.agricola.game:Agricola",
}

app = FastAPI(title="bgsim")
JOBS: dict[str, dict] = {}


# ------------------------------------------------------------------ storage
def _meta_path(pid: str) -> Path:
    return DATA / pid / "meta.json"


def _load(pid: str) -> dict:
    p = _meta_path(pid)
    if not p.exists():
        raise HTTPException(404, "project not found")
    return json.loads(p.read_text())


def _save(meta: dict) -> None:
    d = DATA / meta["id"]
    d.mkdir(exist_ok=True)
    _meta_path(meta["id"]).write_text(json.dumps(meta, indent=2))


def _engine_for(meta: dict):
    if meta.get("builtin"):
        return make_game(BUILTIN[meta["builtin"]])
    game_py = DATA / meta["id"] / "game.py"
    if not game_py.exists():
        raise HTTPException(400, "no engine yet — generate one first")
    spec = importlib.util.spec_from_file_location(f"gen_{meta['id']}", game_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for obj in vars(mod).values():
        if isinstance(obj, type) and hasattr(obj, "initial_state"):
            return obj()
    raise HTTPException(500, "generated file has no engine class")


def _job(kind: str, pid: str) -> dict:
    j = {"id": uuid.uuid4().hex[:10], "kind": kind, "project": pid,
         "status": "running", "started": time.time(), "detail": "", "error": None}
    JOBS[j["id"]] = j
    return j


def _run_in_thread(job: dict, fn) -> None:
    def wrap():
        try:
            fn(job)
            job["status"] = "done"
        except Exception as e:  # surfaced to the UI, not swallowed
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
        job["elapsed"] = round(time.time() - job["started"], 1)
    threading.Thread(target=wrap, daemon=True).start()


# ------------------------------------------------------------------- models
class NewProject(BaseModel):
    name: str
    rulebook: str = ""
    builtin: str | None = None  # "splendor" | "agricola"


class SimRequest(BaseModel):
    games: int = 500
    players: int = 2
    agents: str = "scoregreedy"  # one spec for all seats, or comma list
    rotate: bool = True          # cycle agents through seats between games


# ---------------------------------------------------------------- endpoints
@app.get("/api/projects")
def list_projects():
    out = []
    for p in sorted(DATA.glob("*/meta.json")):
        out.append(json.loads(p.read_text()))
    return out


@app.post("/api/projects")
def create_project(req: NewProject):
    if req.builtin and req.builtin not in BUILTIN:
        raise HTTPException(400, f"unknown built-in {req.builtin!r}")
    if not req.builtin and not req.rulebook.strip():
        raise HTTPException(400, "provide a rulebook or pick a built-in engine")
    pid = uuid.uuid4().hex[:8]
    meta = {"id": pid, "name": req.name.strip() or "Untitled game",
            "builtin": req.builtin, "created": time.time(),
            "engine": "builtin" if req.builtin else "none",
            "validation": None, "has_report": False}
    _save(meta)
    if req.rulebook.strip():
        (DATA / pid / "rulebook.txt").write_text(req.rulebook)
    return meta


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _load(pid)


@app.post("/api/projects/{pid}/generate")
def generate(pid: str):
    meta = _load(pid)
    if meta.get("builtin"):
        raise HTTPException(400, "this project uses a built-in engine")
    rb = DATA / pid / "rulebook.txt"
    if not rb.exists():
        raise HTTPException(400, "no rulebook on this project")
    job = _job("generate", pid)

    def work(j):
        j["detail"] = "asking the model for an engine"
        code = llm.generate_engine(rb.read_text())
        (DATA / pid / "game.py").write_text(code)
        j["detail"] = "validating: random games with invariants on"
        game = _engine_for(meta)
        for seed in range(30):
            agents = [make_agent("random", seed * 10 + i) for i in range(2)]
            play_game(game, agents, seed, debug=True)
        meta["engine"] = "generated"
        meta["validation"] = "30 random games, invariants held on every action"
        _save(meta)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/simulate")
def run_sim(pid: str, req: SimRequest):
    meta = _load(pid)
    _engine_for(meta)  # fail fast with a clear message before starting a job
    specs = req.agents.split(",")
    if len(specs) == 1:
        specs = specs * req.players
    elif len(specs) != req.players:
        raise HTTPException(400, f"agents list has {len(specs)} seats but "
                                 f"players is {req.players} — make them match")
    if meta.get("builtin") != "splendor" and "expert" in specs:
        raise HTTPException(400, "the 'expert' agent is Splendor-only; use "
                                 "score-greedy for other games")
    job = _job("simulate", pid)

    def work(j):
        j["detail"] = f"playing {req.games} games"
        game_name = BUILTIN[meta["builtin"]] if meta.get("builtin") else None
        if game_name:
            records, secs = simulate(game_name, specs, req.games, seed=0,
                                     workers=2, rotate=req.rotate)
            game = make_game(game_name)
        else:  # generated engines: run in-process (no worker pool yet)
            game = _engine_for(meta)
            records = []
            t0 = time.time()
            for seed in range(req.games):
                order = list(range(len(specs)))
                if req.rotate:
                    k = seed % len(specs)
                    order = order[k:] + order[:k]
                agents = [make_agent(specs[j], seed * 100 + i)
                          for i, j in enumerate(order)]
                rec = play_game(game, agents, seed)
                rec.extra["seat_agent"] = [specs[j] for j in order]
                records.append(rec)
            secs = time.time() - t0
        j["detail"] = "writing the report"
        md = make_report(records, game)
        header = (f"_{req.games} games · {req.players} players · seats: "
                  f"{req.agents}{' · seats rotated' if req.rotate else ''} · "
                  f"{secs:.0f}s_\n\n")
        (DATA / pid / "report.md").write_text(header + md)
        meta["has_report"] = True
        _save(meta)

    _run_in_thread(job, work)
    return job


@app.get("/api/projects/{pid}/report")
def get_report(pid: str):
    p = DATA / pid / "report.md"
    if not p.exists():
        raise HTTPException(404, "no report yet — run a simulation first")
    return {"markdown": p.read_text()}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[jid]


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"),
          name="static")
