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
import re
import shutil
import json
import os
import sys
import threading
import traceback
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
from bgsim.trace import record_trace, print_schema
from web import sandbox
from bgsim.verify import load_checkers
from web import llm
from web.quality import quality_report, second_opinion

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
    return json.loads(p.read_text(encoding="utf-8"))


def _save(meta: dict) -> None:
    d = DATA / meta["id"]
    d.mkdir(exist_ok=True)
    rec = SPEND.get(meta["id"])
    if rec:
        meta["spend_usd"] = rec["usd"]
        meta["spend_by_model"] = rec["by_model"]
        meta["spend_ledger"] = rec.get("ledger", [])
    _meta_path(meta["id"]).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _engine_for(meta: dict, filename: str = "game.py"):
    if meta.get("builtin"):
        return make_game(BUILTIN[meta["builtin"]])
    game_py = DATA / meta["id"] / filename
    if not game_py.exists():
        raise HTTPException(400, "no engine yet — generate one first")
    spec = importlib.util.spec_from_file_location(f"gen_{meta['id']}_{filename[:-3]}", game_py)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclasses resolve annotations via sys.modules
    spec.loader.exec_module(mod)
    for obj in vars(mod).values():
        if isinstance(obj, type) and hasattr(obj, "initial_state"):
            return obj()
    raise HTTPException(500, "generated file has no engine class")


def _validate_file(pid: str, filename: str = "game.py", players=(2, 3, 4), seeds: int = 12) -> str | None:
    """Random games with invariants, in a killable child. None if clean, else
    the failure text (a traceback, or the hang explanation)."""
    if _load(pid).get("builtin"):
        return None
    try:
        return sandbox.run("validate", timeout=240, path=str(DATA / pid / filename), players=players, seeds=seeds)
    except sandbox.EngineHung as e:
        return str(e)


def _mark_stale(meta: dict) -> None:
    """After the engine changes, earlier checks describe a different engine."""
    for k in ("quality", "second_opinion"):
        if meta.get(k):
            meta[f"{k}_stale"] = True
    meta["needs_regeneration"] = False
    # strategies are weights over THIS engine's features: derived ones are dropped,
    # designer-described ones are kept but must be re-translated
    meta.pop("feature_names", None)
    meta.pop("balance_scan", None); meta.pop("matrix", None); meta.pop("counter_findings", None)
    meta.pop("competent_report", None); meta.pop("players", None); meta.pop("analysis", None)
    meta["strategies"] = [dict(st, stale=True) for st in meta.get("strategies", []) if st.get("source") == "designer"]


def _write_engine(pid: str, code: str, label: str) -> None:
    """Write game.py, archiving the previous version so any change is reversible."""
    d = DATA / pid
    cur = d / "game.py"
    vdir = d / "versions"
    vdir.mkdir(exist_ok=True)
    meta = _load(pid)
    versions = meta.get("versions", [])
    if cur.exists():
        n = len(versions)
        (vdir / f"v{n}.py").write_text(cur.read_text(encoding="utf-8"), encoding="utf-8")
        versions.append({"v": n, "label": meta.get("current_label", "engine"),
                         "time": time.time()})
    cur.write_text(code, encoding="utf-8")
    meta["versions"] = versions
    meta["current_label"] = label
    _mark_stale(meta)
    _save(meta)


MUTATING = {"generate", "checkers", "jury", "fix", "resolve_answer", "second_opinion",
            "workshop", "run_all"}


def _running_for(pid: str):
    return [j for j in JOBS.values() if j.get("project") == pid and j["status"] == "running"
            and j.get("kind") in MUTATING and not j.get("parent")]


def _job(kind: str, pid: str) -> dict:
    parent = getattr(_current, "job", None)
    nested = parent is not None and parent.get("project") == pid and parent["status"] == "running"
    if kind in MUTATING and not nested:   # a job may start its own sub-jobs
        busy = _running_for(pid)
        if busy:
            b = busy[0]
            raise HTTPException(409, f"'{b['kind']}' is already running on this game "
                                     f"({b.get('detail') or 'working'}); wait for it to finish")
    j = _StagedJob({"id": uuid.uuid4().hex[:10], "kind": kind, "project": pid,
                    "status": "running", "started": time.time(), "detail": "", "error": None})
    if nested:
        j["parent"] = parent["id"]
    JOBS[j["id"]] = j
    return j


DEV_MODE = os.environ.get("BGSIM_DEV", "1") != "0"   # production sets BGSIM_DEV=0
_current = threading.local()


def _check_cancel() -> None:
    j = getattr(_current, "job", None)
    if j is not None and j.get("cancel"):
        raise llm.Cancelled("stopped by the user")


llm.cancel_check = lambda: bool(getattr(_current, "job", None) and _current.job.get("cancel"))


def _progress(msg: str) -> None:
    j = getattr(_current, "job", None)
    if j is not None:
        dict.__setitem__(j, "progress", msg)   # not via detail: keep the stage label


llm.progress = _progress


def _on_cap(spent: float, cap: float):
    """Hold the job at a call boundary until the designer continues or stops.
    Work already done stays; nothing is re-run."""
    j = getattr(_current, "job", None)
    if j is None:
        return None
    dict.__setitem__(j, "paused", {"spent": round(spent, 2), "cap": round(cap, 2),
                                   "offer": round(cap + max(cap, 0.5), 2)})
    t0 = time.time()
    while time.time() - t0 < 6 * 3600:          # wait up to six hours for an answer
        if j.get("cancel"):
            dict.__setitem__(j, "paused", None)
            return None
        raised = j.get("cap_raised")
        if raised:
            dict.__setitem__(j, "paused", None); dict.__setitem__(j, "cap_raised", None)
            dict.__setitem__(j, "cap_usd", raised)
            return raised
        time.sleep(1)
    dict.__setitem__(j, "paused", None)
    return None


llm.on_cap = _on_cap


def learned_estimate(kind: str, tier: str) -> float:
    """Median cost of the last few finished runs of this kind (any project),
    once there are three; the table otherwise."""
    costs = []
    for rec in SPEND.values():
        by_job = {}
        for e in rec.get("ledger", []):
            k = e["stage"].split(":")[0]
            if k == kind:
                by_job.setdefault(int(e["time"] // 1800), 0.0)   # half-hour buckets ≈ one run
                by_job[int(e["time"] // 1800)] += e["usd"]
        costs += list(by_job.values())
    costs = sorted(costs)[-9:]
    if len(costs) >= 3:
        return round(1.15 * costs[len(costs) // 2], 2)
    return estimate_for(kind, tier)


# Rough per-run estimates in USD by kind and tier; the cap is 3x the estimate.
ESTIMATES = {
    "light": {"workshop": 0.15, "generate": 0.35, "checkers": 0.45, "jury": 0.60, "fix": 0.60,
              "resolve_answer": 0.60, "second_opinion": 0.50, "run_all": 1.60, "review": 0.15,
              "simulate": 0.0, "quality": 0.0},
    "heavy": {"workshop": 0.30, "generate": 1.50, "checkers": 0.90, "jury": 1.20, "fix": 1.50,
              "resolve_answer": 1.50, "second_opinion": 1.80, "run_all": 5.00, "review": 0.30,
              "simulate": 0.0, "quality": 0.0},
}


def estimate_for(kind: str, tier: str) -> float:
    return ESTIMATES.get(tier, ESTIMATES["light"]).get(kind, 0.50)


class _StagedJob(dict):
    """Job dict whose 'detail' updates also label model calls for the ledger."""
    def __setitem__(self, k, v):
        super().__setitem__(k, v)
        if k == "detail":
            super().__setitem__("progress", "")
            llm.set_stage(f"{self.get('kind', '?')}: {v}")


def _run_in_thread(job: dict, fn) -> None:
    def wrap():
        _current.job = job
        try:
            m = _load(job["project"]) if job.get("project") else {}
            cx = m.get("complexity") or {"tier": "light", "budget_usd": BUDGET_USD["light"]}
            est = learned_estimate(job.get("kind", ""), cx["tier"])
            job["estimate_usd"] = est
            job["cap_usd"] = round(3 * est, 2) if est > 0 else None
            if job.get("parent"):     # a sub-step spends against its parent's cap
                parent = JOBS.get(job["parent"])
                job["cap_usd"] = parent.get("cap_usd") if parent else None
                job["spent_usd"] = parent.get("spent_usd", 0.0) if parent else 0.0
            llm.configure(cx["tier"], job.get("project"), cx["budget_usd"], m.get("spend_usd", 0.0),
                          job_cap_usd=job["cap_usd"])
            if job.get("parent"):
                llm._ctx.job_spent = job.get("spent_usd", 0.0)
            llm.set_stage(f"{job.get('kind', '?')}")
            fn(job)
            job["status"] = "done"
        except llm.Cancelled:
            job["status"] = "cancelled"
            job["error"] = "stopped by the user"
        except Exception as e:  # surfaced to the UI, not swallowed
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
            job["trace"] = traceback.format_exc()
            print(job["trace"], file=sys.stderr)
        job["elapsed"] = round(time.time() - job["started"], 1)
    threading.Thread(target=wrap, daemon=True).start()


# ------------------------------------------------------------------- models
class NewProject(BaseModel):
    name: str
    rulebook: str = ""
    builtin: str | None = None  # "splendor" | "agricola"
    components: list[dict] = []  # optional [{"name": ..., "text": ...}] tables given up front


BUDGET_USD = {"light": 8.0, "heavy": 15.0}   # per-project model spend guard


def _complexity(meta: dict) -> dict:
    """Measured from the workshop's own outputs, no model call."""
    ws = meta.get("workshop") or {}
    rb = DATA / meta["id"] / "rulebook.txt"
    words = len(rb.read_text(encoding="utf-8").split()) if rb.exists() else 0
    actions = len(ws.get("cost_table", []))
    lines = sum(len(sec["lines"]) for sec in ws.get("outline", []))
    uncertain = len(ws.get("items", []))
    steps = len(ws.get("walk", []))
    score = words / 400 + actions * 1.5 + lines / 4 + uncertain + steps / 2
    tier = "heavy" if (words > 4000 or actions > 12 or lines > 60 or score > 45) else "light"
    return {"score": round(score, 1), "tier": tier, "words": words, "actions": actions,
            "outline_lines": lines, "uncertain_items": uncertain, "budget_usd": BUDGET_USD[tier]}


SPEND: dict[str, dict] = {}  # pid -> {"usd": float, "by_model": {...}}; merged into every _save


def _record_cost(pid: str, model: str, usd: float, info: dict | None = None) -> None:
    j = getattr(_current, "job", None)
    if j is not None:
        dict.__setitem__(j, "spent_usd", round(j.get("spent_usd", 0.0) + usd, 4))
        parent = JOBS.get(j.get("parent") or "")
        if parent is not None:
            dict.__setitem__(parent, "spent_usd", round(parent.get("spent_usd", 0.0) + usd, 4))
    rec = SPEND.setdefault(pid, {"usd": 0.0, "by_model": {}, "ledger": []})
    rec["usd"] = round(rec["usd"] + usd, 4)
    rec["by_model"][model] = round(rec["by_model"].get(model, 0.0) + usd, 4)
    info = info or {}
    rec.setdefault("ledger", []).append({
        "time": time.time(), "stage": info.get("stage", "?"), "model": model.replace("claude-", ""),
        "in": info.get("in", 0), "out": info.get("out", 0), "usd": round(usd, 4)})
    rec["ledger"] = rec["ledger"][-300:]


llm.on_cost = _record_cost


MAX_WORKSHOP_PASSES = 5
MAX_RULEBOOK_WORDS = 12000


class WorkshopAnswer(BaseModel):
    item_id: str
    answer: str = ""      # empty answer + confirm=True means "the assumption is right"
    confirm: bool = False


class Answers(BaseModel):
    answers: list[dict]  # [{"question": str, "answer": str}]


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
        out.append(json.loads(p.read_text(encoding="utf-8")))
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
        (DATA / pid / "rulebook.txt").write_text(req.rulebook, encoding="utf-8")
    for comp in req.components:
        name, text = str(comp.get("name", "")).strip(), str(comp.get("text", "")).strip()
        if name and "\n" in text:
            upload_component(pid, ComponentUpload(name=name, text=text))
    return meta


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    return _load(pid)


@app.get("/api/projects/{pid}/rulebook")
def get_rulebook(pid: str):
    _load(pid)
    rb = DATA / pid / "rulebook.txt"
    return {"text": rb.read_text(encoding="utf-8") if rb.exists() else ""}


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    """Remove a game and everything under it. Refuses while a job is running."""
    _load(pid)
    if _running_for(pid):
        raise HTTPException(409, "a job is running on this game — stop it first")
    shutil.rmtree(DATA / pid, ignore_errors=True)
    SPEND.pop(pid, None)
    return {"deleted": pid}


# ======================= clarifications (undoable) ========================
CLAR_HEADER = "## Clarifications from the designer"


def _rebuild_clarifications(pid: str, meta: dict) -> None:
    """The rulebook's clarifications section is always regenerated from the
    structured store, so removing a record removes its text."""
    rb = DATA / pid / "rulebook.txt"
    text = rb.read_text(encoding="utf-8")
    if CLAR_HEADER in text:
        text = text[:text.index(CLAR_HEADER)].rstrip()
    recs = meta.get("clarifications_store", [])
    if recs:
        text += "\n\n" + CLAR_HEADER + "\n"
        for r in recs:
            text += f"- {r['question']}\n  Designer: {r['answer']}\n"
    rb.write_text(text.rstrip() + "\n", encoding="utf-8")


def _set_clarification(pid: str, meta: dict, key: str, question: str, answer: str) -> None:
    store = [r for r in meta.get("clarifications_store", []) if r["key"] != key]
    store.append({"key": key, "question": question, "answer": answer, "time": time.time()})
    meta["clarifications_store"] = store
    _rebuild_clarifications(pid, meta)


def _drop_clarification(pid: str, meta: dict, key: str) -> None:
    meta["clarifications_store"] = [r for r in meta.get("clarifications_store", []) if r["key"] != key]
    _rebuild_clarifications(pid, meta)


# ============================ component data ===============================

def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:60] or "component"


def _component_section(pid: str) -> str:
    """The 'Component data' section appended to the rulebook for every model
    call: the designer's tables, verbatim."""
    meta = _load(pid)
    comps = meta.get("component_data", {})
    if not comps:
        return ""
    out = ["", "", "## Component data", "Authoritative tables supplied by the designer.", ""]
    for name, rec in comps.items():
        txt = (DATA / pid / "components" / rec["file"]).read_text(encoding="utf-8")
        out += [f"### {name} ({rec['rows']} rows)", "```", txt.strip()[:40_000], "```", ""]
    return "\n".join(out)


def _component_summary(pid: str) -> str:
    """Short form for the workshop: names and row counts only."""
    comps = _load(pid).get("component_data", {})
    if not comps:
        return ""
    return "\n\n## Component data\n" + "\n".join(
        f"- {n}: {r['rows']} rows supplied by the designer (values listed)" for n, r in comps.items()) + "\n"


def _rulebook_for_model(pid: str) -> str:
    return (DATA / pid / "rulebook.txt").read_text(encoding="utf-8") + _component_section(pid)


def _refresh_readiness(meta: dict) -> None:
    ws = meta.get("workshop")
    if ws:
        ws["readiness"] = _readiness(ws, meta)


def _missing_components(meta: dict) -> list[dict]:
    """Component sets the workshop found unlisted and the designer has neither
    uploaded nor waived."""
    ws = meta.get("workshop") or {}
    have = set(meta.get("component_data", {}))
    waived = set(meta.get("components_waived", []))
    return [c for c in ws.get("components", []) if not c.get("listed")
            and c["name"] not in have and c["name"] not in waived]


class ComponentUpload(BaseModel):
    name: str
    text: str


@app.post("/api/projects/{pid}/components")
def upload_component(pid: str, req: ComponentUpload):
    """Save a table (CSV or a pasted table) for one component set."""
    meta = _load(pid)
    txt = req.text.strip()
    if not txt or "\n" not in txt:
        raise HTTPException(400, "paste the whole table — a header row plus one row per item")
    if len(txt) > 200_000:
        raise HTTPException(400, "table too large (200k characters max)")
    cdir = DATA / pid / "components"; cdir.mkdir(exist_ok=True)
    fname = _slug(req.name) + ".csv"
    (cdir / fname).write_text(txt + "\n", encoding="utf-8")
    meta.setdefault("component_data", {})[req.name] = {
        "file": fname, "rows": max(0, txt.count("\n")), "preview": "\n".join(txt.splitlines()[:4])}
    meta.setdefault("components_waived", [])
    if req.name in meta["components_waived"]:
        meta["components_waived"].remove(req.name)
    _refresh_readiness(meta)
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/components/waive")
def waive_component(pid: str, req: WorkshopAnswer):
    """Designer says the values are meant to be random/invented: not a blocker."""
    meta = _load(pid)
    meta.setdefault("components_waived", [])
    if req.item_id not in meta["components_waived"]:
        meta["components_waived"].append(req.item_id)
    _refresh_readiness(meta)
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/components/unwaive")
def unwaive_component(pid: str, req: WorkshopAnswer):
    meta = _load(pid)
    meta["components_waived"] = [n for n in meta.get("components_waived", []) if n != req.item_id]
    _refresh_readiness(meta)
    _save(meta)
    return meta


@app.delete("/api/projects/{pid}/components/{name}")
def delete_component(pid: str, name: str):
    meta = _load(pid)
    rec = meta.get("component_data", {}).pop(name, None)
    if rec:
        (DATA / pid / "components" / rec["file"]).unlink(missing_ok=True)
    _refresh_readiness(meta)
    _save(meta)
    return meta


# ============================ rules workshop ==============================
def _readiness(ws: dict, meta: dict | None = None) -> int:
    items = ws.get("items", [])
    missing = len(_missing_components(meta)) if meta else 0
    total = len(items) + missing
    if not total:
        return 100
    done = sum(1 for it in items if it["status"] != "open")
    return int(round(100 * done / total))


@app.post("/api/projects/{pid}/workshop")
def run_workshop(pid: str):
    """Phase A: restate, walk a turn, ask + triage. Produces one open-items list."""
    meta = _load(pid)
    rb = DATA / pid / "rulebook.txt"
    if not rb.exists():
        raise HTTPException(400, "no rulebook on this project")
    ws = meta.get("workshop") or {"pass": 0, "items": [], "history": []}
    if ws["pass"] >= MAX_WORKSHOP_PASSES:
        raise HTTPException(400, f"this rulebook has used its {MAX_WORKSHOP_PASSES} workshop "
                                 f"passes; {sum(1 for i in ws['items'] if i['status']=='open')} "
                                 f"item(s) still open usually means the rulebook needs an edit")
    text = rb.read_text(encoding="utf-8")
    if len(text.split()) > MAX_RULEBOOK_WORDS:
        raise HTTPException(400, f"rulebook is over {MAX_RULEBOOK_WORDS:,} words")
    job = _job("workshop", pid)

    def work(j):
        rulebook = rb.read_text(encoding="utf-8") + _component_summary(pid)
        items = []
        j["detail"] = "restating the rules as a structured outline"
        restated = llm.restate_rules(rulebook)
        for sec in restated["outline"]:
            for ln in sec["lines"]:
                if ln["basis"] in ("inferred", "unclear"):
                    items.append({"kind": ln["basis"], "source": "outline",
                                  "where": sec["section"], "text": ln["text"],
                                  "assumption": ln["assumption"], "quote": ln["quote"]})
        for row in restated["cost_table"]:
            if row["basis"] in ("inferred", "unclear"):
                items.append({"kind": row["basis"], "source": "cost table", "where": row["action"],
                              "text": f"{row['action']}: costs {row['cost']}; effect: {row['effect']}",
                              "assumption": "", "quote": ""})
        ws["components"] = restated.get("components", [])
        j["detail"] = "walking through a sample round"
        steps = llm.walk_turn(rulebook)
        for st in steps:
            if st["assumption"]:
                items.append({"kind": "inferred", "source": "sample round", "where": "",
                              "text": st["text"], "assumption": st["assumption"], "quote": ""})
        j["detail"] = "looking for gaps and contradictions"
        review = llm.review_rules(rulebook)
        for r in review:
            items.append({"kind": r["kind"], "source": "review", "where": "", "text": r["question"],
                          "assumption": "", "quote": r["quote"]})
        # gatekeeper: one cheap pass classifies every candidate before the designer sees it
        j["detail"] = "checking which items are real rules questions"
        qs = [(it["assumption"] or it["text"]) if it["source"] != "review" else it["text"] for it in items]
        try:
            triage = llm.triage_questions(rulebook, qs) if qs else []
        except Exception as e:   # the gatekeeper is a convenience, never a reason to lose the pass
            print(f"gatekeeper failed, showing all items: {e}", file=sys.stderr)
            triage = [{"kind": "gap", "answered": False, "answer": "", "quote": ""} for _ in qs]
            ws["gatekeeper_note"] = "the automatic filter failed this pass, so every item is shown"
        auto = 0
        gate_labels = {"example": "(example choice, not a rule)", "design": "(design question, not a rule)",
                       "duplicate": "(duplicate of another item)"}
        for it, t in zip(items, triage):
            if t["kind"] == "answered":
                auto += 1
                it["status"] = "auto"; it["answer"] = t["answer"]; it["quote"] = t["quote"] or it.get("quote", "")
                if it["source"] == "review":
                    it["assumption"] = t["answer"]
            elif t["kind"] in gate_labels:
                it["status"] = "dismissed"; it["answer"] = gate_labels[t["kind"]]
        # ids + default status
        for k, it in enumerate(items):
            it.setdefault("status", "open")
            it.setdefault("answer", "")
            it["id"] = f"p{ws['pass'] + 1}-{k}"
        # example choices that slipped through the prompt are not rules questions
        junk = ("chose", "arbitrarily", "specifically", "assumed the existence", "example card",
                "exact card", "picked", "since any combination", "since any colour")
        for it in items:
            if it["source"] == "sample round" and any(w in it["assumption"].lower() for w in junk):
                it["status"] = "dismissed"; it["answer"] = "(example choice, not a rule)"
        # carry forward: anything the designer already settled stays settled,
        # matched by content because the model rephrases between passes
        import difflib
        prior = [i for i in ws.get("items", []) if i["status"] in ("confirmed", "answered", "dismissed")]
        def key(i):
            return (i.get("text", "") + " " + i.get("assumption", "")).lower()
        carried = 0
        for it in items:
            if it["status"] != "open":
                continue
            best, score = None, 0.0
            for pr in prior:
                r = difflib.SequenceMatcher(None, key(it), key(pr)).ratio()
                if r > score:
                    best, score = pr, r
            if best and score >= 0.55:
                it["status"] = best["status"]; it["answer"] = best["answer"]
                it["clar_key"] = best.get("clar_key", f"ws:{best['id']}")
                carried += 1
        # anything settled before that the model no longer raises stays in the
        # list as settled, so the count and the undo list remain honest
        seen = {key(i) for i in items}
        for pr in prior:
            if not any(difflib.SequenceMatcher(None, key(pr), k).ratio() >= 0.55 for k in seen):
                pr = dict(pr); pr["id"] = f"p{ws['pass'] + 1}-kept-{pr['id']}"
                pr.setdefault("clar_key", f"ws:{prior[0]['id']}" if False else pr.get("clar_key", f"ws:{pr['id'].split('-kept-')[-1]}"))
                items.append(pr)
        ws["pass"] += 1
        ws["items"] = items
        ws["outline"] = restated["outline"]
        ws["cost_table"] = restated["cost_table"]
        ws["walk"] = steps
        ws["history"].append({"pass": ws["pass"], "open": sum(1 for i in items if i["status"] == "open"),
                              "auto_answered": auto, "carried": carried, "cost": llm.cost_note()})
        ws["readiness"] = _readiness(ws, meta)
        meta["workshop"] = ws
        meta["complexity"] = _complexity(meta)
        _save(meta)

    _run_in_thread(job, work)
    return job


class Budget(BaseModel):
    budget_usd: float


@app.post("/api/projects/{pid}/budget")
def set_budget(pid: str, req: Budget):
    meta = _load(pid)
    cx = meta.get("complexity") or {"tier": "light", "score": 0}
    cx["budget_usd"] = max(0.0, float(req.budget_usd))
    meta["complexity"] = cx
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/workshop/answer")
def workshop_answer(pid: str, req: WorkshopAnswer):
    """Confirm an assumption or answer an item; the result is appended to the
    rulebook as a clarification in the designer's words."""
    meta = _load(pid)
    ws = meta.get("workshop")
    if not ws:
        raise HTTPException(400, "run the workshop first")
    it = next((i for i in ws["items"] if i["id"] == req.item_id), None)
    if not it:
        raise HTTPException(404, "no such item")
    if req.confirm and not req.answer.strip():
        if not it.get("assumption"):
            raise HTTPException(400, "nothing to confirm — type an answer")
    if req.answer.strip():
        import difflib
        q = (it.get("text") or "").strip().lower(); ans = req.answer.strip().lower()
        # only a near-verbatim restatement, or something phrased as a question, is refused
        if ans.endswith("?") or (q and difflib.SequenceMatcher(None, q, ans).ratio() > 0.95):
            raise HTTPException(400, "that's the question again, not an answer — state what the rule is")
        it["status"] = "confirmed"; it["answer"] = it["assumption"]
    elif req.answer.strip():
        it["status"] = "answered"; it["answer"] = req.answer.strip()
    else:
        raise HTTPException(400, "type an answer or confirm the assumption")
    q = it["text"] if it["source"] == "review" else (it["assumption"] or it["text"])
    it.setdefault("clar_key", f"ws:{it['id']}")
    _set_clarification(pid, meta, it["clar_key"], q, it["answer"])
    ws["readiness"] = _readiness(ws, meta)
    meta["workshop"] = ws
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/workshop/undo")
def workshop_undo(pid: str, req: WorkshopAnswer):
    """Reopen a settled workshop item and remove its clarification from the rulebook."""
    meta = _load(pid); ws = meta.get("workshop") or {}
    it = next((i for i in ws.get("items", []) if i["id"] == req.item_id), None)
    if not it:
        raise HTTPException(404, "no such item")
    it["status"] = "open"; it["answer"] = ""; it.pop("deferred", None)
    _drop_clarification(pid, meta, it.get("clar_key", f"ws:{it['id']}"))
    ws["readiness"] = _readiness(ws, meta)
    meta["workshop"] = ws
    _save(meta)
    return meta


class Clarify(BaseModel):
    text: str


@app.post("/api/projects/{pid}/workshop/clarify")
def workshop_clarify(pid: str, req: Clarify):
    """A rule the designer adds on their own initiative (e.g. from a report finding)."""
    meta = _load(pid)
    text = req.text.strip()
    if len(text) < 8:
        raise HTTPException(400, "write the rule out")
    key = f"designer:{int(time.time())}"
    _set_clarification(pid, meta, key, "Designer-added rule", text)
    if meta.get("engine") == "generated":
        meta["needs_regeneration"] = True
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/workshop/unclarify")
def workshop_unclarify(pid: str, req: WorkshopAnswer):
    meta = _load(pid)
    if not any(r["key"] == req.item_id for r in meta.get("clarifications_store", [])):
        raise HTTPException(404, "no such rule")
    _drop_clarification(pid, meta, req.item_id)
    if meta.get("engine") == "generated":
        meta["needs_regeneration"] = True
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/workshop/dismiss")
def workshop_dismiss(pid: str, req: WorkshopAnswer):
    """'Not a rules question' — settles the item without a clarification."""
    meta = _load(pid); ws = meta.get("workshop") or {}
    it = next((i for i in ws.get("items", []) if i["id"] == req.item_id), None)
    if not it:
        raise HTTPException(404, "no such item")
    it["status"] = "dismissed"; it["answer"] = "(not a rules question)"; it.pop("deferred", None)
    ws["readiness"] = _readiness(ws, meta)
    meta["workshop"] = ws
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/workshop/reset-passes")
def workshop_reset_passes(pid: str):
    if not DEV_MODE:
        raise HTTPException(403, "disabled in production")
    meta = _load(pid); ws = meta.get("workshop") or {}
    ws["pass"] = 0
    meta["workshop"] = ws
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/workshop/skip")
def workshop_skip(pid: str, req: WorkshopAnswer):
    """'Not sure' — leaves the item open but marks it deferred so it is not nagging."""
    meta = _load(pid); ws = meta.get("workshop") or {}
    it = next((i for i in ws.get("items", []) if i["id"] == req.item_id), None)
    if not it:
        raise HTTPException(404, "no such item")
    it["deferred"] = True
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/review")
def review(pid: str):
    """Stage 1: the model reads the rulebook and returns questions for the designer."""
    meta = _load(pid)
    rb = DATA / pid / "rulebook.txt"
    if not rb.exists():
        raise HTTPException(400, "no rulebook on this project")
    job = _job("review", pid)

    def work(j):
        j["detail"] = "reading the rulebook for gaps, ambiguities and contradictions"
        meta["review"] = llm.review_rules(rb.read_text(encoding="utf-8"))
        meta["review_cost"] = llm.cost_note()
        meta["clarified"] = False
        _save(meta)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/clarify")
def clarify(pid: str, req: Answers):
    """Append the designer's answers to the rulebook as clarifications."""
    meta = _load(pid)
    rb = DATA / pid / "rulebook.txt"
    answered = [a for a in req.answers if str(a.get("answer", "")).strip()]
    for k, a in enumerate(answered):
        _set_clarification(pid, meta, f"review:{k}:{a['question'][:40]}", a["question"].strip(), a["answer"].strip())
    meta["clarified"] = True
    meta["clarifications"] = len(answered)
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/checkers")
def build_checkers(pid: str, force: bool = False):
    """Stage 3: compile the rulebook into independent checkers and run them.
    force=True replaces an existing set (old files archived, rulings kept)."""
    meta = _load(pid)
    game = _engine_for(meta)
    rb = DATA / pid / "rulebook.txt"
    if not rb.exists():
        raise HTTPException(400, "no rulebook on this project")
    job = _job("checkers", pid)

    def work(j):
        rulebook = _rulebook_for_model(pid)
        cdir = DATA / pid / "checkers"
        cdir.mkdir(exist_ok=True)
        if force:
            old = DATA / pid / f"checkers_old_{int(time.time())}"
            old.mkdir(exist_ok=True)
            for f in list(cdir.glob("*.py")):
                f.rename(old / f.name)
            m0 = _load(pid)
            m0["retired_auditors"] = []; m0["checkers"] = []; m0["checker_questions"] = {}
            _save(m0)
        j["detail"] = "recording a sample game for the trace format"
        schema = (print_schema(record_trace(game, ["random", "random"], 2, 0)) if meta.get("builtin")
                  else sandbox.run("schema", timeout=120, path=str(DATA / pid / "game.py")))
        j["detail"] = "splitting the rulebook into auditable sections"
        sections = llm.plan_checkers(rulebook)
        for k, sec in enumerate(sections):
            j["detail"] = f"writing checker {k + 1}/{len(sections)}: {sec['name']}"
            (cdir / f"{sec['name']}.py").write_text(
                llm.write_checker(sec["name"], sec["text"], schema), encoding="utf-8")
        j["detail"] = "auditing 30 games with the new checkers"
        run_checkers(meta, game, cdir)
        meta["checkers_cost"] = llm.cost_note()
        _save(meta)

    _run_in_thread(job, work)
    return job


SPECS_BY_SEED = [["random", "random"], ["scoregreedy", "scoregreedy"],
                 ["random", "random", "random", "random"]]


def _narrate_moment(game, finding: str, before: int = 4, after: int = 2) -> str:
    """Replay the flagged game and narrate the steps around the flagged one in
    the engine's own plain-language voice (describe_state / describe_action).
    This is what the designer reads, so it must contain no code terms."""
    import re
    g = re.search(r"game (\d+)", finding); st = re.search(r"step (\d+)", finding)
    if not g or not st:
        return ""
    seed = int(g.group(1)); step = int(st.group(1))
    specs = SPECS_BY_SEED[seed % len(SPECS_BY_SEED)]
    ds = getattr(game, "describe_state", None); da = getattr(game, "describe_action", None)
    if not (ds and da):
        return ""
    try:
        agents = [make_agent(sp, (500 + seed) * 100 + i) for i, sp in enumerate(specs)]
        state = game.initial_state(len(specs), 500 + seed)
        lines, i = [], 0
        while not game.is_terminal(state) and i <= step + after:
            p = game.current_player(state)
            action = agents[p].act(game, state, p)
            if i >= step - before:
                mark = "  <-- the flagged moment" if i == step else ""
                lines.append(f"{ds(state)}\n   Player {p + 1}: {sandbox._desc(da, state, action)}{mark}")
            state = game.apply(state, action)
            i += 1
        return "\n".join(lines)
    except Exception as e:
        return f"(could not replay the moment: {e})"


def _step_record(game, finding: str) -> str:
    """Action + engine description + state diff at the step a finding names."""
    import re
    from bgsim.trace import state_diff
    g = re.search(r"game (\d+)", finding); st = re.search(r"step (\d+)", finding)
    if not g or not st:
        return ""
    seed = int(g.group(1)); step = int(st.group(1))
    specs = SPECS_BY_SEED[seed % len(SPECS_BY_SEED)]
    try:
        tr = record_trace(game, specs, len(specs), 500 + seed)
        s = tr["steps"][step]
        return (f"action: {s['action']}\nengine says: {s.get('describe', '?')}\n"
                f"state changes: {state_diff(s['before'], s['after'])}")
    except Exception as e:
        return f"(could not replay: {e})"


def _verify_auditor_claims(meta: dict, game) -> None:
    """Before findings reach the designer or the jury: check a sample of each
    auditor's claims against the record. An auditor whose claims the record
    contradicts is marked suspect, not presented as a finding."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return
    for c in meta.get("checkers") or []:
        if not c["findings"]:
            continue
        verdicts = []
        for f in c["findings"][:3]:
            rec = _step_record(game, f)
            if not rec or rec.startswith("("):
                continue
            try:
                verdicts.append(llm.verify_claim(f, rec))
            except Exception:
                continue
        if verdicts and all(v["contradicted"] for v in verdicts):
            c["suspect"] = verdicts[0]["why"]


def _moment_context(game, finding: str) -> str:
    """Re-record the flagged game and describe what the engine was doing at
    the flagged step: phase, acting player, action. Grounds the jury's framing
    in the record rather than in the auditor's wording."""
    import re
    g = re.search(r"game (\d+)", finding); st = re.search(r"step (\d+)", finding)
    if not g or not st:
        return ""
    seed = int(g.group(1)); step = int(st.group(1))
    specs = SPECS_BY_SEED[seed % len(SPECS_BY_SEED)]
    try:
        tr = record_trace(game, specs, len(specs), 500 + seed)
        s = tr["steps"][step]
        nxt = tr["steps"][step + 1] if step + 1 < len(tr["steps"]) else None
        from bgsim.trace import state_diff
        out = (f"At the flagged step the phase was '{s['phase']}', player {s['player']} was acting, "
               f"and the action taken was {s['action']}; afterwards the phase was '{s['after'].get('phase', '?')}'.\n"
               f"WHAT ACTUALLY CHANGED in the state at that step (this is the record, more reliable than "
               f"any description of it): {state_diff(s['before'], s['after'])}")
        if nxt:
            out += (f" The very next step was phase '{nxt['phase']}', player {nxt['player']}, "
                    f"action {nxt['action']}.")
        ds = getattr(game, "describe_state", None)
        return out
    except Exception as e:
        return f"(could not re-record the moment: {e})"


def _audit_inprocess(game, cdir, n_games):
    """Built-in engines are trusted to terminate; audit them in-process."""
    checkers = load_checkers(cdir)
    names = [getattr(c, "NAME", c.__checker_file__) for c in checkers]
    results = {n: {"rule": getattr(c, "RULE", ""), "gaps": [], "findings": []} for n, c in zip(names, checkers)}
    hit = {n: set() for n in names}
    for seed in range(n_games):
        _check_cancel()
        specs = SPECS_BY_SEED[seed % len(SPECS_BY_SEED)]
        tr = record_trace(game, specs, len(specs), 500 + seed)
        for name, c in zip(names, checkers):
            try:
                msgs = c.check(tr) or []
            except Exception as e:
                msgs = [f"CHECKER CRASHED: {type(e).__name__}: {e}"]
            for m in msgs:
                bucket = "gaps" if m.startswith("SCHEMA GAP") else "findings"
                if bucket == "findings":
                    hit[name].add(seed)
                if m not in results[name][bucket]:
                    results[name][bucket].append(m if bucket == "gaps" else f"game {seed}: {m}")
    return results, {n: len(v) for n, v in hit.items()}


def run_checkers(meta: dict, game, cdir: Path, n_games: int = 30) -> None:
    if meta.get("builtin"):
        path = None
    else:
        path = str(DATA / meta["id"] / "game.py")
    if path:
        jb = getattr(_current, "job", None)
        results, games_hit = sandbox.run("audit", timeout=600, path=path, checker_dir=str(cdir),
                                         n_games=n_games, specs_by_seed=SPECS_BY_SEED,
                                         on_progress=(lambda t: dict.__setitem__(jb, "progress", f"games {t}")) if jb is not None else None)
    else:
        results, games_hit = _audit_inprocess(game, cdir, n_games)
    rulings = meta.get("rulings", {})
    meta["checkers"] = []
    for n, r in results.items():
        entry = {"name": n, "rule": r["rule"], "gaps": r["gaps"][:3],
                 "findings": r["findings"][:8], "n_findings": len(r["findings"]),
                 "games_hit": games_hit[n], "games_total": n_games}
        if r["findings"] and n in rulings:   # the designer already ruled: never ask again
            entry.update({"decision": rulings[n]["decision"], "note": rulings[n].get("note", ""),
                          "applied": True, "attempts": rulings[n].get("attempts", 0), "stuck": True})
        meta["checkers"].append(entry)
    meta["checkers_games"] = n_games
    # plain-language questions for every auditor that found something
    _verify_auditor_claims(meta, game)
    with_findings = {n: r["findings"] for n, r in results.items()
                     if r["findings"] and not any(c.get("suspect") for c in meta["checkers"] if c["name"] == n)}
    if with_findings and os.environ.get("ANTHROPIC_API_KEY"):
        rb = DATA / meta["id"] / "rulebook.txt"
        if rb.exists():
            try:
                moments = {n: _narrate_moment(game, f[0]) for n, f in with_findings.items()}
                meta["checker_questions"] = llm.explain_findings(
                    rb.read_text(encoding="utf-8"), with_findings, moments)
                for c in meta["checkers"]:   # the explainer's verdict counts like the record check
                    q = meta["checker_questions"].get(c["name"], {})
                    if isinstance(q, dict) and q.get("record_contradicts_auditor") and not c.get("suspect"):
                        c["suspect"] = q.get("simulation_did", "the record contradicts the auditor's account")[:300]
            except Exception as e:  # explanations are a convenience, never a blocker
                meta["checker_questions"] = {"_error": str(e)}
    else:
        meta["checker_questions"] = {}


@app.post("/api/projects/{pid}/checkers/rerun")
def rerun_checkers(pid: str):
    meta = _load(pid)
    cdir = DATA / pid / "checkers"
    if not cdir.exists():
        raise HTTPException(400, "no checkers yet")
    job = _job("checkers", pid)

    def work(j):
        j["detail"] = "auditing 30 games"
        run_checkers(meta, _engine_for(meta), cdir)
        _save(meta)

    _run_in_thread(job, work)
    return job


class Resolution(BaseModel):
    name: str                 # auditor name
    decision: str             # "engine" | "auditor" | "rulebook"
    note: str = ""            # clarification text when decision == "rulebook"


@app.post("/api/projects/{pid}/checkers/decide")
def checkers_decide(pid: str, req: Resolution):
    """Record the designer's decision on a finding without running anything.
    'Apply decisions' later takes them all in one repair + one re-audit."""
    meta = _load(pid)
    entry = next((c for c in meta.get("checkers") or [] if c["name"] == req.name), None)
    if not entry:
        raise HTTPException(404, "no such auditor")
    if req.decision not in ("sim_right", "auditor_right", "answer", "rulebook", "engine", "auditor"):
        raise HTTPException(400, "unknown decision")
    if req.decision in ("answer", "rulebook") and not req.note.strip():
        raise HTTPException(400, "write it first")
    q = meta.get("checker_questions", {}).get(req.name, {})
    entry["decision"] = req.decision; entry["note"] = req.note.strip(); entry["applied"] = False
    entry.pop("stuck", None)
    meta.setdefault("rulings", {})[req.name] = {"decision": req.decision, "note": req.note.strip(), "attempts": 0}
    # the designer's statement of the rule goes into the rulebook straight away
    stmt = {"sim_right": q.get("answer_if_simulation_right", ""),
            "auditor_right": q.get("answer_if_auditor_right", ""),
            "answer": req.note.strip(), "rulebook": req.note.strip()}.get(req.decision, "")
    if stmt:
        _set_clarification(pid, meta, f"finding:{req.name}", q.get("question") or entry["rule"], stmt)
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/checkers/undecide")
def checkers_undecide(pid: str, req: Resolution):
    meta = _load(pid)
    entry = next((c for c in meta.get("checkers") or [] if c["name"] == req.name), None)
    if not entry or (entry.get("applied") and not entry.get("stuck")):
        raise HTTPException(400, "nothing to undo")
    entry.pop("decision", None); entry.pop("note", None); entry.pop("applied", None); entry.pop("stuck", None)
    meta.get("rulings", {}).pop(req.name, None)
    _drop_clarification(pid, meta, f"finding:{req.name}")
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/fix-invariant")
def fix_invariant(pid: str):
    """The engine failed its own invariant in the quality check: find a
    failing game, hand the traceback to the repair loop (verified as usual)."""
    meta = _load(pid)
    if meta.get("builtin") or meta.get("engine") != "generated":
        raise HTTPException(400, "no generated engine")
    job = _job("fix", pid)

    def work(j):
        j["detail"] = "searching for a game where the engine's invariant fails"
        tb = sandbox.run("find_failure", timeout=600, path=str(DATA / pid / "game.py"),
                         on_progress=lambda t: dict.__setitem__(j, "progress", f"games {t}"))
        if not tb:
            j["detail"] = "no failure found in 1,200 random games"
            m = _load(pid); m["quality_stale"] = True; _save(m)
            return
        m = _load(pid)
        _repair_and_reaudit(pid, m, j, (
            "The engine crashed on its OWN invariant check during play (so the state became one it "
            "considers impossible). Find the action path that corrupts the state and fix it; do not "
            "weaken the invariant.\n" + tb[-3000:]), [], "invariant failure")
        j["detail"] = "re-checking engine quality"
        m = _load(pid)
        m["quality"] = sandbox.run("quality", timeout=900, path=str(DATA / pid / "game.py"))
        m["quality_stale"] = False
        _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/checkers/retry")
def checkers_retry(pid: str, req: Resolution):
    meta = _load(pid)
    entry = next((c for c in meta.get("checkers") or [] if c["name"] == req.name), None)
    if not entry or not entry.get("stuck"):
        raise HTTPException(400, "nothing to retry")
    entry["applied"] = False; entry.pop("stuck", None)
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/checkers/apply")
def checkers_apply(pid: str):
    """Take every decided-but-unapplied finding: dismiss the auditors the
    designer ruled against, and fix the engine ONCE for everything else."""
    meta = _load(pid)
    pending = [c for c in meta.get("checkers") or [] if c.get("decision") and not c.get("applied")]
    if not pending:
        raise HTTPException(400, "no decisions to apply")
    cdir = DATA / pid / "checkers"
    job = _job("fix", pid)

    def work(j):
        m = _load(pid)
        to_fix, dismissed = [], []
        for c in [c for c in m.get("checkers") or [] if c.get("decision") and not c.get("applied")]:
            d = c["decision"]
            if d == "answer":   # the designer stated a fact: work out who it sides with
                j["detail"] = f"comparing your answer for {c['name']}"
                g = _engine_for(m)
                ctx = _moment_context(g, c["findings"][0]) if c["findings"] else ""
                fr = llm.frame_finding(c["name"], c["rule"], c["findings"], ctx)
                side = llm.compare_answer(fr["engine_value"], fr["auditor_value"], c["note"])
                d = "sim_right" if side == "A" else "auditor_right"
                c["resolved_as"] = d
            c["applied"] = True
            if d in ("sim_right", "auditor"):
                dismissed.append(c["name"])
                f = cdir / f"{c['name']}.py"
                if f.exists():
                    f.rename(cdir / f"_{c['name']}.py")
                m.setdefault("retired_auditors", []).append(c["name"])
                m.setdefault("jury_trail", []).append({"auditor": c["name"], "time": time.time(),
                    "verdict": "you ruled for the simulation — auditor dismissed"})
            else:
                to_fix.append(c)
        m["checkers"] = [c for c in m["checkers"] if c["name"] not in dismissed]
        _save(m)
        if not to_fix:
            return
        reason = "The designer has ruled on these; fix ALL of them:\n"
        findings = []
        for c in to_fix:
            stmt = c.get("note") or m.get("checker_questions", {}).get(c["name"], {}).get("answer_if_auditor_right", "")
            reason += f"\n[{c['name']}] rule: {c['rule']}\n  designer's ruling: {stmt or 'the auditor is right'}"
            findings += c["findings"][:6]
        names = [c["name"] for c in to_fix]
        for rnd in range(2):
            for n in names:
                m.setdefault("rulings", {}).setdefault(n, {})["attempts"] = m["rulings"][n].get("attempts", 0) + 1
            _save(m)
            _repair_and_reaudit(pid, m, j, reason, findings, ", ".join(names))
            m = _load(pid)
            still = [c["name"] for c in m.get("checkers") or [] if c["name"] in names and c["n_findings"]]
            for n in names:
                if n not in still:
                    m.setdefault("jury_trail", []).append({"auditor": n, "time": time.time(),
                        "verdict": "you ruled against the simulation — fixed"})
            _save(m)
            if not still or rnd == 1:
                for n in still:
                    m.setdefault("jury_trail", []).append({"auditor": n, "time": time.time(),
                        "verdict": "you ruled against the simulation, but two repairs did not clear the "
                                   "auditor — the ruling is kept; press 'try the fix again' or look at the diff"})
                _save(m)
                break
            j["detail"] = f"the fix did not clear: {', '.join(still)} — trying again"
            names = still
            reason = ("PREVIOUS FIX ATTEMPT DID NOT WORK: the auditor still reports the same violations "
                      "after your change. Re-read the rule and change the code path that actually "
                      "produces the state the auditor inspects.\n" + reason)
            findings = [f for c in m["checkers"] if c["name"] in still for f in c["findings"][:6]]
            _refresh = None

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/checkers/resolve")
def resolve_finding(pid: str, req: Resolution):
    """The designer's verdict on an auditor's finding routes the fix."""
    meta = _load(pid)
    entry = next((c for c in meta.get("checkers", []) if c["name"] == req.name), None)
    if not entry:
        raise HTTPException(404, "no such auditor")
    cdir = DATA / pid / "checkers"
    q = meta.get("checker_questions", {}).get(req.name, {})
    if req.decision in ("sim_right", "auditor_right"):
        chosen = q.get("answer_if_simulation_right" if req.decision == "sim_right" else "answer_if_auditor_right", "")
        if q.get("question") and chosen:
            _set_clarification(pid, meta, f"finding:{req.name}", q["question"], chosen)
            _save(meta)
        req.decision = "auditor" if req.decision == "sim_right" else "engine"
    if req.decision == "auditor":
        # retire the auditor: rename so load_checkers skips it
        f = cdir / f"{req.name}.py"
        if f.exists():
            f.rename(cdir / f"_{req.name}.py")
        meta["checkers"] = [c for c in meta["checkers"] if c["name"] != req.name]
        meta.setdefault("retired_auditors", []).append(req.name)
        meta.get("checker_questions", {}).pop(req.name, None)
        _save(meta)
        return meta
    if req.decision == "rulebook":
        rb = DATA / pid / "rulebook.txt"
        text = rb.read_text(encoding="utf-8").rstrip()
        if "## Clarifications from the designer" not in text:
            text += "\n\n## Clarifications from the designer\n"
        q = meta.get("checker_questions", {}).get(req.name, {}).get("question", req.name)
        text += f"- Q: {q}\n  A: {req.note.strip() or '(see designer note)'}\n"
        rb.write_text(text + "\n", encoding="utf-8")
        meta["needs_regeneration"] = True
        _save(meta)
        return meta
    if req.decision == "engine":
        job = _job("fix", pid)
        findings = entry["findings"]

        def work(j):
            _repair_and_reaudit(pid, meta, j, (
                f"An independent auditor of the rule '{entry['rule']}' reports these "
                "violations, and the designer has confirmed the engine is wrong:"),
                findings, req.name)

        _run_in_thread(job, work)
        return job
    if req.decision == "answer":
        # the designer states the fact; the app works out who was wrong
        if not req.note.strip():
            raise HTTPException(400, "write the answer first")
        job = _job("resolve_answer", pid)

        def work(j):
            m = _load(pid)
            game = _engine_for(m)
            ctx = _moment_context(game, entry["findings"][0]) if entry["findings"] else ""
            j["detail"] = "reading the moment"
            fr = llm.frame_finding(req.name, entry["rule"], entry["findings"], ctx)
            j["detail"] = "comparing your answer"
            side = llm.compare_answer(fr["engine_value"], fr["auditor_value"], req.note)
            _set_clarification(pid, m, f"finding:{req.name}", f"{fr['question']} (scenario: {fr['scenario']})", req.note.strip())
            if side == "A":
                f = cdir / f"{req.name}.py"
                if f.exists():
                    f.rename(cdir / f"_{req.name}.py")
                m["checkers"] = [c for c in m["checkers"] if c["name"] != req.name]
                m.setdefault("retired_auditors", []).append(req.name)
                m.setdefault("jury_trail", []).append({"auditor": req.name, "verdict": "designer's answer matches the simulation — auditor dismissed", "time": time.time()})
                _save(m)
                return
            _save(m)
            reason = (f"The designer has stated the rule for this scenario. Scenario: {fr['scenario']} "
                      f"Question: {fr['question']} Designer's answer: {req.note.strip()}. "
                      f"The engine currently does: {fr['engine_value']}.")
            _repair_and_reaudit(pid, m, j, reason, entry["findings"], f"designer answer: {req.name}")
            m = _load(pid)
            m.setdefault("jury_trail", []).append({"auditor": req.name, "verdict": "designer's answer overrode the simulation — fixed", "time": time.time()})
            _save(m)

        _run_in_thread(job, work)
        return job
    raise HTTPException(400, "decision must be engine, auditor, rulebook or answer")


@app.get("/api/projects/{pid}/diff/{v}")
def version_diff(pid: str, v: int):
    """Unified diff between saved version v and the current engine, plus a
    plain count so a no-op 'fix' is visible at a glance."""
    import difflib
    meta = _load(pid)
    old = DATA / pid / "versions" / f"v{v}.py"
    cur = DATA / pid / "game.py"
    if not old.exists() or not cur.exists():
        raise HTTPException(404, "no such version")
    a_lines = old.read_text(encoding="utf-8").splitlines()
    b_lines = cur.read_text(encoding="utf-8").splitlines()
    diff = list(difflib.unified_diff(a_lines, b_lines, f"v{v}", "current", lineterm="", n=2))
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return {"added": added, "removed": removed, "diff": "\n".join(diff)[:60000]}


@app.get("/api/projects/{pid}/versions")
def list_versions(pid: str):
    meta = _load(pid)
    return {"current": meta.get("current_label", "engine"), "versions": meta.get("versions", [])}


@app.post("/api/projects/{pid}/versions/{v}/revert")
def revert_version(pid: str, v: int):
    meta = _load(pid)
    f = DATA / pid / "versions" / f"v{v}.py"
    if not f.exists():
        raise HTTPException(404, "no such version")
    label = next((x["label"] for x in meta.get("versions", []) if x["v"] == v), f"v{v}")
    _write_engine(pid, f.read_text(encoding="utf-8"), f"reverted to v{v} ({label})")
    m = _load(pid)
    m["checkers"] = None; m["checker_questions"] = {}; m["quality"] = None; m["second_opinion"] = None
    _save(m)
    return m


# ================================= jury ===================================
def _wait_job(jid: str, timeout: float = 3600, parent: dict | None = None) -> dict:
    t0 = time.time()
    parent = parent or getattr(_current, "job", None)
    while JOBS[jid]["status"] == "running":
        if time.time() - t0 > timeout:
            raise RuntimeError("sub-step timed out")
        if parent is not None and JOBS[jid].get("detail"):
            parent["detail"] = f"{JOBS[jid]['kind']}: {JOBS[jid]['detail']}"
            dict.__setitem__(parent, "progress", JOBS[jid].get("progress", ""))
        if parent is not None and parent.get("cancel"):
            JOBS[jid]["cancel"] = True
        # a paused child surfaces on the parent; continuing the parent continues the child
        if parent is not None:
            dict.__setitem__(parent, "paused", JOBS[jid].get("paused"))
            if parent.get("cap_raised"):
                dict.__setitem__(JOBS[jid], "cap_raised", parent["cap_raised"])
                dict.__setitem__(parent, "cap_usd", parent["cap_raised"]); dict.__setitem__(parent, "cap_raised", None)
        time.sleep(1)
    return JOBS[jid]


def _signature(pid: str, filename: str = "game.py"):
    try:
        return sandbox.run("signature", timeout=240, path=str(DATA / pid / filename))
    except Exception:
        return None


def _repair_and_reaudit(pid: str, meta: dict, j: dict, reason: str, findings: list[str], label: str):
    """Repair, then prove the repair did something: replay the same seeded
    games before and after. An unchanged fingerprint means a no-op fix; retry
    once with the failure spelled out and the stronger model."""
    rulebook = _rulebook_for_model(pid)
    code = (DATA / pid / "game.py").read_text(encoding="utf-8")
    error = reason + "\n" + "\n".join(findings[:25])
    j["detail"] = "recording how the engine plays before the fix"
    before = _signature(pid)
    for attempt in range(2):
        j["detail"] = f"fixing the engine: {label}" + (" (second attempt, stronger model)" if attempt else "")
        new_code = llm.repair_engine(rulebook, code, error, escalate=bool(attempt))
        _write_engine(pid, new_code, f"fixed: {label}" + (" (2nd attempt)" if attempt else ""))
        # a fix that breaks the engine is handed back to the model, not to the designer
        for vround in range(3):
            j["detail"] = "re-validating"
            _err = _validate_file(pid, players=(2,))
            if not _err:
                break
            if vround == 2:
                _write_engine(pid, code, f"reverted: '{label}' fix broke validation")
                raise RuntimeError("the fix broke the engine three times running; the engine was put back "
                                   "as it was. Last failure:\n" + _err[-1500:])
            j["detail"] = f"the fix broke something — repairing ({vround + 1}/3)"
            broken = (DATA / pid / "game.py").read_text(encoding="utf-8")
            new_code = llm.repair_engine(rulebook, broken, (
                "Your fix for the rule below is right in intent but the engine now crashes in play. "
                "Keep the rule fix and make the engine consistent with it — if an internal assertion "
                "or invariant was written for the old behaviour, update the invariant, do not undo the "
                "rule.\n\nRULE FIX: " + reason + "\n\nCRASH:\n" + _err), escalate=True)
            _write_engine(pid, new_code, f"fixed: {label} (made consistent {vround + 1})")
        after = _signature(pid)
        if before is None or after is None or after != before:
            break
        # nothing observable changed: tell the model exactly that and try once more
        _write_engine(pid, code, f"reverted: '{label}' fix changed nothing observable")
        j["detail"] = "the fix changed nothing observable — retrying"
        error = (reason + "\n\nYOUR PREVIOUS FIX CHANGED NOTHING: the same seeded games played out "
                 "identically (same turn counts, same final scores) before and after it. The rule "
                 "is still not enforced in play. Make sure the change reaches legal_actions/apply/"
                 "is_terminal — the code paths that decide what actually happens — not just a flag.\n"
                 + "\n".join(findings[:25]))
    else:
        m0 = _load(pid)
        m0.setdefault("jury_trail", []).append({"auditor": label, "time": time.time(),
            "verdict": "two repair attempts changed nothing observable — the engine was left as it was; "
                       "this needs a look"})
        _save(m0)
    game = _engine_for(meta)
    j["detail"] = "re-auditing"
    m2 = _load(pid)
    run_checkers(m2, game, DATA / pid / "checkers")
    m2["fix_rounds"] = m2.get("fix_rounds", 0) + 1
    _save(m2)
    return m2


def run_jury(pid: str, j: dict) -> None:
    """For every auditor with findings: frame a scenario, ask the proxy cold,
    route by agreement. Unanimous -> fix; proxy sides with engine -> mute;
    rulebook silent -> provisional ruling + inbox question."""
    meta = _load(pid)
    rulebook = _rulebook_for_model(pid)
    trail = meta.get("jury_trail", [])
    inbox = meta.get("inbox", [])
    fixes = 0
    to_fix = []
    for c in list(meta.get("checkers") or []):
        if not c["n_findings"] or c.get("suspect"):
            continue
        j["detail"] = f"jury: {c['name']}"
        g = _engine_for(meta)
        ctx = _moment_context(g, c["findings"][0]) if c["findings"] else ""
        story = _narrate_moment(g, c["findings"][0]) if c["findings"] else ""
        if story:
            ctx = ctx + "\n\nReplay in the game's own words:\n" + story
        fr = llm.frame_finding(c["name"], c["rule"], c["findings"], ctx)
        px = llm.proxy_answer(rulebook, fr["scenario"], fr["question"])
        entry = {"auditor": c["name"], "rule": c["rule"], **fr, "proxy": px, "time": time.time()}
        if px["basis"] == "unclear" or not px["answer"]:
            entry["verdict"] = "rulebook silent — provisional ruling kept, question queued"
            inbox.append({"id": f"q{len(inbox) + 1}", "auditor": c["name"], "scenario": fr["scenario"],
                          "question": fr["question"], "engine_value": fr["engine_value"],
                          "auditor_value": fr["auditor_value"], "proxy_note": px["note"],
                          "provisional": fr["engine_value"], "status": "open", "answer": ""})
            c["status"] = "undecided"
        else:
            side = llm.compare_answer(fr["engine_value"], fr["auditor_value"], px["answer"])
            entry["proxy_agrees_with"] = {"A": "engine", "B": "auditor"}.get(side, "neither")
            if side == "B":
                entry["verdict"] = "auditor + proxy agree the engine is wrong — fixed"
                trail.append(entry)
                to_fix.append((c, px))
                meta["jury_trail"] = trail; meta["inbox"] = inbox; _save(meta)
                continue
            if side == "A":
                entry["verdict"] = "proxy sides with the engine — auditor muted"
                f = DATA / pid / "checkers" / f"{c['name']}.py"
                if f.exists():
                    f.rename(DATA / pid / "checkers" / f"_{c['name']}.py")
                meta["checkers"] = [x for x in meta["checkers"] if x["name"] != c["name"]]
                meta.setdefault("retired_auditors", []).append(c["name"])
            else:
                entry["verdict"] = "three different answers — question queued"
                inbox.append({"id": f"q{len(inbox) + 1}", "auditor": c["name"], "scenario": fr["scenario"],
                              "question": fr["question"], "engine_value": fr["engine_value"],
                              "auditor_value": fr["auditor_value"], "proxy_note": f"expert read: {px['answer']}",
                              "provisional": fr["engine_value"], "status": "open", "answer": ""})
                c["status"] = "undecided"
        trail.append(entry)
        meta["jury_trail"] = trail; meta["inbox"] = inbox; _save(meta)
    if to_fix:
        # one repair covering every confirmed violation, one validation, one re-audit
        reason = ("Independent auditors and an independent rules expert (reading the rulebook only) "
                  "agree the engine is wrong on these points; fix ALL of them:\n")
        for c, px in to_fix:
            reason += (f"\n[{c['name']}] expert: {px['answer']} (\"{px['quote']}\")\n  " +
                       "\n  ".join(c["findings"][:6]))
        _repair_and_reaudit(pid, _load(pid), j, reason, [], ", ".join(c["name"] for c, _ in to_fix))
        fixes = len(to_fix)
    m = _load(pid)
    m["jury_trail"] = trail; m["inbox"] = inbox; m["jury_fixes"] = m.get("jury_fixes", 0) + fixes
    m["jury_cost"] = llm.cost_note()
    _save(m)


@app.post("/api/projects/{pid}/jury")
def jury(pid: str):
    meta = _load(pid)
    if not meta.get("checkers"):
        raise HTTPException(400, "build the rule checkers first")
    job = _job("jury", pid)
    _run_in_thread(job, lambda j: run_jury(pid, j))
    return job


class InboxAnswer(BaseModel):
    item_id: str
    answer: str


@app.post("/api/projects/{pid}/inbox/answer")
def inbox_answer(pid: str, req: InboxAnswer):
    """Designer settles a queued question. The answer is saved as a
    clarification straight away; nothing runs. 'Apply answers' later takes
    every pending answer in one repair + one re-audit."""
    meta = _load(pid)
    it = next((i for i in meta.get("inbox", []) if i["id"] == req.item_id), None)
    if not it:
        raise HTTPException(404, "no such question")
    if not req.answer.strip():
        raise HTTPException(400, "write the answer first")
    it["status"] = "answered"; it["answer"] = req.answer.strip(); it["applied"] = False
    it.pop("outcome", None)
    _set_clarification(pid, meta, f"inbox:{it['id']}",
                       f"{it['question']} (scenario: {it['scenario']})", it["answer"])
    _save(meta)
    return {"meta": meta, "job": None}


@app.post("/api/projects/{pid}/inbox/apply")
def inbox_apply(pid: str):
    """Take every answered-but-unapplied question: those agreeing with the
    engine are closed; the rest go into ONE combined repair, then one
    validation and one re-audit."""
    meta = _load(pid)
    pending = [i for i in meta.get("inbox", []) if i["status"] == "answered" and not i.get("applied")]
    if not pending:
        raise HTTPException(400, "no unapplied answers")
    job = _job("fix", pid)

    def work(j):
        m = _load(pid)
        overturned = []
        for it in [i for i in m["inbox"] if i["status"] == "answered" and not i.get("applied")]:
            j["detail"] = f"comparing your answer: {it['question'][:50]}"
            side = llm.compare_answer(it["engine_value"], it["auditor_value"], it["answer"])
            it["applied"] = True
            if side == "A":
                it["outcome"] = "matches what the engine did — nothing to change"
            else:
                it["outcome"] = "overturned the provisional ruling — engine fixed"
                overturned.append(it)
        _save(m)
        if not overturned:
            return
        reason = "The designer has settled these rules; fix ALL of them:\n" + "\n".join(
            f"- {it['question']} Scenario: {it['scenario']} Answer: {it['answer']}. "
            f"The engine currently does: {it['engine_value']}." for it in overturned)
        _repair_and_reaudit(pid, m, j, reason, [],
                            "designer answers: " + ", ".join(it["auditor"] for it in overturned))
        m = _load(pid)
        for c in m.get("checkers") or []:
            c.pop("status", None)
        _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/inbox/undo")
def inbox_undo(pid: str, req: WorkshopAnswer):
    """Reopen an inbox question and remove its clarification. If the answer had
    triggered an engine fix, the previous engine is still in version history."""
    meta = _load(pid)
    it = next((i for i in meta.get("inbox", []) if i["id"] == req.item_id), None)
    if not it:
        raise HTTPException(404, "no such question")
    it["status"] = "open"; it["answer"] = ""
    fixed = "fixed" in (it.pop("outcome", "") or "")
    _drop_clarification(pid, meta, f"inbox:{it['id']}")
    meta["undo_note"] = ("that answer had changed the engine — use the version list to revert it"
                         if fixed else "")
    _save(meta)
    return meta


# ============================ strategies / balance =========================

class StrategyReq(BaseModel):
    description: str = ""
    index: int | None = None
    players: int = 2
    games_per_pair: int = 60


def _engine_path(meta: dict) -> str | None:
    return None if meta.get("builtin") else str(DATA / meta["id"] / "game.py")


def _feature_names(meta: dict) -> list[str]:
    if meta.get("feature_names"):
        return meta["feature_names"]
    if meta.get("builtin"):
        g = _engine_for(meta)
        names = list(getattr(g, "FEATURE_NAMES", ()) or ())
        n = len(g.features(g.initial_state(2, 0), 0))
        names = (names + [f"feature_{i}" for i in range(n)])[:n]
    else:
        names = sandbox.run("feature_names", timeout=60, path=_engine_path(meta))
    meta["feature_names"] = names
    _save(meta)
    return names


def _strategy_specs(meta: dict):
    """Baselines plus the designer's strategies, as sandbox specs."""
    specs, names = ["scoregreedy"], ["score-greedy (baseline)"]
    n = len(_feature_names(meta))
    for st in meta.get("strategies", []):
        if st.get("stale") or len(st.get("weights", [])) != n:
            continue
        specs.append({"name": st["name"], "weights": st["weights"]}); names.append(st["name"])
    return specs, names


@app.post("/api/projects/{pid}/strategies")
def add_strategy(pid: str, req: StrategyReq):
    """Designer describes a play style in words; a model turns it into weights
    over the engine's named features (one cheap call)."""
    meta = _load(pid)
    if not req.description.strip():
        raise HTTPException(400, "describe the play style first")
    names = _feature_names(meta)
    job = _job("strategy", pid)

    def work(j):
        j["detail"] = "translating the play style into weights"
        st = llm.strategy_weights(names, req.description.strip())
        st["description"] = req.description.strip(); st["source"] = "designer"; st["features"] = names
        m = _load(pid); m.setdefault("strategies", []).append(st); _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/strategies/{index}/retranslate")
def retranslate_strategy(pid: str, index: int):
    meta = _load(pid)
    sts = meta.get("strategies", [])
    if not 0 <= index < len(sts):
        raise HTTPException(404, "no such strategy")
    names = _feature_names(meta)
    desc = sts[index].get("description", "")
    job = _job("strategy", pid)

    def work(j):
        j["detail"] = "re-translating for the rebuilt engine"
        st = llm.strategy_weights(names, desc)
        m = _load(pid)
        m["strategies"][index] = {**st, "description": desc, "source": "designer", "features": names}
        _save(m)

    _run_in_thread(job, work)
    return job


@app.delete("/api/projects/{pid}/strategies/{index}")
def delete_strategy(pid: str, index: int):
    meta = _load(pid)
    sts = meta.get("strategies", [])
    if not 0 <= index < len(sts):
        raise HTTPException(404, "no such strategy")
    sts.pop(index); meta["strategies"] = sts; meta.pop("matrix", None)
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/balance/matrix")
def balance_matrix(pid: str, req: StrategyReq):
    """Round-robin between the baseline and every designer strategy. Free."""
    meta = _load(pid)
    specs, names = _strategy_specs(meta)
    if len(specs) < 2:
        raise HTTPException(400, "add at least one strategy first")
    job = _job("simulate", pid)

    def work(j):
        j["detail"] = f"{len(specs)} strategies, every pair, {req.games_per_pair} games each, {req.players} players"
        if meta.get("builtin"):
            raise RuntimeError("round-robin runs on generated engines only for now")
        table = sandbox.run("matrix", timeout=600, stall=300, path=_engine_path(meta), strategies=specs,
                            n_players=req.players, games_per_pair=req.games_per_pair,
                            on_progress=lambda t: dict.__setitem__(j, "progress", f"games {t}"))
        k = len(names)
        avg = [sum(x for x in row if x is not None) / max(1, k - 1) for row in table]
        findings = []
        for i in range(k):
            others = [x for x in table[i] if x is not None]
            if others and min(others) >= 0.65:
                findings.append(f"**{names[i]}** beats every other strategy tested (never below {min(others):.0%}). "
                                "If nothing counters it, it is a dominant strategy.")
            if others and max(others) <= 0.35:
                findings.append(f"**{names[i]}** loses to everything (never above {max(others):.0%}) — "
                                "either this play style is a trap or the engine undervalues it.")
        m = _load(pid)
        m["matrix"] = {"names": names, "players": req.players, "games_per_pair": req.games_per_pair,
                       "table": table, "average": avg, "findings": findings, "time": time.time()}
        _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/balance/counter")
def balance_counter(pid: str, req: StrategyReq):
    """Evolve a strategy whose only goal is beating the chosen one. Free (CPU)."""
    meta = _load(pid)
    specs, names = _strategy_specs(meta)
    if req.index is None or not 0 <= req.index < len(specs):
        raise HTTPException(400, "pick a strategy to counter")
    target = specs[req.index]
    if isinstance(target, str):   # baseline: derive weights by asking the engine-agnostic greedy to be countered as 'points only'
        n = len(_feature_names(meta)); target = {"name": names[req.index], "weights": [10.0] + [0.0] * (n - 1)}
    job = _job("simulate", pid)

    def work(j):
        j["detail"] = f"evolving a counter to {names[req.index]} ({req.players} players)"
        res = sandbox.run("counter", timeout=900, stall=300, path=_engine_path(meta), target=target,
                          n_players=req.players, on_progress=lambda t: dict.__setitem__(j, "progress", f"candidates {t}"))
        m = _load(pid)
        st = {"name": f"counter to {names[req.index]}", "weights": res["weights"], "source": "counter",
              "description": f"evolved purely to beat '{names[req.index]}'",
              "rationale": f"wins {res['winrate_vs_target']:.0%} of games against it"}
        m.setdefault("strategies", []).append(st)
        verdict = (f"**{names[req.index]} can be countered**: an evolved strategy beats it {res['winrate_vs_target']:.0%} of the time."
                   if res["winrate_vs_target"] >= 0.55 else
                   f"**No counter found to {names[req.index]}**: the best evolved opponent wins only {res['winrate_vs_target']:.0%}. "
                   "If it also tops the round-robin, this looks like a dominant strategy.")
        m.setdefault("counter_findings", []).append(verdict)
        m.pop("matrix", None)   # the table is out of date once a strategy is added
        _save(m)

    _run_in_thread(job, work)
    return job


class AnalysisReq(BaseModel):
    kind: str = "competitive"         # competitive | family
    counts: list[int] = [2, 4]
    primary: int = 2
    target_minutes: int | None = None


@app.post("/api/projects/{pid}/balance/analyze")
def balance_analyze(pid: str, req: AnalysisReq):
    """The one button: train players if needed, measure everything under
    trained play at each count, write one report. CPU only."""
    from web.analysis import run_analysis
    meta = _load(pid)
    if meta.get("builtin") or meta.get("engine") != "generated":
        raise HTTPException(400, "needs a generated engine")
    job = _job("simulate", pid)
    prof = req.model_dump()

    def work(j):
        res = run_analysis(pid, meta, j, prof, sandbox, _engine_path(meta), _load, _save)
        m = _load(pid)
        m["analysis"] = {**res, "engine_version": len(m.get("versions", []))}
        m["balance_profile"] = prof
        _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/analysis/clear-questions")
def clear_analysis_questions(pid: str):
    meta = _load(pid)
    if meta.get("analysis"):
        meta["analysis"]["questions"] = []
        meta["analysis"]["findings"] = [f for f in meta["analysis"]["findings"] if f[0] != "question"]
    _save(meta)
    return meta


@app.post("/api/projects/{pid}/players/train")
def train_players(pid: str, req: StrategyReq):
    """Train player populations for this engine at a player count (CPU only,
    ~10 min). Cached with the engine version; every later measurement uses them."""
    meta = _load(pid)
    if meta.get("builtin") or meta.get("engine") != "generated":
        raise HTTPException(400, "needs a generated engine")
    job = _job("simulate", pid)

    def work(j):
        j["detail"] = f"training players for {req.players}-player games (3 independent populations)"
        res = sandbox.run("train", timeout=1800, stall=600, path=_engine_path(meta), n_players=req.players,
                          populations=3, rounds=2, budget_s=600,
                          on_progress=lambda t: dict.__setitem__(j, "progress", f"step {t}"))
        m = _load(pid)
        m.setdefault("players", {})[str(req.players)] = {**res, "time": time.time(),
                                                        "engine_version": len(m.get("versions", []))}
        _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/balance/scan")
def balance_scan_ep(pid: str, req: StrategyReq):
    """Automatic balance analysis: evolve, double-oracle, equilibrium,
    archetypes, sensitivity. CPU only, no designer input, no model."""
    meta = _load(pid)
    if meta.get("builtin") or meta.get("engine") != "generated":
        raise HTTPException(400, "needs a generated engine")
    job = _job("simulate", pid)

    def work(j):
        j["detail"] = f"balance scan at {req.players} players: evolving strategies, mapping the meta-game"
        res = sandbox.run("balance", timeout=1800, stall=600, hard_max=1500, path=_engine_path(meta), n_players=req.players,
                          budget_s=900, on_progress=lambda t: dict.__setitem__(j, "progress", f"step {t}"))
        m = _load(pid); m["balance_scan"] = {**res, "time": time.time()}
        # the discovered strategies become usable in the strategies stage and the seat picker
        m["strategies"] = [x for x in m.get("strategies", []) if x.get("source") != "scan"]
        for lab, st, arch in zip(res["labels"], res["strategies"], res["archetypes"]):
            if "weights" in st:
                m["strategies"].append({"name": lab, "weights": st["weights"], "source": "scan",
                                        "description": arch, "rationale": "found by the balance scan",
                                        "features": res["feature_names"]})
        # a counter found earlier that beats the scan's top strategy overrides 'dominant'
        m.pop("counter_findings", None)
        _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/balance/competent")
def balance_competent(pid: str, req: StrategyReq):
    """Seat / lock-in / length figures under MCTS play. CPU only."""
    meta = _load(pid)
    if meta.get("builtin") or meta.get("engine") != "generated":
        raise HTTPException(400, "needs a generated engine")
    job = _job("simulate", pid)
    n_games = max(8, min(400, req.games_per_pair))

    def work(j):
        j["detail"] = f"{n_games} games between MCTS players, {req.players} players (slow: strong play)"
        recs = sandbox.run_parallel("competent", n_games=n_games, path=_engine_path(meta), n_players=req.players,
                                    iterations=60, on_progress=lambda t: dict.__setitem__(j, "progress", f"games {t}"))
        md = make_report(recs, None)
        m = _load(pid); m["competent_report"] = {"players": req.players, "games": n_games, "markdown": md, "time": time.time()}
        _save(m)

    _run_in_thread(job, work)
    return job


# =============================== the runner ===============================
@app.post("/api/projects/{pid}/run-all")
def run_all(pid: str, force: bool = False, games: int = 300):
    """Phase B: everything after the workshop, unattended."""
    meta = _load(pid)
    if meta.get("builtin"):
        raise HTTPException(400, "run-all is for generated engines")
    ws = meta.get("workshop")
    if not force and (not ws or ws.get("readiness", 0) < 100):
        raise HTTPException(409, "rules readiness is below 100% — finish the workshop or pass force")
    job = _job("run_all", pid)

    def step(j, label, sub):
        j["detail"] = label
        r = _wait_job(sub["id"])
        j.setdefault("steps", []).append({"step": label, "status": r["status"], "error": r.get("error")})
        return r["status"] == "done"

    def work(j):
        m = _load(pid)
        if m.get("engine") != "generated":
            if not step(j, "building the engine", generate(pid, force=True)):
                raise RuntimeError("engine generation failed; see steps")
        ok = step(j, "rule checkers", build_checkers(pid))
        step(j, "second opinion", run_second_opinion(pid))
        if ok:
            step(j, "jury", jury(pid))
        step(j, "balance report", run_sim(pid, SimRequest(games=games, players=2,
                                                            agents="scoregreedy", rotate=True)))
        m = _load(pid)
        m["last_run"] = {"time": time.time(), "steps": j.get("steps", []),
                         "open_questions": sum(1 for i in m.get("inbox", []) if i["status"] == "open")}
        _save(m)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/fix-from-findings")
def fix_from_findings(pid: str):
    """Hand the checkers' findings to the model as a repair round, then re-audit."""
    meta = _load(pid)
    if meta.get("builtin"):
        raise HTTPException(400, "built-in engines are not repaired here")
    findings = [f for c in meta.get("checkers", []) for f in c["findings"]]
    if not findings:
        raise HTTPException(400, "no findings to fix")
    job = _job("fix", pid)

    def work(j):
        _repair_and_reaudit(pid, meta, j, (
            "Independent rule auditors, written from the rulebook only, report "
            "these violations in games your engine played:"), findings, "auditor findings")

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/generate")
def generate(pid: str, force: bool = False):
    meta = _load(pid)
    if meta.get("builtin"):
        raise HTTPException(400, "this project uses a built-in engine")
    ws = meta.get("workshop")
    if not force:
        if not ws:
            raise HTTPException(409, "run the rules workshop first — the engine is only as good as the rules it reads")
        if ws.get("readiness", 0) < 100:
            open_n = sum(1 for i in ws["items"] if i["status"] == "open")
            raise HTTPException(409, f"rules readiness is {ws['readiness']}% with {open_n} open item(s); "
                                     f"answer them, or build anyway knowing the engine will guess")
    rb = DATA / pid / "rulebook.txt"
    if not rb.exists():
        raise HTTPException(400, "no rulebook on this project")
    job = _job("generate", pid)

    def validate(code: str) -> str | None:
        """Write, load, play random games with invariants on. Returns None if
        clean, else the traceback text to hand back to the model."""
        _write_engine(pid, code, "validating")
        return _validate_file(pid)

    def work(j):
        rulebook = _rulebook_for_model(pid)
        j["detail"] = "asking the model for an engine"
        try:
            code = llm.generate_engine(rulebook)
        except llm.TruncatedOutput as e:
            (DATA / pid / "partial.py").write_text(e.partial, encoding="utf-8")
            raise
        rounds = 0
        spent = [llm.cost_note()]
        j["detail"] = f"validating: random games with invariants on ({spent[-1]})"
        err = validate(code)
        while err and rounds < 3:
            rounds += 1
            esc = rounds >= 2  # two failures: escalate to the stronger model
            j["detail"] = (f"validation failed — asking the model to fix it (round {rounds}"
                           + (", stronger model)" if esc else ")"))
            code = llm.repair_engine(rulebook, code, err, escalate=esc)
            spent.append(llm.cost_note())
            j["detail"] = f"re-validating after repair round {rounds} ({spent[-1]})"
            err = validate(code)
        if err:
            (DATA / pid / "last_error.txt").write_text(err, encoding="utf-8")
            raise RuntimeError(f"engine still failing after {rounds} repair rounds; "
                               f"last error: {err.strip().splitlines()[-1]}")
        j["detail"] = "engine quality checks: seat symmetry, termination, determinism"
        meta["quality"] = (quality_report(_engine_for(meta)) if meta.get("builtin") else
                           sandbox.run("quality", timeout=900, path=str(DATA / pid / "game.py")))
        meta["quality_stale"] = False
        m = _load(pid)  # _write_engine updated versions; merge into the saved copy
        m["engine"] = "generated"
        m["current_label"] = f"generated ({rounds} repair rounds)" if rounds else "generated (first try)"
        m["repair_rounds"] = rounds
        m["validation"] = (f"36 random games at 2-4 players, invariants held"
                           + (f" · {rounds} repair round(s)" if rounds else " · first try")
                           + " · model calls: " + "; ".join(spent))
        _save(m)
        meta.update(m)
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
        else:  # generated engines: run in a killable child process
            game = _engine_for(meta)
            t0 = time.time()
            j["detail"] = f"playing {req.games} games"
            pops = (meta.get("players") or {}).get(str(req.players), {}).get("populations")
            if "trained" in specs and not pops:
                raise RuntimeError(f"no trained players for {req.players}-player games yet — train them first (free, ~10 min)")
            if "trained" in specs and pops:
                # each population plays its own share of the games, so stability can be checked
                records = []
                share = max(1, req.games // len(pops))
                for k, pop in enumerate(pops):
                    sp = [({"population": pop, "name": "trained"} if x == "trained" else x) for x in specs]
                    records += sandbox.run_parallel("play", n_games=share, seed0=k * 10_000, path=str(DATA / pid / "game.py"),
                                                    specs=sp, rotate=req.rotate,
                                                    on_progress=lambda t, k=k: dict.__setitem__(j, "progress", f"population {k + 1}/{len(pops)}, games {t}"))
                play_specs = [({"population": pops[0], "name": "trained"} if x == "trained" else x) for x in specs]
            else:
                play_specs = specs
                records = sandbox.run_parallel("play", n_games=req.games, path=str(DATA / pid / "game.py"),
                                               specs=specs, rotate=req.rotate,
                                               on_progress=lambda t: dict.__setitem__(j, "progress", f"games {t}"))
            j["detail"] = "narrating two sample games"
            try:
                meta["sample_games"] = [sandbox.run("narrate", timeout=120, path=str(DATA / pid / "game.py"),
                                                    specs=play_specs, seed=sd, rotate=req.rotate) for sd in (0, 1)]
            except Exception as e:
                meta["sample_games"] = []
            stuck = [r.seed for r in records if r.extra.get("unfinished")]
            if stuck:
                j["detail"] = f"{len(stuck)} game(s) never ended — replaying one to see why"
                try:
                    tail = sandbox.run("stuck_tail", timeout=300, path=str(DATA / pid / "game.py"),
                                       specs=specs, seed=stuck[0], rotate=req.rotate)
                    meta["stuck_diagnosis"] = {"seeds": stuck[:20], "seed": stuck[0], **tail}
                except Exception as e:
                    meta["stuck_diagnosis"] = {"seeds": stuck[:20], "error": str(e)[-500:]}
            else:
                meta.pop("stuck_diagnosis", None)
            secs = time.time() - t0
        j["detail"] = "writing the report"
        md = make_report(records, game)
        L = (meta.get("players") or {}).get(str(req.players), {}).get("ladder")
        if L:
            md = (f"## How good are the players behind these numbers?\n"
                  f"Rounds to finish a {req.players}-player game: random play {L['random']['rounds']:.0f} · "
                  f"novice (one-move greedy) {L['novice']['rounds']:.0f} · **trained {L['trained']['rounds']:.0f}**. "
                  f"Trained players beat the novice {L['trained_beats_novice']:.0%} of the time. "
                  + ("This run used the trained players." if "trained" in req.agents else
                     "**This run used " + req.agents.split(',')[0] + " players; results under trained play can differ.**")
                  + "\n\n") + md
        n = len(records); wins0 = sum((1 / len(r.winners)) for r in records if 0 in r.winners)
        meta["last_report"] = {"games": n, "players": req.players, "agents": req.agents,
                               "seat0": round(wins0 / n, 3) if n else None,
                               "unfinished": sum(1 for r in records if r.extra.get("unfinished")), "time": time.time()}
        for k, sg in enumerate(meta.get("sample_games") or []):
            md += f"\n## Sample game {k + 1} — how these players actually play\n"
            md += f"Final scores {sg['scores']}; {sg['steps']} moves{'' if sg['ended'] else ' (cut off)'}.\n\n<details><summary>every move</summary>\n\n"
            md += "\n".join(sg["lines"]) + "\n\n</details>\n"
        sd = meta.get("stuck_diagnosis")
        if sd:
            md += "\n## Games that never ended — what was happening\n"
            if sd.get("error"):
                md += f"(could not replay: {sd['error']})\n"
            else:
                md += (f"{len(sd['seeds'])} game(s) hit the action cap (game numbers {sd['seeds'][:8]}). "
                       f"The last turns of game {sd['seed']}, in the engine's own words:\n\n")
                md += "\n".join(f"- {t}" for t in sd["tail"]) + "\n\n"
                md += f"Position at the cap: {sd['state']}\n\n"
                md += (f"The player to move has {sd['n_legal']} legal action(s): " + "; ".join(sd["legal"]) + "\n\n"
                       "**Finding**: this is a position the rules let repeat forever. Either the rulebook needs "
                       "an ending for it (e.g. \"if every player passes in succession, the game ends\") or the "
                       "engine is not applying such a rule. Add it as a clarification and rebuild.\n")
        header = (f"_{req.games} games · {req.players} players · seats: "
                  f"{req.agents}{' · seats rotated' if req.rotate else ''} · "
                  f"{secs:.0f}s_\n\n")
        m_now = _load(pid)
        prov = [i for i in m_now.get("inbox", []) if i["status"] == "open"]
        if prov:
            header += ("> **Provisional rulings.** This report assumes: " +
                       "; ".join(f"{i['provisional']} ({i['auditor']})" for i in prov[:5]) +
                       ". Answer the open questions to confirm.\n\n")
        (DATA / pid / "report.md").write_text(header + md, encoding="utf-8")
        meta["has_report"] = True
        _save(meta)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/quality")
def run_quality(pid: str):
    meta = _load(pid)
    game = _engine_for(meta)
    job = _job("quality", pid)

    def work(j):
        j["detail"] = "playing random games at 2 and 4 players"
        meta["quality"] = quality_report(game)
        _save(meta)

    _run_in_thread(job, work)
    return job


@app.post("/api/projects/{pid}/second-opinion")
def run_second_opinion(pid: str):
    """Generate an independent second engine from the same rulebook and
    compare agent-free statistics. Agreement = evidence about the rules."""
    meta = _load(pid)
    if meta.get("builtin"):
        raise HTTPException(400, "second opinions are for generated engines")
    rb = DATA / pid / "rulebook.txt"
    if not rb.exists() or meta.get("engine") != "generated":
        raise HTTPException(400, "generate the first engine before asking for a second opinion")
    job = _job("second_opinion", pid)

    def work(j):
        rulebook = _rulebook_for_model(pid)
        j["detail"] = "asking the model for an independent second engine"
        code = llm.generate_engine(rulebook)
        rounds = 0
        while True:
            (DATA / pid / "game2.py").write_text(code, encoding="utf-8")
            try:
                _err = _validate_file(pid, "game2.py", players=(2,))
                if _err:
                    raise RuntimeError(_err)
                break
            except Exception:
                if rounds >= 3:
                    raise RuntimeError("second engine still failing after 3 repair rounds")
                rounds += 1
                j["detail"] = f"second engine failed validation — repair round {rounds}"
                code = llm.repair_engine(rulebook, code, traceback.format_exc())
        j["detail"] = "comparing the two engines on random play"
        meta["second_opinion"] = sandbox.run("second", timeout=900, path_a=str(DATA / pid / "game.py"),
                                             path_b=str(DATA / pid / "game2.py"))
        meta["second_opinion_stale"] = False
        missing = _missing_components(meta)
        if missing and any(r.get("status") == "flag" for r in meta["second_opinion"] if isinstance(r, dict)):
            meta["second_opinion"].insert(0, {
                "check": "invented components", "status": "flag",
                "detail": "Both engines had to invent " + ", ".join(c["name"] for c in missing) +
                          " because the rulebook does not list their values, so they cannot agree on "
                          "anything that depends on them (game length, pacing, lock-in). Upload the "
                          "tables in the workshop and rebuild before reading these disagreements as "
                          "rules problems."})
        meta["second_opinion_cost"] = llm.cost_note()
        _save(meta)

    _run_in_thread(job, work)
    return job


@app.get("/api/projects/{pid}/report")
def get_report(pid: str):
    p = DATA / pid / "report.md"
    if not p.exists():
        raise HTTPException(404, "no report yet — run a simulation first")
    return {"markdown": p.read_text(encoding="utf-8")}


@app.get("/api/projects/{pid}/jobs")
def project_jobs(pid: str):
    """Running jobs for this game, so a freshly loaded page can show them."""
    out = []
    for j in JOBS.values():
        if j.get("project") == pid and j["status"] == "running":
            d = dict(j); d["elapsed"] = round(time.time() - j["started"], 1); out.append(d)
    return out


def _build_stamp() -> str:
    """Short git commit + code file times, so the page can show what's running."""
    import subprocess
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        sha = ""
    newest = max((p.stat().st_mtime for p in [ROOT / "web" / "app.py", ROOT / "web" / "llm.py",
                                                ROOT / "web" / "static" / "index.html"] if p.exists()), default=0)
    return f"{sha or 'no-git'} · code {time.strftime('%b %d %H:%M', time.localtime(newest))} · server up since {time.strftime('%H:%M', time.localtime(SERVER_START))}"


SERVER_START = time.time()


@app.get("/api/config")
def config():
    return {"dev": DEV_MODE, "build": _build_stamp()}


@app.post("/api/jobs/{jid}/cancel")
def cancel_job(jid: str):
    if not DEV_MODE:
        raise HTTPException(403, "stopping jobs is disabled in production")
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    JOBS[jid]["cancel"] = True
    for j in JOBS.values():
        if j.get("parent") == jid and j["status"] == "running":
            j["cancel"] = True
    return {"ok": True}


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    if jid not in JOBS:
        raise HTTPException(404, "unknown job")
    j = dict(JOBS[jid])
    j["elapsed"] = round(time.time() - j["started"], 1)
    return j


@app.post("/api/jobs/{jid}/continue")
def continue_job(jid: str):
    """Raise a paused job's cap so it goes on from where it stopped."""
    j = JOBS.get(jid)
    if not j or not j.get("paused"):
        raise HTTPException(400, "that run is not paused")
    dict.__setitem__(j, "cap_raised", j["paused"]["offer"])
    return {"ok": True, "cap": j["paused"]["offer"]}


@app.get("/api/estimates")
def estimates():
    return ESTIMATES


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html",
                        headers={"Cache-Control": "no-store, must-revalidate"})


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"),
          name="static")
