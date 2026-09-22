"""Model call: rulebook -> engine code. Needs ANTHROPIC_API_KEY."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import threading

MODEL = os.environ.get("BGSIM_MODEL", "claude-sonnet-5")  # legacy default

# Model routing. Roles: engine (generate/repair), audit (workshop, auditors,
# proxy), chore (compare, triage, frame, explain). Tiers come from the
# complexity score; escalation swaps the engine role up one rung.
RATES = {  # $ per million tokens, (input, output)
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-opus-5": (5.0, 25.0),
}
ROUTES = {
    "light": {"engine": "claude-sonnet-5", "audit": "claude-sonnet-5", "chore": "claude-haiku-4-5-20251001"},
    "heavy": {"engine": "claude-opus-5", "audit": "claude-sonnet-5", "chore": "claude-haiku-4-5-20251001"},
}
ESCALATE = {"claude-sonnet-5": "claude-opus-5", "claude-opus-5": "claude-opus-5",
            "claude-haiku-4-5-20251001": "claude-sonnet-5"}
EFFORTS = {"light": "medium", "heavy": "high"}

_ctx = threading.local()  # per-thread: tier, project id, spend cap, spend so far
on_cost = None            # app sets: callable(pid, model, usd, info) after every call
_stage = threading.local() # human label for what the current call is for


def set_stage(label: str) -> None:
    _stage.label = label


class BudgetExceeded(RuntimeError):
    pass


class Cancelled(RuntimeError):
    pass


cancel_check = None  # app sets: callable() -> bool, True when the current job was stopped


def _maybe_cancel():
    if cancel_check and cancel_check():
        raise Cancelled("stopped by the user")


def configure(tier: str = "light", pid: str | None = None, cap_usd: float | None = None,
              spent_usd: float = 0.0) -> None:
    _ctx.tier = tier if tier in ROUTES else "light"
    _ctx.pid = pid
    _ctx.cap = cap_usd
    _ctx.spent = spent_usd


def model_for(role: str, escalate: bool = False) -> str:
    override = os.environ.get(f"BGSIM_MODEL_{role.upper()}")
    if override:
        return override
    tier = getattr(_ctx, "tier", "light")
    m = ROUTES[tier].get(role, MODEL)
    return ESCALATE.get(m, m) if escalate else m


def effort_for(role: str) -> str:
    if role != "engine":
        return "medium"
    return os.environ.get("BGSIM_EFFORT", EFFORTS.get(getattr(_ctx, "tier", "light"), "medium"))


def _supports_effort(model: str) -> bool:
    """The effort setting exists on Opus 4.5+, Sonnet 4.6+ and Sonnet 5; not on Haiku 4.5."""
    return not model.startswith("claude-haiku")


def _spend_check(model: str) -> None:
    cap = getattr(_ctx, "cap", None)
    if cap is not None and getattr(_ctx, "spent", 0.0) >= cap:
        raise BudgetExceeded(f"this project has used ${_ctx.spent:.2f} of its ${cap:.2f} model "
                             f"budget; nothing more will run until the budget is raised")


def _spend_record(model: str) -> float:
    i, o = LAST_USAGE.get("input", 0), LAST_USAGE.get("output", 0)
    ri, ro = RATES.get(model, (2.0, 10.0))
    usd = i / 1e6 * ri + o / 1e6 * ro
    _ctx.spent = getattr(_ctx, "spent", 0.0) + usd
    if on_cost and getattr(_ctx, "pid", None):
        try:
            on_cost(_ctx.pid, model, usd, {"in": i, "out": o,
                                          "stage": getattr(_stage, "label", "?")})
        except Exception:
            pass
    return usd
API = os.environ.get("BGSIM_API_URL", "https://api.anthropic.com/v1/messages")
EFFORT = os.environ.get("BGSIM_EFFORT", "medium")
LAST_USAGE: dict = {}  # tokens of the most recent call, for cost display
ENGINE_SPEC = (Path(__file__).resolve().parent.parent / "bgsim" / "engine.py").read_text(encoding="utf-8")

PROMPT = """You are implementing a board game as a deterministic simulation \
engine for a balance-testing tool. Below is engine.py, defining the `Game` \
protocol every engine targets, then the game's rulebook.

Write a single Python file containing one class implementing the protocol, \
standard library only. States are frozen dataclasses of tuples; `apply` \
returns a new state; all randomness happens in initial_state(n_players, \
seed). Actions are tuples whose first element is a string naming the kind; \
break multi-step decisions into forced sub-phases with their own small \
action sets. Respect the transition-atomicity requirement in the engine \
docstring. legal_actions must be exact. Implement check_invariants with \
every structural rule you can express, plus describe_state, \
describe_action, describe_final, summary, and features (may return ()). If \
the rulebook is ambiguous, choose a reading and mark it with a `# RULING:` \
comment. Keep the file compact: no test code, no example usage, no \
exhaustive comments, and no data tables beyond what the rules require — a \
complete engine for a typical game is 400-1500 lines. Return ONLY the \
complete Python file, no fences, no commentary.

=== engine.py ===
{spec}

=== rulebook ===
{rulebook}
"""


def _call_streaming(key: str, body: dict) -> str:
    """Stream the response so the connection never looks idle; assemble the
    text deltas. Returns the full text."""
    body = dict(body, stream=True)
    _maybe_cancel()
    if not _supports_effort(body.get("model", MODEL)):
        body.pop("output_config", None)
    _spend_check(body.get("model", MODEL))
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01", "accept": "text/event-stream"})
    out = []
    stop = None
    completed = False
    try:
        r = urllib.request.urlopen(req, timeout=120)
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:400]
        raise RuntimeError(f"API {e.code} for model {body.get('model')}: {detail}") from None
    with r:
        for raw in r:
            _maybe_cancel()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if not payload or payload == "[DONE]":
                continue
            evt = json.loads(payload)
            t = evt.get("type")
            if t == "content_block_delta":
                d = evt.get("delta", {})
                if d.get("type") == "text_delta":
                    out.append(d.get("text", ""))
            elif t == "message_start":
                u = evt.get("message", {}).get("usage", {})
                LAST_USAGE.clear(); LAST_USAGE.update(input=u.get("input_tokens", 0), output=0,
                                                      model=body.get("model", MODEL))
            elif t == "message_delta":
                stop = evt.get("delta", {}).get("stop_reason", stop)
                LAST_USAGE["output"] = evt.get("usage", {}).get("output_tokens", LAST_USAGE.get("output", 0))
            elif t == "error":
                raise RuntimeError(f"API error: {evt.get('error')}")
            elif t == "message_stop":
                completed = True
                break
    if not completed:
        # the connection ended without the API's terminal event: treat as a
        # network failure so the caller retries instead of using half a file
        raise ConnectionError("stream ended before message_stop")
    text = "".join(out)
    _spend_record(body.get("model", MODEL))
    if stop == "max_tokens":
        raise TruncatedOutput(text)
    return text


class TruncatedOutput(RuntimeError):
    def __init__(self, partial: str):
        self.partial = partial
        lines = partial.count("\n")
        super().__init__(f"model output was cut off at the length limit after "
                         f"{lines} lines; partial output saved as partial.py")


def generate_engine(rulebook: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to the server environment "
            "to generate engines; built-in engines work without it.")
    body = {
        "model": model_for("engine"),
        "max_tokens": 64000,
        "output_config": {"effort": effort_for("engine")},
        "messages": [{"role": "user", "content": PROMPT.format(
            spec=ENGINE_SPEC, rulebook=rulebook[:120_000])}],
    }
    last = None
    for attempt in range(3):
        try:
            code = _call_streaming(key, body).strip()
            break
        except (ConnectionResetError, TimeoutError, OSError) as e:
            last = e
            time.sleep(3 * (attempt + 1))
    else:
        raise RuntimeError(f"model call failed 3 times (network): {last}")
    if code.startswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0]
    if "def initial_state" not in code:
        raise RuntimeError("model reply does not look like an engine file")
    return code


REPAIR_PROMPT = """You previously wrote this board game engine (below). Running it produced the error at the bottom. Fix the engine so the error cannot recur, keeping every rule from the rulebook intact and the same public API. Return ONLY the complete corrected Python file, no fences, no commentary.

=== engine.py (the protocol) ===
{spec}

=== rulebook ===
{rulebook}

=== your engine ===
{code}

=== error ===
{error}
"""


def _clean(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0]
    return code


def repair_engine(rulebook: str, code: str, error: str, escalate: bool = False) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    body = {"model": model_for("engine", escalate=escalate), "max_tokens": 64000,
            "output_config": {"effort": effort_for("engine")},
            "messages": [{"role": "user", "content": REPAIR_PROMPT.format(
                spec=ENGINE_SPEC, rulebook=rulebook[:120_000], code=code,
                error=error[-4000:])}]}
    fixed = _clean(_call_streaming(key, body))
    if "def initial_state" not in fixed:
        raise RuntimeError("model repair reply does not look like an engine file")
    return fixed


def cost_note() -> str:
    """Human-readable tokens + approximate cost of the last call."""
    i, o = LAST_USAGE.get("input", 0), LAST_USAGE.get("output", 0)
    ri, ro = RATES.get(LAST_USAGE.get("model", MODEL), (2.0, 10.0))
    usd = i / 1e6 * ri + o / 1e6 * ro
    return f"{i:,} in / {o:,} out tokens ≈ ${usd:.2f}"


# =========================== rules review (stage 1) ===========================

REVIEW_PROMPT = """You are a meticulous board game rules editor. Read the rulebook \
below and find everything that would stop a careful reader from resolving every \
situation the game can produce. Look for: missing rules (no end condition, no \
tiebreak, what happens when a supply runs out, a phase referenced but never \
defined), ambiguities (a phrase with two reasonable readings that lead to \
different play), contradictions (two passages that disagree), and undefined \
terms or orphaned components (listed but never used). Do NOT comment on \
balance, fun, or style — only on whether the rules are complete and \
unambiguous.

Return ONLY a JSON array (no fences, no prose). Each item: \
{{"kind": "missing"|"ambiguous"|"contradiction"|"undefined", \
"quote": "<the exact phrase from the rulebook this concerns, or the section name if absent>", \
"question": "<one concrete yes/no or short-answer question for the designer whose \
answer resolves it>"}}. At most 12 items, most important first. If the \
rulebook is complete, return [].

=== rulebook ===
{rulebook}
"""


def _json_call(key: str, prompt: str, max_tokens: int = 8000, role: str = "chore"):
    body = {"model": model_for(role), "max_tokens": max_tokens,
            "output_config": {"effort": effort_for(role)},
            "messages": [{"role": "user", "content": prompt}]}
    text = _clean(_call_streaming(key, body))
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end < 0:
        raise RuntimeError("model did not return a JSON array")
    return json.loads(text[start:end + 1])


def review_rules(rulebook: str) -> list[dict]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    items = _json_call(key, REVIEW_PROMPT.format(rulebook=rulebook[:120_000]), role="audit")
    out = []
    for it in items[:12]:
        if isinstance(it, dict) and it.get("question"):
            out.append({"kind": it.get("kind", "ambiguous"),
                        "quote": str(it.get("quote", ""))[:300],
                        "question": str(it["question"])[:500]})
    return out


# ========================= checker compiler (stage 3) =========================

PLAN_PROMPT = """Split the rulebook below into 3 to 6 self-contained rule sections \
that could each be audited independently over a record of a played game — for \
example: turn structure and turn order; costs and payments; a scoring or \
end-of-game rule; a resource or supply rule; a limit or capacity rule. Prefer \
sections that state exact numbers. Sections must be DISJOINT: each rule \
belongs to exactly one section, and no two sections may cover the same limit, \
cost, or phase. Return ONLY a JSON array: \
[{{"name": "<short_snake_case>", "text": "<the verbatim rulebook text of that \
section, complete>"}}].

=== rulebook ===
{rulebook}
"""

CHECKER_PROMPT = """You are writing an independent auditor for a board game \
simulation. You get ONE section of the rulebook and the trace format. You have \
NOT seen the simulation's code and must not assume anything beyond the trace \
format. Write a single Python file, standard library only, with:

    NAME = "{name}"
    RULE = "<one-sentence restatement of the rule you check>"

    def check(trace: dict) -> list[str]:

`check` audits one full game trace and returns one message per violation \
(empty list if clean). Re-derive the rule ONLY from the section: recompute \
what should have happened and compare with what the trace records. Every \
message carries the step index and the concrete numbers. Audit every step; do \
not sample. Do not use fields absent from the schema; if the schema lacks \
information the rule needs, return one message starting with "SCHEMA GAP:" \
instead of guessing. Be strict on what the section states, silent on what it \
doesn't.

Timing matters: a rule stated for a moment ("at the end of your turn", "at \
harvest", "when the game ends") must be checked only at that moment — for an \
end-of-turn limit, only at transitions where the acting player changes, never \
in mid-turn sub-phases where the engine may be about to enforce the limit. \
Trace contract you may rely on: a transition's `before`/`after` are full \
states; the engine may auto-resolve forced bookkeeping but every rule event \
(payments, penalties, scoring) occurs in a transition whose `phase` names \
it; the phase field's values are engine-specific strings you can observe in \
the trace; actions are lists whose first element names the action kind.

Return ONLY the complete checker file, no fences.

=== rulebook section ===
{section}

=== trace format ===
{schema}
"""


def plan_checkers(rulebook: str) -> list[dict]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    items = _json_call(key, PLAN_PROMPT.format(rulebook=rulebook[:120_000]), max_tokens=16000, role="audit")
    out = []
    for it in items[:6]:
        if isinstance(it, dict) and it.get("name") and it.get("text"):
            name = "".join(ch if ch.isalnum() else "_" for ch in str(it["name"]).lower())[:40]
            out.append({"name": name, "text": str(it["text"])[:20_000]})
    if not out:
        raise RuntimeError("model returned no usable sections")
    return out


def write_checker(name: str, section: str, schema: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    body = {"model": model_for("audit"), "max_tokens": 16000,
            "output_config": {"effort": effort_for("audit")},
            "messages": [{"role": "user", "content": CHECKER_PROMPT.format(
                name=name, section=section, schema=schema)}]}
    code = _clean(_call_streaming(key, body))
    if "def check(" not in code:
        raise RuntimeError(f"checker '{name}' reply does not look like a checker file")
    return code


# ================= plain-language explanations of findings =================

EXPLAIN_PROMPT = """You explain simulation audit results to a board game designer \
who does not program. Below is the designer's rulebook, then findings from \
independent auditors (each auditor checks one rule; its findings are terse \
and technical). For EACH auditor, write one plain-English item: what happened \
in the simulated games in the designer's own terms (rounds, players, \
resources — never "step", "trace", "state" or code words), which rule it \
concerns (quote the rulebook phrase), and a single FACTUAL question about that concrete moment — "with 11 \
tokens after taking three, what must this player do before the turn ends?" — \
never "should the simulation…" or "does the rulebook allow…". Do not decide \
who is right; the engine that played the games, the \
auditor, or the rulebook's wording could each be at fault.

Return ONLY a JSON array: [{{"name": "<auditor name exactly as given>", \
"simulation_did": "<1-2 plain sentences: what the simulated players/engine \
actually did at that moment, with the numbers>", \
"auditor_expected": "<1 sentence: what the auditor says should have happened>", \
"rule": "<the rulebook phrase this concerns>", \
"question": "<one factual question about that moment whose answer settles it>"}}]

=== rulebook ===
{rulebook}

=== auditor findings ===
{findings}
"""


def explain_findings(rulebook: str, findings: dict[str, list[str]]) -> list[dict]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    blob = "\n".join(f"[{n}]\n" + "\n".join(f"  {m}" for m in msgs[:6])
                     for n, msgs in findings.items())
    items = _json_call(key, EXPLAIN_PROMPT.format(rulebook=rulebook[:60_000], findings=blob),
                       max_tokens=6000)
    out = {}
    for it in items:
        if isinstance(it, dict) and it.get("name") in findings:
            out[it["name"]] = {k: str(it.get(k, ""))[:800]
                               for k in ("simulation_did", "auditor_expected", "rule", "question")}
            out[it["name"]]["what_happened"] = out[it["name"]]["simulation_did"]  # back-compat
    return out


# ============================ rules workshop ===============================

RESTATE_PROMPT = """You are restating a board game rulebook as a precise structured \
outline, the way a careful stranger would read it. Sections: setup, turn \
structure, actions (one line per action: name, exact cost, exact effect), \
limits and capacities, end of game, scoring. Every line MUST carry a basis: \
"stated" (the text says it outright), "inferred" (you combined passages or \
assumed a common default), or "unclear" (the text does not settle it). For \
inferred and unclear lines, include the assumption you made.

Return ONLY a JSON object: {{"outline": [{{"section": "<name>", "lines": \
[{{"text": "<one rule, one line>", "basis": "stated"|"inferred"|"unclear", \
"assumption": "<what you assumed, or empty>", "quote": "<short rulebook phrase \
this rests on, or empty>"}}]}}], "cost_table": [{{"action": "<name>", \
"cost": "<exact>", "effect": "<exact>", "basis": "stated"|"inferred"|"unclear"}}]}}

=== rulebook ===
{rulebook}
"""

WALK_PROMPT = """From the rulebook below, narrate one complete sample round of play \
for {n} players, in plain language, as a rules expert would when teaching the \
game: who acts, what they do, what it costs, what changes. Whenever you have \
to assume something the text does not settle, say so in that step. Keep it to \
the shortest round that exercises the main actions. Return ONLY a JSON \
object: {{"steps": [{{"text": "<one step>", "assumption": "<empty, or the \
assumption this step needed>"}}]}}

=== rulebook ===
{rulebook}
"""

TRIAGE_PROMPT = """Below is a rulebook and a list of questions a reviewer raised \
about it. For each question, decide whether the rulebook text actually \
answers it. If it does, give the answer and quote the passage. If it does \
not, say so. Do not guess, do not use knowledge of other games. Return ONLY a \
JSON array in the same order: [{{"answered": true|false, "answer": "<the \
answer from the text, or empty>", "quote": "<the passage, or empty>"}}]

=== rulebook ===
{rulebook}

=== questions ===
{questions}
"""


def _json_obj_call(key: str, prompt: str, max_tokens: int = 16000, role: str = "chore"):
    body = {"model": model_for(role), "max_tokens": max_tokens,
            "output_config": {"effort": effort_for(role)},
            "messages": [{"role": "user", "content": prompt}]}
    text = _clean(_call_streaming(key, body))
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise RuntimeError("model did not return a JSON object")
    return json.loads(text[start:end + 1])


def restate_rules(rulebook: str) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    data = _json_obj_call(key, RESTATE_PROMPT.format(rulebook=rulebook[:120_000]), role="audit")
    outline = []
    for sec in data.get("outline", []):
        lines = [{"text": str(l.get("text", ""))[:400], "basis": l.get("basis", "stated"),
                  "assumption": str(l.get("assumption", ""))[:400],
                  "quote": str(l.get("quote", ""))[:200]}
                 for l in sec.get("lines", []) if isinstance(l, dict) and l.get("text")]
        outline.append({"section": str(sec.get("section", ""))[:80], "lines": lines})
    table = [{"action": str(r.get("action", ""))[:80], "cost": str(r.get("cost", ""))[:200],
              "effect": str(r.get("effect", ""))[:300], "basis": r.get("basis", "stated")}
             for r in data.get("cost_table", []) if isinstance(r, dict)]
    return {"outline": outline, "cost_table": table}


def walk_turn(rulebook: str, n_players: int = 2) -> list[dict]:
    key = os.environ.get("ANTHROPIC_API_KEY")
    data = _json_obj_call(key, WALK_PROMPT.format(rulebook=rulebook[:120_000], n=n_players),
                          max_tokens=8000, role="audit")
    return [{"text": str(s.get("text", ""))[:500], "assumption": str(s.get("assumption", ""))[:400]}
            for s in data.get("steps", []) if isinstance(s, dict) and s.get("text")]


def triage_questions(rulebook: str, questions: list[str]) -> list[dict]:
    if not questions:
        return []
    key = os.environ.get("ANTHROPIC_API_KEY")
    qtext = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    items = _json_call(key, TRIAGE_PROMPT.format(rulebook=rulebook[:120_000], questions=qtext),
                       max_tokens=8000)
    out = []
    for i in range(len(questions)):
        it = items[i] if i < len(items) and isinstance(items[i], dict) else {}
        out.append({"answered": bool(it.get("answered")), "answer": str(it.get("answer", ""))[:500],
                    "quote": str(it.get("quote", ""))[:300]})
    return out


# ================================ jury ====================================

FRAME_PROMPT = """Below are findings from an auditor that checks one rule of a board \
game against simulated games. Turn them into ONE concrete scenario and ONE \
factual question a rules expert could answer from the rulebook alone, without \
seeing these findings. Describe the scenario in game terms (round, players, \
resources) with the exact numbers; never mention steps, traces, engines or \
auditors. Also extract, as short values, what the simulation actually did and \
what the auditor says should have happened.

The "moment in the game record" below is what the simulation was actually \
doing at the flagged step — its phase and pending action. If it shows the \
moment was mid-turn (e.g. a discard or payment still pending), the scenario \
must say so; the auditor may have judged the wrong moment.

Return ONLY a JSON object: {{"scenario": "<2-3 sentences>", "question": "<one \
factual question, e.g. 'How much food does this family owe?'>", \
"engine_value": "<what the simulation did, short>", "auditor_value": "<what \
the auditor expected, short>"}}

=== auditor: {name} — {rule} ===
{findings}

=== moment in the game record ===
{context}
"""

PROXY_PROMPT = """You are a rules expert. Using ONLY the rulebook below, answer the \
question about the scenario. If the rulebook settles it, answer and quote the \
passage. If the rulebook does not settle it, say so plainly — do not guess and \
do not use knowledge of other games.

Return ONLY a JSON object: {{"answer": "<short answer, or empty if unsettled>", \
"basis": "stated"|"unclear", "quote": "<the passage, or empty>", \
"note": "<if unclear: what the rulebook would need to say>"}}

=== rulebook ===
{rulebook}

=== scenario ===
{scenario}

=== question ===
{question}
"""

COMPARE_PROMPT = """Two candidate outcomes for a board game scenario, and an expert's \
answer. Say which candidate the expert's answer agrees with. Return ONLY a JSON \
object: {{"agrees_with": "A"|"B"|"neither"}}

Candidate A: {a}
Candidate B: {b}
Expert's answer: {answer}
"""


def frame_finding(name: str, rule: str, findings: list[str], context: str = "") -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    d = _json_obj_call(key, FRAME_PROMPT.format(name=name, rule=rule,
                                                findings="\n".join(findings[:8]),
                                                context=context or "(not available)"), max_tokens=3000)
    return {k: str(d.get(k, ""))[:600] for k in ("scenario", "question", "engine_value", "auditor_value")}


def proxy_answer(rulebook: str, scenario: str, question: str) -> dict:
    key = os.environ.get("ANTHROPIC_API_KEY")
    d = _json_obj_call(key, PROXY_PROMPT.format(rulebook=rulebook[:120_000], scenario=scenario,
                                                question=question), max_tokens=3000, role="audit")
    return {"answer": str(d.get("answer", ""))[:400], "basis": d.get("basis", "unclear"),
            "quote": str(d.get("quote", ""))[:300], "note": str(d.get("note", ""))[:400]}


def compare_answer(a: str, b: str, answer: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    d = _json_obj_call(key, COMPARE_PROMPT.format(a=a, b=b, answer=answer), max_tokens=200)
    v = str(d.get("agrees_with", "neither")).upper()
    return v if v in ("A", "B") else "neither"
