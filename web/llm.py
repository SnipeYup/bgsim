"""Model call: rulebook -> engine code. Needs ANTHROPIC_API_KEY."""
from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

MODEL = os.environ.get("BGSIM_MODEL", "claude-sonnet-5")
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
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01", "accept": "text/event-stream"})
    out = []
    stop = None
    completed = False
    with urllib.request.urlopen(req, timeout=120) as r:
        for raw in r:
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
                LAST_USAGE.clear(); LAST_USAGE.update(input=u.get("input_tokens", 0), output=0)
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
        "model": MODEL,
        "max_tokens": 64000,
        "output_config": {"effort": EFFORT},
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


def repair_engine(rulebook: str, code: str, error: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    body = {"model": MODEL, "max_tokens": 64000,
            "output_config": {"effort": EFFORT},
            "messages": [{"role": "user", "content": REPAIR_PROMPT.format(
                spec=ENGINE_SPEC, rulebook=rulebook[:120_000], code=code,
                error=error[-4000:])}]}
    fixed = _clean(_call_streaming(key, body))
    if "def initial_state" not in fixed:
        raise RuntimeError("model repair reply does not look like an engine file")
    return fixed


def cost_note() -> str:
    """Human-readable tokens + approximate cost of the last call (Sonnet 5 rates)."""
    i, o = LAST_USAGE.get("input", 0), LAST_USAGE.get("output", 0)
    usd = i / 1e6 * 2 + o / 1e6 * 10
    return f"{i:,} in / {o:,} out tokens ≈ ${usd:.2f}"
