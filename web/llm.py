"""Model call: rulebook -> engine code. Needs ANTHROPIC_API_KEY."""
from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

MODEL = os.environ.get("BGSIM_MODEL", "claude-sonnet-4-6")
API = "https://api.anthropic.com/v1/messages"
ENGINE_SPEC = (Path(__file__).resolve().parent.parent / "bgsim" / "engine.py").read_text()

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
comment. Return ONLY the complete Python file, no fences, no commentary.

=== engine.py ===
{spec}

=== rulebook ===
{rulebook}
"""


def generate_engine(rulebook: str) -> str:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to the server environment "
            "to generate engines; built-in engines work without it.")
    body = {
        "model": MODEL,
        "max_tokens": 32000,
        "messages": [{"role": "user", "content": PROMPT.format(
            spec=ENGINE_SPEC, rulebook=rulebook[:120_000])}],
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"})
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.load(r)
    code = "".join(b.get("text", "") for b in data.get("content", []))
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0]
    if "def initial_state" not in code:
        raise RuntimeError("model reply does not look like an engine file")
    return code
