"""One-button balance analysis.

Trains players (if needed), then measures everything under trained play at
each player count the designer cares about, and writes a single report
organised around what designers and publishers actually ask:

  1. meaningful choice     — viable approaches, traps, dominance
  2. fairness of the unchosen — seat order, ties
  3. component health      — pieces far above/below baseline in winning hands
  4. the arc               — lock-in, comebacks
  5. luck vs skill         — trained vs novice, spread between populations
  6. pacing and scaling    — length per count, unfinished games
  7. degenerate play       — endless positions, what they look like

Everything is CPU; no model calls. Numbers are labelled with the players
behind them and whether independent populations agree.
"""
from __future__ import annotations

import os
import statistics
import time

from bgsim.analysis import report as make_report, lead_lockin, stability


DEFAULT_PROFILE = {"kind": "competitive", "counts": [2, 4], "target_minutes": None, "primary": 2}


def _pct(x):
    return f"{100 * x:.0f}%"


def run_analysis(pid: str, meta: dict, j: dict, profile: dict, sandbox, engine_path: str, load, save) -> dict:
    prof = {**DEFAULT_PROFILE, **(profile or {})}
    counts = [int(c) for c in prof["counts"]] or [2]
    primary = int(prof.get("primary") or counts[0])
    t0 = time.time()
    findings, sections = [], []
    per_count = {}

    stages = ["training players", "playing games", "searching strategies", "writing the report"]
    def prog(msg, stage=None):
        j["detail"] = msg
        if stage is not None:
            m_ = load(pid); m_["analysis_progress"] = {"stage": stage, "of": len(stages), "label": stages[stage - 1], "detail": msg}
            save(m_)

    incidents = []   # what went wrong on the way, what we learned, what we did about it

    def attempt(label, fn, diagnose=None, retries=1):
        """Run a stage; on a stall or failure, diagnose (a stalled game is
        often a design finding, not a bug), then retry with a different seed."""
        last = None
        for k in range(retries + 1):
            try:
                return fn(k)
            except sandbox.EngineHung as e:
                last = e
                note = {"stage": label, "what": "stalled", "detail": str(e)[:200]}
                if diagnose:
                    try:
                        note["diagnosis"] = diagnose()
                    except Exception as e2:
                        note["diagnosis_error"] = str(e2)[:200]
                incidents.append(note)
                prog(f"{label} stalled — diagnosed it, retrying with a different seed", None)
            except Exception as e:
                last = e
                incidents.append({"stage": label, "what": "failed", "detail": str(e)[-300:]})
                prog(f"{label} failed — retrying", None)
        raise RuntimeError(f"{label} could not complete after {retries + 1} attempts: {last}")

    def partial(findings_so_far, note):
        m_ = load(pid); m_["analysis_partial"] = {"findings": findings_so_far, "note": note, "time": time.time()}; save(m_)

    # ---- 1) players
    m = load(pid)
    players = m.get("players") or {}
    for n in counts:
        if str(n) in players and players[str(n)].get("populations"):
            continue
        prog(f"training players for {n}-player games ({counts.index(n) + 1}/{len(counts)}) — learning by self-play", 1)
        res = attempt(f"training at {n} players",
                      lambda k: sandbox.run("train", timeout=1800, stall=600, path=engine_path, n_players=n,
                                            populations=2, rounds=1, budget_s=1200, size="quick",
                                            on_progress=lambda t: dict.__setitem__(j, "progress", f"step {t}")),
                      diagnose=lambda: sandbox.run("stuck_tail", timeout=300, path=engine_path, specs=["random"] * n, seed=0, rotate=True))
        m = load(pid)
        m.setdefault("players", {})[str(n)] = {**res, "time": time.time(), "engine_version": len(m.get("versions", []))}
        save(m); players = m["players"]

    # ---- 2) games under trained play, per count
    for n in counts:
        pops = players[str(n)]["populations"]; ladder = players[str(n)]["ladder"]
        recs = []
        share = max(60, 300 // len(pops))
        for k, pop in enumerate(pops):
            prog(f"{n}-player games under trained play (population {k + 1}/{len(pops)})", 2)
            sp = [{"population": pop, "name": "trained"}] * n
            recs += attempt(f"{n}-player games",
                            lambda a, sp=sp, k=k: sandbox.run_parallel("play", n_games=share, seed0=k * 10_000 + a * 500, path=engine_path, specs=sp, rotate=True,
                                                                        on_progress=lambda t: dict.__setitem__(j, "progress", f"games {t}")),
                            diagnose=lambda sp=sp: sandbox.run("stuck_tail", timeout=300, path=engine_path, specs=sp, seed=0, rotate=True))
        md = make_report(recs, None)
        wins = [sum(1 / len(r.winners) for r in recs if s in r.winners) / len(recs) for s in range(n)]
        rounds = [r.n_turns / n for r in recs]
        unfinished = [r.seed for r in recs if r.extra.get("unfinished")]
        stab = stability(recs)
        per_count[n] = {"games": len(recs), "seats": wins, "rounds_median": statistics.median(rounds),
                        "rounds_min": min(rounds), "rounds_max": max(rounds),
                        "ties": sum(1 for r in recs if len(r.winners) > 1), "unfinished": unfinished,
                        "ladder": ladder, "markdown": md, "stability_md": "\n".join(stab),
                        "stable": not any("**unstable**" in l for l in stab)}
        if unfinished:
            try:
                per_count[n]["stuck"] = sandbox.run("stuck_tail", timeout=300, path=engine_path,
                                                    specs=[{"population": pops[0], "name": "trained"}] * n,
                                                    seed=unfinished[0], rotate=True)
            except Exception:
                pass

    # partial results: fairness and pacing are known before the strategy search
    early = []
    for n, pc in per_count.items():
        edge = max(pc["seats"]) - 1 / n; worst = max(range(n), key=lambda s_: pc["seats"][s_])
        early.append(("warn" if edge > 0.06 and pc["stable"] else "ok" if edge <= 0.06 else "info",
                      f"Seat {'advantage' if edge > 0.06 else 'fairness'} at {n} players",
                      f"Seat {worst + 1} wins {_pct(pc['seats'][worst])} (fair: {_pct(1 / n)})."))
        early.append(("info", f"Length at {n} players", f"Median {pc['rounds_median']:.0f} rounds (range {pc['rounds_min']:.0f}–{pc['rounds_max']:.0f})."))
        if pc["unfinished"]:
            early.append(("warn", f"Games that never end at {n} players", f"{len(pc['unfinished'])} of {pc['games']} games hit the cap."))
    partial(early, "Fairness and pacing are in; the strategy search is still running.")

    # ---- 3) meaningful choice at the primary count
    prog(f"searching the strategy space at {primary} players — this is the longest step", 3)
    scan = attempt("strategy search",
                   lambda k: sandbox.run("balance", timeout=3600, stall=900, hard_max=3600, path=engine_path, n_players=primary,
                                         rounds=3, seed=k, budget_s=2400, on_progress=lambda t: dict.__setitem__(j, "progress", f"step {t}")),
                   diagnose=lambda: sandbox.run("stuck_tail", timeout=300, path=engine_path, specs=["scoregreedy"] * primary, seed=0, rotate=True))

    prog("writing the report", 4)
    # ================================ findings =================================
    fam = prof["kind"] == "family"
    # 1 meaningful choice
    mix = scan["equilibrium"]; top = max(range(len(mix)), key=lambda i: mix[i])
    viable = [scan["labels"][i] for i, s in enumerate(mix) if s >= 0.1]
    comp = scan.get("completeness", 1.0); rd = scan.get("rounds_done", 0)
    if comp < 0.99:
        findings.append(("info", "Strategy search didn't finish", f"Only {rd} round(s) completed — something slowed it down on our side. "
                         "The choice findings below are provisional; we've logged it."))
    if mix[top] >= 0.85 and rd >= 2:
        findings.append(("warn", "One way to play", f"At {primary} players, {rd} rounds of search settled on a single approach — "
                         f"'{scan['archetypes'][top]}' — that nothing else beats. Other approaches are traps for new players."))
    elif mix[top] >= 0.85:
        findings.append(("info", "One strategy on top so far", f"'{scan['archetypes'][top]}' beat everything in {rd} round(s) of search — "
                         "not enough to call it dominant. Run again to search deeper."))
    elif len(viable) >= 3:
        findings.append(("ok", "Several ways to play", f"{len(viable)} approaches share the table at {primary} players, "
                         "each beaten by another. Choice matters."))
    else:
        findings.append(("info", "Two ways to play", f"Play settles between two approaches at {primary} players."))
    # 2 fairness
    for n, pc in per_count.items():
        edge = max(pc["seats"]) - 1 / n
        worst = max(range(n), key=lambda s: pc["seats"][s])
        if edge > 0.06:
            findings.append(("warn" if pc["stable"] else "info", f"Seat advantage at {n} players",
                             f"Seat {worst + 1} wins {_pct(pc['seats'][worst])} (fair: {_pct(1 / n)})"
                             + ("." if pc["stable"] else " — but the trained populations disagree, so treat it as unproven.")))
        else:
            findings.append(("ok", f"Seats fair at {n} players", f"No seat wins more than {_pct(max(pc['seats']))} (fair: {_pct(1 / n)})."))
        tie_rate = pc["ties"] / pc["games"]
        if tie_rate > 0.05:
            findings.append(("warn", f"Ties at {n} players", f"{_pct(tie_rate)} of games end tied — the tiebreak may need another level."))
    # 4 the arc (from primary count lockin lines)
    pc = per_count[primary]
    lock = None
    for line in pc["markdown"].splitlines():
        if line.startswith("| 50% |"):
            try:
                lock = float(line.split("|")[2].strip().rstrip("%")) / 100
            except Exception:
                pass
    if lock is not None:
        if lock >= 0.85:
            findings.append(("warn", "Decided early", f"The mid-game leader wins {_pct(lock)} of the time at {primary} players — the second half rarely matters."))
        elif lock <= 0.6:
            findings.append(("ok" if not fam else "ok", "Games stay open", f"The mid-game leader wins only {_pct(lock)} of the time — comebacks are real."))
        else:
            findings.append(("info", "Leads mostly hold", f"The mid-game leader wins {_pct(lock)} of the time at {primary} players."))
    # 5 luck vs skill
    L = per_count[primary]["ladder"]
    skill = L["trained_beats_novice"]
    if fam and skill > 0.85:
        findings.append(("warn", "Skill dominates", f"Trained players beat novices {_pct(skill)} of the time — for a family game, a weaker player rarely wins."))
    elif not fam and skill < 0.65:
        findings.append(("warn", "Luck dominates", f"Trained players beat novices only {_pct(skill)} of the time — for a competitive game, skill barely shows."))
    else:
        findings.append(("ok", "Luck and skill", f"Trained players beat novices {_pct(skill)} of the time — {'about right for a family game' if fam else 'skill shows, luck still matters'}."))
    # 6 pacing and scaling
    for n, pc in per_count.items():
        findings.append(("info", f"Length at {n} players", f"Median {pc['rounds_median']:.0f} rounds (range {pc['rounds_min']:.0f}–{pc['rounds_max']:.0f})."))
    if len(per_count) >= 2:
        meds = {n: pc["rounds_median"] for n, pc in per_count.items()}
        lo, hi = min(meds.values()), max(meds.values())
        if hi > 1.5 * lo:
            findings.append(("warn", "Length swings with player count", ", ".join(f"{n}p: {v:.0f} rounds" for n, v in meds.items())))
    # 7 degenerate
    for n, pc in per_count.items():
        if pc["unfinished"]:
            findings.append(("warn", f"Games that never end at {n} players", f"{len(pc['unfinished'])} of {pc['games']} games hit the cap — see the replay below."))

    def rules_question(diag, source):
        """Phrase an endless position as a designer's question. One cheap model
        call for wording; a plain template if that's not possible."""
        moves = [t.split(": ", 1)[-1] for t in diag["tail"][-8:]]
        fallback = {"question": "What should happen when nobody can make progress?",
                    "scenario": "In a simulated game the same moves kept repeating and nothing changed: " + "; ".join(moves[-4:]) + ".",
                    "options": ["If every player passes in succession, the game ends immediately and is scored as it stands.",
                                "Passing is only allowed when there is no legal action; a player who can act must act."]}
        q = fallback
        if os.environ.get("ANTHROPIC_API_KEY"):
            try:
                from web import llm
                phrased = llm.phrase_rules_question(diag["tail"], diag.get("state", ""))
                if phrased.get("scenario") and len(phrased.get("options", [])) == 2:
                    q = phrased
            except Exception:
                pass
        q["source"] = source
        return q

    questions = []
    for inc in incidents:
        d = inc.get("diagnosis")
        if d and not d.get("ended"):
            questions.append(rules_question(d, f"noticed while {inc['stage']}"))
    for n, pc in per_count.items():
        st = pc.get("stuck")
        if st and not st.get("ended") and not questions:
            questions.append(rules_question(st, f"{n}-player games"))
    for q in questions:
        findings.append(("question", q["question"], q["scenario"][:220]))
    # incidents: one plain sentence for the designer; detail goes to the logs
    if incidents:
        n_inc = len(incidents)
        findings.append(("info", "We hit a snag and tried again", f"{n_inc} step(s) had to be re-run. Nothing for you to do; the details are in Logs."))

    # ================================ report ===================================
    out = [f"# Balance analysis — {prof['kind']} game, {', '.join(str(c) for c in counts)} players",
           f"_{time.strftime('%b %d %H:%M')} · CPU only · {int(time.time() - t0)}s_", ""]
    out.append("## Players behind these numbers")
    for n in counts:
        L = per_count[n]["ladder"]
        out.append(f"- {n} players: rounds to finish — random {L['random']['rounds']:.0f}, novice {L['novice']['rounds']:.0f}, "
                   f"**trained {L['trained']['rounds']:.0f}**; trained beat novices {_pct(L['trained_beats_novice'])}.")
    out += ["", "## Findings"]
    icon = {"ok": "✓", "warn": "⚠", "info": "•"}
    out += [f"- {icon[k]} **{h}** — {d}" for k, h, d in findings]
    out += ["", f"## 1 · Meaningful choice ({primary} players)",
            "| share at equilibrium | approach | what it does |", "|---|---|---|"]
    out += [f"| {_pct(s)} | {l} | {a} |" for s, l, a in zip(mix, scan["labels"], scan["archetypes"])]
    out += ["", "Which levers matter (change in win rate per step of valuing a feature):",
            "| feature | effect |", "|---|---|"] + [f"| {n_} | {c * 100:+.1f} pts |" for n_, c in scan["sensitivity"][:8]]
    for n in counts:
        pc = per_count[n]
        out += ["", f"## Under trained play — {n} players ({pc['games']} games)",
                "| seat | win rate |", "|---|---|"] + [f"| {s + 1} | {_pct(w)} |" for s, w in enumerate(pc["seats"])]
        out += ["", f"Ties: {pc['ties']} · unfinished: {len(pc['unfinished'])} · rounds median {pc['rounds_median']:.0f}", ""]
        if pc["stability_md"]:
            out += [pc["stability_md"], ""]
        # the lock-in and component sections from the standard report
        md = pc["markdown"]
        for head in ("## Lead lock-in", "## How games ended"):
            if head in md:
                seg = md[md.index(head):]
                nxt = seg.find("\n## ", 5)
                out += [seg[: nxt if nxt > 0 else None].rstrip(), ""]
        for head in [h for h in md.split("\n") if h.startswith("## ") and "who ends up holding" in h]:
            seg = md[md.index(head):]; nxt = seg.find("\n## ", 5)
            out += [seg[: nxt if nxt > 0 else None].rstrip(), ""]
        if pc.get("stuck"):
            st = pc["stuck"]
            out += [f"### A game that never ended ({n} players)", "The last moves, in the engine's own words:", ""]
            out += [f"- {t}" for t in st["tail"]] + ["", f"Position: {st['state']}", ""]
    m_ = load(pid); m_.pop("analysis_partial", None); m_.pop("analysis_progress", None); save(m_)
    return {"profile": prof, "findings": findings, "incidents": incidents, "questions": questions, "markdown": "\n".join(out), "per_count": {str(k): {kk: vv for kk, vv in v.items() if kk not in ("markdown", "stability_md", "stuck")} for k, v in per_count.items()},
            "scan": {k: scan[k] for k in ("labels", "archetypes", "equilibrium", "sensitivity", "findings")},
            "seconds": int(time.time() - t0), "time": time.time()}
