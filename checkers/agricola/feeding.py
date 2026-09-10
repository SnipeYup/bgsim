"""
feeding.py — independent auditor for the Agricola (revised) harvest rules.

Derived ONLY from rulebook page 9 (Harvest: Field Phase, Feeding Phase,
Breeding Phase) and the trace schema, plus one supplied engine fact: with an
improvement, building resources (wood, clay, reed, stone) may be converted to
food during feeding via a ["convert", <resource>] action.

Granularity: the trace may bundle several rule events into one before/after
pair (a conversion that also settles feeding, breeds, and advances the round),
so every rule is audited as a NET EFFECT per player, never per action.

Window rule (whitelist):
  * A harvest window is exactly one maximal consecutive run of transitions
    whose BEFORE state has phase "feeding". Its round R is the before-round of
    the first transition in the run. Transitions in any other phase, whatever
    its name ("main", "farm_expansion", "adjust", "fencing", ...), are
    work-phase machinery and contribute nothing to harvest accounting.
  * Baselines (food, goods, animals, begging, family, born) come from the
    before-state of the first transition of the run.
  * The terminal transition of the run is the one whose after-state carries
    round R+1 (else the last transition of the run); it alone carries breeding
    for every player, audited as each player's animal delta across it.
  * Per player: feeding accounting runs from the baseline to the after-state of
    that player's own last action inside the run (their settlement). If a
    player takes no action in the run, their settlement is the last transition
    in which their food, goods or begging changed.

Interpretation notes:
  * `family` is the number of people a player has; `born` is how many of them
    were added this round. Food owed = 2 * (family - born) + 1 * born.
  * "improvements" in the section is read as len(majors) > 0.
  * `goods` indices are unlabeled; every ["convert", <kind>] action that
    lowers exactly one goods index teaches which index <kind> is, and the
    mapping is checked for self-consistency.
  * Whatever left a player's supply before settlement is what they ate:
      - grain/vegetable: exactly 1 food each without improvements, >= 1 with;
      - animals: only with improvements, >= 2 food each;
      - building resources: only with improvements AND a matching
        ["convert", <resource>] action by that player in the run, >= 1 each.
    With improvements the payment is therefore a lower bound.
  * A player with an improvement may gain food in the terminal transition of
    the run (start-of-round income from a major improvement); any other food
    change after a player's settlement is flagged.
"""

NAME = "feeding"
RULE = (
    "At the end of rounds 4, 7, 9, 11, 13 and 14 each player harvests 1 crop "
    "from every field, then pays 2 food per person (1 per newborn added that "
    "round) taking 1 begging marker per missing food, then every animal type "
    "with 2+ animals breeds exactly 1 newborn if it can be accommodated, and "
    "no animal may be turned into food during breeding; begging markers can "
    "never be removed and grain/vegetables are worth 1 food each without "
    "improvements."
)

HARVEST_ROUNDS = (4, 7, 9, 11, 13, 14)
FEEDING = "feeding"
CROP_KINDS = ("grain", "vegetable")
ANIMAL_FOOD_MIN = 2     # smallest food value of a cooked animal (lower bound only)
RESOURCE_FOOD_MIN = 1   # smallest food value of a converted building resource

SCHEMA_GAP = (
    "SCHEMA GAP: (1) Field phase — `fields` is described only as a list with no "
    "per-field crop type or count, so 'take exactly 1 crop from each field' "
    "cannot be recomputed; only the ordering of crop gains relative to feeding "
    "is audited. (2) Breeding accommodation — neither the schema nor this "
    "rulebook section defines animal capacity, so a pair that does not breed is "
    "not flagged; pairs, at-most-one-newborn-per-type, and no-food-conversion "
    "during breeding are audited. (3) `goods` is an unlabeled 6-int list, so "
    "grain/vegetable indices are learned from `convert` actions and checked for "
    "self-consistency rather than assumed."
)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def check(trace: dict) -> list[str]:
    out = [SCHEMA_GAP]
    try:
        out.extend(_audit(trace))
    except Exception as exc:  # an auditor must never die silently
        out.append(f"AUDIT ERROR: checker raised {type(exc).__name__}: {exc}")
    return out


# --------------------------------------------------------------------------- #
# state chain
# --------------------------------------------------------------------------- #
def _build_chain(trace, msgs):
    """Return (states, trans). trans[j] describes states[j] -> states[j+1]."""
    states, trans = [], []
    prev_i = None
    for pos, st in enumerate(trace.get("steps") or []):
        i = st.get("i", pos)
        b, a = st.get("before"), st.get("after")
        if b is None or a is None:
            msgs.append(f"step {i}: missing before/after state")
            continue
        if not states:
            states.append(b)
        elif states[-1] != b:
            states.append(b)
            trans.append(_tr(f"step {prev_i}->{i} (unrecorded transition)", i, None, None, None))
        states.append(a)
        trans.append(_tr(f"step {i}", i, st.get("action"), st.get("player"), st.get("phase")))
        prev_i = i
    final = trace.get("final")
    if final is not None:
        if not states:
            states.append(final)
        elif states[-1] != final:
            states.append(final)
            trans.append(_tr(f"step {prev_i}->final (unrecorded transition)", prev_i, None, None, None))
    return states, trans


def _tr(label, step, action, actor, phase):
    return {"label": label, "step": step, "action": action, "actor": actor, "phase": phase}


def _is_convert(t):
    a = t["action"]
    return isinstance(a, (list, tuple)) and len(a) >= 1 and a[0] == "convert"


def _kind(t):
    a = t["action"]
    return str(a[1]).lower() if isinstance(a, (list, tuple)) and len(a) > 1 else ""


def _majors(pl):
    return len(pl.get("majors") or [])


def _owed(pl, label, p, msgs):
    family, born = pl["family"], pl["born"]
    if born < 0 or born > family:
        msgs.append(f"{label}: P{p} has born={born} but family={family}; "
                    f"newborns cannot exceed people")
        born = max(0, min(born, family))
    adults = family - born
    return 2 * adults + born, adults, born


def _span_label(R, j0, j1, trans):
    first, last = trans[j0]["label"], trans[j1]["label"]
    if first == last:
        return f"{first} (round-{R} harvest)"
    return f"{first}..{last} (round-{R} harvest)"


# --------------------------------------------------------------------------- #
# harvest windows: maximal runs of transitions whose before-phase is "feeding"
# --------------------------------------------------------------------------- #
def _feeding_runs(states, trans):
    runs, cur = [], []
    for j in range(len(trans)):
        if states[j].get("phase") == FEEDING:
            cur.append(j)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return runs


# --------------------------------------------------------------------------- #
# main audit
# --------------------------------------------------------------------------- #
def _audit(trace):
    msgs = []
    states, trans = _build_chain(trace, msgs)
    if len(states) < 2:
        msgs.append("trace contains no state transitions to audit")
        return msgs
    n = len(states[0]["players"])

    runs = _feeding_runs(states, trans)
    rounds_seen = {s.get("round") for s in states if isinstance(s.get("round"), int)}
    max_round = max(rounds_seen, default=0)
    in_harvest = set()
    harvests = {}  # R -> list of runs
    for run in runs:
        R = states[run[0]].get("round")
        in_harvest.update(run)
        if R not in HARVEST_ROUNDS:
            msgs.append(f"{_span_label(R, run[0], run[-1], trans)}: feeding phase in round {R}, "
                        f"but harvests occur only at the end of rounds "
                        f"{', '.join(map(str, HARVEST_ROUNDS))}")
        else:
            harvests.setdefault(R, []).append(run)
    for R in HARVEST_ROUNDS:
        if R in harvests:
            if len(harvests[R]) > 1:
                labels = ", ".join(_span_label(R, r[0], r[-1], trans) for r in harvests[R])
                msgs.append(f"round {R}: {len(harvests[R])} separate feeding runs ({labels}); "
                            f"expected one harvest")
        elif R in rounds_seen or max_round > R:
            msgs.append(f"harvest at end of round {R}: no feeding phase recorded although "
                        f"the trace reaches round {max_round}")

    # ---- global invariants (hold at every transition) --------------------- #
    for j, t in enumerate(trans):
        pre, post = states[j], states[j + 1]
        lab = t["label"]
        for p in range(n):
            a, b = pre["players"][p], post["players"][p]
            if b["food"] < 0:
                msgs.append(f"{lab}: P{p} food is {b['food']} (negative)")
            if b["begging"] < 0:
                msgs.append(f"{lab}: P{p} begging is {b['begging']} (negative)")
            if b["begging"] < a["begging"]:
                msgs.append(f"{lab}: P{p} begging fell {a['begging']}->{b['begging']}; "
                            f"there is no way to get rid of begging markers")
            if b["begging"] > a["begging"] and j not in in_harvest:
                msgs.append(f"{lab}: P{p} begging rose {a['begging']}->{b['begging']} in round "
                            f"{pre.get('round')} phase '{pre.get('phase')}' outside any feeding "
                            f"phase")

    # ---- learn which goods index each convertible kind is ----------------- #
    kind_index = _learn_kind_indices(states, trans, n, msgs)

    # ---- each harvest, net effect per player ----------------------------- #
    for R in HARVEST_ROUNDS:
        for run in harvests.get(R, []):
            for p in range(n):
                _audit_harvest(R, run, p, states, trans, kind_index, msgs)

    # ---- final / summary -------------------------------------------------- #
    final = trace.get("final")
    summ = trace.get("summary") or {}
    sb = summ.get("begging")
    if final is not None and isinstance(sb, list):
        for p in range(min(n, len(sb))):
            fb = final["players"][p]["begging"]
            if sb[p] != fb:
                msgs.append(f"final: summary.begging[{p}]={sb[p]} but final state records "
                            f"{fb} begging markers for P{p}")
    return msgs


# --------------------------------------------------------------------------- #
# goods-index inference for every convert kind (crops and building resources).
# Self-consistency only; bundled steps may move more than one index, so only
# unambiguous single-index decreases are used.
# --------------------------------------------------------------------------- #
def _learn_kind_indices(states, trans, n, msgs):
    kind_index = {}
    for j, t in enumerate(trans):
        if not _is_convert(t):
            continue
        kind = _kind(t)
        p = t["actor"]
        if not kind or not isinstance(p, int) or not (0 <= p < n):
            continue
        a, b = states[j]["players"][p], states[j + 1]["players"][p]
        down = [k for k in range(len(a["goods"])) if b["goods"][k] < a["goods"][k]]
        if len(down) != 1:
            continue  # animal conversion, bundled step, or nothing consumed
        idx = down[0]
        known = kind_index.setdefault(kind, idx)
        if known != idx:
            msgs.append(f"{t['label']}: P{p} convert '{kind}' drew from goods index {idx} but "
                        f"earlier '{kind}' conversions drew from index {known}")
    seen = {}
    for kind, idx in sorted(kind_index.items()):
        if idx in seen:
            msgs.append(f"goods index {idx} is used for both '{seen[idx]}' and '{kind}' "
                        f"conversions")
        seen[idx] = kind
    return kind_index


# --------------------------------------------------------------------------- #
# one player, one harvest — net effect over one feeding run
# --------------------------------------------------------------------------- #
def _audit_harvest(R, run, p, states, trans, kind_index, msgs):
    h0 = run[0]
    # terminal transition: the one whose after-state is round R+1, else the last
    f = next((j for j in run if states[j + 1].get("round") == R + 1), run[-1])
    e = f + 1  # index of the state after the terminal transition
    lab = _span_label(R, h0, f, trans)
    start = states[h0]["players"][p]
    majors = max(_majors(states[j]["players"][p]) for j in range(h0, e + 1))
    n_goods = len(start["goods"])
    n_anim = len(start["animals"])
    crop_idx = {kind_index[k] for k in CROP_KINDS if k in kind_index}
    crops_known = len(crop_idx) == len(CROP_KINDS)
    idx_kind = {v: k for k, v in kind_index.items()}

    # ---- settlement: the player's own last action inside the run ---------- #
    own = [j for j in range(h0, e) if trans[j]["actor"] == p and trans[j]["action"] is not None]
    if own:
        settle = own[-1]
    else:
        changed = [j for j in range(h0, e)
                   if any(states[j]["players"][p][k] != states[j + 1]["players"][p][k]
                          for k in ("food", "goods", "begging"))]
        settle = changed[-1] if changed else f
    settled = states[settle + 1]["players"][p]
    slab = trans[settle]["label"]

    # ---- what left the supply up to settlement ---------------------------- #
    goods_out = [0] * n_goods
    anim_out_pre = [0] * n_anim
    for j in range(h0, settle + 1):
        a, b = states[j]["players"][p], states[j + 1]["players"][p]
        for k in range(n_goods):
            if b["goods"][k] < a["goods"][k]:
                goods_out[k] += a["goods"][k] - b["goods"][k]
        for k in range(n_anim):
            if b["animals"][k] < a["animals"][k]:
                anim_out_pre[k] += a["animals"][k] - b["animals"][k]
    animals_eaten = sum(anim_out_pre)

    # ---- classify goods that left: crops vs building resources ------------ #
    # A non-crop convert action by this player in the run may cover index k if
    # its kind maps to k, or if its kind's index is still unknown.
    resource_converts = [_kind(trans[j]) for j in range(h0, settle + 1)
                         if _is_convert(trans[j]) and trans[j]["actor"] == p
                         and _kind(trans[j]) not in CROP_KINDS]
    crops_eaten = 0
    resources_eaten = 0
    for k in range(n_goods):
        units = goods_out[k]
        if not units:
            continue
        if k in crop_idx or not crops_known:
            crops_eaten += units  # crop, or indistinguishable from one yet
            continue
        name = idx_kind.get(k, f"goods index {k}")
        if majors == 0:
            msgs.append(f"{lab}: P{p} {name} fell by {units} during feeding with no improvement "
                        f"(only grain and vegetables are food without improvements)")
            continue
        matching = [kd for kd in resource_converts
                    if kind_index.get(kd, k) == k]
        if not matching:
            msgs.append(f"{lab}: P{p} {name} fell by {units} during feeding with no matching "
                        f"['convert', '{name}'] action")
            continue
        resources_eaten += units

    # ---- nothing of the player's food economy may move after settlement --- #
    # Exception: a player with an improvement may GAIN food in the terminal
    # transition only (start-of-round income from a major improvement).
    for j in range(settle + 1, e):
        a, b = states[j]["players"][p], states[j + 1]["players"][p]
        for key in ("food", "goods", "begging"):
            if a[key] == b[key]:
                continue
            if key == "food" and j == f and majors > 0 and b["food"] > a["food"]:
                continue
            msgs.append(f"{trans[j]['label']}: P{p} {key} changed {a[key]}->{b[key]} after "
                        f"their feeding settled at {slab}")

    # ---- animals: lost only via an improvement; bred only in terminal step - #
    lost_total = 0
    newborn = [0] * n_anim
    for j in range(h0, e):
        a, b = states[j]["players"][p], states[j + 1]["players"][p]
        for k in range(n_anim):
            d = b["animals"][k] - a["animals"][k]
            if d < 0:
                lost_total -= d
            elif d > 0 and j != f:
                msgs.append(f"{trans[j]['label']}: P{p} animal type {k} rose "
                            f"{a['animals'][k]}->{b['animals'][k]} before the breeding "
                            f"transition ({trans[f]['label']}); animals breed after feeding")
                newborn[k] += d
    if lost_total and majors == 0:
        msgs.append(f"{lab}: P{p} lost {lost_total} animal(s) during the harvest with no "
                    f"improvement (animals do not provide food per se and may not be "
                    f"converted during breeding)")
    base, after = states[f]["players"][p]["animals"], states[e]["players"][p]["animals"]
    for k in range(n_anim):
        d = after[k] - base[k]
        if d > 0:
            newborn[k] += d
            if base[k] < 2:
                msgs.append(f"{trans[f]['label']}: P{p} animal type {k} rose {base[k]}->{after[k]} "
                            f"in breeding but at least 2 animals of a type are needed to breed")
            if d > 1:
                msgs.append(f"{trans[f]['label']}: P{p} animal type {k} rose by {d} at once "
                            f"({base[k]}->{after[k]}); breeding gives exactly 1 newborn")
        if newborn[k] > 1:
            msgs.append(f"{lab}: P{p} animal type {k} gained {newborn[k]} newborns "
                        f"(at most 1 newborn of each type per harvest)")

    # ---- feeding arithmetic ----------------------------------------------- #
    owed, adults, born = _owed(start, lab, p, msgs)
    dbeg = settled["begging"] - start["begging"]
    head = f"{slab}: P{p} owed {owed} food ({adults} adults, {born} newborn)"

    if majors == 0:
        # exact: every crop that left is worth exactly 1 food, nothing else feeds
        available = start["food"] + crops_eaten
        paid = available - settled["food"]
        detail = (f"paid {paid} ({start['food']} food + {crops_eaten} crops - "
                  f"{settled['food']} food kept)")
        if paid < 0:
            msgs.append(f"{head}, but food rose {start['food']}->{settled['food']} with only "
                        f"{crops_eaten} crops converted (1 food each) and no improvement")
            return
        if paid > owed:
            msgs.append(f"{head}, {detail}; overpaid by {paid - owed}")
        expected = max(0, owed - paid)
        if dbeg != expected:
            msgs.append(f"{head}, {detail}, begging rose by {dbeg} (expected {expected})")
        if dbeg > 0 and settled["food"] > 0:
            msgs.append(f"{head}, went begging for {dbeg} while still holding "
                        f"{settled['food']} food after feeding")
    else:
        # improvement rates unknown: crops >= 1, resources >= 1, animals >= 2,
        # so `paid` is a lower bound and begging can only be bounded from above
        available_min = (start["food"] + crops_eaten
                         + RESOURCE_FOOD_MIN * resources_eaten
                         + ANIMAL_FOOD_MIN * animals_eaten)
        paid_min = available_min - settled["food"]
        detail = (f"paid at least {paid_min} ({start['food']} food + {crops_eaten} crops + "
                  f"{resources_eaten} resources x{RESOURCE_FOOD_MIN} + "
                  f"{animals_eaten} animals x{ANIMAL_FOOD_MIN} - {settled['food']} food kept)")
        if paid_min > owed:
            msgs.append(f"{head}, {detail}; overpaid by at least {paid_min - owed}")
        if dbeg > max(0, owed - paid_min):
            msgs.append(f"{head}, {detail}, begging rose by {dbeg} "
                        f"(at most {max(0, owed - paid_min)} food could be missing)")
        if dbeg > 0 and settled["food"] > 0:
            msgs.append(f"{head}, went begging for {dbeg} while still holding "
                        f"{settled['food']} food after feeding")
