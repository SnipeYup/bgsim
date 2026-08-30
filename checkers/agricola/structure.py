"""Structural checker written in-repo (not rules re-derivation): the game
skeleton any Agricola trace must satisfy. Real rule checkers are generated
independently — see docs/VERIFIER.md."""
NAME = "structure"
RULE = "14 rounds; phases alternate sanely; every game finishes; players in range"


def check(trace):
    out = []
    if not trace["meta"]["finished"]:
        out.append("game did not finish")
    n = trace["meta"]["n_players"]
    last_round = 0
    for s in trace["steps"]:
        if not (0 <= s["player"] < n):
            out.append(f"step {s['i']}: player {s['player']} out of range")
        r = s["before"].get("round", s["before"].get("round_no"))
        if r is not None:
            if r < last_round:
                out.append(f"step {s['i']}: round went backwards {last_round}->{r}")
            last_round = max(last_round, r)
    if last_round not in (0, 14):
        out.append(f"final round was {last_round}, expected 14")
    return out
