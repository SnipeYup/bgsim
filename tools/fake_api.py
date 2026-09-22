"""Fake Anthropic Messages API for offline testing of web/llm.py.

Speaks the real SSE protocol (message_start, thinking + text content blocks,
message_delta with stop_reason/usage, message_stop). The request body's
first user message content decides the scenario via a marker substring:

  [SCENARIO:ok]        stream thinking, then a valid engine file with unicode
  [SCENARIO:truncate]  stream half an engine, stop_reason=max_tokens
  [SCENARIO:reset]     first call drops the socket mid-stream, later calls ok
  [SCENARIO:error]     API-style error event
  (no marker)          same as ok

Run: python tools/fake_api.py 8555
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

ENGINE = open(__file__.replace("fake_api.py", "../bgsim/games/agricola/game.py"),
              encoding="utf-8").read().replace('"""', '"""Agricola — fake model output\n', 1)
STATE = {"reset_done": False}


def sse(event: dict) -> bytes:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        assert body.get("stream") is True, "client must stream"
        if body["model"].startswith("claude-haiku"):
            # the real API rejects effort on Haiku 4.5; the client must not send it
            if "output_config" in body:
                self.send_response(400); self.send_header("content-type", "application/json"); self.end_headers()
                self.wfile.write(b'{"type":"error","error":{"type":"invalid_request_error","message":"output_config: not supported for this model"}}')
                return
        else:
            assert body["output_config"]["effort"] in ("low", "medium", "high"), "effort missing"
        if "[SCENARIO:400]" in json.dumps(body):
            self.send_response(400); self.send_header("content-type", "application/json"); self.end_headers()
            self.wfile.write(b'{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: must be <= 64000"}}')
            return
        assert body["max_tokens"] >= 100
        text = body["messages"][0]["content"]
        # ---- non-engine prompts: review / plan / checker
        if "Turn them into ONE concrete scenario" in text:
            return self._stream_text(json.dumps({"scenario": "Round 9 harvest: a family of 4 adults and 1 newborn with 7 food.",
                "question": "How much food does this family owe?", "engine_value": "8 food", "auditor_value": "9 food"}))
        if "You are a rules expert. Using ONLY the rulebook" in text:
            if "[JURY:unclear]" in text: return self._stream_text(json.dumps({"answer": "", "basis": "unclear", "quote": "", "note": "the rulebook never defines newborn"}))
            if "[JURY:engine]" in text: return self._stream_text(json.dumps({"answer": "8 food", "basis": "stated", "quote": "2 per person", "note": ""}))
            return self._stream_text(json.dumps({"answer": "9 food: 2 per adult and 1 for the newborn", "basis": "stated", "quote": "newborn only requires 1 food", "note": ""}))
        if "Say which candidate the expert's answer agrees with" in text:
            ans = text.split("Expert's answer:")[1]
            side = "A" if "8 food" in ans else "B" if "9" in ans else "neither"
            return self._stream_text(json.dumps({"agrees_with": side}))
        if "restating a board game rulebook as a precise structured outline" in text:
            return self._stream_text(json.dumps({"outline": [
                {"section": "setup", "lines": [
                    {"text": "Each player starts with 3 food.", "basis": "stated", "assumption": "", "quote": "each other player gets 3 food"},
                    {"text": "The first field may be placed anywhere.", "basis": "inferred", "assumption": "no adjacency needed for the first field", "quote": ""}]},
                {"section": "end of game", "lines": [
                    {"text": "Ties are unresolved.", "basis": "unclear", "assumption": "shared victory", "quote": ""}]}],
                "cost_table": [{"action": "Build room", "cost": "5 wood 2 reed", "effect": "one room", "basis": "stated"},
                               {"action": "Build stable", "cost": "2 wood", "effect": "one stable", "basis": "inferred"}]}))
        if "narrate one complete sample round of play" in text:
            return self._stream_text(json.dumps({"steps": [
                {"text": "Player 1 takes 3 wood from the Forest.", "assumption": ""},
                {"text": "Player 2 plows a field next to their house.", "rule_gap": "whether the first field must be adjacent to the house"},
                {"text": "Player 1 takes white, blue and green gems.", "rule_gap": ""}]}))
        if "decide whether the rulebook text actually" in text:
            n = text.split("=== questions ===")[1].count("\n")
            return self._stream_text(json.dumps([{"answered": True, "answer": "Exactly three; the text says 'three different colours'.", "quote": "three gem tokens of different colours"}] + [{"answered": False, "answer": "", "quote": ""}] * max(0, n - 1)))
        if "meticulous board game rules editor" in text:
            return self._stream_text(json.dumps([
                {"kind": "ambiguous", "quote": "take 3 gem tokens of different colours",
                 "question": "May a player take fewer than 3 different tokens by choice?"},
                {"kind": "missing", "quote": "End of the game",
                 "question": "If two players tie on points and card count, who wins?"}]))
        if "You explain a disagreement about a board game" in text:
            import re
            names = re.findall(r"^\[(\w+)\]", text.split("=== auditors ===")[1], re.M)
            return self._stream_text(json.dumps([{
                "name": n, "simulation_did": "Turn 13: Player 2 took three gems and, holding 11, returned one to the supply before the turn passed.",
                "rule": "You may never end your turn with more than 10 tokens.",
                "question": "Is returning excess gems down to 10 a step within the turn, or an extra action?",
                "answer_if_simulation_right": "Returning excess gems is a step of the turn: after your action you must return gems until you hold 10.",
                "answer_if_auditor_right": "Nothing beyond the four actions may happen on a turn; a player may never hold more than 10 gems at any point."} for n in names]))
        if "Split the rulebook below into" in text:
            return self._stream_text(json.dumps([
                {"name": "turn_order", "text": "Players take turns in order. Each turn one action."},
                {"name": "end_game", "text": "The game ends after round 14."}]))
        if "You are writing an independent auditor" in text:
            name = text.split('NAME = "')[1].split('"')[0]
            checker = (f'NAME = "{name}"\nRULE = "fake checker for {name}"\n\n'
                       f'def check(trace):\n'
                       f'    out = []\n'
                       f'    if trace["meta"]["n_players"] < 1: out.append("step 0: impossible")\n'
                       f'    return out\n')
            return self._stream_text(checker)
        scenario = "ok"
        for s in ("truncate", "reset", "error"):
            if f"[SCENARIO:{s}]" in text:
                scenario = s
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        w = self.wfile
        w.write(sse({"type": "message_start", "message": {"usage": {"input_tokens": 12345}}}))
        # a thinking block first, like adaptive thinking would produce
        w.write(sse({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}))
        w.write(sse({"type": "content_block_delta", "index": 0,
                     "delta": {"type": "thinking_delta", "thinking": "planning the engine..."}}))
        w.write(sse({"type": "content_block_stop", "index": 0}))
        if scenario == "error":
            w.write(sse({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}))
            return
        w.write(sse({"type": "content_block_start", "index": 1, "content_block": {"type": "text"}}))
        code = ENGINE
        if scenario == "truncate":
            code = code[: len(code) // 2]
        chunks = [code[i:i + 2000] for i in range(0, len(code), 2000)]
        for k, ch in enumerate(chunks):
            if scenario == "reset" and not STATE["reset_done"] and k == 3:
                STATE["reset_done"] = True
                self.connection.close()  # simulate WinError 10054 mid-stream
                return
            w.write(sse({"type": "content_block_delta", "index": 1,
                         "delta": {"type": "text_delta", "text": ch}}))
            w.flush()
        w.write(sse({"type": "content_block_stop", "index": 1}))
        w.write(sse({"type": "message_delta",
                     "delta": {"stop_reason": "max_tokens" if scenario == "truncate" else "end_turn"},
                     "usage": {"output_tokens": 25000}}))
        w.write(sse({"type": "message_stop"}))


    def _stream_text(self, text: str):
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        w = self.wfile
        w.write(sse({"type": "message_start", "message": {"usage": {"input_tokens": 3000}}}))
        w.write(sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text"}}))
        w.write(sse({"type": "content_block_delta", "index": 0,
                     "delta": {"type": "text_delta", "text": text}}))
        w.write(sse({"type": "content_block_stop", "index": 0}))
        w.write(sse({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                     "usage": {"output_tokens": 800}}))
        w.write(sse({"type": "message_stop"}))


def serve(port: int):
    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8555
    print(f"fake API on http://127.0.0.1:{port}/v1/messages")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
