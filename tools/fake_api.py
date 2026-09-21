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
        assert body["output_config"]["effort"] in ("low", "medium", "high"), "effort missing"
        assert body["max_tokens"] >= 64000
        text = body["messages"][0]["content"]
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


def serve(port: int):
    srv = HTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8555
    print(f"fake API on http://127.0.0.1:{port}/v1/messages")
    HTTPServer(("127.0.0.1", port), H).serve_forever()
