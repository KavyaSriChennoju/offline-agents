"""
A pretend llama-server for testing the plumbing without a model.

    python tests/fake_llama_server.py --port 8080

It speaks the same JSON as llama-server (including the `timings` block) and
answers with canned text that mentions how many images it received.
Facilitators: handy for checking a room's Python setup before models finish downloading.
"""
import argparse
import os
import sys
import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def fake_answer(messages):
    last = messages[-1]
    content = last["content"]
    n_img, n_audio, text = 0, 0, content
    if isinstance(content, list):
        n_img = sum(1 for p in content if p.get("type") == "image_url")
        n_audio = sum(1 for p in content if p.get("type") == "input_audio")
        text = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    sys_prompt = messages[0]["content"] if messages[0]["role"] == "system" else ""
    if "HEARD:" in sys_prompt and n_audio:
        return "HEARD: what am I holding\nYou're holding a red square. It looks very square."
    if n_audio and "Describe this sound" in text:
        return "A short high beep."
    if "one word: ready" in text:
        return "ready"
    if "two shapes" in text:
        return "A red square and a blue circle."
    if "SECRET" in sys_prompt and text == "Start the game.":
        return "SECRET: coffee mug\nI spy with my little eye something white and round."
    if "ACTION:" in sys_prompt and "remember" in text:
        return "ACTION: save_note | buy coffee\nGot it, noted."
    if "ACTION:" in sys_prompt and "picture" in text:
        return "ACTION: take_photo | desk\nSnap, saved."
    if "YES or NO" in sys_prompt:
        return "YES" if "see" in text or "hair" in text else "NO"
    if "Transcribe" in text:
        return "testing one two three"
    return f"I got {n_img} image(s), {n_audio} audio clip(s) and {len(messages)} messages. You said: {text[:40]}. That is all."


def prompt_tokens(messages):
    n = 0
    for m in messages:
        c = m["content"]
        if isinstance(c, str):
            n += len(c) // 4
        else:
            for p in c:
                n += 256 if p.get("type") == "image_url" else len(p.get("text", "")) // 4
    return n


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._json({"status": "ok"})
        if self.path == "/v1/models":
            return self._json({"data": [{"id": "fake-vlm-Q4_0.gguf"}]})
        self._json({"error": "nope"}, 404)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = req["messages"]
        answer = fake_answer(msgs)
        if req.get("response_format", {}).get("type") == "json_schema":
            schema = str(req["response_format"])
            answer = '{"sequence": ["red", "blue"]}' if "sequence" in schema else '{"color": "green"}'
        # Simulate a thinking model (FAKE_THINKING=1): unless thinking is switched off,
        # every token goes to reasoning_content and the answer comes back empty.
        thinks = os.environ.get("FAKE_THINKING") == "1" and \
            (req.get("chat_template_kwargs") or {}).get("enable_thinking", True)
        words = re.findall(r"\S+\s*", answer)[: req.get("max_tokens", 256)]
        pn = prompt_tokens(msgs)
        timings = {"prompt_n": pn, "prompt_ms": pn * 0.5, "prompt_per_second": 2000,
                   "predicted_n": len(words), "predicted_ms": len(words) * 20, "predicted_per_second": 50.0}
        if not req.get("stream"):
            time.sleep(0.01)
            msg = {"role": "assistant", "content": "", "reasoning_content": "".join(words)} if thinks \
                else {"role": "assistant", "content": "".join(words)}
            return self._json({"choices": [{"message": msg}],
                               "timings": timings})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for w in words:
            chunk = {"choices": [{"delta": {"reasoning_content" if thinks else "content": w}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.005)
        final = {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": timings}
        self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode())


if __name__ == "__main__":
    if "--help" in sys.argv:
        print("--reasoning [on|off|auto]   use reasoning/thinking in the chat (fake help text)")
        raise SystemExit
    if "--version" in sys.argv:
        print("version: 0000 (fake llama-server for tests)")
        raise SystemExit
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    port = ap.parse_known_args()[0].port   # ignores llama-server flags like -m / --mmproj / -c
    print(f"fake llama-server on http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
