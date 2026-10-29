#!/usr/bin/env python3
"""Fake OpenAI/vLLM endpoint for CPU smoke tests of vlm_judge.py and uc19/translate.py.

/v1/models, /v1/chat/completions. With response_format json_schema it answers a schema-valid instance (checked
with jsonschema), varying enum choices by a hash of the request; without one it answers plain text. A request
whose text contains FAIL400 gets HTTP 400 with a vLLM-like body. Requests without
chat_template_kwargs.enable_thinking=false for a qwen3 text model get empty content (like the real parser).

  python3 fake_vllm.py --port 18004 --model qwen3-vl-8b
"""
import argparse, hashlib, json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jsonschema import validate

WORDS = ["red fox", "snow", "forest", "sitting", "winter", "1girl", "source_anime", "masterpiece", "nonexistent tag xyz"]


def inst(s, h, name=""):
    t = s.get("type")
    if "enum" in s:
        return s["enum"][h % len(s["enum"])]
    if t == "object":
        return {k: inst(v, h + i, k) for i, (k, v) in enumerate(s.get("properties", {}).items())}
    if t == "array":
        n = min(s.get("maxItems", 3), 6 if name == "tags" else 2)
        return [inst(s["items"], h + i, name)[: s["items"].get("maxLength", 99)] for i in range(n)] if name == "tags" else \
            [WORDS[(h + i) % 3] for i in range(n)]
    if t == "string":
        if name == "tags":
            return WORDS[h % len(WORDS)]
        if name == "prompt":
            return "a red fox sitting in fresh snow in a winter forest, (soft light:1.2), photo"
        return "blurry"
    if t == "integer":
        return 3
    return None


def handler(model):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.endswith("/models"):
                return self._send(200, {"data": [{"id": model}]})
            if self.path.endswith("/health"):
                return self._send(200, {})
            self._send(404, {})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            text = json.dumps(body["messages"][-1])
            if "FAIL400" in text:
                return self._send(400, {"object": "error", "message": "fake: schema rejected", "code": 400})
            h = int(hashlib.sha256(text.encode()).hexdigest()[:6], 16)
            rf = body.get("response_format")
            if rf and rf.get("type") == "json_schema":
                schema = rf["json_schema"]["schema"]
                ans = inst(schema, h)
                validate(ans, schema)
                content = json.dumps(ans)
            else:
                content = "A red fox sitting in fresh snow in a winter forest."
            if "qwen3" in model and "vl" not in model and (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False:
                content = ""
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 1000, "completion_tokens": 50}})
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18004)
    ap.add_argument("--model", default="qwen3-vl-8b")
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(a.model)).serve_forever()
