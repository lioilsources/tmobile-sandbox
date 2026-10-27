#!/usr/bin/env python3
"""toolshim — minimal OpenAI-compatible proxy that adds tool calling to a vLLM
server started WITHOUT --enable-auto-tool-choice (e.g. the L40S batch profile L).

How: tools stay in the request (vLLM renders them into the Qwen3 chat template),
but tool_choice is forced to "none"; the model then emits Hermes-style
<tool_call>{json}</tool_call> blocks as plain text, which the shim parses back
into OpenAI `tool_calls`. Upstream is always called non-streaming; if the client
asked for `stream: true`, the shim replays the answer as SSE chunks.

Only meant for smoke tests of the OpenCode swarm config. Real runs use a vLLM
started with `--enable-auto-tool-choice --tool-call-parser qwen3_coder|hermes`.

Usage: toolshim.py --upstream http://127.0.0.1:8000/v1 --port 18099 [--log calls.jsonl]
                   [--ctx 4096]   # clamp max_tokens so prompt+completion fits
"""
import argparse, json, re, sys, time, uuid, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ARGS = None


def upstream(path, body=None, method="GET"):
    req = urllib.request.Request(ARGS.upstream.rstrip("/") + path, method=method,
                                 data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw)
        except Exception:
            return e.code, {"error": {"message": raw.decode(errors="replace")}}


def count_prompt_tokens(body):
    root = ARGS.upstream.rstrip("/")
    root = root[:-3] if root.endswith("/v1") else root
    req = urllib.request.Request(root + "/tokenize", method="POST",
                                 data=json.dumps({"model": body.get("model"), "messages": body.get("messages", []),
                                                  "tools": body.get("tools")}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("count")
    except Exception:
        return None


def parse_tool_calls(text):
    calls = []
    for m in TOOL_RE.finditer(text or ""):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        calls.append({"id": "call_" + uuid.uuid4().hex[:12], "type": "function",
                      "function": {"name": obj.get("name", ""),
                                   "arguments": args if isinstance(args, str) else json.dumps(args)}})
    rest = TOOL_RE.sub("", text or "").strip()
    return calls, rest


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        st, js = upstream(self.path[3:] if self.path.startswith("/v1") else self.path)
        self._json(st, js)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self.path.endswith("/chat/completions"):
            st, js = upstream(self.path[3:], body, "POST")
            return self._json(st, js)
        if ARGS.dump:
            with open(ARGS.dump, "a") as f:
                f.write(json.dumps(body) + "\n")
        stream = bool(body.pop("stream", False))
        body.pop("stream_options", None)
        has_tools = bool(body.get("tools"))
        if ARGS.slim_tools and has_tools:
            # Smoke-only: shorten tool descriptions so OpenCode fits a 4k-context server.
            for t in body["tools"]:
                f = t.get("function", {})
                d = f.get("description") or ""
                if len(d) > ARGS.slim_tools:
                    f["description"] = d[:ARGS.slim_tools].rsplit(" ", 1)[0] + " ..."
                for prop in (f.get("parameters") or {}).get("properties", {}).values():
                    if isinstance(prop, dict) and len(prop.get("description", "")) > 120:
                        prop["description"] = prop["description"][:120]
        if has_tools:
            body["tool_choice"] = "none"
            body.pop("parallel_tool_calls", None)
        ptok = count_prompt_tokens(body) if (ARGS.ctx or ARGS.log) else None
        if ARGS.ctx and ptok:
            room = ARGS.ctx - ptok - 8
            mt = body.get("max_tokens") or body.get("max_completion_tokens")
            if room < 64:
                pass  # let upstream produce the real context-length error
            elif not mt or mt > room:
                body["max_tokens"] = room
                body.pop("max_completion_tokens", None)
        t0 = time.time()
        st, js = upstream("/chat/completions", body, "POST")
        dt = time.time() - t0
        if st == 200 and has_tools:
            msg = js["choices"][0]["message"]
            calls, rest = parse_tool_calls(msg.get("content"))
            if calls:
                msg["tool_calls"] = calls
                msg["content"] = rest or None
                js["choices"][0]["finish_reason"] = "tool_calls"
        if ARGS.log:
            with open(ARGS.log, "a") as f:
                f.write(json.dumps({"ts": time.time(), "status": st, "secs": round(dt, 2),
                                    "prompt_tokens_est": ptok, "n_tools": len(body.get("tools") or []),
                                    "n_msgs": len(body.get("messages", [])),
                                    "usage": js.get("usage") if st == 200 else None,
                                    "tool_calls": [c["function"]["name"] for c in
                                                   (js["choices"][0]["message"].get("tool_calls") or [])] if st == 200 else None,
                                    "error": js.get("error") if st != 200 else None}) + "\n")
        if st != 200 or not stream:
            return self._json(st, js)
        self._sse(js)

    def _sse(self, js):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        base = {"id": js.get("id", "chatcmpl-shim"), "object": "chat.completion.chunk",
                "created": js.get("created", int(time.time())), "model": js.get("model")}
        ch = js["choices"][0]
        msg = ch["message"]

        def emit(delta, finish=None, usage=None):
            c = dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}])
            if usage is not None:
                c = dict(base, choices=[], usage=usage)
            self.wfile.write(b"data: " + json.dumps(c).encode() + b"\n\n")
            self.wfile.flush()

        emit({"role": "assistant", "content": ""})
        if msg.get("reasoning_content"):
            emit({"reasoning_content": msg["reasoning_content"]})
        if msg.get("content"):
            emit({"content": msg["content"]})
        for i, tc in enumerate(msg.get("tool_calls") or []):
            emit({"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                  "function": {"name": tc["function"]["name"],
                                               "arguments": tc["function"]["arguments"]}}]})
        emit({}, ch.get("finish_reason") or "stop")
        if js.get("usage"):
            emit({}, usage=js["usage"])
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--port", type=int, default=18099)
    ap.add_argument("--log")
    ap.add_argument("--ctx", type=int, default=0)
    ap.add_argument("--slim-tools", type=int, default=0,
                    help="truncate tool descriptions to N chars (smoke tests on tiny contexts only)")
    ap.add_argument("--dump", help="append raw request bodies (jsonl) — debugging only")
    ARGS = ap.parse_args()
    print(f"toolshim :{ARGS.port} -> {ARGS.upstream}", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
