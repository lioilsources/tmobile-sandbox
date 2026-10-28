"""Mini proxy pro OpenEvolve: do každého /v1/chat/completions vloží chat_template_kwargs
{"enable_thinking": false} (OpenEvolve 0.4 neumí poslat extra body a vLLM s --reasoning-parser qwen3
jinak vrací myšlení v reasoning_content a content=None). Bez závislostí.

  python nothink_proxy.py --port 8009 --upstream http://127.0.0.1:8000
"""
import argparse
import json
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UP = "http://127.0.0.1:8000"


class P(BaseHTTPRequestHandler):
    def _fwd(self, body=None):
        if body is not None and self.path.endswith("/chat/completions"):
            d = json.loads(body)
            d.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
            body = json.dumps(d).encode()
        req = urllib.request.Request(UP + self.path, body, {"Content-Type": "application/json"}, method=self.command)
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                code, data = r.status, r.read()
        except urllib.error.HTTPError as e:
            code, data = e.code, e.read()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._fwd()

    def do_POST(self):
        self._fwd(self.rfile.read(int(self.headers.get("Content-Length", 0))))

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8009)
    ap.add_argument("--upstream", default=UP)
    a = ap.parse_args()
    UP = a.upstream.rstrip("/")
    ThreadingHTTPServer(("127.0.0.1", a.port), P).serve_forever()
