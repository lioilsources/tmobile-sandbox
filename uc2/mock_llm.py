#!/usr/bin/env python3
"""UC2: náhrada LLM pro zkoušku smyčky bez GPU: vrací náhodný produkční filtr jako „dítě“ (OpenAI chat API).

  mock_llm.py --shaders /srv/work/uc2/MirrorBooth/mirrorbooth/shaders --port 8090
"""
import argparse, json, pathlib, random
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ap = argparse.ArgumentParser(); ap.add_argument("--shaders", required=True); ap.add_argument("--port", type=int, default=8090)
a = ap.parse_args(); FRAGS = [p.read_text() for p in pathlib.Path(a.shaders).glob("*.frag")]


class H(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        body = json.dumps({"choices": [{"message": {"content": "```glsl\n" + random.choice(FRAGS) + "```"}}],
                           "usage": {"completion_tokens": 0}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)

    def log_message(self, *_): pass


ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
