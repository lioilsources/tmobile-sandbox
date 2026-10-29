#!/usr/bin/env python3
"""Fake ComfyUI for CPU smoke tests: same API shapes as ComfyUI 0.42 (/prompt, /history, /view, /queue,
/object_info, /system_stats). /prompt validates the graph with comfy.validate against the object_info
snapshot and answers 400 + node_errors like the real server. Prompts "execute" one at a time in
`--exec-s` seconds; a prompt whose text contains FAILME ends with an execution_error.

  python3 fake_comfy.py --port 18188 --object-info ../object_info.json --exec-s 0.05
"""
import argparse, io, json, os, sys, threading, time, uuid, zlib, struct
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import comfy  # noqa: E402


def tiny_png(seed: int) -> bytes:
    w = h = 8
    raw = b"".join(b"\x00" + bytes([(seed + x * 7 + y * 3) % 256 for x in range(w) for _ in range(3)]) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class State:
    def __init__(self, oi, exec_s, out, images=()):
        self.oi, self.exec_s, self.out, self.images = oi, exec_s, out, list(images)
        self.queue, self.history, self.files, self.lock = [], {}, {}, threading.Lock()

    def worker(self):
        while True:
            with self.lock:
                item = self.queue[0] if self.queue else None
            if not item:
                time.sleep(0.01)
                continue
            pid, g = item
            t0 = int(time.time() * 1000)
            time.sleep(self.exec_s)
            t1 = int(time.time() * 1000)
            text = json.dumps(g)
            with self.lock:
                self.queue.pop(0)
                if "FAILME" in text:
                    self.history[pid] = {"status": {"status_str": "error", "completed": False, "messages": [
                        ["execution_start", {"prompt_id": pid, "timestamp": t0}],
                        ["execution_error", {"prompt_id": pid, "node_type": "KSampler", "exception_type": "RuntimeError",
                                             "exception_message": "fake failure", "timestamp": t1}]]}, "outputs": {}}
                    continue
                save = next(n for n in g.values() if n["class_type"] == "SaveImage")
                pref = save["inputs"]["filename_prefix"]
                sub, name = os.path.dirname(pref), os.path.basename(pref) + "_00001_.png"
                if self.images:  # real PNGs (smoke test of the scorers on real pictures)
                    self.files[(sub, name)] = open(self.images[hash(pid) % len(self.images)], "rb").read()
                else:
                    self.files[(sub, name)] = tiny_png(hash(pid) & 0xFF)
                if self.out:
                    os.makedirs(os.path.join(self.out, sub), exist_ok=True)
                    open(os.path.join(self.out, sub, name), "wb").write(self.files[(sub, name)])
                self.history[pid] = {"status": {"status_str": "success", "completed": True, "messages": [
                    ["execution_start", {"prompt_id": pid, "timestamp": t0}],
                    ["execution_success", {"prompt_id": pid, "timestamp": t1}]]},
                    "outputs": {"save": {"images": [{"filename": name, "subfolder": sub, "type": "output"}]}}}


def handler(st: State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj=None, raw=None, ctype="application/json"):
            b = raw if raw is not None else json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/system_stats":
                return self._send(200, {"system": {"comfyui_version": "fake"}, "devices": []})
            if u.path == "/object_info":
                return self._send(200, st.oi)
            if u.path.startswith("/history/"):
                pid = u.path.split("/")[-1]
                with st.lock:
                    h = st.history.get(pid)
                return self._send(200, {pid: h} if h else {})
            if u.path == "/queue":
                with st.lock:
                    q = [[i, pid] for i, (pid, _) in enumerate(st.queue)]
                return self._send(200, {"queue_running": q[:1], "queue_pending": q[1:]})
            if u.path == "/view":
                q = parse_qs(u.query)
                key = (q.get("subfolder", [""])[0], q["filename"][0])
                with st.lock:
                    b = st.files.get(key)
                return self._send(200, raw=b, ctype="image/png") if b else self._send(404, {"error": "nf"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if u.path == "/prompt":
                errs = comfy.validate(body["prompt"], st.oi)
                if errs:
                    return self._send(400, {"error": {"type": "prompt_outputs_failed_validation"},
                                            "node_errors": {"x": {"errors": errs}}})
                pid = str(uuid.uuid4())
                with st.lock:
                    st.queue.append((pid, body["prompt"]))
                return self._send(200, {"prompt_id": pid, "number": len(st.queue), "node_errors": {}})
            if u.path == "/queue":
                with st.lock:
                    dele = set(body.get("delete", []))
                    st.queue = [q for i, q in enumerate(st.queue) if i == 0 or q[0] not in dele]
                return self._send(200, {})
            self._send(404, {"error": "not found"})
    return H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18188)
    ap.add_argument("--object-info", required=True)
    ap.add_argument("--exec-s", type=float, default=0.05)
    ap.add_argument("--out", default="", help="also write files like ComfyUI's output dir")
    ap.add_argument("--images-glob", default="", help="serve these real PNGs instead of 8×8 dummies")
    a = ap.parse_args()
    import glob
    st = State(json.load(open(a.object_info)), a.exec_s, a.out, sorted(glob.glob(a.images_glob))[:64] if a.images_glob else ())
    threading.Thread(target=st.worker, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", a.port), handler(st)).serve_forever()
