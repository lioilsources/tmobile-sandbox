"""ComfyUI API: workflow builders (SDXL family, FLUX.1-dev), offline validator against /object_info, HTTP client.

validate(graph, object_info) checks what ComfyUI's /prompt would reject before a single GPU second is spent:
unknown class, missing required input, link to a missing node / output index, link type mismatch
(MODEL into CLIP …), enum value not offered (ckpt_name file missing, sampler typo), number out of range,
no output node. object_info comes from the live server or from dump_object_info.py (CPU, no server).
"""
from __future__ import annotations

import json, time, urllib.error, urllib.parse, urllib.request

from models import CKPTS, WIDTH, HEIGHT


# ---- workflows -----------------------------------------------------------------------------------
def build(ckpt_id: str, prompt: str, negative: str, seed: int, filename_prefix: str,
          width: int = WIDTH, height: int = HEIGHT, batch: int = 1) -> dict:
    c = CKPTS[ckpt_id]
    g = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": c["file"]}}}
    clip = ["ckpt", 1]
    if c.get("clip_skip", 1) > 1:
        g["clipskip"] = {"class_type": "CLIPSetLastLayer", "inputs": {"clip": clip, "stop_at_clip_layer": -c["clip_skip"]}}
        clip = ["clipskip", 0]
    g["pos"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": prompt}}
    g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": negative}}
    pos = ["pos", 0]
    if c["family"] == "flux":
        g["guid"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": pos, "guidance": c.get("guidance", 3.5)}}
        pos = ["guid", 0]
        g["lat"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": batch}}
    else:
        g["lat"] = {"class_type": "EmptyLatentImage", "inputs": {"width": width, "height": height, "batch_size": batch}}
    g["ks"] = {"class_type": "KSampler", "inputs": {
        "model": ["ckpt", 0], "positive": pos, "negative": ["neg", 0], "latent_image": ["lat", 0], "seed": int(seed),
        "steps": c["steps"], "cfg": c["cfg"], "sampler_name": c["sampler"], "scheduler": c["scheduler"], "denoise": 1.0}}
    g["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["ks", 0], "vae": ["ckpt", 2]}}
    g["save"] = {"class_type": "SaveImage", "inputs": {"images": ["dec", 0], "filename_prefix": filename_prefix}}
    return g


# ---- validator -----------------------------------------------------------------------------------
def _spec_type(spec):
    """object_info input spec → (type, options-or-None, cfg). Handles V1 [type, cfg] and [[enum], cfg], V3 COMBO."""
    t = spec[0]
    cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    if isinstance(t, list):
        return "COMBO", t, cfg
    if t == "COMBO":
        return "COMBO", cfg.get("options", []), cfg
    return t, None, cfg


def validate(graph: dict, oi: dict) -> list[str]:
    errs = []
    has_out = False
    for nid, node in graph.items():
        ct = node.get("class_type")
        if ct not in oi:
            errs.append(f"{nid}: unknown class_type {ct!r}")
            continue
        info = oi[ct]
        has_out |= bool(info.get("output_node"))
        req = info["input"].get("required", {})
        opt = info["input"].get("optional", {})
        inputs = node.get("inputs", {})
        for name in req:
            if name not in inputs:
                errs.append(f"{nid}({ct}): missing required input {name!r}")
        for name, val in inputs.items():
            spec = req.get(name) or opt.get(name)
            if spec is None:
                errs.append(f"{nid}({ct}): unknown input {name!r}")
                continue
            typ, options, cfg = _spec_type(spec)
            if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str) and isinstance(val[1], int):
                src = graph.get(val[0])
                if src is None:
                    errs.append(f"{nid}({ct}).{name}: link to missing node {val[0]!r}")
                    continue
                outs = oi.get(src.get("class_type"), {}).get("output", [])
                if val[1] >= len(outs):
                    errs.append(f"{nid}({ct}).{name}: {val[0]} has no output {val[1]}")
                    continue
                if typ not in ("*", outs[val[1]]) and options is None:
                    errs.append(f"{nid}({ct}).{name}: type {outs[val[1]]} → expected {typ}")
                continue
            if options is not None:
                if val not in options:
                    errs.append(f"{nid}({ct}).{name}: {val!r} not offered ({len(options)} options)")
            elif typ == "INT":
                if not isinstance(val, int) or isinstance(val, bool):
                    errs.append(f"{nid}({ct}).{name}: {val!r} is not INT")
                elif ("min" in cfg and val < cfg["min"]) or ("max" in cfg and val > cfg["max"]):
                    errs.append(f"{nid}({ct}).{name}: {val} outside [{cfg.get('min')}, {cfg.get('max')}]")
            elif typ == "FLOAT":
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    errs.append(f"{nid}({ct}).{name}: {val!r} is not FLOAT")
                elif ("min" in cfg and val < cfg["min"]) or ("max" in cfg and val > cfg["max"]):
                    errs.append(f"{nid}({ct}).{name}: {val} outside [{cfg.get('min')}, {cfg.get('max')}]")
            elif typ == "STRING":
                if not isinstance(val, str):
                    errs.append(f"{nid}({ct}).{name}: {val!r} is not STRING")
            elif typ == "BOOLEAN":
                if not isinstance(val, bool):
                    errs.append(f"{nid}({ct}).{name}: {val!r} is not BOOLEAN")
            else:
                errs.append(f"{nid}({ct}).{name}: literal {val!r} where a {typ} link is required")
    if not has_out:
        errs.append("graph has no output node")
    return errs


# ---- client --------------------------------------------------------------------------------------
class ComfyError(RuntimeError):
    pass


class Comfy:
    def __init__(self, base: str = "http://127.0.0.1:8188", timeout: float = 30):
        self.base, self.timeout = base.rstrip("/"), timeout

    def _req(self, path, data=None, raw=False):
        body = json.dumps(data).encode() if data is not None else None
        req = urllib.request.Request(self.base + path, body, {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                b = r.read()
                return b if raw else (json.loads(b) if b else {})
        except urllib.error.HTTPError as e:  # ComfyUI explains a rejected prompt in the body (node_errors)
            raise ComfyError(f"HTTP {e.code} {path}: {e.read().decode(errors='replace')[:3000]}") from None

    def alive(self) -> bool:
        try:
            self._req("/system_stats")
            return True
        except Exception:
            return False

    def system_stats(self):
        return self._req("/system_stats")

    def object_info(self):
        return self._req("/object_info")

    def submit(self, graph: dict, client_id: str) -> str:
        r = self._req("/prompt", {"prompt": graph, "client_id": client_id})
        if r.get("node_errors"):
            raise ComfyError("node_errors: " + json.dumps(r["node_errors"])[:3000])
        return r["prompt_id"]

    def history(self, pid: str) -> dict | None:
        h = self._req(f"/history/{pid}")
        return h.get(pid)

    def queue(self) -> tuple[int, int]:
        q = self._req("/queue")
        return len(q.get("queue_running", [])), len(q.get("queue_pending", []))

    def delete_pending(self, pids: list[str]) -> None:
        if pids:
            self._req("/queue", {"delete": pids})

    def view(self, filename: str, subfolder: str, typ: str = "output") -> bytes:
        q = urllib.parse.urlencode({"filename": filename, "subfolder": subfolder, "type": typ})
        return self._req(f"/view?{q}", raw=True)


def parse_history(h: dict) -> dict:
    """→ {status: success|error|running, images: [{filename, subfolder, type}], exec_s, error}"""
    st = h.get("status", {}) or {}
    msgs = {m[0]: m[1] for m in st.get("messages", []) if isinstance(m, list) and len(m) == 2}
    out = {"status": "running", "images": [], "exec_s": None, "error": None}
    if st.get("completed") or st.get("status_str") == "success":
        out["status"] = "success"
    if st.get("status_str") == "error":
        out["status"] = "error"
        e = msgs.get("execution_error", {})
        out["error"] = f"{e.get('node_type')}: {e.get('exception_type')}: {e.get('exception_message')}"[:2000]
    t0 = msgs.get("execution_start", {}).get("timestamp")
    t1 = (msgs.get("execution_success") or msgs.get("execution_error") or {}).get("timestamp")
    if t0 and t1:
        out["exec_s"] = round((t1 - t0) / 1000, 3)
    for node_out in (h.get("outputs") or {}).values():
        out["images"] += node_out.get("images", [])
    return out


def wait_alive(c: Comfy, timeout_s: float) -> bool:
    t = time.time()
    while time.time() - t < timeout_s:
        if c.alive():
            return True
        time.sleep(5)
    return False
