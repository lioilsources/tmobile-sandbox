"""Shared helpers for UC18/UC19: thread limits, incremental JSONL, SFW text check, GPU sampler, logging."""
from __future__ import annotations

import json, os, re, subprocess, sys, threading, time
from datetime import datetime


def limit_threads(n: int = 4) -> None:
    """8 vCPU VM with ~1 core steal: never let BLAS/torch take all 8 (MODEL_NOTES, UC25 head 1 h timeout)."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(k, str(n))
    try:
        import torch
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    except Exception:
        pass


def log(msg: str, logfile: str | None = None) -> None:
    line = f"{datetime.now().isoformat(timespec='seconds')} {msg}"
    print(line, file=sys.stderr, flush=True)
    if logfile:
        with open(logfile, "a") as f:
            f.write(line + "\n")


class JsonlWriter:
    """Append-only, one flushed line per record → a crash/VM loss keeps everything written so far."""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.f = open(path, "a", buffering=1)
        self.lock = threading.Lock()

    def write(self, rec: dict) -> None:
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()
            os.fsync(self.f.fileno())

    def close(self) -> None:
        self.f.close()


def read_jsonl(path: str) -> list[dict]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:  # torn last line after a crash
                pass
    return out


def done_keys(path: str, key: str = "id", ok_only: bool = True) -> set:
    return {r[key] for r in read_jsonl(path) if key in r and (not ok_only or not r.get("error"))}


# ---- SFW text gate ------------------------------------------------------------------------------
# Word-boundary match, so Danbooru "1girl"/"1boy" pass while prose "girl"/"boy" (age-ambiguous) do not.
SFW_BLOCK = [
    r"nsfw", r"nude", r"nudity", r"naked", r"nipples?", r"explicit", r"sex", r"sexy", r"sexual", r"erotic\w*", r"porn\w*",
    r"lingerie", r"underwear", r"panties", r"bra", r"bikini", r"cleavage", r"breasts?", r"topless", r"bottomless",
    r"seductive", r"suggestive", r"gore", r"gory", r"blood\w*", r"corpse", r"decapitat\w*", r"torture",
    r"child", r"children", r"kids?", r"teen\w*", r"schoolgirl", r"school uniform", r"loli\w*", r"shota\w*",
    r"toddler", r"infant", r"baby", r"girl", r"girls", r"boy", r"boys", r"minor", r"underage",
    # Czech (UC19 inputs)
    r"nah[áa]\w*", r"sex\w*", r"erot\w*", r"d[íi]t[ěe]\w*", r"d[ěe]ti", r"d[íi]vk\w*", r"holk\w*", r"kluk\w*",
    r"chlapec\w*", r"chlapc\w*", r"miminko\w*", r"krev", r"krv\w*", r"spodn\w* pr[áa]dl\w*",
]
_SFW_RE = re.compile(r"(?<![\w-])(" + "|".join(SFW_BLOCK) + r")(?![\w-])", re.IGNORECASE)


def sfw_hits(text: str) -> list[str]:
    return sorted({m.group(0).lower() for m in _SFW_RE.finditer(text or "")})


# ---- GPU sampler ---------------------------------------------------------------------------------
class GpuSampler(threading.Thread):
    """nvidia-smi every `every` s → samples [(t, util%, mem_MiB, power_W)]; optional log file."""

    def __init__(self, every: float = 5.0, logfile: str | None = None):
        super().__init__(daemon=True)
        self.every, self.logfile, self.samples, self._stop = every, logfile, [], threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                      "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
                u, m, p = (float(x) for x in out.strip().splitlines()[0].split(","))
                self.samples.append((time.time(), u, m, p))
                if self.logfile:
                    with open(self.logfile, "a") as f:
                        f.write(f"{datetime.now().isoformat(timespec='seconds')},{u:.0f},{m:.0f},{p:.0f}\n")
            except Exception:
                pass  # no nvidia-smi (CPU smoke test) → no samples
            self._stop.wait(self.every)

    def stop(self):
        self._stop.set()

    def mean_util(self, since: float = 0.0) -> float | None:
        xs = [u for t, u, _, _ in self.samples if t >= since]
        return sum(xs) / len(xs) if xs else None

    def peak_mem(self) -> float | None:
        return max((m for _, _, m, _ in self.samples), default=None)
