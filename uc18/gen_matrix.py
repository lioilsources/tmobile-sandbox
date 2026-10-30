#!/usr/bin/env python3
"""UC18/UC19 image matrix through the ComfyUI API — resumable, batched, self-checking.

  # UC18: 300 prompts × checkpoints × seeds, each checkpoint gets the prompt in its native dialect
  gen_matrix.py --prompts uc18/prompts/uc18_prompts.jsonl --ckpts all --seeds 1,2 \
      --out-dir /cold/w2/uc18/img --manifest /cold/w2/uc18/manifest.jsonl
  # UC19: explicit jobs (prompt/negative per row, from uc19/build_jobs.py)
  gen_matrix.py --jobs /cold/w2/uc19/jobs.jsonl --out-dir /cold/w2/uc19/img --manifest /cold/w2/uc19/manifest.jsonl
  # dry run: build + validate every workflow against a /object_info snapshot, no server needed
  gen_matrix.py --prompts … --dry-run --object-info uc18/object_info.json

Behaviour (each line is a MODEL_NOTES lesson):
- keeps `--depth` prompts queued in ComfyUI (default 6) so the GPU never waits for the client
  (ACE-Step one-by-one = 38 % util); PNG download/write/hash runs in a thread pool off the loop.
- order: seed-major, then checkpoint, then prompt → one model load per (seed, checkpoint) and a timeout
  still leaves a complete seed-1 matrix.
- resumable: an id whose PNG exists is skipped (manifest row re-created if missing). Manifest = JSONL,
  one fsynced line per finished image (ok or error), so a lost VM keeps everything done.
- per-item errors are recorded and the run continues; the FIRST error is printed in full to stderr.
- health check after `--health-s` (300 s): images/min vs expectation from models.est_s, mean GPU util,
  error share → exits 6 LOUDLY if any is off. Afterwards a progress line every 10 min + progress.json.
- stall guard: no image finished for `--stall-s` → exit 7; ComfyUI unreachable > 120 s → exit 8.
- on SIGTERM/SIGINT (queue timeout) our pending prompts are deleted from the ComfyUI queue.
"""
from __future__ import annotations

import argparse, hashlib, json, os, signal, statistics, sys, time, uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import comfy  # noqa: E402
from common import GpuSampler, JsonlWriter, log, read_jsonl  # noqa: E402
from models import CKPTS, native_prompt  # noqa: E402

EXIT_HEALTH, EXIT_STALL, EXIT_DOWN = 6, 7, 8


def seed_for(pid: str, seed_idx: int) -> int:
    """Same seed for a (prompt, seed_idx) on every checkpoint → pairs differ only by the model."""
    return int(hashlib.sha256(f"{pid}#{seed_idx}".encode()).hexdigest()[:8], 16)


def make_jobs(a) -> list[dict]:
    jobs = []
    if a.prompts:
        recs = read_jsonl(a.prompts)
        if a.styles:
            recs = [r for r in recs if r["style"] in a.styles.split(",")]
        if a.limit:
            recs = recs[: a.limit]
        ckpts = list(CKPTS) if a.ckpts == "all" else a.ckpts.split(",")
        for s in [int(x) for x in a.seeds.split(",")]:
            for ck in ckpts:
                for r in recs:
                    jobs.append({"id": f"{ck}/{r['id']}/s{s}", "pid": r["id"], "style": r["style"], "ckpt": ck,
                                 "seed_idx": s, "prompt": native_prompt(ck, r), "negative": CKPTS[ck]["negative"],
                                 "variant": "", "meta": {"prompt_set": os.path.basename(a.prompts)}})
    else:
        rows = read_jsonl(a.jobs)
        if a.limit:
            rows = rows[: a.limit]
        for r in rows:
            v = r.get("variant", "")
            s = int(r.get("seed_idx", 1))
            jobs.append({"id": f"{v + '/' if v else ''}{r['ckpt']}/{r['pid']}/s{s}", "pid": r["pid"],
                         "style": r.get("style", ""), "ckpt": r["ckpt"], "seed_idx": s, "prompt": r["prompt"],
                         "negative": r.get("negative", CKPTS[r["ckpt"]]["negative"]), "variant": v,
                         "meta": r.get("meta", {})})
        # same ordering rule: seed, checkpoint, then input order
        jobs.sort(key=lambda j: (j["seed_idx"], list(CKPTS).index(j["ckpt"])))
    for j in jobs:
        if j["ckpt"] not in CKPTS:
            raise SystemExit(f"unknown checkpoint {j['ckpt']!r}")
        j["seed"] = seed_for(j["pid"], j["seed_idx"])
        j["path"] = os.path.join(a.out_dir, j["variant"], j["ckpt"], f"{j['pid'].replace('/', '_')}_s{j['seed_idx']}.png")
        j["prefix"] = f"{a.comfy_prefix}/{j['variant'] + '/' if j['variant'] else ''}{j['ckpt']}/{j['pid'].replace('/', '_')}_s{j['seed_idx']}"
    return jobs


def est_seconds(jobs) -> float:
    return sum(CKPTS[j["ckpt"]]["est_s"] for j in jobs) + 25 * len({(j["seed_idx"], j["ckpt"]) for j in jobs})


def dry_run(a, jobs) -> int:
    oi = json.load(open(a.object_info)) if a.object_info else comfy.Comfy(a.api).object_info()
    bad, shown = 0, set()
    for j in jobs:
        g = comfy.build(j["ckpt"], j["prompt"], j["negative"], j["seed"], j["prefix"])
        errs = comfy.validate(g, oi)
        if errs:
            bad += 1
            if bad <= 5:
                print(f"INVALID {j['id']}: {errs}", file=sys.stderr)
        if j["ckpt"] not in shown and a.dump_dir:
            os.makedirs(a.dump_dir, exist_ok=True)
            json.dump(g, open(os.path.join(a.dump_dir, f"wf_{j['ckpt']}.json"), "w"), indent=1)
            shown.add(j["ckpt"])
    per_ck = {}
    for j in jobs:
        per_ck[j["ckpt"]] = per_ck.get(j["ckpt"], 0) + 1
    print(json.dumps({"jobs": len(jobs), "invalid": bad, "per_ckpt": per_ck,
                      "est_hours": round(est_seconds(jobs) / 3600, 2),
                      "existing_outputs": sum(os.path.exists(j["path"]) for j in jobs)}, indent=1))
    return 1 if bad else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts")
    src.add_argument("--jobs")
    ap.add_argument("--ckpts", default="all")
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--styles", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="/cold/w2/uc18/img")
    ap.add_argument("--manifest", default="/cold/w2/uc18/manifest.jsonl")
    ap.add_argument("--api", default="http://127.0.0.1:8188")
    ap.add_argument("--comfy-out", default="/srv/work/comfy-out", help="ComfyUI --output-directory (copy removed after download)")
    ap.add_argument("--comfy-prefix", default="uc18")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--health-s", type=float, default=300)
    ap.add_argument("--min-util", type=float, default=35, help="abort below this mean GPU util %% in the health window (SDXL přes ComfyUI běží na 52–64 %, 29. 10.; skutečnou kontrolou je rychlost)")
    ap.add_argument("--min-speed", type=float, default=0.5, help="abort below this × expected images/min")
    ap.add_argument("--max-err", type=float, default=0.2, help="abort above this error share")
    ap.add_argument("--stall-s", type=float, default=600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--object-info", help="snapshot from dump_object_info.py (dry run without a server)")
    ap.add_argument("--dump-dir", help="dry run: write one workflow JSON per checkpoint here")
    ap.add_argument("--progress", default="", help="progress JSON path (default <manifest dir>/progress.json)")
    a = ap.parse_args(argv)

    jobs = make_jobs(a)
    if a.dry_run:
        return dry_run(a, jobs)

    logfile = os.path.join(os.path.dirname(os.path.abspath(a.manifest)), "gen.log")
    progress_path = a.progress or os.path.join(os.path.dirname(os.path.abspath(a.manifest)), "progress.json")
    have = {r["id"] for r in read_jsonl(a.manifest) if not r.get("error")}
    man = JsonlWriter(a.manifest)
    todo = deque()
    recovered = 0
    for j in jobs:
        if os.path.exists(j["path"]):
            if j["id"] not in have:  # PNG survived, manifest line did not (crash between write and log)
                man.write(record(j, {"exec_s": None}, j["path"], recovered=True))
                recovered += 1
            continue
        todo.append(j)
    total = len(todo)
    log(f"gen_matrix: {len(jobs)} jobs, {len(jobs) - total} done ({recovered} recovered), {total} to do, "
        f"estimate {est_seconds(list(todo)) / 3600:.2f} h", logfile)
    if not total:
        return 0

    c = comfy.Comfy(a.api)
    if not c.alive():
        log(f"ComfyUI {a.api} unreachable — start profile D first", logfile)
        return EXIT_DOWN
    # validate against the live server once per checkpoint before queueing anything
    oi = c.object_info()
    for ck in sorted({j["ckpt"] for j in todo}):
        j = next(x for x in todo if x["ckpt"] == ck)
        errs = comfy.validate(comfy.build(ck, j["prompt"], j["negative"], j["seed"], j["prefix"]), oi)
        if errs:
            log(f"workflow for {ck} INVALID against live /object_info: {errs}", logfile)
            return 2

    gpu = GpuSampler(5, os.path.join(os.path.dirname(os.path.abspath(a.manifest)), "gpu.csv"))
    gpu.start()
    client_id = "uc18-" + uuid.uuid4().hex[:8]
    inflight: dict[str, tuple[dict, float]] = {}
    pool = ThreadPoolExecutor(3)
    futures = deque()
    stats = {"ok": 0, "err": 0, "exec": [], "first_error": None}
    t_start = time.time()
    t_first_done = None
    t_last_done = time.time()
    t_down = None
    health_done = False
    next_progress = time.time() + 600

    def on_signal(sig, _frm):
        pend = list(inflight)
        log(f"signal {sig}: deleting {len(pend)} queued prompts from ComfyUI and exiting", logfile)
        try:
            c.delete_pending(pend)
        except Exception:
            pass
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    def finish(j, h):
        """thread pool: download PNG → /cold, hash, remove ComfyUI's copy, manifest line."""
        info = comfy.parse_history(h)
        if info["status"] == "error" or not info["images"]:
            err = info["error"] or "no image in outputs"
            man.write({**record(j, info, None), "error": err})
            return False, err
        im = info["images"][0]
        data = c.view(im["filename"], im.get("subfolder", ""), im.get("type", "output"))
        os.makedirs(os.path.dirname(j["path"]), exist_ok=True)
        tmp = j["path"] + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, j["path"])
        src = os.path.join(a.comfy_out, im.get("subfolder", ""), im["filename"])
        if a.comfy_out and os.path.isfile(src):
            try:
                os.remove(src)
            except OSError:
                pass
        man.write(record(j, info, j["path"], sha=hashlib.sha256(data).hexdigest(), size=len(data)))
        return True, info["exec_s"]

    def check_health(now) -> str | None:
        n = stats["ok"] + stats["err"]
        if n >= 5 and stats["err"] / n > a.max_err:
            return f"error share {stats['err']}/{n} > {a.max_err}; first error: {stats['first_error']}"
        if stats["ok"] == 0:
            return f"no image finished in {now - t_start:.0f} s"
        warm = [x for x in stats["exec"][1:]] or stats["exec"]
        med = statistics.median(warm)
        done_jobs = [j for j in jobs if j["id"] in stats.setdefault("ids", set())]
        exp = statistics.median(CKPTS[j["ckpt"]]["est_s"] for j in done_jobs) if done_jobs else med
        ipm = stats["ok"] / ((now - (t_first_done or t_start)) / 60) if t_first_done and now > t_first_done else 0
        util = gpu.mean_util(since=t_first_done or t_start)
        msg = (f"HEALTH {stats['ok']} ok / {stats['err']} err in {now - t_start:.0f} s; warm median {med:.2f} s/img "
               f"(expected {exp:.2f}), {ipm:.1f} img/min after first image, GPU util "
               f"{'n/a' if util is None else f'{util:.0f} %'}, peak VRAM {gpu.peak_mem()} MiB")
        log(msg, logfile)
        if med > exp / a.min_speed:
            return f"too slow: {med:.2f} s/img vs expected {exp:.2f} (limit ×{1 / a.min_speed:.1f})"
        if util is not None and util < a.min_util:
            return f"GPU util {util:.0f} % < {a.min_util} % — GPU not doing the work (CPU fallback? shared GPU?)"
        if util is not None and util < 85:
            log(f"WARNING GPU util {util:.0f} % < 85 % target — consider --depth {a.depth * 2}", logfile)
        rem = len(todo) + len(inflight)
        log(f"ETA {rem * med / 3600:.2f} h for {rem} remaining at {med:.2f} s/img → queue timeout ≥ "
            f"{rem * med * 1.5 / 3600:.1f} h (×1.5)", logfile)
        return None

    try:
        while todo or inflight or futures:
            now = time.time()
            # top up the ComfyUI queue
            while todo and len(inflight) < a.depth:
                j = todo.popleft()
                g = comfy.build(j["ckpt"], j["prompt"], j["negative"], j["seed"], j["prefix"])
                try:
                    pid = c.submit(g, client_id)
                    inflight[pid] = (j, now)
                except comfy.ComfyError as e:
                    stats["err"] += 1
                    stats["first_error"] = stats["first_error"] or str(e)
                    if stats["err"] == 1:
                        print(f"first error ({j['id']}): {e}", file=sys.stderr, flush=True)
                    man.write({**record(j, {"exec_s": None}, None), "error": str(e)[:2000]})
                except OSError:
                    todo.appendleft(j)
                    break
            # collect finished prompts
            try:
                for pid in list(inflight):
                    h = c.history(pid)
                    if not h or comfy.parse_history(h)["status"] == "running":
                        continue
                    j, _ = inflight.pop(pid)
                    futures.append((j, pool.submit(finish, j, h)))
                t_down = None
            except (OSError, comfy.ComfyError) as e:
                t_down = t_down or now
                if now - t_down > 120:
                    log(f"ComfyUI unreachable for {now - t_down:.0f} s ({e}) — abort", logfile)
                    return EXIT_DOWN
            while futures and futures[0][1].done():
                j, f = futures.popleft()
                try:
                    ok, x = f.result()
                except Exception as e:  # download/write failed: record, keep going
                    ok, x = False, f"finish: {e}"
                    man.write({**record(j, {"exec_s": None}, None), "error": x[:2000]})
                if ok:
                    stats["ok"] += 1
                    stats.setdefault("ids", set()).add(j["id"])
                    if x:
                        stats["exec"].append(x)
                    t_first_done = t_first_done or time.time()
                else:
                    stats["err"] += 1
                    if stats["first_error"] is None:
                        stats["first_error"] = x
                        print(f"first error ({j['id']}): {x}", file=sys.stderr, flush=True)
                t_last_done = time.time()
            now = time.time()
            if stats["ok"] == 0 and stats["err"] >= 5:  # don't wait for the health window when nothing works
                log(f"!!! first {stats['err']} images all failed — ABORTING: {stats['first_error']}", logfile)
                c.delete_pending(list(inflight))
                return EXIT_HEALTH
            if not health_done and now - t_start >= a.health_s:
                health_done = True
                bad = check_health(now)
                if bad:
                    log(f"!!! HEALTH CHECK FAILED — ABORTING: {bad}", logfile)
                    print(f"\n!!! UC18 gen_matrix HEALTH CHECK FAILED: {bad}\n", file=sys.stderr, flush=True)
                    c.delete_pending(list(inflight))
                    return EXIT_HEALTH
            if inflight and now - t_last_done > a.stall_s:
                log(f"!!! STALL: nothing finished for {now - t_last_done:.0f} s with {len(inflight)} queued — abort", logfile)
                c.delete_pending(list(inflight))
                return EXIT_STALL
            if now >= next_progress:
                next_progress = now + 600
                med = statistics.median(stats["exec"][-50:]) if stats["exec"] else None
                p = {"ok": stats["ok"], "err": stats["err"], "remaining": len(todo) + len(inflight), "total": total,
                     "median_s": med, "gpu_util_10min": gpu.mean_util(since=now - 600), "elapsed_s": round(now - t_start),
                     "eta_h": round((len(todo) + len(inflight)) * med / 3600, 2) if med else None}
                json.dump(p, open(progress_path, "w"))
                log(f"progress {json.dumps(p)}", logfile)
            time.sleep(0.4)
    finally:
        gpu.stop()
        pool.shutdown(wait=True)
        man.close()
    wall = time.time() - t_start
    med = statistics.median(stats["exec"]) if stats["exec"] else None
    summary = {"ok": stats["ok"], "err": stats["err"], "wall_s": round(wall), "img_per_h": round(stats["ok"] / wall * 3600),
               "exec_median_s": med, "gpu_util_mean": gpu.mean_util(), "vram_peak_mib": gpu.peak_mem()}
    json.dump(summary, open(progress_path, "w"))
    log(f"gen_matrix done {json.dumps(summary)}", logfile)
    return 0 if stats["ok"] else 1


def record(j, info, path, sha=None, size=None, recovered=False) -> dict:
    c = CKPTS[j["ckpt"]]
    r = {"id": j["id"], "pid": j["pid"], "style": j["style"], "ckpt": j["ckpt"], "ckpt_file": c["file"],
         "gallery_id": c["gallery_id"], "variant": j["variant"], "seed_idx": j["seed_idx"], "seed": j["seed"],
         "prompt": j["prompt"], "negative": j["negative"], "steps": c["steps"], "cfg": c["cfg"],
         "sampler": c["sampler"], "scheduler": c["scheduler"], "path": path, "exec_s": info.get("exec_s"),
         "meta": j.get("meta", {}), "t": round(time.time(), 1)}
    if sha:
        r.update(sha256=sha, size=size)
    if recovered:
        r["recovered"] = True
    if c.get("licence_restricted"):
        r["licence_restricted"] = c["licence_restricted"]
    return r


if __name__ == "__main__":
    sys.exit(main())
