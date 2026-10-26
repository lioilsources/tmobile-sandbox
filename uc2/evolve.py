#!/usr/bin/env python3
"""UC2: evoluce GLSL filtrů pro MirrorBooth bez člověka v loopu (MAP-Elites, LLM jako mutace).

Semena = produkční filtry MirrorBoothu. Každé dítě napíše LLM (Qwen3-30B-A3B ve vLLM, profil L) jako variaci
rodiče (+ volitelně inspirace z jiného elitního shaderu) na zadané téma. Hodnocení bez LLM:
  1. kontrakt aplikace (MirrorBooth pipeline/checks/contract.py) — uniformy, FlutterFragCoord, zakázané konstrukce,
  2. ShaderBattle evaluátor (AiSwarmBattle, HTTP :8081): glslang, statická cena (ALU, čtení textur), běh přes EGL
     na L40S 1080×1920 (NaN/Inf, černé a konstantní snímky, medián ms/snímek) → composite 0–100,
  3. render nad syntetickými selfie (MirrorBooth preview/render.py, moderngl EGL) → PickScore vůči textu tématu
     (estetika), CLIP podobnost s původní selfie (obličej zůstal poznat), pohyb za 1 s a blikání (změna za 1/30 s
     vůči změně za 1 s).
Archiv MAP-Elites na téma: buňka = (odhad ALU, energie pohybu), v buňce nejlepší fitness.

  evolve.py --mirrorbooth /srv/work/uc2/MirrorBooth --selfies /srv/work/uc2/selfies --out /srv/work/uc2/runs/evo1 \
            --themes kaleido,watercolor,vhs --children 400
"""
import argparse, asyncio, concurrent.futures, json, pathlib, random, re, sys, time
import httpx
import numpy as np

THEMES = {
    "kaleido": "kaleidoscope mirror selfie filter with neon colors and smooth flowing motion",
    "watercolor": "watercolor painting of a selfie, soft paper texture, gently bleeding colors",
    "vhs": "retro VHS tape selfie filter with scanlines, chromatic aberration and a subtle glitch",
}
ALU_BINS = [500, 1000, 2000, 4000]           # → 5 sloupců
MOTION_BINS = [0.005, 0.02, 0.05, 0.1]        # průměrná změna pixelu za 1 s (0..1) → 5 řádků
TIMES = (0.0, 1.0, 1.0 + 1 / 30)

SYSTEM = """You write GLSL fragment shaders for the MirrorBooth Flutter app (live selfie camera filters).
Hard contract (the app cannot run anything else):
- First line: #include <flutter/runtime_effect.glsl>
- Uniforms exactly in this order: uniform sampler2D uTexture; uniform vec2 uResolution; then optionally
  uniform float uTime; then optionally uniform vec2 uFaceCenter; uniform float uFaceScale; (no other uniforms, no arrays)
- out vec4 fragColor; read the pixel position with FlutterFragCoord().xy; uv = FlutterFragCoord().xy / uResolution
- The camera image is uTexture; sample it with texture(uTexture, uv).
- Forbidden: #version, gl_FragCoord, fwidth/dFdx/dFdy, texelFetch, textureSize, uint/uvec, iChannel*, uniform arrays.
- Mobile budget: at most 16 texture reads per pixel, loops with constant bounds <= 32, no recursion.
- The face must stay recognisable: it is a camera filter, not a replacement picture.
Answer with exactly one ```glsl code block containing the complete shader and nothing else."""


def extract(text):
    m = re.search(r"```(?:glsl)?\s*\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).strip() + "\n"


def cell(alu, motion):
    return sum(alu > b for b in ALU_BINS), sum(motion > b for b in MOTION_BINS)


class Scorer:
    """Render + PickScore/CLIP; běží v jediném vlákně (GL kontext i model patří jednomu vláknu)."""

    def __init__(self, mb, selfies):
        sys.path.insert(0, str(mb / "pipeline"))
        from preview.render import create_context, render
        from PIL import Image
        import torch
        from transformers import AutoModel, AutoProcessor
        self.render, self.torch = render, torch
        self.ctx = create_context()
        self.inputs = [np.asarray(Image.open(p).convert("RGB").resize((384, 512)), dtype=np.uint8) for p in sorted(selfies.glob("*.png"))[:2]]
        self.proc = AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
        self.model = AutoModel.from_pretrained("yuvalkirstain/PickScore_v1", torch_dtype=torch.float16).eval().cuda()
        self.text = {}
        self.in_emb = self._img(self.inputs)

    def _img(self, arrays):
        with self.torch.no_grad():
            x = self.proc(images=list(arrays), return_tensors="pt")["pixel_values"].cuda().half()
            e = self.model.get_image_features(pixel_values=x)
        e = getattr(e, "pooler_output", e)  # transformers 5 vrací výstup, projekce je v pooler_output
        return e / e.norm(dim=-1, keepdim=True)

    def _txt(self, theme):
        if theme not in self.text:
            with self.torch.no_grad():
                t = self.proc(text=[THEMES[theme]], padding=True, truncation=True, max_length=77, return_tensors="pt").to("cuda")
                e = self.model.get_text_features(**t)
                e = getattr(e, "pooler_output", e)
            self.text[theme] = e / e.norm(dim=-1, keepdim=True)
        return self.text[theme]

    def score(self, code, theme, needs_time, needs_face):
        frames, big, small, errs = [], [], [], []
        for img in self.inputs:
            try:
                r = self.render(code, img, needs_time, needs_face, times=TIMES, ctx=self.ctx)
            except Exception as e:  # kompilace na desktop GL
                return {"render_error": str(e)[:300]}
            f = [x.astype(np.float32) for x in r.frames]
            if (r.frames[0].max(axis=2) <= 8).mean() > 0.3: errs.append("black")
            if (np.abs(f[0] - img.astype(np.float32)).max(axis=2) <= 2).mean() > 0.98: errs.append("identity")
            if len(f) == 3:
                big.append(float(np.abs(f[1] - f[0]).mean() / 255)); small.append(float(np.abs(f[2] - f[1]).mean() / 255))
            frames.append(r.frames[0])
        emb = self._img(frames)
        pick = float((self.model.logit_scale.exp() * emb @ self._txt(theme).T).mean())
        keep = float((emb * self.in_emb).sum(-1).mean())
        motion = float(np.mean(big)) if big else 0.0
        flicker = float(np.mean(small) / (np.mean(big) + 1e-3)) if big else 0.0
        return {"pick": round(pick, 3), "face_keep": round(keep, 3), "motion": round(motion, 4), "flicker": round(flicker, 3),
                "errors": sorted(set(errs)), "frame": frames[0]}


def fitness(m, ref):
    """PickScore normovaný na semena + kvalita z evaluátoru − penalizace (ztracený obličej, blikání).
    Statický filtr se nepenalizuje: pohyb je osa archivu, ne kritérium kvality."""
    z = (m["pick"] - ref[0]) / max(ref[1], 1e-3)
    return round(z + 2 * (m["composite"] / 100 - 0.9) + 3 * min(m["face_keep"] - 0.6, 0)
                 - 2 * max(m["flicker"] - 0.3, 0), 3)


async def evaluate(http, code, nt, nf):
    uni = [{"name": "uTexture", "type": "sampler2D", "semantic": "camera"},
           {"name": "uResolution", "type": "vec2", "default": [1080.0, 1920.0], "semantic": "resolution"}]
    if nt: uni.append({"name": "uTime", "type": "float", "default": 0.0, "semantic": "time"})
    if nf: uni += [{"name": "uFaceCenter", "type": "vec2", "default": [0.5, 0.42]}, {"name": "uFaceScale", "type": "float", "default": 0.35}]
    r = (await http.post("http://127.0.0.1:8081/evaluate", json={"request_id": "uc2", "glsl_es": code, "uniforms": uni,
         "config": {"test_duration_frames": 30, "target_resolutions": [[1080, 1920]]}}, timeout=120)).json()
    rt = (r.get("runtime", {}).get("per_resolution") or [{}])[0]
    return {"compile_ok": r["compile"]["ok"], "composite": r["score"]["composite"], "alu": r["static"].get("estimated_alu", 0),
            "tex_reads": r["static"].get("tex_reads", 0), "frame_ms": rt.get("median_frame_ms"),
            "nan": rt.get("nan_inf_detected", False), "compile_errors": [e.get("message", "") for e in r["compile"]["errors"]][:3]}


async def run_theme(theme, a, scorer, pool, http, out):
    sys.path.insert(0, str(a.mirrorbooth / "pipeline"))
    from checks.contract import check_contract, uniform_shape
    loop = asyncio.get_running_loop(); ev_sem = asyncio.Semaphore(4)
    tdir = out / theme; (tdir / "shaders").mkdir(parents=True, exist_ok=True); (tdir / "frames").mkdir(exist_ok=True)
    log = open(tdir / "children.jsonl", "a"); archive, stats = {}, {"deti": 0, "kontrakt": 0, "kompilace": 0, "render": 0, "platne": 0}

    async def assess(name, code, parent):
        rec = {"id": name, "parent": parent, "t": round(time.time() - t0, 1)}
        errs = check_contract(code); shape = uniform_shape(code)
        if errs or shape is None:
            rec.update(stage="kontrakt", errors=errs[:3]); return rec
        nt, nf = shape
        async with ev_sem:
            ev = await evaluate(http, code, nt, nf)
        rec.update(ev)
        if not ev["compile_ok"] or ev["nan"] or ev["composite"] < 60:
            rec["stage"] = "evaluator"; return rec
        sc = await loop.run_in_executor(pool, scorer.score, code, theme, nt, nf)
        if "render_error" in sc or sc["errors"]:
            rec.update(stage="render", errors=sc.get("errors") or [sc.get("render_error")]); return rec
        from PIL import Image
        Image.fromarray(sc.pop("frame")).save(tdir / "frames" / f"{name}.jpg", quality=80)
        rec.update(sc, stage="ok", needs_time=nt, needs_face=nf)
        (tdir / "shaders" / f"{name}.frag").write_text(code)
        return rec

    def insert(rec):
        c = cell(rec["alu"], rec["motion"]); k = f"{c[0]},{c[1]}"
        if k not in archive or rec["fit"] > archive[k]["fit"]: archive[k] = rec; return True
        return False

    t0 = time.time()
    seeds = sorted((a.mirrorbooth / "mirrorbooth" / "shaders").glob("*.frag"))
    seed_recs = [await assess(f"seed_{p.stem}", p.read_text(), None) for p in seeds]
    ok = [r for r in seed_recs if r["stage"] == "ok"]
    picks = [r["pick"] for r in ok]; ref = (float(np.mean(picks)), float(np.std(picks)))
    for r in ok: r["fit"] = fitness(r, ref); insert(r)
    for r in seed_recs: log.write(json.dumps(r) + "\n")
    print(f"[{theme}] semena: {len(ok)}/{len(seeds)} platných, PickScore {ref[0]:.2f} ± {ref[1]:.2f}, buněk {len(archive)}", flush=True)
    best_seed = max(r["fit"] for r in ok)

    async def child(i):
        elites = list(archive.values())
        parent = max(random.sample(elites, min(3, len(elites))), key=lambda r: r["fit"])
        insp = random.choice(elites) if random.random() < 0.5 and len(elites) > 1 else None
        pcode = (tdir / "shaders" / f"{parent['id']}.frag").read_text()
        msg = f"Theme: {THEMES[theme]}.\nCreate a new variation of this filter that fits the theme better and looks great on a selfie.\n\nParent shader:\n```glsl\n{pcode}```"
        if insp:
            msg += f"\n\nYou may borrow techniques from this other filter:\n```glsl\n{(tdir / 'shaders' / (insp['id'] + '.frag')).read_text()}```"
        tl = time.time()
        r = await http.post(f"{a.llm}/v1/chat/completions", json={"model": a.model, "temperature": 0.9, "max_tokens": 3000,
                            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": msg}]}, timeout=600)
        j = r.json(); code = extract(j["choices"][0]["message"]["content"])
        rec = await assess(f"c{i:04d}", code, parent["id"])
        rec.update(llm_s=round(time.time() - tl, 1), out_tokens=j["usage"]["completion_tokens"], insp=insp and insp["id"])
        stats["deti"] += 1
        for st, key in (("kontrakt", "kontrakt"), ("evaluator", "kompilace"), ("render", "render")):
            if rec["stage"] == st: stats[key] += 1
        if rec["stage"] == "ok":
            stats["platne"] += 1; rec["fit"] = fitness(rec, ref); rec["novy_elit"] = insert(rec)
        log.write(json.dumps(rec) + "\n"); log.flush()

    sem = asyncio.Semaphore(a.concurrency)
    async def guarded(i):
        async with sem:
            try: await child(i)
            except Exception as e: print(f"[{theme}] dítě {i}: {e}", flush=True)
    tasks = [asyncio.create_task(guarded(i)) for i in range(a.children)]
    for k, t in enumerate(asyncio.as_completed(tasks)):
        await t
        if (k + 1) % 50 == 0:
            best = max(r["fit"] for r in archive.values())
            print(f"[{theme}] {k + 1}/{a.children} dětí, platných {stats['platne']}, buněk {len(archive)}, nejlepší fit {best:.2f} (semena {best_seed:.2f})", flush=True)
    summary = {"tema": theme, "semen_platnych": len(ok), "pickscore_semena": ref, "nejlepsi_fit_semena": best_seed,
               "nejlepsi_fit": max(r["fit"] for r in archive.values()), "bunek": len(archive), **stats, "sekund": round(time.time() - t0)}
    (tdir / "archive.json").write_text(json.dumps(sorted(archive.values(), key=lambda r: -r["fit"]), indent=1))
    (tdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


async def main_async(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pool = concurrent.futures.ThreadPoolExecutor(1)
    scorer = await asyncio.get_running_loop().run_in_executor(pool, Scorer, a.mirrorbooth, a.selfies)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.concurrency + 8)) as http:
        res = [await run_theme(t, a, scorer, pool, http, out) for t in a.themes.split(",")]
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirrorbooth", type=pathlib.Path, required=True); ap.add_argument("--selfies", type=pathlib.Path, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--themes", default="kaleido,watercolor,vhs")
    ap.add_argument("--children", type=int, default=400); ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--llm", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="qwen3-30b-a3b")
    a = ap.parse_args(); random.seed(2); asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
