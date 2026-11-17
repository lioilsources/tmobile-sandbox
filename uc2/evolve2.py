#!/usr/bin/env python3
"""UC2 kolo 2: evoluce shaderů po rodinách (Prompts/08 plánu MirrorBooth). Navazuje na evolve.py (kolo 1, jedno téma).

Rozdíly proti kolu 1: rodina = ostrov s kartou techniky (families.py), rodič se losuje rovnoměrně přes buňky archivu,
kvóta linie 20 %, dynamický zákaz přepoužívaných identifikátorů, čtyři mutační operátory, osy archivu pohyb × síla zásahu,
tvrdé brány (kontrakt, karta, evaluátor, impellerc vč. SkSL, render, face_keep podle rodiny), skóre = shoda s tématem +
novost (CLIP embedding a kód) + malá váha PickScore, deduplikace, checkpoint po vlně dětí, měření RSS, restart evaluátoru.

  evolve2.py --mirrorbooth /srv/work/uc2/MirrorBooth --selfies /srv/work/uc2/selfies --flutter-root /srv/work/uc4/flutter \
             --seeds /srv/work/uc2/MirrorBooth/mirrorbooth/shaders --seeds /srv/work/uc2/roundb \
             --out /srv/work/uc2/runs/r2 --families mozaika,optika --children 100
Opakované spuštění naváže z checkpointu (archive.json, emb.npz, state.json v adresáři rodiny).
"""
import argparse, asyncio, concurrent.futures, hashlib, json, os, pathlib, random, re, subprocess, sys, time
from collections import Counter
import httpx
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import evolve as E  # noqa: E402
import probe as P  # noqa: E402
from families import FAMILIES, assign_seeds, card_text, check_card, gate, probe_errors, strip_comments  # noqa: E402

MOTION_BINS = [0.005, 0.02, 0.05, 0.1]          # → 5 sloupců
STRENGTH_BINS = [0.03, 0.08, 0.15, 0.25]        # průměrná |render − selfie| (0..1) → 5 řádků
OPERATORS = ["variace", "kříž", "import", "zjednodušení"]
GLSL_WORDS = set("""float vec2 vec3 vec4 mat2 mat3 int bool void return if else for while uniform out in const sampler2D
main texture mix clamp smoothstep step fract floor ceil abs sin cos tan pow exp log sqrt length dot cross normalize min max
mod sign reflect atan uv uTime uResolution uTexture uFaceCenter uFaceScale fragColor FlutterFragCoord xy xyz rgb rgba""".split())


def idents(code):
    s = strip_comments(code)
    return {w for w in re.findall(r"[A-Za-z_]\w{3,}", s) if w not in GLSL_WORDS and not w[0].isdigit()}


def builtins_used(code):
    return frozenset(re.findall(r"\b(sin|cos|pow|exp|floor|fract|mod|step|smoothstep|mix|length|dot|abs|texture|clamp|min|max|"
                                r"sqrt|normalize|reflect|sign)\s*\(", strip_comments(code)))


def code_hash(code):
    return hashlib.sha1(re.sub(r"\s+", "", strip_comments(code)).encode()).hexdigest()


def cell(motion, strength):
    return sum(motion > b for b in MOTION_BINS), sum(strength > b for b in STRENGTH_BINS)


def rss_mb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"): return int(l.split()[1]) // 1024
    return 0


def sysmem():
    """RSS vlastního procesu, volná paměť a RSS kontejnerů (evaluátor, vLLM) v MB."""
    out = {"self": rss_mb()}
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable"): out["avail"] = int(l.split()[1]) // 1024
        r = subprocess.run(["docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"], capture_output=True, text=True, timeout=30)
        for l in r.stdout.splitlines():
            n, _, u = l.partition(" ")
            if n in ("swarm-evaluator", "vllm"):
                v, unit = re.match(r"([\d.]+)(\w+)", u).groups()
                out[n] = int(float(v) * {"GiB": 1024, "MiB": 1, "KiB": 1 / 1024, "B": 1 / 1048576}.get(unit, 1))
    except Exception as e:
        out["err"] = str(e)[:80]
    return out


def hist_dist(a, b):
    """Colour-histogram distance (8×8×8 bins, total variation, 0–1) between a render and its source selfie."""
    def h(x):
        q = (np.asarray(x)[..., :3] // 32).reshape(-1, 3).astype(np.int64)
        c = np.bincount(q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2], minlength=512)
        return c / c.sum()
    return float(np.abs(h(a) - h(b)).sum() / 2)


class Scorer2(E.Scorer):
    """Render + CLIP embedding. PickScore vrací embedding obrázku i cosine k textu rodiny; GL i model patří jednomu vláknu."""

    # round 3 (UC2_SET=r3): selfies are center-cropped to 3:4 instead of stretched, and each gets its real face anchor
    # (checked with marker renders 12. 11.; the renderer's fixed (0.5, 0.5) put lipstick on the phone in selfie_0)
    FACES = [((0.745, 0.37), 0.33), ((0.475, 0.47), 0.32)]

    def __init__(self, mb, selfies):
        super().__init__(mb, selfies)
        self.faces = None
        if os.environ.get("UC2_SET") == "r3":
            from PIL import Image
            ims = [Image.open(p).convert("RGB") for p in sorted(selfies.glob("*.png"))[:2]]
            self.inputs = [np.asarray(im.crop(((im.width - im.height * 3 // 4) // 2, 0, (im.width + im.height * 3 // 4) // 2, im.height))
                                      .resize((384, 512)), dtype=np.uint8) for im in ims]
            self.in_emb = self._img(self.inputs); self.faces = self.FACES

    def _anchor(self, i):
        """Set the renderer's face uniforms for selfie i (module globals read at render time)."""
        if self.faces:
            R = sys.modules["preview.render"]; R.FACE_CENTER, R.FACE_SCALE = self.faces[i]

    def score2(self, code, family, needs_time, needs_face):
        frames, big, small, diff, errs, hist, local = [], [], [], [], [], [], []
        for k, img in enumerate(self.inputs):
            self._anchor(k)
            try:
                r = self.render(code, img, needs_time, needs_face, times=E.TIMES, ctx=self.ctx)
            except Exception as e:
                return {"render_error": str(e)[:300]}
            f = [x.astype(np.float32) for x in r.frames]
            if (r.frames[0].max(axis=2) <= 8).mean() > 0.3: errs.append("black")
            d0 = float(np.abs(f[0] - img.astype(np.float32)).mean() / 255)
            same = np.abs(f[0] - img.astype(np.float32)).max(axis=2) <= 2
            if FAMILIES[family].get("identity_head"):     # round 3: a lip or eye effect changes ~1 % of the frame
                hm = self._head(same.shape, k); same = (same * hm).sum() / hm.sum()
            else:
                same = same.mean()
            if same > 0.98: errs.append("identity")
            diff.append(d0); hist.append(hist_dist(r.frames[0], img))
            dm = np.abs(f[0] - img.astype(np.float32)).mean(axis=2); local.append(float((dm * self._head(dm.shape, k)).sum() / (dm.sum() + 1e-6)))
            if len(f) == 3:
                big.append(float(np.abs(f[1] - f[0]).mean() / 255)); small.append(float(np.abs(f[2] - f[1]).mean() / 255))
            frames.append(r.frames[0])
        emb = self._img(frames)
        text = self._txt(family)
        try:
            pr = P.probe(code, self.ctx)
        except Exception as e:  # noqa: BLE001
            pr = {"probe_error": str(e)[:200]}
        pick = float((self.model.logit_scale.exp() * emb @ text.T).mean())
        keep = float((emb * self.in_emb).sum(-1).mean())
        motion = float(np.mean(big)) if big else 0.0
        return {"pick": round(pick, 3), "theme_cos": round(pick / float(self.model.logit_scale.exp()), 4), "face_keep": round(keep, 3),
                "motion": round(motion, 4), "strength": round(float(np.mean(diff)), 4), "hist": round(float(np.mean(hist)), 3), "local": round(float(np.mean(local)), 3),
                "flicker": round(float(np.mean(small) / (np.mean(big) + 1e-3)) if big else 0.0, 3), "errors": sorted(set(errs)),
                "probe": pr,
                # effect embedding = render − source per selfie: whole-frame CLIP is dominated by the face
                # (trial 4, 11. 11.: different seed shaders had cos 0.95–0.997), so dedup/novelty compare the change only
                "frame": frames[0], "emb": (emb - self.in_emb).float().mean(0).cpu().numpy()}

    def _head(self, shape, k=0):
        """Soft head ellipse at the face anchor of selfie k (renderer default (0.5, 0.5), 0.35 without anchors):
        face coords q = (uv - c) / s, head ~ |q.x| < 0.8, q.y from -1.05 to 1.15 (chin and forehead)."""
        (cx, cy), sc = self.faces[k] if self.faces else ((0.5, 0.5), 0.35)
        h, w = shape; y, x = np.mgrid[0:h, 0:w]
        qx = (x / w - cx) / sc * (w / h); qy = (y / h - cy) / sc
        r = np.sqrt((qx / 0.8) ** 2 + ((qy - 0.05) / 1.1) ** 2)
        return np.clip((1.15 - r) / 0.3, 0.0, 1.0)

    def preview(self, code, needs_time, needs_face, path):
        from PIL import Image
        self._anchor(0); sel = self.inputs[0]; times = tuple(i * 2 / 12 for i in range(12)) if needs_time else (0.0,)
        res = self.render(code, sel, needs_time, needs_face, times=times, ctx=self.ctx)
        fr = [Image.fromarray(f) for f in res.frames]
        fr[0].save(path, save_all=True, append_images=fr[1:], duration=167, loop=0, quality=70)


class Island:
    """Jedna rodina: archiv 5×5, linie, zákazy, checkpoint."""

    def __init__(self, family, a, out, scorer, pool, http):
        self.f, self.a, self.scorer, self.pool, self.http = family, a, scorer, pool, http
        self.dir = out / family; (self.dir / "shaders").mkdir(parents=True, exist_ok=True); (self.dir / "frames").mkdir(exist_ok=True)
        self.archive, self.embs, self.recs, self.hashes = {}, {}, {}, set()
        self.ban_until, self.stats, self.next_i, self.rss = {}, {"deti": 0, "kontrakt": 0, "karta": 0, "kompilace": 0, "impellerc": 0,
                                                              "render": 0, "face": 0, "sila": 0, "lokalita": 0, "sonda": 0, "dup": 0, "platne": 0}, 0, []
        self.ops = {o: 0 for o in OPERATORS}
        E.THEMES[family] = FAMILIES[family]["desc"]
        self.t0 = time.time()

    # ---- checkpoint
    def load(self):
        st = self.dir / "state.json"
        if not st.exists(): return False
        s = json.loads(st.read_text()); self.stats, self.next_i, self.ops, self.rss = s["stats"], s["next_i"], s["ops"], s["rss"]
        self.ban_until = s["ban_until"]; self.ref = tuple(s["ref"])
        npz = np.load(self.dir / "emb.npz"); self.embs = {k: npz[k] for k in npz.files}
        for r in json.loads((self.dir / "recs.json").read_text()):
            self.recs[r["id"]] = r; self.hashes.add(r["hash"])
        for r in json.loads((self.dir / "archive.json").read_text()): self.archive[r["cell"]] = r
        return True

    def save(self):
        (self.dir / "recs.json").write_text(json.dumps(list(self.recs.values())))
        (self.dir / "archive.json").write_text(json.dumps(sorted(self.archive.values(), key=lambda r: -r["score"]), indent=1))
        np.savez(self.dir / "emb.npz", **self.embs)
        (self.dir / "state.json").write_text(json.dumps({"stats": self.stats, "next_i": self.next_i, "ops": self.ops, "rss": self.rss,
                                                          "ban_until": self.ban_until, "ref": list(self.ref)}))

    # ---- hodnocení
    async def assess(self, name, code, parent, lineage, op, seed=False):
        from checks.contract import check_contract, uniform_shape
        from checks.compile import compile_shader
        loop = asyncio.get_running_loop()
        rec = {"id": name, "family": self.f, "parent": parent, "lineage": lineage, "op": op, "t": round(time.time() - self.t0, 1)}
        errs = check_contract(code); shape = uniform_shape(code)
        if errs or shape is None:
            return {**rec, "stage": "kontrakt", "errors": errs[:3]}
        ce = check_card(code, self.f)
        if ce: return {**rec, "stage": "karta", "errors": ce[:3]}
        h = code_hash(code)
        if h in self.hashes: return {**rec, "stage": "dup", "errors": ["hash"]}
        nt, nf = shape
        async with self.a.ev_sem:
            ev = await E.evaluate(self.http, code, nt, nf)
        rec.update(ev)
        if not ev["compile_ok"] or ev["nan"] or ev["composite"] < 60 or (ev["frame_ms"] or 0) > self.a.max_frame_ms or ev["alu"] > self.a.max_alu:
            return {**rec, "stage": "kompilace"}
        comp = await loop.run_in_executor(self.a.imp_pool, lambda: compile_shader(code, flutter_root=self.a.flutter_root))
        rec["impellerc"] = bool(comp.impeller_verified)
        if not comp.impeller_verified:
            return {**rec, "stage": "impellerc", "errors": comp.errors[:3]}
        sc = await loop.run_in_executor(self.pool, self.scorer.score2, code, self.f, nt, nf)
        if "render_error" in sc or sc["errors"]:
            return {**rec, "stage": "render", "errors": sc.get("errors") or [sc.get("render_error")]}
        emb = sc.pop("emb"); frame = sc.pop("frame")
        if sc["face_keep"] < (self.a.face_floor if self.a.face_floor is not None else FAMILIES[self.f]["face"]) or sc["flicker"] > self.a.max_flicker:
            return {**rec, **sc, "stage": "face"}
        mh = gate(self.f, "max_hist")
        if not seed and (sc["face_keep"] > gate(self.f, "max_keep") or sc["strength"] < gate(self.f, "min_strength") or (mh and sc["hist"] > mh)):
            return {**rec, **sc, "stage": "sila"}
        ml = gate(self.f, "min_local")
        if ml and not seed and sc["local"] < ml:
            return {**rec, **sc, "stage": "lokalita"}
        pe = probe_errors(self.f, sc["probe"])
        if pe and not seed:
            return {**rec, **sc, "stage": "sonda", "errors": pe}
        near = max(((float(emb @ e / (np.linalg.norm(emb) * np.linalg.norm(e) + 1e-9)), k) for k, e in self.embs.items()),
                   default=(0.0, ""))
        sc["near_cos"] = round(near[0], 4)
        if near[0] > self.a.dup_cos and not seed:
            return {**rec, **sc, "stage": "dup", "errors": [f"render~{near[1]}"]}
        rec.update(sc, stage="ok", needs_time=nt, needs_face=nf, hash=h, ident=sorted(idents(code)), blt=sorted(builtins_used(code)))
        rec["_emb"], rec["_frame"] = emb, frame
        return rec

    def novelty(self, rec):
        e = rec["_emb"]; mc = 0.0; mj = 0.0; ids = set(rec["ident"])
        for k, o in self.embs.items():
            mc = max(mc, float(e @ o / (np.linalg.norm(e) * np.linalg.norm(o))))
            oi = set(self.recs[k]["ident"]); mj = max(mj, len(ids & oi) / max(len(ids | oi), 1))
        return 0.5 * min(1.0, (1 - mc) * 10) + 0.5 * (1 - mj) if self.embs else 0.5

    def score(self, rec):
        z = (rec["theme_cos"] * 100 - self.ref[0]) / max(self.ref[1], self.a.theme_floor)
        a = self.a
        return round(a.w_theme * z + a.w_novel * self.novelty(rec) + a.w_pick * (rec["pick"] - 20) / 2 + 0.5 * (rec["composite"] / 100 - 0.9), 3)

    def accept(self, rec):
        """Vloží platné dítě do archivu; vrací True, když se stalo elitou buňky."""
        rec["score"] = self.score(rec); c = cell(rec["motion"], rec["strength"]); rec["cell"] = f"{c[0]},{c[1]}"
        emb, frame = rec.pop("_emb"), rec.pop("_frame")
        (self.dir / "shaders" / f"{rec['id']}.frag").write_text(rec["_code"]); rec.pop("_code")
        from PIL import Image
        Image.fromarray(frame).save(self.dir / "frames" / f"{rec['id']}.jpg", quality=80)
        self.recs[rec["id"]] = rec; self.embs[rec["id"]] = emb; self.hashes.add(rec["hash"])
        old = self.archive.get(rec["cell"])
        if old is None or rec["score"] > old["score"]:
            self.archive[rec["cell"]] = rec; return True
        return False

    # ---- rodiče, kvóty, zákazy
    def lineage_shares(self):
        kids = [r for r in self.recs.values() if r["parent"] is not None]
        n = len(kids)
        return {l: sum(r["lineage"] == l for r in kids) / n for l in {r["lineage"] for r in kids}} if n else {}, n

    def banned(self, i):
        arch = list(self.archive.values())
        if len(arch) >= 5:
            cnt = {}
            for r in arch:
                for w in r["ident"]: cnt[w] = cnt.get(w, 0) + 1
            for w, c in cnt.items():
                if c / len(arch) > 0.30 and self.ban_until.get(w, -1) < i: self.ban_until[w] = i + self.a.ban_n
        return sorted(w for w, u in self.ban_until.items() if u >= i)

    def pick_parent(self, locked, exclude=None):
        # lineage first, then a cell elite of it (full run 11. 11.: seeds sharing a cell were displaced at start and
        # their lineages never got a child: polotón 186/186 kids from labDither); a lineage without an elite uses its best rec
        recs = [r for r in self.recs.values() if r["id"] != exclude]
        lins = sorted({r["lineage"] for r in recs} - set(locked)) or sorted({r["lineage"] for r in recs})
        if not lins: return random.choice(list(self.archive.values()))
        lin = random.choice(lins)
        cells = [r for r in self.archive.values() if r["lineage"] == lin and r["id"] != exclude]
        return random.choice(cells) if cells else max((r for r in recs if r["lineage"] == lin), key=lambda r: r.get("score", 0))

    def code(self, rid):
        return (self.dir / "shaders" / f"{rid}.frag").read_text()

    def prompt(self, i, seeds_other):
        shares, n = self.lineage_shares()
        locked = {l for l, s in shares.items() if s >= 0.20} if n >= 10 else set()
        op = OPERATORS[i % 4] if random.random() < 0.5 else random.choice(OPERATORS)
        if op == "kříž" and len(self.archive) < 2: op = "variace"
        p = self.pick_parent(locked)
        card = card_text(self.f)
        ban = self.banned(i)
        extra = f"\nDo NOT use these identifiers or function names (they are overused in this family): {', '.join(ban[:25])}." if ban else ""
        head = f"{card}\nFamily theme sentence: {FAMILIES[self.f]['desc']}.{extra}\n\n"
        pc = self.code(p["id"]); parents = [p["id"]]
        if op == "variace":
            msg = head + f"Change the parameters and replace one function of this parent shader, keep it inside the family.\n```glsl\n{pc}```"
        elif op == "kříž":
            q = self.pick_parent(locked, exclude=p["id"]); parents.append(q["id"])
            msg = head + (f"Take the main technique of shader A and apply it to the output of shader B (same family).\n"
                          f"Shader A:\n```glsl\n{pc}```\nShader B:\n```glsl\n{self.code(q['id'])}```")
        elif op == "import" and seeds_other:
            o = random.choice(seeds_other); parents.append(o[0])
            msg = head + (f"Transfer ONE element of the other filter (below) into this parent. The family card above still applies.\n"
                          f"Parent:\n```glsl\n{pc}```\nOther filter:\n```glsl\n{o[1]}```")
        else:
            op = "zjednodušení"
            msg = head + f"Remove one effect layer of this shader but keep its character; the result must be shorter.\n```glsl\n{pc}```"
        return op, parents, p["lineage"], msg

    # ---- běh
    async def run(self, seeds, other_seeds, n_children):
        if not self.load():
            recs = []
            for p in seeds:
                rec = await self.assess(f"seed_{p.stem}", p.read_text(), None, p.stem, "seed", seed=True)
                rec["_code"] = p.read_text(); recs.append(rec)
            ok = [r for r in recs if r["stage"] == "ok"]
            print(f"[{self.f}] semena: {len(ok)}/{len(seeds)} v kartě a platných; vyřazená: "
                  f"{[(r['id'], r['stage'], r.get('errors')) for r in recs if r['stage'] != 'ok']}", flush=True)
            if not ok: raise SystemExit(f"rodina {self.f}: žádný platný seed")
            tc = [r["theme_cos"] * 100 for r in ok]; self.ref = (float(np.mean(tc)), float(np.std(tc)))
            for r in ok: self.accept(r)
            self.save()
        log = open(self.dir / "children.jsonl", "a")
        sem = asyncio.Semaphore(self.a.concurrency); done = 0
        target = n_children                                   # celkový počet dětí rodiny (i po navázání)
        while self.next_i < target:
            wave = list(range(self.next_i, min(self.next_i + self.a.concurrency, target)))
            jobs = [self.prompt(i, other_seeds) for i in wave]      # zákazy a kvóty se přepočítají jednou za vlnu

            async def child(i, job):
                op, parents, lineage, msg = job
                async with sem:
                    try:
                        r = await self.http.post(f"{self.a.llm}/v1/chat/completions", json={"model": self.a.model, "temperature": 0.9, "max_tokens": 3000,
                                                 "messages": [{"role": "system", "content": E.SYSTEM}, {"role": "user", "content": msg}]}, timeout=600)
                        j = r.json(); code = E.extract(j["choices"][0]["message"]["content"])
                        rec = await self.assess(f"c{i:04d}", code, ",".join(parents), lineage, op)
                        rec["out_tokens"] = j["usage"]["completion_tokens"]; rec["_code"] = code
                        return rec
                    except Exception as e:
                        return {"id": f"c{i:04d}", "family": self.f, "stage": "chyba", "errors": [str(e)[:120]], "op": op}
            res = await asyncio.gather(*(child(i, j) for i, j in zip(wave, jobs)))
            for rec in res:     # vložení do archivu serializovaně po vlně
                self.stats["deti"] += 1; st = rec["stage"]
                if st in self.stats: self.stats[st] += 1
                if st == "ok":
                    self.stats["platne"] += 1; self.ops[rec["op"]] += 1
                    rec["novy_elit"] = self.accept(rec)
                else:
                    rec.pop("_code", None)
                log.write(json.dumps({k: v for k, v in rec.items() if not k.startswith("_") and k not in ("ident", "blt")}) + "\n")
            log.flush(); self.next_i = wave[-1] + 1
            m = sysmem(); self.rss.append({"deti": self.stats["deti"], **m}); self.save()
            sh, n = self.lineage_shares()
            print(f"[{self.f}] {self.stats['deti']} dětí, platných {self.stats['platne']}, buněk {len(self.archive)}, "
                  f"linie max {max(sh.values()) * 100 if sh else 0:.0f} %, RSS {m}", flush=True)
            if self.stats["deti"] % self.a.restart_every < len(wave):
                subprocess.run(["docker", "restart", "swarm-evaluator"], capture_output=True)
                for _ in range(60):
                    try:
                        if (await self.http.get("http://127.0.0.1:8081/healthz")).status_code == 200: break
                    except Exception: pass
                    await asyncio.sleep(2)
        await self.finish()

    def export_set(self):
        """Archive elites first, then the best other valid children; no lineage above --export-lineage of the export
        (MirrorBooth 11. 11.: the 20 % quota applies to parent choice and to the export, not to the archive)."""
        n = self.a.export_n; cap = self.a.export_cap or max(2, int(self.a.export_lineage * n))  # ≥ 2: families with 3–4 seeds still fill 8 slots
        elite = {r["id"] for r in self.archive.values()}
        # seeds stay out (the app already has them); near-duplicate effects (cos > --export-dup) too
        pool = sorted((r for r in self.recs.values() if r["parent"] is not None),
                      key=lambda r: (r["id"] not in elite, -r.get("score", 0)))
        out, per = [], {}
        for r in pool:
            if len(out) >= n: break
            if per.get(r["lineage"], 0) >= cap: continue
            e = self.embs[r["id"]]
            if any(float(e @ self.embs[o["id"]] / (np.linalg.norm(e) * np.linalg.norm(self.embs[o["id"]]) + 1e-9)) > self.a.export_dup
                   for o in out):
                continue
            per[r["lineage"]] = per.get(r["lineage"], 0) + 1; out.append(r)
        return out

    async def finish(self):
        loop = asyncio.get_running_loop(); meta = []
        ex = self.dir / "export"; ex.mkdir(exist_ok=True)
        for r in self.export_set():
            code = self.code(r["id"]); (ex / f"{r['id']}.frag").write_text(code)
            try:
                await loop.run_in_executor(self.pool, self.scorer.preview, code, r.get("needs_time", False), r.get("needs_face", False),
                                           ex / f"{r['id']}.webp")
            except Exception as e:
                print("náhled", r["id"], e)
            meta.append({k: v for k, v in r.items() if k not in ("blt",)})
        (ex / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
        kids = [r for r in self.recs.values() if r["parent"] is not None]
        sh, n = self.lineage_shares()
        techs = {tuple(r["blt"]) for r in self.recs.values()}
        summ = {"rodina": self.f, "export": len(meta), "export_linie": dict(Counter(r["lineage"] for r in meta)), "sekund": round(time.time() - self.t0), **self.stats, "operatory_platne": self.ops, "bunek": len(self.archive),
                "linie_max_pct": round(100 * max(sh.values()), 1) if sh else 0, "linie_podily": {k: round(v, 3) for k, v in sorted(sh.items(), key=lambda x: -x[1])[:6]},
                "ruznych_technik": len(techs), "lokalita_export_pct": round(100 * sum(r.get("local", 0) >= 0.6 for r in meta) / max(len(meta), 1)), "polarni_warp": sum(bool(re.search(r"\batan\s*\(", self.code(r["id"]))) for r in kids),
                "rss_max_self_mb": max((x["self"] for x in self.rss), default=0), "rss_max_evaluator_mb": max((x.get("swarm-evaluator", 0) for x in self.rss), default=0)}
        (self.dir / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
        print(json.dumps(summ, ensure_ascii=False), flush=True)


async def main_async(a):
    sys.path.insert(0, str(a.mirrorbooth / "pipeline"))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pool = concurrent.futures.ThreadPoolExecutor(1); a.imp_pool = concurrent.futures.ThreadPoolExecutor(3); a.ev_sem = asyncio.Semaphore(4)
    paths = [p for d in a.seeds for p in pathlib.Path(d).glob("*.frag")]
    assigned = assign_seeds(paths)
    print({f: len(l) for f, l in assigned.items()}, flush=True)
    scorer = await asyncio.get_running_loop().run_in_executor(pool, Scorer2, a.mirrorbooth, a.selfies)
    res = []
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=a.concurrency + 8)) as http:
        for f in a.families.split(","):
            isl = Island(f, a, out, scorer, pool, http)
            others = [(p.stem, p.read_text()) for g, l in assigned.items() if g not in (f, "nezařazeno") for p in l]
            try:
                cm = dict(x.split("=") for x in a.children_map.split(",") if x)
                await isl.run(assigned[f], others, int(cm.get(f, a.children)))
                res.append(json.loads((isl.dir / "summary.json").read_text()))
            except BaseException as e:          # pád jedné rodiny nezahodí ostatní; checkpoint už je na disku
                print(f"[{f}] PÁD: {e!r}", flush=True)
                if isinstance(e, (KeyboardInterrupt, asyncio.CancelledError)): raise
    (out / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirrorbooth", type=pathlib.Path, required=True); ap.add_argument("--selfies", type=pathlib.Path, required=True)
    ap.add_argument("--flutter-root", required=True); ap.add_argument("--seeds", action="append", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--children", type=int, default=100, help="celkem dětí na rodinu")
    ap.add_argument("--children-map", dest="children_map", default="", help="per-family override, e.g. malba=200,barva=200")
    ap.add_argument("--concurrency", type=int, default=32); ap.add_argument("--llm", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen3-30b-a3b")
    ap.add_argument("--w-theme", dest="w_theme", type=float, default=1.0); ap.add_argument("--w-novel", dest="w_novel", type=float, default=0.5)
    ap.add_argument("--w-pick", dest="w_pick", type=float, default=0.5)
    # calib2 on Olin's like/dislike table (11. 11.): seed spread of theme_cos is 0.2–1.0, a 0.3 floor turned tiny
    # differences into z ±6; floor 1.0 + w_novel 0.5 + w_pick 0.5 lifted AUC 0.60 → 0.64 (all) and 0.61 → 0.79 (past gates)
    ap.add_argument("--theme-floor", dest="theme_floor", type=float, default=1.0)
    ap.add_argument("--max-frame-ms", dest="max_frame_ms", type=float, default=16.0); ap.add_argument("--max-alu", dest="max_alu", type=int, default=8000)
    ap.add_argument("--max-flicker", dest="max_flicker", type=float, default=0.6)
    ap.add_argument("--export-n", dest="export_n", type=int, default=25)
    ap.add_argument("--export-lineage", dest="export_lineage", type=float, default=0.2)
    ap.add_argument("--export-cap", dest="export_cap", type=int, default=0, help="max children per lineage in the export (0 = from --export-lineage)")
    # Olin 11. 11.: a recognisable face is not important; the face gate is then only a guard against a broken image
    ap.add_argument("--face-floor", dest="face_floor", type=float, default=None, help="face_keep gate for every family instead of the card value")
    ap.add_argument("--export-dup", dest="export_dup", type=float, default=0.93, help="export: drop effects with cos above this to a chosen one")
    ap.add_argument("--dup-cos", dest="dup_cos", type=float, default=0.97, help="dedup: cosine of effect embeddings (render − source)")
    ap.add_argument("--ban-n", dest="ban_n", type=int, default=40); ap.add_argument("--restart-every", dest="restart_every", type=int, default=150)
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args(); random.seed(a.seed); asyncio.run(main_async(a))


if __name__ == "__main__":
    main()
