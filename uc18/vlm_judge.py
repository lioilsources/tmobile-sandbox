#!/usr/bin/env python3
"""Qwen3-VL-8B judge through vLLM (:8004, guided JSON) — pairwise (UC18) or single-image (UC19).

  # UC18: per prompt, every checkpoint pair of the same seed; order randomised, SWAP_FRAC re-asked reversed
  vlm_judge.py pair   --manifest /cold/w2/uc18/manifest.jsonl --refs uc18/prompts/uc18_prompts.jsonl \
                      --out /cold/w2/uc18/scores/vlm_pairs.jsonl [--seed-idx 1] [--swap-frac 0.15]
  # UC19: does the image match the Czech request? (1–5 + missing / contradicting elements)
  vlm_judge.py single --manifest /cold/w2/uc19/manifest.jsonl --refs uc19/prompts/cz_prompts.jsonl \
                      --out /cold/w2/uc19/scores/vlm_single.jsonl
  vlm_judge.py schema-check        # validates both schemas against the JSON Schema 2020-12 metaschema

MODEL_NOTES rules applied: schema checked against the metaschema before the first request (an invalid one
= HTTP 400 on every request); every array has maxItems (+ presence_penalty) so the model can't loop;
the first error is printed with the response BODY; error share watched from the first 20 answers (abort
exit 4); one fsynced line per answer (resume skips done ids); a missing image is a per-item error.
Judge identity = model alias @ JUDGE_VERSION, stored on every row.
"""
from __future__ import annotations

import argparse, base64, hashlib, io, itertools, json, os, random, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import JsonlWriter, done_keys, log, read_jsonl  # noqa: E402

JUDGE_VERSION = "uc18-judge-v1"
MAX_SIDE = 768  # 2 images × ~576 visual tokens + ~400 text tokens ≪ 8k context

_LIST = lambda d: {"type": "array", "items": {"type": "string", "maxLength": 48}, "maxItems": 4, "description": d}  # noqa: E731
_AB = {"type": "string", "enum": ["first", "second", "tie"]}
PAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "missing_in_first": _LIST("prompt elements absent or wrong in the first image"),
        "missing_in_second": _LIST("prompt elements absent or wrong in the second image"),
        "adherence": _AB,
        "quality": _AB,
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": ["missing_in_first", "missing_in_second", "adherence", "quality", "confidence"],
    "additionalProperties": False,
}
SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "missing": {"type": "array", "items": {"type": "string", "maxLength": 48}, "maxItems": 5},
        "contradictions": {"type": "array", "items": {"type": "string", "maxLength": 48}, "maxItems": 3},
        "match": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "quality": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
    },
    "required": ["missing", "contradictions", "match", "quality"],
    "additionalProperties": False,
}

PAIR_SYSTEM = (
    "You are a strict rater in a text-to-image benchmark. You see two images generated from the same text prompt. "
    "First list the prompt elements (subject, attributes, count, action, setting, style/medium, lighting, camera) "
    "that are missing or wrong in each image, then decide which image follows the prompt better (adherence) and "
    "which is technically and aesthetically better (quality: anatomy, artifacts, composition, coherence). "
    "Say \"tie\" only when you genuinely cannot separate them. Image order carries no meaning. Answer with JSON only."
)
SINGLE_SYSTEM = (
    "You are a strict rater in a text-to-image benchmark. A user wrote an image request in Czech; an English gist of "
    "the intended meaning is given too. List requested elements that are missing or wrong, list depicted things the "
    "request explicitly excludes (e.g. 'without glasses' but glasses are visible), then rate how well the image "
    "matches the request (match 1–5: 5 = everything requested is there and nothing excluded) and its visual "
    "quality (1–5). Answer with JSON only."
)


def schema_check() -> None:
    from jsonschema import Draft202012Validator
    for s in (PAIR_SCHEMA, SINGLE_SCHEMA):
        Draft202012Validator.check_schema(s)

        def walk(n):
            if isinstance(n, dict):
                if n.get("type") == "array":
                    assert "maxItems" in n, f"array without maxItems: {n}"
                for k, v in n.items():
                    if k == "properties":
                        for vv in v.values():
                            walk(vv)
                    elif isinstance(v, (dict, list)):
                        walk(v)
            elif isinstance(n, list):
                for x in n:
                    walk(x)
        walk(s)


def data_url(path: str) -> str:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def load_manifest(path):
    last = {}
    for r in read_jsonl(path):
        if not r.get("error") and r.get("path"):
            last[r["id"]] = r
    return list(last.values())


def pair_items(rows, refs, seed_idx, swap_frac, ckpts=None, variant=None):
    by = {}
    for r in rows:
        if r.get("seed_idx") != seed_idx or (ckpts and r["ckpt"] not in ckpts) or (variant is not None and r.get("variant", "") != variant):
            continue
        by.setdefault(r["pid"], {})[r["ckpt"]] = r
    items = []
    for pid, d in sorted(by.items()):
        if pid not in refs:
            continue
        for a, b in itertools.combinations(sorted(d), 2):
            h = int(hashlib.sha256(f"{pid}|{a}|{b}".encode()).hexdigest()[:8], 16)
            first, second = (a, b) if h % 2 == 0 else (b, a)
            base = {"pid": pid, "style": refs[pid].get("style") or d[a].get("style", ""), "seed_idx": seed_idx}
            items.append({**base, "id": f"{pid}/s{seed_idx}/{a}|{b}", "first": first, "second": second,
                          "paths": [d[first]["path"], d[second]["path"]], "swapped": False})
            if random.Random(h).random() < swap_frac:
                items.append({**base, "id": f"{pid}/s{seed_idx}/{a}|{b}/swap", "first": second, "second": first,
                              "paths": [d[second]["path"], d[first]["path"]], "swapped": True})
    return items


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pair", "single", "schema-check"])
    ap.add_argument("--manifest")
    ap.add_argument("--refs")
    ap.add_argument("--out")
    ap.add_argument("--base-url", default="http://127.0.0.1:8004/v1")
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--presence-penalty", type=float, default=1.0)
    ap.add_argument("--seed-idx", type=int, default=1)
    ap.add_argument("--swap-frac", type=float, default=0.15)
    ap.add_argument("--ckpts", default="")
    ap.add_argument("--variant", default=None, help="pair: only manifest rows of this variant")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="build items + one request body, no HTTP")
    a = ap.parse_args(argv)

    schema_check()
    if a.mode == "schema-check":
        print("schemas OK (Draft 2020-12, every array has maxItems)")
        return 0
    import requests

    refs = {}
    for r in read_jsonl(a.refs):
        refs[r.get("id") or r.get("cid")] = r
    rows = load_manifest(a.manifest)
    if a.mode == "pair":
        items = pair_items(rows, refs, a.seed_idx, a.swap_frac, set(a.ckpts.split(",")) if a.ckpts else None, a.variant)
    else:
        items = [{"id": r["id"], "pid": r["pid"], "ckpt": r["ckpt"], "variant": r.get("variant", ""),
                  "style": refs.get(r["pid"], {}).get("style", ""), "paths": [r["path"]]}
                 for r in rows if r["pid"] in refs]
    done = done_keys(a.out) if a.out else set()
    todo = [it for it in items if it["id"] not in done]
    if a.limit:
        todo = todo[: a.limit]
    log(f"vlm_judge {a.mode}: {len(items)} items, {len(done)} done, {len(todo)} to do")

    schema = PAIR_SCHEMA if a.mode == "pair" else SINGLE_SCHEMA

    def body_for(it):
        ref = refs[it["pid"]]
        if a.mode == "pair":
            text = f"Prompt: {ref.get('prose') or ref.get('ref_en')}"
            content = [{"type": "text", "text": "First image:"}, {"type": "image_url", "image_url": {"url": data_url(it["paths"][0])}},
                       {"type": "text", "text": "Second image:"}, {"type": "image_url", "image_url": {"url": data_url(it["paths"][1])}},
                       {"type": "text", "text": text}]
            system = PAIR_SYSTEM
        else:
            text = f"Request (Czech): {ref['cz']}\nEnglish gist: {ref['ref_en']}"
            content = [{"type": "image_url", "image_url": {"url": data_url(it["paths"][0])}}, {"type": "text", "text": text}]
            system = SINGLE_SYSTEM
        return {"model": a.model, "temperature": 0, "max_tokens": a.max_tokens, "presence_penalty": a.presence_penalty,
                "response_format": {"type": "json_schema", "json_schema": {"name": JUDGE_VERSION.replace("-", "_") + "_" + a.mode,
                                                                            "schema": schema, "strict": True}},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]}

    if a.dry_run:
        b = body_for(todo[0]) if todo else {}
        for m in b.get("messages", []):
            if isinstance(m["content"], list):
                for c in m["content"]:
                    if c["type"] == "image_url":
                        c["image_url"]["url"] = c["image_url"]["url"][:40] + "…"
        print(json.dumps({"items": len(items), "todo": len(todo), "example_id": todo[0]["id"] if todo else None,
                          "body": b}, ensure_ascii=False, indent=1)[:4000])
        return 0

    w = JsonlWriter(a.out)
    sess = requests.Session()
    lock = threading.Lock()
    st = {"ok": 0, "err": 0, "first": None}

    def one(it):
        t = time.perf_counter()
        rec = {k: v for k, v in it.items() if k != "paths"}
        rec["judge"] = f"{a.model}@{JUDGE_VERSION}"
        try:
            body = body_for(it)
        except Exception as e:  # missing/corrupt image → per-item error
            return {**rec, "error": f"image: {e}"[:500]}
        try:
            r = sess.post(a.base_url.rstrip("/") + "/chat/completions", json=body, timeout=180)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:1500]}")
            ch = r.json()["choices"][0]
            if ch.get("finish_reason") == "length":
                raise RuntimeError(f"truncated at max_tokens={a.max_tokens}: {str(ch['message']['content'])[-200:]}")
            ans = json.loads(ch["message"]["content"])
            rec.update(ans)
            if a.mode == "pair":  # resolve first/second → model ids
                for k in ("adherence", "quality"):
                    rec[f"{k}_winner"] = {"first": it["first"], "second": it["second"]}.get(ans[k], "tie")
            u = r.json().get("usage", {})
            rec["tokens"] = [u.get("prompt_tokens"), u.get("completion_tokens")]
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:2000]
        rec["ms"] = round((time.perf_counter() - t) * 1000)
        return rec

    t0 = time.time()
    abort = False
    with ThreadPoolExecutor(a.concurrency) as pool:
        futs = [pool.submit(one, it) for it in todo]
        for f in as_completed(futs):
            rec = f.result()
            w.write(rec)
            with lock:
                if "error" in rec:
                    st["err"] += 1
                    if st["first"] is None:
                        st["first"] = rec["error"]
                        print(f"first error ({rec['id']}): {rec['error']}", file=sys.stderr, flush=True)
                else:
                    st["ok"] += 1
                n = st["ok"] + st["err"]
                if n == 20 and st["err"] > 10 and not abort:
                    abort = True
                    log(f"!!! vlm_judge: {st['err']}/20 first answers failed — aborting. First: {st['first'][:600]}")
                    for ff in futs:
                        ff.cancel()
                if n % 500 == 0:
                    el = time.time() - t0
                    log(f"vlm_judge {a.mode}: {n}/{len(todo)} ({st['err']} err, {n / el * 3600:.0f}/h)")
    w.close()
    el = time.time() - t0
    print(json.dumps({"vlm_judge": a.mode, "model": a.model, "ok": st["ok"], "errors": st["err"], "wall_s": round(el, 1),
                      "per_h": round(st["ok"] / el * 3600) if el and st["ok"] else None}), file=sys.stderr)
    if abort:
        return 4
    return 0 if st["ok"] or not todo else 1


if __name__ == "__main__":
    sys.exit(main())
