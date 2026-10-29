#!/usr/bin/env python3
"""UC19 translator techniques through vLLM :8000 (profile A = qwen3-32b or S = qwen3-8b).

  translate.py --cz uc19/prompts/cz_prompts.jsonl --fewshot uc19/prompts/fewshot.jsonl \
      --out /cold/w2/uc19/translations.jsonl [--techniques t1,t2] [--dialects pony,illustrious,sdxl,flux] \
      [--base-url http://127.0.0.1:8000/v1] [--model auto] [--concurrency auto]
  translate.py … --dry-run          # no HTTP: schemas, one request body per technique/dialect, post-processing of few-shots

t1 = baseline: plain CZ→EN translation, the same text for every dialect (one request per prompt).
t2 = few-shot per dialect: 15 hand-written examples as chat turns + guided JSON (tags list maxItems 40 /
     prompt maxLength) + deterministic post-processing (WD14 vocabulary, no quality/rating tags, weight syntax,
     CLIP-L ≤ 75 tokens for SDXL, no weights for FLUX, SFW filter).
Thinking is disabled with chat_template_kwargs {"enable_thinking": false} (with --reasoning-parser qwen3 a
"/no_think" in the text is ignored and content comes back empty — MODEL_NOTES). Concurrency auto: 16 for
qwen3-32b (48 overloads it: UC11), 48 for qwen3-8b. Resumable per (cid, technique, dialect); one fsynced line
per answer; first error printed with the response body; abort (exit 4) when the first 10 answers all fail.
"""
from __future__ import annotations

import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed

H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, H)
sys.path.insert(0, os.path.join(H, "..", "uc18"))
import dialects as D  # noqa: E402
from common import JsonlWriter, log, read_jsonl  # noqa: E402


def key(cid, tech, dia):
    return f"{cid}/{tech}/{dia}"


def build_body(model, tech, dialect, cz, fewshot):
    base = {"model": model, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    if tech == "t1":
        return {**base, "max_tokens": 300, "messages": [{"role": "system", "content": D.BASELINE_SYSTEM},
                                                        {"role": "user", "content": cz}]}
    return {**base, "max_tokens": D.MAX_TOKENS[dialect],
            "presence_penalty": 1.0 if dialect in ("pony", "illustrious") else 0.3,
            "response_format": {"type": "json_schema", "json_schema": {"name": f"uc19_{dialect}", "schema": D.SCHEMA[dialect],
                                                                        "strict": True}},
            "messages": [{"role": "system", "content": D.SYSTEM[dialect]}] + D.fewshot_messages(dialect, fewshot)
            + [{"role": "user", "content": cz}]}


VOCAB_MODE = "report"


def post(tech, dialect, content, vocab, tok):
    if tech == "t1":
        return D.post_baseline(content, tok)
    ans = json.loads(content)
    if dialect in ("pony", "illustrious"):
        r = D.post_tags(dialect, ans.get("tags", []), vocab, VOCAB_MODE)
        r["n_raw"] = len(ans.get("tags", []))
        return r
    if dialect == "sdxl":
        return D.post_sdxl(ans.get("prompt", ""), ans.get("negative", ""), tok)
    return D.post_flux(ans.get("prompt", ""), tok)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cz", required=True)
    ap.add_argument("--fewshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--techniques", default="t1,t2")
    ap.add_argument("--dialects", default=",".join(D.DIALECTS))
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="auto")
    ap.add_argument("--concurrency", default="auto")
    ap.add_argument("--vocab", default=None, help="WD14 selected_tags.csv (default: HF cache)")
    ap.add_argument("--vocab-mode", default="report", choices=["report", "filter"],
                    help="out-of-WD14-vocabulary tags: keep + list (report) or drop like prompt-tags@v1 (filter)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    global VOCAB_MODE
    VOCAB_MODE = a.vocab_mode

    from jsonschema import Draft202012Validator
    for s in D.SCHEMA.values():
        Draft202012Validator.check_schema(s)
    cz = read_jsonl(a.cz)[: a.limit or None]
    fs = read_jsonl(a.fewshot)
    evset = {r["cz"] for r in cz}
    assert not any(f["cz"] in evset for f in fs), "few-shot example leaks into the eval set"
    techs, dias = a.techniques.split(","), a.dialects.split(",")
    vocab, tok = D.load_vocab(a.vocab), D.Tokens()
    log(f"translate: {len(cz)} prompts × {techs} × {dias}; vocab {'%d tags' % len(vocab) if vocab else 'MISSING'}, "
        f"CLIP tokenizer {'ok' if tok.clip else 'MISSING'}, T5 tokenizer {'ok' if tok.t5 else 'MISSING'}")

    if a.dry_run:
        out = {}
        for t in techs:
            for d in (dias if t != "t1" else ["*"]):
                b = build_body("MODEL", t, d, cz[0]["cz"], fs)
                out[f"{t}/{d}"] = {"n_messages": len(b["messages"]), "max_tokens": b["max_tokens"],
                                   "has_schema": "response_format" in b, "last_user": b["messages"][-1]["content"]}
        # the few-shot answers themselves must survive post-processing (catches vocab/limit rules that are too strict)
        pp = {}
        for d in dias:
            res = [post("t2", d, json.dumps({"tags": f[d]} if d in ("pony", "illustrious") else
                                            {"prompt": f[d], "negative": "blurry"} if d == "sdxl" else {"prompt": f[d]}),
                        vocab, tok) for f in fs]
            pp[d] = {"dropped": sorted({x for r in res for x in r.get("dropped", [])}),
                     "oov": sorted({x for r in res for x in r.get("oov", [])}),
                     "clip_tokens_max": max((r.get("clip_tokens_raw") or 0) for r in res),
                     "t5_tokens": [r.get("t5_tokens") for r in res][:5], "truncated": sum(bool(r.get("truncated")) for r in res),
                     "unsafe": [r.get("unsafe") or r.get("unsafe_removed") for r in res if r.get("unsafe") or r.get("unsafe_removed")]}
        print(json.dumps({"bodies": out, "fewshot_postprocess": pp}, ensure_ascii=False, indent=1))
        return 0

    import requests
    sess = requests.Session()
    model = a.model
    if model == "auto":
        model = sess.get(a.base_url.rstrip("/") + "/models", timeout=10).json()["data"][0]["id"]
    conc = int(a.concurrency) if a.concurrency != "auto" else (16 if "32b" in model else 48)
    done = {r["key"] for r in read_jsonl(a.out) if not r.get("error")}
    items = []
    for r in cz:
        for t in techs:
            if t == "t1":
                if key(r["cid"], "t1", "*") not in done:
                    items.append((r, "t1", "*"))
            else:
                items += [(r, t, d) for d in dias if key(r["cid"], t, d) not in done]
    log(f"translate: model {model}, concurrency {conc}, {len(items)} requests to do ({len(done)} done)")
    w = JsonlWriter(a.out)
    st = {"ok": 0, "err": 0, "first": None}
    lock = threading.Lock()

    def one(it):
        r, t, d = it
        rec = {"key": key(r["cid"], t, d), "cid": r["cid"], "technique": t, "dialect": d, "category": r["category"],
               "style": r["style"], "cz": r["cz"], "model": model, "version": f"{model}@{D.PROMPT_VERSION}", "vocab_mode": VOCAB_MODE}
        t0 = time.perf_counter()
        try:
            resp = sess.post(a.base_url.rstrip("/") + "/chat/completions", json=build_body(model, t, d, r["cz"], fs), timeout=180)
            if resp.status_code >= 400:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:1500]}")
            j = resp.json()
            ch = j["choices"][0]
            content = ch["message"].get("content") or ""
            if not content.strip():
                raise RuntimeError("empty content (thinking not disabled? reasoning parser ate the answer)")
            if ch.get("finish_reason") == "length":
                raise RuntimeError(f"truncated at max_tokens: {content[-200:]}")
            rec["raw"] = content
            rec.update(post(t, d, content, vocab, tok))
            rec["usage"] = j.get("usage", {})
        except Exception as e:  # noqa: BLE001
            rec["error"] = str(e)[:2000]
        rec["ms"] = round((time.perf_counter() - t0) * 1000)
        return rec

    abort = False
    t0 = time.time()
    with ThreadPoolExecutor(conc) as pool:
        futs = [pool.submit(one, it) for it in items]
        for f in as_completed(futs):
            rec = f.result()
            w.write(rec)
            with lock:
                if "error" in rec:
                    st["err"] += 1
                    if not st["first"]:
                        st["first"] = rec["error"]
                        print(f"first error ({rec['key']}): {rec['error']}", file=sys.stderr, flush=True)
                else:
                    st["ok"] += 1
                if st["ok"] == 0 and st["err"] >= 10 and not abort:
                    abort = True
                    log(f"!!! translate: first {st['err']} answers failed — abort. {st['first'][:500]}")
                    for ff in futs:
                        ff.cancel()
    w.close()
    el = time.time() - t0
    print(json.dumps({"translate_model": model, "ok": st["ok"], "errors": st["err"], "wall_s": round(el, 1),
                      "req_per_s": round(st["ok"] / el, 2) if el else None}), file=sys.stderr)
    return 4 if abort else (0 if st["ok"] or not items else 1)


if __name__ == "__main__":
    sys.exit(main())
