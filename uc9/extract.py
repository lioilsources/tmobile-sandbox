#!/usr/bin/env python3
"""UC9: dávková LLM extrakce nad Registrem smluv (contracts.jsonl z parse.py) → extracted.jsonl.
N workerů drží souběh, výstup se přidává (lze navázat po pádu), na konci report.json s propustností,
tokeny a prefix-cache hit rate z /metrics vLLM.

  extract.py --inp contracts.jsonl --out runs/<run> --concurrency 384 [--limit 5000] [--model qwen3-30b-a3b]
"""
import argparse, asyncio, json, pathlib, re, statistics, time
import httpx

ODVETVI = ["IT a software", "telekomunikace", "zdravotnictví a léčiva", "laboratoř a věda", "stavebnictví a údržba",
           "doprava a vozidla", "energie a utility", "potraviny a stravování", "kancelář a spotřební materiál",
           "vzdělávání a školení", "kultura, sport a volný čas", "sociální služby", "právní a poradenské služby",
           "marketing, média a tisk", "úklid a facility", "nemovitosti a pronájem", "finance a pojištění",
           "zemědělství, lesnictví a životní prostředí", "bezpečnost a ochrana", "jiné"]
TYP = ["smlouva", "objednávka", "dodatek", "rámcová smlouva", "dotace / dar", "nájem / pacht", "licence", "jiné"]
PLNENI = ["dodávky", "služby", "stavební práce", "dotace / dar", "nájem", "převod majetku", "jiné"]

SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["typ", "plneni", "odvetvi", "dodavatel_ico", "it_relevance", "popis"],
          "properties": {
              "typ": {"enum": TYP}, "plneni": {"enum": PLNENI}, "odvetvi": {"enum": ODVETVI},
              "dodavatel_ico": {"type": "string", "maxLength": 10},
              "it_relevance": {"type": "integer", "minimum": 0, "maximum": 3},
              "popis": {"type": "string", "maxLength": 90}}}

RF_SCHEMA = {"type": "json_schema", "json_schema": {"name": "smlouva", "schema": SCHEMA}}


def parse(txt):
    """JSON odpovídající SCHEMA, jinak None."""
    try:
        x = json.loads(txt)
    except Exception:
        return None
    ok = (isinstance(x, dict) and set(x) == set(SCHEMA["required"]) and x["typ"] in TYP and x["plneni"] in PLNENI
          and x["odvetvi"] in ODVETVI and isinstance(x["it_relevance"], int) and 0 <= x["it_relevance"] <= 3
          and isinstance(x["dodavatel_ico"], str) and isinstance(x["popis"], str))
    return x if ok else None


# Statická část jde první a je pro všechny dotazy stejná → prefix caching ji počítá jednou.
SYSTEM = f"""Jsi analytik veřejných zakázek. Dostaneš metadata jedné smlouvy z českého Registru smluv.
Vrať JSON:
- typ: druh dokumentu ({", ".join(TYP)}). Objednávka = jednorázový nákup na objednávku; dodatek = změna dřívější smlouvy.
- plneni: co protistrana plní ({", ".join(PLNENI)}).
- odvetvi: jedno odvětví plnění ze seznamu: {"; ".join(ODVETVI)}.
- dodavatel_ico: IČO strany, která zboží/službu/práci DODÁVÁ (u dotace příjemce dotace; u nájmu pronajímatel). Pokud IČO chybí, prázdný řetězec.
- it_relevance: 0 = nesouvisí s IT; 1 = jen IT příslušenství a spotřební materiál (tiskárny, tonery, kabely, kancelářská technika); zdravotnický, laboratorní a jiný materiál = 0; 2 = IT/telco plnění (HW, SW, licence, sítě, servis);
  3 = vývoj software, data, AI, digitalizace, kybernetická bezpečnost, cloud.
- popis: stručný věcný popis plnění česky, max. 12 slov, bez názvů stran.
Odpovídej jen JSONem."""


def user_msg(r):
    def st(s):
        return f"{s['nazev']} (IČO {s['ico'] or '—'}{', příjemce platby' if s['prijemce'] == '1' else ''})"
    hod = r["bez_dph"] and f"{r['bez_dph']} Kč bez DPH" or r["s_dph"] and f"{r['s_dph']} Kč s DPH" or r["cizi_mena"] or (r["cena_neuvedena"] or "neuvedena")
    return (f"Zveřejnil: {st(r['subjekt'])}\nProtistrany: {'; '.join(st(s) for s in r['strany'][:4]) or '—'}\n"
            f"Předmět: {r['predmet']}\nHodnota: {hod}\nPřílohy: {', '.join(r['prilohy'][:3]) or '—'}")


async def metrics(cl, base):
    try:
        t = (await cl.get(base + "/metrics")).text
    except Exception:
        return {}
    out = {}
    for k in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"):
        m = re.search(rf"^{k}\{{[^}}]*\}} ([0-9.e+]+)", t, re.M)
        if m: out[k] = float(m.group(1))
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/srv/work/uc9/contracts.jsonl"); ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000"); ap.add_argument("--model", default="qwen3-30b-a3b")
    ap.add_argument("--concurrency", type=int, default=384); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--guided", choices=["schema", "json", "none"], default="json",
                    help="schema = json_schema (enumy); json = json_object; none = volný text")
    ap.add_argument("--ids", help="soubor s idSmlouvy (jedno na řádek) – zpracovat jen tyto (vzorek pro rozhodčího)")
    ap.add_argument("--no-think", action="store_true", help="Qwen3 hybrid: vypnout reasoning (chat_template_kwargs)")
    ap.add_argument("--skip", type=int, default=0, help="přeskočit prvních N smluv (oddělení pilotu od ostré dávky)")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done = set()
    pf = out / "extracted.jsonl"
    if pf.exists():
        done = {json.loads(l)["id"] for l in open(pf) if l.strip()}
    rows = []
    want = set(open(a.ids).read().split()) if a.ids else None
    with open(a.inp) as f:
        for i, l in enumerate(f):
            if i < a.skip: continue
            if a.limit and len(rows) >= a.limit: break
            r = json.loads(l)
            if want is not None and r["id"] not in want: continue
            if r["id"] not in done: rows.append(r)
    q = asyncio.Queue()
    for r in rows: q.put_nowait(r)
    st = {"ok": 0, "err": 0, "bad_json": 0, "opraveno": 0, "tin": 0, "tout": 0, "lat": []}
    fo = open(pf, "a", encoding="utf-8")

    async def call(cl, body):
        for attempt in range(3):
            try:
                resp = await cl.post(a.base + "/v1/chat/completions", json=body); resp.raise_for_status(); j = resp.json()
                u = j.get("usage", {}); st["tin"] += u.get("prompt_tokens", 0); st["tout"] += u.get("completion_tokens", 0)
                return j["choices"][0]["message"]["content"], None
            except Exception as e:
                err = repr(e); await asyncio.sleep(2 * (attempt + 1))
        return None, err

    async def worker(cl):
        while True:
            try: r = q.get_nowait()
            except asyncio.QueueEmpty: return
            body = {"model": a.model, "temperature": 0, "max_tokens": 160,
                    "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(r)}]}
            if a.no_think: body["chat_template_kwargs"] = {"enable_thinking": False}
            if a.guided == "schema": body["response_format"] = RF_SCHEMA
            elif a.guided == "json": body["response_format"] = {"type": "json_object"}
            t0 = time.time()
            txt, err = await call(cl, body); x = parse(txt)
            if txt is not None and x is None and a.guided != "schema":
                # volný/JSON výstup mimo výčet → jeden opakovaný dotaz s vynuceným schématem
                st["opraveno"] += 1; body["response_format"] = RF_SCHEMA
                txt, err = await call(cl, body); x = parse(txt)
            st["lat"].append(time.time() - t0)
            if txt is None:
                st["err"] += 1; fo.write(json.dumps({"id": r["id"], "err": err}, ensure_ascii=False) + "\n"); continue
            st["ok" if x else "bad_json"] += 1
            fo.write(json.dumps({"id": r["id"], "x": x}, ensure_ascii=False) + "\n")

    async with httpx.AsyncClient(timeout=600, limits=httpx.Limits(max_connections=a.concurrency + 8)) as cl:
        m0 = await metrics(cl, a.base); t0 = time.time()
        tasks = [asyncio.create_task(worker(cl)) for _ in range(a.concurrency)]
        last = 0
        while not all(t.done() for t in tasks):
            await asyncio.sleep(30)
            n = st["ok"] + st["err"] + st["bad_json"]; el = time.time() - t0
            print(f"{n}/{len(rows)}  {n / el:.1f} dok/s  +{(n - last) / 30:.1f}/s  out {st['tout'] / el:.0f} tok/s", flush=True); last = n
            fo.flush()
        wall = time.time() - t0; m1 = await metrics(cl, a.base)
    fo.close()
    n = st["ok"] + st["err"] + st["bad_json"]
    hits = m1.get("vllm:prefix_cache_hits_total", 0) - m0.get("vllm:prefix_cache_hits_total", 0)
    qs = m1.get("vllm:prefix_cache_queries_total", 0) - m0.get("vllm:prefix_cache_queries_total", 0)
    lat = sorted(st["lat"]) or [0]
    rep = {"smluv": n, "ok": st["ok"], "bad_json": st["bad_json"], "opraveno_schematem": st["opraveno"], "chyb": st["err"], "souběh": a.concurrency, "guided": a.guided,
           "sekund": round(wall), "dok_za_s": round(n / wall, 2), "dok_za_hodinu": round(n / wall * 3600),
           "tok_vstup_s": round(st["tin"] / wall), "tok_vystup_s": round(st["tout"] / wall),
           "tokeny_prumer": {"vstup": round(st["tin"] / max(n, 1)), "vystup": round(st["tout"] / max(n, 1))},
           "prefix_cache_hit_%": round(100 * hits / qs, 1) if qs else None,
           "latence_s": {"p50": round(lat[len(lat) // 2], 2), "p95": round(lat[int(len(lat) * .95)], 2)}}
    (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
