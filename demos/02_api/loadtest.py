#!/usr/bin/env python3
"""Load test AI API se třemi tenanty přes gateway (127.0.0.1:8080). Tři fáze:
  1. klid       A + B běžný provoz
  2. nápor C    C pošle naráz 40 souběžných dlouhých úloh + přepisy hovorů (limit tenanta 16 → 429 + opakování)
  3. klid       zase jen A + B
Měří TTFT a dobu po tenantech a fázích → <out>/loadtest.json. Účtování dělá gateway (kniha usage.jsonl).

  loadtest.py --tenants ~/tmobile-sandbox/56_api/tenants.json --out /srv/work/demo2/runs/<run> [--phase 120]
"""
import argparse, asyncio, collections, json, pathlib, random, statistics, time
import httpx

GW = "http://127.0.0.1:8080/v1"
QUESTIONS = ["Kolik stojí roaming v Chorvatsku?", "Jak si aktivuji eSIM v novém telefonu?", "Proč mám na faktuře položku za prémiové SMS?",
             "Můžu si změnit tarif v polovině období?", "Jak dlouho trvá přenos čísla od jiného operátora?", "Nefunguje mi mobilní internet, co mám zkusit?",
             "Jak zablokuji ukradený telefon?", "Kde najdu PUK ke své SIM kartě?", "Jaký je rozdíl mezi tarifem s 20 a s neomezenými daty?",
             "Jak vypovím smlouvu a jaká je výpovědní lhůta?", "Můžu sdílet data s rodinou?", "Proč je můj účet vyšší než minulý měsíc?"]

def pct(xs, q):
    xs = sorted(xs); return round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None

class Stats:
    def __init__(self): self.rows = []
    def add(self, **kw): self.rows.append(kw)

async def call(cl, key, path, stats, tenant, phase_of, **kw):
    t0 = time.time(); ph = phase_of(t0); retries = 0
    while True:
        try:
            if kw.get("stream"):
                first = None
                async with cl.stream("POST", f"{GW}{path}", headers={"Authorization": f"Bearer {key}"}, json=kw["json"]) as r:
                    if r.status_code == 429: await r.aread(); raise httpx.HTTPStatusError("429", request=r.request, response=r)
                    r.raise_for_status()
                    async for line in r.aiter_lines():
                        if first is None and line.startswith("data: {") and '"content"' in line: first = time.time()
                stats.add(tenant=tenant, phase=ph, ok=True, ttft=(first or time.time()) - t0, dur=time.time() - t0, retries=retries); return
            r = await cl.post(f"{GW}{path}", headers={"Authorization": f"Bearer {key}"}, **{k: v for k, v in kw.items() if k != "stream"})
            if r.status_code == 429: raise httpx.HTTPStatusError("429", request=r.request, response=r)
            r.raise_for_status()
            stats.add(tenant=tenant, phase=ph, ok=True, dur=time.time() - t0, retries=retries); return
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and retries < 60:
                retries += 1; await asyncio.sleep(1 + random.random()); continue
            stats.add(tenant=tenant, phase=ph, ok=False, dur=time.time() - t0, retries=retries, err=str(e.response.status_code)); return
        except httpx.HTTPError as e:
            stats.add(tenant=tenant, phase=ph, ok=False, dur=time.time() - t0, retries=retries, err=type(e).__name__); return

async def poisson(rate, until, make):
    tasks = []
    while time.time() < until:
        tasks.append(asyncio.create_task(make())); await asyncio.sleep(random.expovariate(rate))
    await asyncio.gather(*tasks)

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tenants", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--phase", type=int, default=120); ap.add_argument("--a-rate", type=float, default=4); ap.add_argument("--c-workers", type=int, default=40)
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keys = {t["name"]: t["keys"][0] for t in json.load(open(a.tenants))["tenants"]}
    KA, KB, KC = keys["A-chatbot"], keys["B-rag"], keys["C-batch"]
    calls = sorted(pathlib.Path("/srv/work/demo1/calls").glob("call_*.json"))
    texts = [r["text"] for c in calls for r in json.loads(c.read_text())["repliky"]]
    context = " ".join(texts[:400])[:9000]               # ~3 000 tokenů „dokumentace“ pro RAG
    wavs = sorted(pathlib.Path("/srv/work/demo1/audio").glob("call_*.wav"))
    st = Stats(); T = [time.time()]; P = a.phase
    phase_of = lambda t: 1 if t - T[0] < P else (2 if t - T[0] < 2 * P else 3)   # T[0] se po zahřátí posune
    limits = httpx.Limits(max_connections=400, max_keepalive_connections=200)
    async with httpx.AsyncClient(timeout=600, limits=limits) as cl:
        def chat_a():
            return call(cl, KA, "/chat/completions", st, "A-chatbot", phase_of, stream=True, json={"model": "chat", "stream": True,
                        "max_tokens": random.randint(120, 300), "messages": [{"role": "system", "content": "Jsi stručný zákaznický asistent mobilního operátora."},
                                                                               {"role": "user", "content": random.choice(QUESTIONS)}]})
        def rag_b():
            if random.random() < 0.5:
                return call(cl, KB, "/embeddings", st, "B-rag", phase_of, json={"model": "embed", "input": random.sample(texts, 32)})
            return call(cl, KB, "/chat/completions", st, "B-rag", phase_of, json={"model": "chat", "max_tokens": 200, "messages": [
                {"role": "system", "content": "Odpovídej jen z dodaného kontextu."},
                {"role": "user", "content": f"Kontext:\n{context}\n\nOtázka: {random.choice(QUESTIONS)}"}]})
        async def burst_c(until):
            async def worker(i):
                while time.time() < until:
                    if i % 5 == 0 and wavs:
                        w = random.choice(wavs)
                        await call(cl, KC, "/audio/transcriptions", st, "C-batch", phase_of, files={"file": (w.name, w.read_bytes(), "audio/wav")},
                                   data={"model": "whisper", "language": "cs", "response_format": "verbose_json"})
                    else:
                        await call(cl, KC, "/chat/completions", st, "C-batch", phase_of, json={"model": "chat", "max_tokens": 800, "messages": [
                            {"role": "user", "content": f"Napiš podrobný report pro vedení (cca 600 slov) o tomto tématu: {random.choice(QUESTIONS)}"}]})
            await asyncio.gather(*(worker(i) for i in range(a.c_workers)))
        # zahřátí: každý typ požadavku víckrát, ať se rozběhnou všechny workery LiteLLM a CUDA grafy vLLM
        warm = Stats(); wp = lambda t: 0
        for _ in range(3):
            await asyncio.gather(*[call(cl, KA, "/chat/completions", warm, "warm", wp, stream=True, json={"model": "chat", "stream": True, "max_tokens": 16,
                                   "messages": [{"role": "user", "content": "ahoj"}]}) for _ in range(8)],
                                 *[call(cl, KB, "/embeddings", warm, "warm", wp, json={"model": "embed", "input": ["ahoj"]}) for _ in range(8)],
                                 *[call(cl, KB, "/chat/completions", warm, "warm", wp, json={"model": "chat", "max_tokens": 8, "messages": [{"role": "user", "content": "ahoj"}]}) for _ in range(8)],
                                 *[call(cl, KC, "/audio/transcriptions", warm, "warm", wp, files={"file": (wavs[0].name, wavs[0].read_bytes(), "audio/wav")},
                                        data={"model": "whisper", "language": "cs"}) for _ in range(2)])
        print(f"zahřáto: {len(warm.rows)} požadavků, chyb {sum(not r['ok'] for r in warm.rows)}", flush=True)
        T[0] = T0 = time.time(); end = T0 + 3 * P
        print(f"fáze 1–3 po {P} s, A {a.a_rate} req/s, B 2 req/s, C {a.c_workers} souběžných ve fázi 2", flush=True)
        await asyncio.gather(poisson(a.a_rate, end, chat_a), poisson(2, end, rag_b),
                             burst_c_after(P, T0, burst_c))
    res = {"faze_s": P, "a_rate": a.a_rate, "c_workers": a.c_workers, "start": T0, "konec": time.time(), "vysledky": {}}
    for t in ("A-chatbot", "B-rag", "C-batch"):
        for ph in (1, 2, 3):
            rows = [r for r in st.rows if r["tenant"] == t and r["phase"] == ph]
            if not rows: continue
            ok = [r for r in rows if r["ok"]]; tt = [r["ttft"] for r in ok if "ttft" in r]
            res["vysledky"][f"{t}/{ph}"] = {"pozadavku": len(rows), "chyb": len(rows) - len(ok), "opakovani_429": sum(r["retries"] for r in rows),
                "chyby_typ": dict(collections.Counter(r.get("err") for r in rows if not r["ok"])),
                "ttft_p50": pct(tt, .5), "ttft_p95": pct(tt, .95), "doba_p50": pct([r["dur"] for r in ok], .5), "doba_p95": pct([r["dur"] for r in ok], .95)}
    (out / "loadtest.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for k, v in res["vysledky"].items(): print(f"{k:14s} {v}")

async def burst_c_after(P, T0, burst_c):
    await asyncio.sleep(max(0, T0 + P - time.time())); await burst_c(T0 + 2 * P)

if __name__ == "__main__": asyncio.run(main())
