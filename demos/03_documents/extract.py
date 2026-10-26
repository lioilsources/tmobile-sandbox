#!/usr/bin/env python3
"""PDF → stránky (150 dpi) → Qwen3-VL (vLLM, profil C) → JSON se strukturovaným výstupem (json_schema).

  extract.py --docs /srv/work/demo3/docs --out /srv/work/demo3/runs/<run> --concurrency 8
"""
import argparse, asyncio, base64, json, pathlib, time
import httpx, pymupdf

API = "http://127.0.0.1:8000/v1/chat/completions"
N = lambda t: {"type": [t, "null"]}
SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "typ": {"enum": ["faktura", "smlouva", "jiny"]},
    # faktura
    "cislo_faktury": N("string"), "variabilni_symbol": N("string"), "datum_vystaveni": N("string"),
    "datum_splatnosti": N("string"), "duzp": N("string"), "dodavatel_nazev": N("string"), "dodavatel_ico": N("string"),
    "dodavatel_dic": N("string"), "odberatel_nazev": N("string"), "odberatel_ico": N("string"), "zaklad_dph": N("number"),
    "sazba_dph": N("number"), "dph": N("number"), "celkem": N("number"), "bankovni_ucet": N("string"), "pocet_polozek": N("integer"),
    # smlouva
    "typ_smlouvy": {"enum": ["smlouva_o_dilo", "servisni_smlouva", "jina", None]}, "cislo_smlouvy": N("string"),
    "strana1_nazev": N("string"), "strana1_ico": N("string"), "strana2_nazev": N("string"), "strana2_ico": N("string"),
    "datum_uzavreni": N("string"), "ucinnost_od": N("string"), "doba_urcita": N("boolean"), "konec_platnosti": N("string"),
    "cena_bez_dph": N("number"), "vypovedni_lhuta_mesice": N("integer"), "smluvni_pokuta_procent_denne": N("number")}}
SCHEMA["required"] = list(SCHEMA["properties"])
PROMPTS = {"v1": """Jsi systém pro vytěžování českých účetních a právních dokumentů. Urči typ dokumentu a vyplň pole.
Pravidla: data ve formátu RRRR-MM-DD; částky jako čísla bez měny a mezer (desetinná tečka); IČO jen 8 číslic;
u smlouvy strana1 = první uvedená smluvní strana; konec_platnosti jen u smlouvy na dobu určitou;
pocet_polozek = počet řádků v tabulce položek. Pole, která pro daný typ dokumentu neexistují nebo v dokumentu nejsou, nastav na null.""",
"v2": """Jsi systém pro vytěžování českých účetních a právních dokumentů. Urči typ dokumentu a vyplň pole.

Faktura – přiřazení rolí (nejčastější chyba, postupuj pečlivě):
- DODAVATEL = vystavitel faktury, ten kdo dostane zaplaceno. Poznáš ho podle: štítku „Dodavatel“, „Vystavil“, „Prodávající“;
  bankovní účet pro úhradu patří dodavateli; firma v hlavičce bez štítku nebo v patičce dokumentu je obvykle dodavatel.
- ODBĚRATEL = příjemce faktury, ten kdo platí; štítek „Odběratel“, „Kupující“, „Příjemce“.
- Dodavatel a odběratel jsou vždy dvě různé firmy. Nejprve najdi na celé stránce (včetně patičky) obě firmy, pak teprve přiřaď role.
- DIČ v ČR = „CZ“ + IČO.

Formát: data RRRR-MM-DD; částky jako čísla (desetinná tečka, bez měny a mezer); IČO jen 8 číslic;
pocet_polozek = počet řádků v tabulce položek.
Smlouva: strana1 = první uvedená smluvní strana; typ_smlouvy podle nadpisu; datum_uzavreni = datum u podpisů („V … dne“);
ucinnost_od = datum nabytí účinnosti; konec_platnosti jen u smlouvy na dobu určitou.
Pole, která pro daný typ dokumentu neexistují nebo v dokumentu nejsou, nastav na null."""}

def pages_b64(pdf, dpi):
    return [base64.b64encode(p.get_pixmap(dpi=dpi).tobytes("png")).decode() for p in pymupdf.open(pdf)]

async def one(cl, sem, pdf, out, model, a):
    async with sem:
        t0 = time.time(); imgs = pages_b64(pdf, a.dpi); t1 = time.time()
        body = {"model": model, "temperature": 0, "max_tokens": 1200,
                "response_format": {"type": "json_schema", "json_schema": {"name": "doc", "schema": SCHEMA}},
                "messages": [{"role": "system", "content": PROMPTS[a.prompt]}, {"role": "user", "content":
                    [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in imgs]
                    + [{"type": "text", "text": "Vytěž dokument do JSON."}]}]}
        r = await cl.post(API, json=body); r.raise_for_status(); j = r.json(); t2 = time.time()
        res = {"_soubor": pdf.name, "_stran": len(imgs), "_render_s": round(t1 - t0, 2), "_llm_s": round(t2 - t1, 2),
               "_usage": j["usage"], "pred": json.loads(j["choices"][0]["message"]["content"])}
        (out / f"{pdf.stem}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
        print(f"{pdf.name:14s} {len(imgs)} str. {t2 - t1:5.1f}s  {j['usage']['prompt_tokens']:5d}+{j['usage']['completion_tokens']} tok", flush=True)

async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--docs", default="/srv/work/demo3/docs")
    ap.add_argument("--out", required=True); ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--dpi", type=int, default=150); ap.add_argument("--prompt", default="v2", choices=list(PROMPTS))
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=600) as cl:
        model = (await cl.get("http://127.0.0.1:8000/v1/models")).json()["data"][0]["id"]
        pdfs = sorted(pathlib.Path(a.docs).glob("*.pdf")); sem = asyncio.Semaphore(a.concurrency); t0 = time.time()
        await asyncio.gather(*(one(cl, sem, p, out, model, a) for p in pdfs))
        wall = time.time() - t0
    (out / "_run.json").write_text(json.dumps({"model": model, "docs": len(pdfs), "concurrency": a.concurrency, "dpi": a.dpi, "prompt": a.prompt,
                                               "wall_s": round(wall, 1)}, indent=1))
    print(f"\n{len(pdfs)} dokumentů za {wall:.0f} s (souběh {a.concurrency})")

if __name__ == "__main__": asyncio.run(main())
