#!/usr/bin/env python3
"""LLM (profil B) napíše syntetické hovory se zadanými vlastnostmi → ground truth = zadání.

  scenarios.py --n 40 --out /srv/work/demo1/calls
"""
import argparse, asyncio, json, pathlib, random
import httpx
from common import LLM, LABELS, BOOLS, OPERATOR, DEF_SENTIMENT, DEF_KROK

DESC = {
 "fakturace": "nesrovnalost na vyúčtování (nečekaná položka, vyšší částka, prémiové SMS, dvojí platba)",
 "zmena_tarifu": "zákazník chce jiný tarif (více dat, levnější, rodinný, firemní)",
 "roaming": "roaming v zahraničí (vysoký účet, nefunguje data, nastavení před cestou)",
 "technicka_zavada": "výpadek nebo špatný signál, pomalý internet, nefungují hovory",
 "vypoved_smlouvy": "zákazník chce ukončit smlouvu nebo přejít ke konkurenci",
 "prenos_cisla": "přenos čísla od jiného operátora nebo k jinému, OKU kód",
 "zarizeni_a_pojisteni": "rozbitý nebo ukradený telefon, reklamace, pojištění zařízení, splátky",
 "sim_a_esim": "ztráta SIM, aktivace eSIM, výměna SIM, PUK",
}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["repliky"], "properties": {"repliky": {"type": "array", "minItems": 8, "maxItems": 22,
          "items": {"type": "object", "additionalProperties": False, "required": ["mluvci", "text"],
                    "properties": {"mluvci": {"enum": ["operator", "zakaznik"]}, "text": {"type": "string", "minLength": 2}}}}}}

def labels(rng):
    l = {k: rng.choice(v) for k, v in LABELS.items()}
    l.update({k: rng.random() < .5 for k in BOOLS})
    if l["kategorie"] == "vypoved_smlouvy": l["riziko_odchodu"] = True
    if l["sentiment_zakaznika"] == "pozitivni": l["riziko_odchodu"] = False
    if l["vyreseno"] and l["dalsi_krok"] == "eskalace_na_specialistu": l["dalsi_krok"] = "zadny"
    return l

def prompt(l, name):
    return f"""Napiš realistický přepis telefonního hovoru na zákaznickou linku mobilního operátora {OPERATOR} v češtině.
Operátorka se jmenuje {name}. Volá muž (zákazník), mluví v mužském rodě. Hovor má 10–18 replik, repliky jsou přirozené,
mluvené (žádné odrážky, emotikony ani scénické poznámky), čísla a částky piš slovy nebo běžně (např. „čtyři sta devadesát korun“).
Vlastnosti hovoru, které MUSÍ z textu jednoznačně vyplynout:
- téma: {DESC[l['kategorie']]}
- nálada zákazníka na konci hovoru: {l['sentiment_zakaznika']} – {DEF_SENTIMENT[l['sentiment_zakaznika']]}; poslední replika zákazníka to musí jasně ukázat
- problém {'je během hovoru vyřešen' if l['vyreseno'] else 'během hovoru NENÍ vyřešen'}
- zákazník {'výslovně zmíní, že zvažuje odchod ke konkurenci nebo zrušení smlouvy' if l['riziko_odchodu'] else 'nezmiňuje odchod ani zrušení smlouvy'}
- operátorka {'na začátku ověří identitu zákazníka (datum narození nebo heslo / poslední čtyřčíslí)' if l['overeni_identity'] else 'identitu zákazníka NEOVĚŘUJE'}
- další krok: {DEF_KROK[l['dalsi_krok']]}. Operátorka v předposlední replice jednou větou výslovně shrne tento jediný dohodnutý krok;
  žádný jiný krok (e-mail, SMS, zpětné volání, předání, technik) v hovoru nedomlouvej ani nenabízej."""

async def one(cl, sem, i, rng, out):
    l = labels(rng); name = rng.choice(["Petra", "Jana", "Lucie", "Kateřina", "Tereza", "Martina"])
    f = out / f"call_{i:03d}.json"
    if f.exists(): return                     # navázat na předchozí běh
    for attempt in range(4):                  # json_schema občas uvízne ve smyčce bílých znaků → zkusit znovu
        async with sem:
            r = await cl.post(f"{LLM}/chat/completions", json={"model": MODEL, "temperature": 0.8, "max_tokens": 2500,
                "frequency_penalty": 0.3, "seed": i * 10 + attempt, "chat_template_kwargs": {"enable_thinking": False},
                "response_format": {"type": "json_schema", "json_schema": {"name": "hovor", "schema": SCHEMA}},
                "messages": [{"role": "user", "content": prompt(l, name)}]})
            r.raise_for_status()
        ch = r.json()["choices"][0]
        try:
            turns = json.loads(ch["message"]["content"])["repliky"]
            if ch["finish_reason"] == "stop": break
        except (json.JSONDecodeError, KeyError): pass
        print(f"call_{i:03d} pokus {attempt + 1} selhal ({ch['finish_reason']})", flush=True)
    else:
        print(f"call_{i:03d} VYNECHÁN", flush=True); return
    f.write_text(json.dumps({"id": f"call_{i:03d}", "labels": l, "operatorka": name, "repliky": turns}, ensure_ascii=False, indent=1))
    print(f"call_{i:03d} {l['kategorie']:22s} {len(turns)} replik", flush=True)

async def main():
    global MODEL
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--out", default="/srv/work/demo1/calls")
    ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); rng = random.Random(a.seed)
    async with httpx.AsyncClient(timeout=600) as cl:
        MODEL = (await cl.get(f"{LLM}/models")).json()["data"][0]["id"]
        sem = asyncio.Semaphore(16)
        await asyncio.gather(*(one(cl, sem, i, rng, out) for i in range(a.n)))

if __name__ == "__main__": asyncio.run(main())
