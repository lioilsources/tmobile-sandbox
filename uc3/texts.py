#!/usr/bin/env python3
"""UC3: ke každému kvízu jedna věta příběhu + nápověda česky, jazykem Chess4Princesses
(Princezna = dáma, Koník, Motýlek, Věžka na kolečkách, Král tatínek, družičky = pěšci).
Qwen3-32B přes vLLM (profil A), výstup vynucený JSON schématem; nápověda nesmí prozradit pole
ani tah (kontrola regexem, jinak druhý pokus). Výstup: quizzes_cs.json.

  texts.py --inp quizzes.json --out quizzes_cs.json [--limit 2000]
"""
import argparse, asyncio, json, re
import chess
import httpx

FIG = {"king": "Král tatínek", "queen": "Princezna", "rook": "Věžka", "bishop": "Motýlek", "knight": "Koník", "pawn": "družička"}
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["pribeh", "napoveda"], "properties": {
    "pribeh": {"type": "string", "maxLength": 160}, "napoveda": {"type": "string", "maxLength": 120}}}
SYSTEM = """Píšeš texty do šachové hry pro pětileté děti. Figury mají jména: Princezna (dáma), Král tatínek (král),
Věžka na kolečkách (věž), Motýlek (střelec), Koník (jezdec), družičky (pěšci). Soupeř je zlý drak se svou armádou.
Ke kvízu napiš:
- pribeh: jedna krátká veselá věta, proč je úkol důležitý (max. 20 slov), bez násilí a strašení.
- napoveda: jedna věta, která navede k nápadu, ale NEPROZRADÍ tah, pole ani souřadnice (žádné „e4“, „na h7“).
Piš spisovně, jednoduše, v 2. osobě („Pomoz…“, „Podívej se…“). Odpovídej jen JSONem."""
SQUARE = re.compile(r"\b[a-h][1-8]\b", re.I)


def describe(q):
    b = chess.Board(q["fen"]); m = chess.Move.from_uci(q["solution"][0])
    who = FIG[q["piece"]]
    victim = b.piece_at(m.to_square)
    b.push(m)
    what = {"m08": "dát drakovu králi mat jedním tahem", "m07": "sníst nechráněnou figuru draka",
            "m11_fork": "jedním tahem zaútočit na dvě dračí figury naráz (vidlička)",
            "m11_pin": "přišpendlit dračí figuru, která nesmí uhnout (vazba)"}
    key = "m08" if q["lesson"].startswith("m08") else q["lesson"] if q["lesson"].startswith("m11") else "m07"
    extra = f", bere {chess.piece_name(victim.piece_type)}" if victim else ""
    return f"Lekce: {q['lesson_name']}. Úkol: {what[key]}. Táhne {who}{extra}. Obtížnost {q['difficulty']}/5."


async def one(cl, sem, a, q):
    msg = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": describe(q)}]
    body = {"model": a.model, "temperature": 0.7, "max_tokens": 200, "messages": msg,
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"name": "kviz", "schema": SCHEMA}}}
    async with sem:
        for attempt in range(3):
            try:
                r = await cl.post(a.url, json=body); r.raise_for_status()
                x = json.loads(r.json()["choices"][0]["message"]["content"])
                if not SQUARE.search(x["napoveda"]):
                    return {**q, **x, "pokusu": attempt + 1}
            except Exception:
                await asyncio.sleep(1)
    return {**q, "pribeh": None, "napoveda": None, "pokusu": 3}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions"); ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--concurrency", type=int, default=64)
    a = ap.parse_args()
    qs = json.load(open(a.inp))
    # rovnoměrně přes lekce, aby limit nevzal jen první lekci
    by = {}
    for q in qs: by.setdefault(q["lesson"], []).append(q)
    pick, i = [], 0
    while len(pick) < min(a.limit, len(qs)):
        for l in by.values():
            if i < len(l) and len(pick) < a.limit: pick.append(l[i])
        i += 1
    sem = asyncio.Semaphore(a.concurrency)
    async with httpx.AsyncClient(timeout=300) as cl:
        res = await asyncio.gather(*(one(cl, sem, a, q) for q in pick))
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    ok = sum(r["pribeh"] is not None for r in res)
    print(json.dumps({"kvizu": len(res), "s_textem": ok, "opakovano": sum(r["pokusu"] > 1 for r in res)}))


if __name__ == "__main__":
    asyncio.run(main())
