#!/usr/bin/env python3
# UC9: contracts.jsonl + extracted.jsonl → SQLite (leads.sqlite) + souhrn pro report (summary.json):
# odvětví, typy, IT poptávka podle zadavatelů (leady) a největší IT dodavatelé.
#   build_db.py --inp contracts.jsonl --ext runs/full/extracted.jsonl --db leads.sqlite --out summary.json
import argparse, json, sqlite3


def value_kc(r):
    for k, f in (("bez_dph", 1.0), ("s_dph", 1 / 1.21)):
        try:
            v = float(r[k]) * f
            if 0 < v < 1e11: return round(v)
        except ValueError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="/srv/work/uc9/contracts.jsonl"); ap.add_argument("--ext", required=True)
    ap.add_argument("--db", default="/srv/work/uc9/leads.sqlite"); ap.add_argument("--out", default="/srv/work/uc9/summary.json")
    a = ap.parse_args()
    ext = {}
    for l in open(a.ext):
        j = json.loads(l)
        if j.get("x"): ext[j["id"]] = j["x"]
    db = sqlite3.connect(a.db); db.execute("DROP TABLE IF EXISTS smlouvy")
    db.execute("""CREATE TABLE smlouvy (id TEXT PRIMARY KEY, zverejneno TEXT, uzavreno TEXT, odkaz TEXT,
        zadavatel_ico TEXT, zadavatel TEXT, predmet TEXT, hodnota_kc INTEGER,
        typ TEXT, plneni TEXT, odvetvi TEXT, dodavatel_ico TEXT, dodavatel TEXT, it_relevance INTEGER, popis TEXT)""")
    batch = []
    for l in open(a.inp):
        r = json.loads(l); x = ext.get(r["id"])
        if not x: continue
        names = {s["ico"]: s["nazev"] for s in r["strany"] + [r["subjekt"]] if s["ico"]}
        batch.append((r["id"], r["zverejneno"], r["uzavreno"], r["odkaz"], r["subjekt"]["ico"], r["subjekt"]["nazev"],
                      r["predmet"], value_kc(r), x["typ"], x["plneni"], x["odvetvi"], x["dodavatel_ico"],
                      names.get(x["dodavatel_ico"], ""), x["it_relevance"], x["popis"]))
    db.executemany("INSERT OR REPLACE INTO smlouvy VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch); db.commit()
    for c in ("odvetvi", "zadavatel_ico", "dodavatel_ico", "it_relevance"):
        db.execute(f"CREATE INDEX IF NOT EXISTS i_{c} ON smlouvy({c})")
    q = lambda sql: [list(r) for r in db.execute(sql)]
    s = {
        "smluv": q("SELECT count(*), sum(hodnota_kc), count(hodnota_kc) FROM smlouvy")[0],
        "odvetvi": q("SELECT odvetvi, count(*), sum(hodnota_kc) FROM smlouvy GROUP BY 1 ORDER BY 2 DESC"),
        "typ": q("SELECT typ, count(*) FROM smlouvy GROUP BY 1 ORDER BY 2 DESC"),
        "it_relevance": q("SELECT it_relevance, count(*), sum(hodnota_kc) FROM smlouvy GROUP BY 1 ORDER BY 1"),
        # leady: zadavatelé s nejvíc smlouvami na vývoj SW / data / AI / kyberbezpečnost (it=3)
        "leady_it3": q("""SELECT zadavatel, zadavatel_ico, count(*) n, sum(hodnota_kc), count(DISTINCT dodavatel_ico)
                          FROM smlouvy WHERE it_relevance=3 GROUP BY 2 ORDER BY 3 DESC LIMIT 25"""),
        "leady_it3_hodnota": q("""SELECT zadavatel, zadavatel_ico, count(*), sum(hodnota_kc) s FROM smlouvy
                                  WHERE it_relevance=3 AND hodnota_kc IS NOT NULL GROUP BY 2 ORDER BY s DESC LIMIT 15"""),
        "dodavatele_it3": q("""SELECT dodavatel, dodavatel_ico, count(*) n, sum(hodnota_kc), count(DISTINCT zadavatel_ico)
                               FROM smlouvy WHERE it_relevance=3 AND dodavatel_ico!='' GROUP BY 2 ORDER BY 3 DESC LIMIT 20"""),
        "ukazky_it3": q("""SELECT zadavatel, dodavatel, popis, hodnota_kc, odkaz FROM smlouvy
                           WHERE it_relevance=3 AND hodnota_kc > 500000 ORDER BY random() LIMIT 12"""),
    }
    open(a.out, "w").write(json.dumps(s, ensure_ascii=False, indent=1))
    print(json.dumps({"radku": len(batch), "smluv": s["smluv"], "it": s["it_relevance"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
