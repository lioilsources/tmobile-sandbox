#!/usr/bin/env python3
# UC9: měsíční dumpy Registru smluv (data.smlouvy.gov.cz, XML ISRS 1.2) → contracts.jsonl.
# Bere jen platné záznamy, u každé smlouvy poslední verzi.
#   parse.py /srv/work/uc9/raw/dump_2026_0*.xml --out /srv/work/uc9/contracts.jsonl
import argparse
import json
import xml.etree.ElementTree as ET

NS = "{http://portal.gov.cz/rejstriky/ISRS/1.2/}"


def txt(el, path):
    x = el.find("/".join(NS + p for p in path.split("/")))
    return x.text.strip() if x is not None and x.text else ""


def strana(el):
    return {k: txt(el, k) for k in ("nazev", "ico", "adresa", "prijemce")}


def record(z):
    s = z.find(NS + "smlouva")
    r = {
        "id": txt(z, "identifikator/idSmlouvy"), "verze": int(txt(z, "identifikator/idVerze") or 0),
        "odkaz": txt(z, "odkaz"), "zverejneno": txt(z, "casZverejneni")[:10],
        "subjekt": strana(s.find(NS + "subjekt")),
        "strany": [strana(x) for x in s.findall(NS + "smluvniStrana")],
        "predmet": txt(s, "predmet"), "uzavreno": txt(s, "datumUzavreni"),
        "bez_dph": txt(s, "hodnotaBezDph"), "s_dph": txt(s, "hodnotaVcetneDph"),
        "cizi_mena": " ".join(filter(None, (txt(s, "ciziMena/hodnota"), txt(s, "ciziMena/mena")))),
        "cena_neuvedena": txt(s, "cenaNeuvedena"),
        "navazany": txt(s, "navazanyZaznam"),
        "prilohy": [txt(p, "nazevSouboru") for p in z.findall(f"{NS}prilohy/{NS}priloha")][:5],
    }
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    best, n_all, n_invalid = {}, 0, 0
    for fn in a.dumps:
        for _, el in ET.iterparse(fn, events=("end",)):
            if el.tag != NS + "zaznam":
                continue
            n_all += 1
            if txt(el, "platnyZaznam") != "1":
                n_invalid += 1
            else:
                r = record(el)
                if r["id"] not in best or best[r["id"]]["verze"] < r["verze"]:
                    best[r["id"]] = r
            el.clear()
    with open(a.out, "w", encoding="utf-8") as f:
        for r in best.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"zaznamu": n_all, "neplatnych": n_invalid, "smluv": len(best)}))


if __name__ == "__main__":
    main()
