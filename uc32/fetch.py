#!/usr/bin/env python3
"""UC32 corpus: current wording of public Czech laws from e-Sbírka, one chunk per section (§).

  fetch.py --out /cold/uc32/corpus.jsonl

Stdlib only. Output rows: {"id", "law", "cite", "title", "text"}.
"""
import argparse, html, json, re, time, urllib.parse, urllib.request

API = "https://e-sbirka.gov.cz/sbr-cache/dokumenty-sbirky/"
LAWS = ["/sb/2005/127",   # electronic communications
        "/sb/1992/634",   # consumer protection
        "/sb/2019/110",   # personal data processing
        "/sb/2006/262",   # labour code
        "/sb/2004/235",   # VAT
        "/sb/2012/89"]    # civil code
MAX_CHARS = 4000


def get(path, suffix=""):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(API + urllib.parse.quote(path, safe="") + suffix, timeout=60) as r:
                return json.load(r)
        except Exception:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def fetch(path, title):
    meta = get(path)
    url, law = meta["staleUrl"], meta["zkracenaCitace"]  # staleUrl = current wording, e.g. /sb/2005/127/2026-11-01
    frags, page, pages = [], 0, 1
    while page < pages:
        d = get(url, "/fragmenty?cisloStranky=%d" % page)
        frags += d["seznam"]
        pages, page = d["pocetStranek"], page + 1
    chunks, cur = [], None
    for f in frags:
        kind, text = f["kodTypuFragmentu"], clean(f.get("xhtml") or "")
        if kind == "Paragraf":
            cur = {"cite": f["zkracenaCitace"], "head": text, "parts": []}
            chunks.append(cur)
        elif cur is not None and text and not kind.startswith(("Virtual", "Cast", "Hlava", "Dil", "Oddil", "Pododdil")):
            cur["parts"].append(text)
    out = []
    for c in chunks:
        body = "\n".join(c["parts"]).strip()
        if len(body) < 40 or "zrušen" in body[:30]:
            continue
        for i in range(0, len(body), MAX_CHARS):
            out.append({"id": "%s#%d" % (c["cite"], i // MAX_CHARS), "law": law, "cite": c["cite"],
                        "title": title, "text": "%s\n%s" % (c["head"], body[i:i + MAX_CHARS])})
    return out


def clean(x):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for p in LAWS:
        title = get(p)["nazev"]
        got = fetch(p, title)
        print(p, len(got), flush=True)
        rows += got
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("chunks", len(rows))


if __name__ == "__main__":
    main()
