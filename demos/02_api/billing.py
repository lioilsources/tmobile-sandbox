#!/usr/bin/env python3
"""Vyúčtování z knihy gateway (usage.jsonl) za časové okno → tabulka po tenantech a modelech (JSON + Markdown).

  billing.py --ledger /srv/work/demo2/usage.jsonl [--since <unix ts>] [--until <unix ts>] --out <dir>
"""
import argparse, collections, datetime as dt, json, pathlib

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ledger", default="/srv/work/demo2/usage.jsonl")
    ap.add_argument("--since", type=float, default=0); ap.add_argument("--until", type=float, default=1e12); ap.add_argument("--out", required=True)
    a = ap.parse_args(); agg = collections.defaultdict(lambda: collections.Counter()); bad = 0
    for line in open(a.ledger, errors="replace"):
        try: u = json.loads(line)
        except json.JSONDecodeError: bad += 1; continue   # např. nuly po tvrdém pádu VM
        ts = dt.datetime.fromisoformat(u["ts"].replace("Z", "+00:00")).timestamp()
        if not a.since <= ts <= a.until: continue
        c = agg[(u["tenant"], u["model"])]
        c["pozadavku"] += 1; c["chyb"] += u["status"] >= 400; c["vstup_tok"] += u["prompt_tokens"]; c["vystup_tok"] += u["completion_tokens"]
        c["audio_min"] += (u.get("audio_s") or 0) / 60; c["kc"] += u["cost_czk"]
    rows = [{"tenant": t, "model": m, **{k: round(v, 2) for k, v in c.items()}} for (t, m), c in sorted(agg.items())]
    tot = collections.defaultdict(float)
    for r in rows: tot[r["tenant"]] += r["kc"]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "billing.json").write_text(json.dumps({"radky": rows, "celkem_kc": {k: round(v, 2) for k, v in tot.items()}}, ensure_ascii=False, indent=1))
    md = ["| Tenant | Model | Požadavků | Vstup tok. | Výstup tok. | Audio min | Kč |", "|---|---|---|---|---|---|---|"]
    md += [f"| {r['tenant']} | {r['model']} | {r.get('pozadavku', 0):.0f} | {r.get('vstup_tok', 0):,.0f} | {r.get('vystup_tok', 0):,.0f} | {r.get('audio_min', 0):.1f} | {r.get('kc', 0):.2f} |".replace(",", " ") for r in rows]
    md += [f"| **{t}** | | | | | | **{v:.2f}** |" for t, v in sorted(tot.items())]
    if bad: md.append(f"\n*Přeskočeno {bad} poškozených řádků knihy.*")
    (out / "billing.md").write_text("\n".join(md)); print("\n".join(md))

if __name__ == "__main__": main()
