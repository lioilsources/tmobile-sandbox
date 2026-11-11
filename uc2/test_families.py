#!/usr/bin/env python3
"""Každý seed ze SEED_MAP (kromě neonu: polární warp) musí projít kartou své rodiny.
  python3 -I test_families.py <adresář-se-seedy> ...   (původní filter_*.frag z MirrorBoothu a ručně psané lab*.frag)"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from families import SEED_MAP, check_card

EXCLUDED = {"neon"}
files = {p.stem.replace("filter_", ""): p for d in sys.argv[1:] for p in pathlib.Path(d).glob("*.frag")}
bad, missing = [], []
for fam, names in SEED_MAP.items():
    for n in names.split():
        if n in EXCLUDED: continue
        if n not in files: missing.append((fam, n)); continue
        e = check_card(files[n].read_text(), fam)
        if e: bad.append((fam, n, e))
for b in bad: print("MIMO KARTU", *b)
for m in missing: print("CHYBÍ SOUBOR", *m)
print("OK" if not bad and not missing else "SELHÁNÍ", f"({sum(len(v.split()) for v in SEED_MAP.values()) - len(EXCLUDED)} seedů)")
sys.exit(1 if bad or missing else 0)
