"""UC24 krok 2 — mapování prompt → AudioSet třídy (pro PANNs class check) přes lokální LLM.

  python prompt_classes.py --manifest /cold/uc24/synth/manifest.jsonl --out /cold/uc24/prompt_classes.json
Model dostane celý seznam 527 tříd a vrátí 1–5 přesných názvů; neznámé názvy se zahodí.
"""
from __future__ import annotations

import argparse
import csv
import json
import os

from llm import chat_json

LABELS = os.environ.get("AUDIOSET_LABELS", "/cold/uc24/ext/panns/class_labels_indices.csv")
SYSTEM = ("You map a sound description to AudioSet ontology classes. Answer ONLY JSON "
          "{\"classes\": [1-5 class names copied EXACTLY from the list, most specific first]}.\nClasses: ")


def labels() -> list[str]:
    return [r["display_name"] for r in csv.DictReader(open(LABELS))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/cold/uc24/synth/manifest.jsonl")
    ap.add_argument("--out", default="/cold/uc24/prompt_classes.json")
    a = ap.parse_args()
    L = labels()
    known = set(L)
    cache = json.load(open(a.out)) if os.path.exists(a.out) else {}
    prompts = sorted({json.loads(line)["prompt"] for line in open(a.manifest)})
    sys_p = SYSTEM + "; ".join(L)
    for p in prompts:
        if p in cache:
            continue
        d, meta = chat_json(sys_p, f"Sound description: {p}", temperature=0.0, max_tokens=120)
        got = [c for c in d.get("classes", []) if c in known]
        cache[p] = {"classes": got, "dropped": [c for c in d.get("classes", []) if c not in known], "model": meta["model"]}
        print(p, "→", got)
    json.dump(cache, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
