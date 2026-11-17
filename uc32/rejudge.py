#!/usr/bin/env python3
"""UC32: re-score results.jsonl with an independent judge (profile A, Qwen3-32B) to remove self-judging bias.

  rejudge.py --dir /cold/uc32
"""
import argparse, json, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import run

run.LLM_MODEL = "qwen3-32b"

def judge(q, ref, ans):
    r = run.post(run.LLM + "/chat/completions", {
        "model": "qwen3-32b", "temperature": 0, "max_tokens": 50,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": run.JUDGE.format(q=q, ref=ref, ans=ans)}]})
    try:
        return int(run.parse_json(r["choices"][0]["message"]["content"])["score"])
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    d = Path(ap.parse_args().dir)
    qa = {q["id"]: q for q in map(json.loads, open(d / "qa.jsonl"))}
    rows = [json.loads(l) for l in open(d / "results.jsonl")]

    def one(r):
        q = qa[r["id"]]
        r["score32"] = {k: judge(q["question"], q["answer"], a) for k, a in r["answers"].items()}
        return r

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, rows))
    (d / "results_32b.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    lines = ["", "## Nezávislý soudce Qwen3-32B (0–2 body)", "", "| | průměr | správně (2) | špatně (0) | shoda s 30B |", "|---|---|---|---|---|"]
    for name, k in (("bez kontextu", "closed"), ("embedding top 5", "emb"), ("rerank top 5", "rerank")):
        s = [r["score32"][k] for r in rows if r["score32"][k] is not None]
        agree = sum(1 for r in rows if r["score32"][k] is not None and r["score32"][k] == r["score"][k]) / len(s)
        lines.append("| %s | %.2f | %.0f %% | %.0f %% | %.0f %% |" % (name, statistics.mean(s), 100 * s.count(2) / len(s), 100 * s.count(0) / len(s), 100 * agree))
    with open(d / "report.md", "a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
