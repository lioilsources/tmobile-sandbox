#!/usr/bin/env python3
"""UC4: vyhodnocení agenta po sektorech → křivka obtížnosti, heatmapy smrtí a zásahů, replay pro GIF.

Pro každý sektor --episodes epizod s novou lodí: natrénovaný agent (vzorkování z policy)
a pro srovnání náhodný hráč. Výstup: eval.json (po sektorech), deaths.json, replays/*.json
(trajektorie lodi, nepřátel a střel po krocích – podklad pro GIF).

  evaluate.py --repo /srv/work/uc4/kiran --agent runs/ppo1/agent.pt --out runs/ppo1/eval --episodes 30
"""
import argparse, json, pathlib
import numpy as np
import torch
from kirian_env import DIRS, OBS, OBS_V2, KirianProc, encode, kstats
from ppo import Agent


def run(envs, sectors, policy, max_t, replay_sector=None, v2=False):
    """Každé prostředí hraje svoje epizody; vrací seznam výsledků epizod."""
    queue = list(sectors); res = []; live = {}
    rep = None
    def start(i):
        if not queue: return
        s = queue.pop(0); o = envs[i].req({"cmd": "reset", "sector": s})
        live[i] = {"s": s, "o": o, "hits": [], "hp0": o["v"][2], "frames": [] if (s == replay_sector and rep is None) else None}
    for i in range(len(envs)): start(i)
    while live:
        ids = list(live)
        obs = np.stack([encode(live[i]["o"], 18, v2) for i in ids])
        acts = policy(obs)
        for i, a in zip(ids, acts):
            d = DIRS[a % 9]; envs[i].send({"cmd": "step", "a": [d[0], d[1], int(a // 9)], "k": 4})
        for i in ids:
            e = live[i]; prev = e["o"]; o = envs[i].recv(); e["o"] = o
            W, H = o["field"]
            if (o["v"][2] + o["v"][4]) < (prev["v"][2] + prev["v"][4]): e["hits"].append((o["v"][0] / W, o["v"][1] / H))
            if e["frames"] is not None:
                e["frames"].append({"v": o["v"][:2], "hp": o["v"][2] / max(o["v"][3], 1), "h": [x[:2] + [x[3]] for x in o["h"]], "p": [x[:2] for x in o["p"]], "score": o["score"]})
            end = "smrt" if o["over"] else "dokonceno" if o["done"] else "cas" if o["t"] >= max_t else None
            if end:
                kills, count, _, _, esc = kstats(o)
                r = {"sector": e["s"], "end": end, "hp": o["v"][2] / max(o["v"][3], 1), "t": o["t"], "score": o["score"], "hits": e["hits"],
                     "kills": kills, "count": count, "esc": esc,
                     "death": (o["v"][0] / W, o["v"][1] / H) if end == "smrt" else None}
                res.append(r)
                if e["frames"] is not None: rep = {"sector": e["s"], "end": end, "field": o["field"], "frames": e["frames"]}
                del live[i]; start(i)
    return res, rep


def summarize(res):
    by = {}
    for r in res: by.setdefault(r["sector"], []).append(r)
    out = {}
    for s, rs in sorted(by.items()):
        ok = [r for r in rs if r["end"] == "dokonceno"]
        out[s] = {"epizod": len(rs), "uspesnost_%": round(100 * len(ok) / len(rs), 1),
                  "hp_pri_dokonceni": round(float(np.mean([r["hp"] for r in ok])), 2) if ok else None,
                  "cas_s": round(float(np.mean([r["t"] for r in rs])), 1), "skore": round(float(np.mean([r["score"] for r in rs]))),
                  "sestreleno_%": round(100 * float(np.mean([min(r["kills"] / max(r["count"], 1), 1) for r in rs])), 1),
                  "vycisteno_%": round(100 * sum(r["end"] == "dokonceno" and r["kills"] >= r["count"] for r in rs) / len(rs), 1),
                  "uletelo": round(float(np.mean([r["esc"] for r in rs])), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--agent", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=30); ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--sectors", default="0-17"); ap.add_argument("--flutter", default="flutter"); ap.add_argument("--base-port", type=int, default=7100)
    ap.add_argument("--max-t", type=float, default=150.0)
    ap.add_argument("--mode", default="survive", choices=["survive", "clear"], help="clear = agent kola 2 (rozšířené pozorování)")
    a = ap.parse_args()
    lo, hi = map(int, a.sectors.split("-")); secs = list(range(lo, hi + 1))
    out = pathlib.Path(a.out); (out / "replays").mkdir(parents=True, exist_ok=True)
    envs = [KirianProc(a.repo, a.base_port + i, flutter=a.flutter, log_dir=str(out)) for i in range(a.envs)]
    v2 = a.mode == "clear"
    agent = Agent(obs=OBS_V2 if v2 else OBS); agent.load_state_dict(torch.load(a.agent, map_location="cpu")); agent.eval()

    def pol(obs):
        with torch.no_grad():
            return agent(torch.tensor(obs))[0].numpy()
    rng = np.random.default_rng(0)
    rnd = lambda obs: rng.integers(0, 18, len(obs))
    result = {}
    for name, p in (("agent", pol), ("nahodny", rnd)):
        res, _ = run(envs, [s for s in secs for _ in range(a.episodes)], p, a.max_t, v2=v2)
        result[name] = summarize(res)
        (out / f"episodes_{name}.json").write_text(json.dumps(res))
    # replay: jedna epizoda agenta pro každý sektor
    for s in secs:
        _, rep = run(envs, [s], pol, a.max_t, replay_sector=s, v2=v2)
        if rep: (out / "replays" / f"sector_{s:02d}.json").write_text(json.dumps(rep))
    (out / "eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
    for e in envs: e.close()
    print(json.dumps({s: (result["agent"][s]["uspesnost_%"], result["agent"][s]["vycisteno_%"], result["nahodny"][s]["uspesnost_%"]) for s in result["agent"]}))


if __name__ == "__main__":
    main()
