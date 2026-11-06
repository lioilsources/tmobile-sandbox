#!/usr/bin/env python3
"""UC4 záznam: agent pro vykreslenou hru (lib/main_record.dart v Kiranu).

Hra je TCP klient, tento skript server. Na jedno spojení = jeden běh (kampaň od sektoru N):
  hra → {"type":"hello"|"obs"|"done"|"shop"|"over","wall":epoch_s,"f":snímek, …pozorování}
  most → {"a":[dx,dy,fire]} na každé "obs" (hra ho použije od dalšího snímku, nikdy nečeká),
         {"op":i} na "shop", {"cmd":"start"} na "hello", {"cmd":"end"} = konec běhu.

Pilot: agent.pt z ppo.py / ppo_campaign.py (OBS, OBS_V2 nebo OBS_V3 podle šířky první vrstvy).
Obchodník: shop.pt (ShopAgent), nebo "greedy" (pravidlo GREEDY z ppo_campaign), nebo "none".
--policy greedy = argmax (deterministicky), sample = vzorkování jako při tréninku.

Pozorování: agenti se učili v poli 600×300 (headless test s oknem 400×800). V jiném poli
(--obs-map train, výchozí) se y přepočítá tak, aby relativní vzdálenosti v px zůstaly
a poloha lodi zůstala poměrná; --obs-map raw pošle pole tak, jak je.

Výstup do --out: events.jsonl (zajímavé okamžiky s časem "wall"), trace.jsonl (stručný průběh),
run.json (souhrn: sektory, výbava před/po, nákupy, konec).

  agent_bridge.py --pilot runs/ppo3/agent.pt --shop runs/ppo3/shop.pt --out /cold/…/run01 --port 7400
"""
import argparse, collections, json, math, os, pathlib, socket, sys, time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import numpy as np
import torch
from kirian_env import DIRS, OBS, OBS_V2, OBS_V3, encode, kstats, shop_obs
from ppo import Agent
from ppo_campaign import GREEDY, SHOP_MAX, ShopAgent

N_SECTORS = 18
TRAIN_H = 300.0
SHOP_NAMES = {0: "odlet", 1: "upgrade přední zbraně", 2: "nová přední zbraň", 3: "levá boční zbraň",
              4: "pravá boční zbraň", 5: "upgrade generátoru", 6: "nová levá boční zbraň", 7: "nová pravá boční zbraň"}
SHOP_NAMES_EN = {0: "launch", 1: "front gun upgrade", 2: "new front weapon", 3: "left side gun",
                 4: "right side gun", 5: "generator upgrade", 6: "new left side weapon", 7: "new right side weapon"}


def load_pilot(path):
    sd = torch.load(path, map_location="cpu"); n = sd["body.0.weight"].shape[1]
    hidden = sd["body.0.weight"].shape[0]
    agent = Agent(hidden=hidden, obs=n); agent.load_state_dict(sd); agent.eval()
    return agent, n


def remap(o, mode):
    """Pole jiné výšky než 600×300 → souřadnice, na jaké je agent zvyklý (viz docstring)."""
    W, H = o["field"]
    if mode == "raw" or abs(H - TRAIN_H) < 1:
        return o
    vx, vy = o["v"][0], o["v"][1]
    vy2 = vy / H * TRAIN_H; dy = vy2 - vy
    o2 = dict(o); o2["field"] = [W, TRAIN_H]
    o2["v"] = [vx, vy2] + list(o["v"][2:])
    o2["h"] = [[x, y + dy] + r for x, y, *r in o["h"]]
    o2["p"] = [[x, y + dy] + r for x, y, *r in o["p"]]
    o2["c"] = [[x, y + dy] + r for x, y, *r in o["c"]]
    return o2


def loadout_str(o):
    d = o.get("dev") or {}
    return {k: d.get(k) for k in ("front", "left", "right", "gen")} | {"credit": o["shop"]["credit"]}


class Run:
    def __init__(self, a):
        self.a = a
        self.pilot, self.obs_n = load_pilot(a.pilot)
        self.v2, self.v3 = self.obs_n >= OBS_V2, self.obs_n == OBS_V3
        self.shop = None
        if a.shop not in ("greedy", "none"):
            self.shop = ShopAgent(); self.shop.load_state_dict(torch.load(a.shop, map_location="cpu")); self.shop.eval()
        self.rng = np.random.default_rng(a.seed)
        self.out = pathlib.Path(a.out); self.out.mkdir(parents=True, exist_ok=True)
        self.ev = open(self.out / "events.jsonl", "w"); self.tr = open(self.out / "trace.jsonl", "w")
        self.summary = {"pilot": str(a.pilot), "shop": str(a.shop), "policy": a.policy, "obs_n": self.obs_n,
                        "obs_map": a.obs_map, "sectors": [], "purchases": [], "end": None}
        self.prev = None; self.sector = None; self.visit = 0; self.visit_n = 0
        self.kill_times = collections.deque(); self.streak = None
        self.near = []  # kandidáti na těsný průlet: (wall, hp+štít, vzdálenost)
        self.last_near = -9; self.boss_seen = False; self.boss_max = 0; self.between = False

    def event(self, kind, wall, **kw):
        rec = {"type": kind, "wall": round(wall, 3), **kw}
        self.ev.write(json.dumps(rec, ensure_ascii=False) + "\n"); self.ev.flush()

    # ---- pilot ----
    def act(self, o):
        x = torch.tensor(encode(remap(o, self.a.obs_map), N_SECTORS, self.v2, self.v3)[: self.obs_n])[None]
        with torch.no_grad():
            logits = self.pilot.pi(self.pilot.body(x))[0]
        if self.a.policy == "greedy":
            a = int(torch.argmax(logits))
        else:
            p = torch.softmax(logits, -1).numpy().astype(np.float64); a = int(self.rng.choice(len(p), p=p / p.sum()))
        d = DIRS[a % 9]
        return [d[0], d[1], int(a // 9)]

    def shop_op(self, o):
        if self.visit_n >= SHOP_MAX:
            return 0
        x, mask = shop_obs(o, o["sector"], N_SECTORS)
        if self.a.shop == "none":
            return 0
        if self.a.shop == "greedy":
            return next((op for op in GREEDY if mask[op]), 0)
        with torch.no_grad():
            h = self.shop.body(torch.tensor(x)[None]); logits = self.shop.pi(h)[0].masked_fill(~torch.tensor(mask), -1e9)
        if self.a.policy == "greedy":
            return int(torch.argmax(logits))
        p = torch.softmax(logits, -1).numpy().astype(np.float64)
        return int(self.rng.choice(len(p), p=p / p.sum()))

    # ---- highlight log ----
    def track(self, o):
        w = o["wall"]; prev = self.prev
        hs = o["v"][2] + o["v"][4]
        kills = kstats(o)[0]
        self.tr.write(json.dumps([round(w, 3), o["sector"], kills, round(hs, 1), len(o["h"]), len(o["p"]), o["score"]]) + "\n")
        if prev is None or prev["sector"] != o["sector"] or prev.get("type") == "shop":
            self.prev = o; return
        dk = kills - kstats(prev)[0]
        for _ in range(max(dk, 0)):
            self.kill_times.append(w)
        while self.kill_times and w - self.kill_times[0] > 3.0:
            self.kill_times.popleft()
        if self.streak is None and len(self.kill_times) >= 5:  # 5 sestřelů za 3 s
            self.streak = {"start": self.kill_times[0], "n": len(self.kill_times), "last": w}
        elif self.streak is not None:
            if dk > 0:
                self.streak["n"] += dk; self.streak["last"] = w
            elif w - self.streak["last"] > 2.0:
                s = self.streak; self.streak = None
                self.event("kill_streak", s["start"], end=round(s["last"], 3), kills=s["n"], sector=o["sector"])
        if kstats(o)[2] > 0 and o["k"][2] > prev["k"][2]:
            self.event("fleet_cleared", w, sector=o["sector"], fleets_cleared=o["k"][2], fleets=o["k"][4])
        dmg = (prev["v"][2] + prev["v"][4]) - hs
        if dmg >= 8:
            self.event("hit", w, dmg=round(dmg, 1), hp=o["v"][2], hp_max=o["v"][3], sector=o["sector"])
        # těsný průlet: střela blíž než --near px a během 0,5 s žádné poškození
        vx, vy = o["v"][0], o["v"][1]
        dmin = min((math.hypot(p[0] - vx, p[1] - vy) for p in o["p"]), default=1e9)
        self.near = [(t0, hs0, d0) for t0, hs0, d0 in self.near if hs >= hs0 - 0.5]  # zásah ruší kandidáty
        for t0, hs0, d0 in [n for n in self.near if w - n[0] >= 0.5]:
            if t0 - self.last_near > 1.0:
                self.event("near_miss", t0, dist=round(d0, 1), sector=o["sector"]); self.last_near = t0
        self.near = [n for n in self.near if w - n[0] < 0.5]
        if dmin < self.a.near and not any(w - n[0] < 0.3 for n in self.near):
            self.near.append((w, hs, dmin))
        boss = o.get("boss")
        if boss and not self.boss_seen:
            self.boss_seen = True; self.boss_max = boss[1]; self.event("boss_appear", w, sector=o["sector"], hp=boss[0])
        if self.boss_seen and not boss:
            self.boss_seen = False
            if not o["over"] and o["v"][2] > 0:
                self.event("boss_kill", w, sector=o["sector"], boss_hp=self.boss_max)
        self.prev = o

    def start_sector(self, o):
        self.sector = {"sector": o["sector"], "wall_start": o["wall"], "loadout_before": loadout_str(o), "kills0": kstats(o)[0]}
        self.prev = None; self.kill_times.clear(); self.streak = None; self.boss_seen = False

    def end_sector(self, o, end):
        if self.sector is None:
            return
        kills, count, _, _, esc = kstats(o)
        self.sector.update(end=end, wall_end=o["wall"], t=round(o["t"], 2), kills=kills, count=count, escaped=esc,
                           score=o["score"], hp=o["v"][2], loadout_after=loadout_str(o))
        self.summary["sectors"].append(self.sector); self.sector = None

    # ---- protocol ----
    def handle(self, o):
        typ = o["type"]
        if typ == "hello":
            self.summary["start_sector"] = o["sector"]; self.summary["loadout_start"] = loadout_str(o)
            self.event("run_start", o["wall"], sector=o["sector"], loadout=loadout_str(o))
            return {"cmd": "start"}
        if typ == "obs":
            if self.between:  # snímky mezi koncem sektoru a odletem z obchodu
                return {"a": [0, 0, 0]}
            if self.sector is None:
                self.start_sector(o)
                if self.summary["sectors"]:
                    self.event("launch", o["wall"], sector=o["sector"], loadout=loadout_str(o))
            self.track(o)
            if o["t"] >= self.a.max_t:
                self.event("timeout", o["wall"], sector=o["sector"])
                self.end_sector(o, "cas"); self.summary["end"] = "cas"
                return {"cmd": "end"}
            return {"a": self.act(o)}
        if typ == "done":
            kills, count, _, _, esc = kstats(o)
            self.event("sector_clear", o["wall"], sector=o["sector"], kills=kills, count=count, t=round(o["t"], 1),
                       all_killed=kills >= count)
            self.end_sector(o, "dokonceno")
            n_done = len(self.summary["sectors"])
            if o["sector"] >= self.a.last_sector or (self.a.max_sectors and n_done >= self.a.max_sectors):
                self.summary["end"] = "posledni_sektor"; self.stop_after_shop = True
            self.visit_n = 0; self.between = True
            return None
        if typ == "shop":
            if getattr(self, "stop_after_shop", False):
                return {"cmd": "end"}
            op = self.shop_op(o)
            if self.visit_n == 0:
                self.event("shop_open", o["wall"], sector=o["sector"], loadout=loadout_str(o))
            elif self.last_shop is not None:  # výsledek předchozího nákupu
                op0, before = self.last_shop
                after = loadout_str(o)
                p = {"wall": round(o["wall"], 3), "sector": o["sector"], "op": op0, "cs": SHOP_NAMES[op0], "en": SHOP_NAMES_EN[op0],
                     "before": before, "after": after}
                self.summary["purchases"].append(p); self.event("purchase", o["wall"], **{k: v for k, v in p.items() if k != "wall"})
            self.visit_n += 1
            self.last_shop = (op, loadout_str(o)) if op else None
            if op == 0:
                self.between = False
                self.event("shop_close", o["wall"], sector=o["sector"], loadout=loadout_str(o))
            return {"op": op}
        if typ == "over":
            self.track(o)
            self.event("death", o["wall"], sector=o["sector"], score=o["score"])
            self.end_sector(o, "smrt"); self.summary["end"] = "smrt"
            return None
        return None

    def close(self):
        (self.out / "run.json").write_text(json.dumps(self.summary, ensure_ascii=False, indent=1))
        self.ev.close(); self.tr.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", required=True); ap.add_argument("--shop", default="greedy", help="shop.pt | greedy | none")
    ap.add_argument("--policy", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--port", type=int, default=7400); ap.add_argument("--out", required=True)
    ap.add_argument("--max-t", type=float, default=150.0, help="s na sektor, pak konec běhu (jako při tréninku)")
    ap.add_argument("--last-sector", type=int, default=17); ap.add_argument("--max-sectors", type=int, default=0)
    ap.add_argument("--obs-map", default="train", choices=["train", "raw"])
    ap.add_argument("--near", type=float, default=40.0, help="px: těsný průlet střely")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--accept-timeout", type=float, default=180)
    a = ap.parse_args()
    torch.set_num_threads(1)
    run = Run(a); run.last_shop = None
    srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port)); srv.listen(1); srv.settimeout(a.accept_timeout)
    print(f"BRIDGE_READY {a.port}", flush=True)
    try:
        conn, _ = srv.accept()
    except socket.timeout:
        print("hra se nepřipojila", file=sys.stderr); run.summary["end"] = "nepripojeno"; run.close(); sys.exit(2)
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = conn.makefile("rw")
    n = 0
    try:
        for line in f:
            o = json.loads(line); n += 1
            r = run.handle(o)
            if r is not None:
                f.write(json.dumps(r) + "\n"); f.flush()
    except (ConnectionError, OSError):
        pass
    run.summary["messages"] = n
    if run.summary["end"] is None: run.summary["end"] = "odpojeno"
    run.close()
    print(json.dumps({"end": run.summary["end"], "sectors": [(s["sector"], s["end"], s["kills"], s["count"]) for s in run.summary["sectors"]],
                      "purchases": len(run.summary["purchases"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
