#!/usr/bin/env python3
"""UC4 kolo 3: kampaň s obchodem. Pilot letí sektory od 0 za sebou se stejnou lodí; po dokončeném sektoru
obchodník (druhá malá síť) utrácí vystřílený kredit za zbraně v ComCenteru, nevyužitý kredit zůstává
na další kolo. Běh kampaně končí smrtí, vypršeným časem v sektoru nebo posledním sektorem.

Pilot: PPO jako v kole 2 (odměna za sestřely, epizoda = jeden sektor), start z agenta kola 2,
pozorování navíc výbava a kredit (OBS_V3).
Obchodník: PPO nad nákupními rozhodnutími. Každé rozhodnutí návštěvy obchodu před sektorem k+1 dostane
výnos G = Σ 0,8^m · R(k+1+m), kde R je odměna pilota v sektoru, takže se počítá i to, co přinese ušetřený
kredit v dalších sektorech. Neplatné nákupy (málo kreditu, max. úroveň, neodemčeno) jsou maskované.

  ppo_campaign.py --repo /srv/work/uc4/kiran --init runs/ppo2/agent.pt --out runs/ppo3 --steps 16000000
  ppo_campaign.py --repo … --eval runs/ppo3 --shop learn|greedy|none --runs 30   (vyhodnocení)
"""
import argparse, json, pathlib, time
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Categorical
from kirian_env import (DIRS, N_ACT, OBS_V2, OBS_V3, SHOP_OBS, SHOP_OPS, KirianProc, encode, kstats, reward,
                        shop_obs)
from ppo import Agent, layer

SHOP_MAX = 12      # nákupů na jednu návštěvu obchodu, pak se odlétá
SECTOR_GAMMA = 0.8
GREEDY = [2, 1, 6, 7, 3, 4, 5]  # srovnávací pravidlo: nejdřív lepší přední zbraň, pak upgrady


class ShopAgent(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.body = nn.Sequential(layer(SHOP_OBS, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.pi = layer(hidden, SHOP_OPS, 0.01); self.v = layer(hidden, 1, 1.0)

    def forward(self, x, mask, a=None):
        h = self.body(x); logits = self.pi(h).masked_fill(~mask, -1e9); d = Categorical(logits=logits)
        if a is None: a = d.sample()
        return a, d.log_prob(a), d.entropy(), self.v(h).squeeze(-1)


class Campaign:
    """N her; každá hraje kampaň od sektoru `start`. step() vrací přechody pilota (epizoda = sektor);
    nákupy mezi sektory dělá `shop_policy` uvnitř step() a hotové návštěvy se sbírají v self.shop_done."""

    def __init__(self, repo, n, shop_policy, start=0, last=17, base_port=7000, flutter="flutter", log_dir="/tmp",
                 frame_skip=4, max_t=150.0):
        self.envs = [KirianProc(repo, base_port + i, flutter=flutter, log_dir=log_dir) for i in range(n)]
        self.shop_policy, self.start, self.last, self.k, self.max_t = shop_policy, start, last, frame_skip, max_t
        self.n_sectors = last + 1
        self.o = [None] * n; self.ep = [None] * n; self.run = [None] * n
        self.shop_done = []   # [(x, mask, a, logp, v, G)]
        self.runs_done = []   # souhrny dokončených kampaní

    def _new_run(self, i):
        o = self.envs[i].req({"cmd": "reset", "sector": self.start})
        self.run[i] = {"sectors": [], "visits": [], "R": [], "spent": 0, "bought": []}
        return self._new_sector(i, o)

    def _new_sector(self, i, o):
        self.o[i] = o; self.ep[i] = {"sector": o["sector"], "ret": 0.0, "steps": 0, "credit0": o["shop"]["credit"]}
        return encode(o, self.n_sectors, True, True)

    def reset(self):
        return np.stack([self._new_run(i) for i in range(len(self.envs))])

    def _shop(self, i):
        """Návštěva obchodu po dokončeném sektoru; vrací pozorování prvního kroku dalšího sektoru."""
        e = self.envs[i]; o = e.req({"cmd": "next"}); visit = []
        for _ in range(SHOP_MAX):
            x, mask = shop_obs(o, o["sector"], self.n_sectors)
            a, logp, v = self.shop_policy(x, mask)
            visit.append((x, mask, a, logp, v))
            if a == 0: break
            c0 = o["shop"]["credit"]; o = e.req({"cmd": "shop", "op": int(a)})
            self.run[i]["spent"] += c0 - o["shop"]["credit"]; self.run[i]["bought"].append((o["sector"], int(a)))
        self.run[i]["visits"].append(visit)
        return e.req({"cmd": "launch"})

    def _finish_run(self, i, last_end):
        r = self.run[i]; R = r["R"]
        # návštěva k proběhla před sektorem k+1 → G_k = Σ_m γ^m R[k+1+m]
        for k, visit in enumerate(r["visits"]):
            G = sum(SECTOR_GAMMA ** m * R[k + 1 + m] for m in range(len(R) - k - 1))
            self.shop_done += [(*d, G) for d in visit]
        s = self.o[i]["shop"]
        self.runs_done.append({"sektoru": len(r["sectors"]), "dokonceno": sum(x["end"] == "dokonceno" for x in r["sectors"]),
                               "konec": last_end, "utraceno": r["spent"], "kredit_na_konci": s["credit"],
                               "vybava": {k: s[k] for k in ("front", "left", "right", "gen")}, "nakupy": r["bought"],
                               "sektory": [{k: x[k] for k in ("sector", "end", "kills", "count", "esc", "credit0", "t")}
                                           for x in r["sectors"]]})

    def step(self, actions):
        for i, a in enumerate(actions):
            d = DIRS[a % 9]; self.envs[i].send({"cmd": "step", "a": [d[0], d[1], int(a // 9)], "k": self.k})
        obs, rews, dones, infos = [], [], [], []
        for i in range(len(self.envs)):
            prev, o = self.o[i], self.envs[i].recv(); self.o[i] = o
            r, end, _ = reward(prev, o, "campaign", self.max_t)
            ep = self.ep[i]; ep["ret"] += r; ep["steps"] += 1
            if end:
                kills, count, _, _, esc = kstats(o)
                ep.update(end=end, kills=kills, count=count, esc=esc, t=o["t"], hp=o["v"][2] / max(o["v"][3], 1))
                infos.append(ep); self.run[i]["sectors"].append(ep); self.run[i]["R"].append(ep["ret"])
                if end == "dokonceno" and o["sector"] < self.last:
                    obs.append(self._new_sector(i, self._shop(i)))
                else:
                    self._finish_run(i, end); obs.append(self._new_run(i))
                dones.append(1.0)
            else:
                obs.append(encode(o, self.n_sectors, True, True)); dones.append(0.0)
            rews.append(r)
        return np.stack(obs), np.asarray(rews, np.float32), np.asarray(dones, np.float32), infos

    def close(self):
        for e in self.envs: e.close()


def make_shop_policy(kind, shop, sample=True):
    def learned(x, mask):
        with torch.no_grad():
            a, logp, _, v = shop(torch.tensor(x)[None], torch.tensor(mask)[None])
        return int(a), float(logp), float(v)
    def greedy(x, mask):
        return next((op for op in GREEDY if mask[op]), 0), 0.0, 0.0
    def none(x, mask):
        return 0, 0.0, 0.0
    return {"learn": learned, "greedy": greedy, "none": none}[kind]


def update_shop(shop, opt, data, epochs=4, mb=128):
    x = torch.tensor(np.stack([d[0] for d in data])); m = torch.tensor(np.stack([d[1] for d in data]))
    a = torch.tensor([d[2] for d in data]); lp = torch.tensor([d[3] for d in data]); v = torch.tensor([d[4] for d in data])
    G = torch.tensor([d[5] for d in data], dtype=torch.float32); adv = G - v
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)  # přes celou dávku: minidávka s 1 prvkem dávala std NaN
    idx = np.arange(len(data))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), mb):
            j = idx[s:s + mb]
            if len(j) < 8: continue
            _, nl, ent, nv = shop(x[j], m[j], a[j]); ratio = (nl - lp[j]).exp()
            ad = adv[j]
            loss = (torch.max(-ad * ratio, -ad * ratio.clamp(0.8, 1.2)).mean() + 0.5 * nn.functional.smooth_l1_loss(nv, G[j])
                    - 0.01 * ent.mean())
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(shop.parameters(), 0.5); opt.step()
    return loss.item(), ent.mean().item()


def load_pilot(path):
    agent = Agent(obs=OBS_V3); sd = torch.load(path, map_location="cpu"); w = sd["body.0.weight"]
    if w.shape[1] < OBS_V3:  # agent kola 1/2: výbava a kredit začnou s nulovými váhami
        sd["body.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], OBS_V3 - w.shape[1])], 1)
    agent.load_state_dict(sd); return agent


def evaluate(a):
    out = pathlib.Path(a.eval); pilot = load_pilot(out / "agent.pt"); pilot.eval()
    shop = ShopAgent()
    if (out / "shop.pt").exists(): shop.load_state_dict(torch.load(out / "shop.pt", map_location="cpu"))
    env = Campaign(a.repo, a.envs, make_shop_policy(a.shop, shop), base_port=a.base_port, flutter=a.flutter, log_dir=str(out))
    obs = env.reset()
    while len(env.runs_done) < a.runs:
        with torch.no_grad():
            act = pilot(torch.tensor(obs))[0].numpy()
        obs, *_ = env.step(act)
    runs = env.runs_done[: a.runs]; env.close()
    (out / f"eval_campaign_{a.shop}.json").write_text(json.dumps(runs, ensure_ascii=False))
    secs = [r["dokonceno"] for r in runs]
    print(json.dumps({"obchod": a.shop, "behu": len(runs), "sektoru_dokonceno_prumer": round(float(np.mean(secs)), 2),
                      "nejdal": max(secs), "utraceno_prumer": round(float(np.mean([r["utraceno"] for r in runs]))),
                      "kredit_na_konci_prumer": round(float(np.mean([r["kredit_na_konci"] for r in runs])))}, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--out"); ap.add_argument("--init")
    ap.add_argument("--envs", type=int, default=8); ap.add_argument("--steps", type=int, default=16_000_000)
    ap.add_argument("--rollout", type=int, default=256); ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--shop-lr", type=float, default=3e-4); ap.add_argument("--shop-batch", type=int, default=512)
    ap.add_argument("--flutter", default="flutter"); ap.add_argument("--base-port", type=int, default=7000)
    ap.add_argument("--eval", help="adresář s agent.pt a shop.pt → jen vyhodnocení kampaní")
    ap.add_argument("--shop", default="learn", choices=["learn", "greedy", "none"]); ap.add_argument("--runs", type=int, default=30)
    a = ap.parse_args()
    if a.eval: return evaluate(a)
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(2)
    pilot = load_pilot(a.init); shop = ShopAgent()
    opt = torch.optim.Adam(pilot.parameters(), lr=a.lr, eps=1e-5); sopt = torch.optim.Adam(shop.parameters(), lr=a.shop_lr, eps=1e-5)
    env = Campaign(a.repo, a.envs, make_shop_policy("learn", shop), base_port=a.base_port, flutter=a.flutter, log_dir=str(out))
    T, N = a.rollout, a.envs
    obs_b = torch.zeros((T, N, OBS_V3)); act_b = torch.zeros((T, N), dtype=torch.long)
    logp_b, rew_b, done_b, val_b = (torch.zeros((T, N)) for _ in range(4))
    obs = torch.tensor(env.reset()); done = torch.zeros(N)
    epf = open(out / "episodes.jsonl", "a"); runf = open(out / "runs.jsonl", "a"); logf = open(out / "train.jsonl", "a")
    gstep, t0, it, shop_log = 0, time.time(), 0, {}
    while gstep < a.steps:
        it += 1; opt.param_groups[0]["lr"] = (1 - gstep / a.steps) * a.lr
        for t in range(T):
            obs_b[t] = obs; done_b[t] = done
            with torch.no_grad():
                act, logp, _, v = pilot(obs)
            act_b[t], logp_b[t], val_b[t] = act, logp, v
            o, r, d, infos = env.step(act.numpy())
            rew_b[t] = torch.tensor(r); obs = torch.tensor(o); done = torch.tensor(d)
            for inf in infos:
                inf["gstep"] = gstep + t * N; epf.write(json.dumps(inf) + "\n")
        for run in env.runs_done: runf.write(json.dumps(run, ensure_ascii=False) + "\n")
        env.runs_done.clear()
        gstep += T * N
        with torch.no_grad():  # GAE pilota (stejné jako ppo.py)
            nv = pilot(obs)[3]; adv = torch.zeros_like(rew_b); last = 0
            for t in reversed(range(T)):
                nnt = 1.0 - (done if t == T - 1 else done_b[t + 1]); nxt = nv if t == T - 1 else val_b[t + 1]
                delta = rew_b[t] + 0.99 * nxt * nnt - val_b[t]
                last = delta + 0.99 * 0.95 * nnt * last; adv[t] = last
            ret = adv + val_b
        bo, ba, bl, badv, bret = obs_b.reshape(-1, OBS_V3), act_b.reshape(-1), logp_b.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        idx = np.arange(T * N); mb = T * N // 4
        for _ in range(4):
            np.random.shuffle(idx)
            for s in range(0, T * N, mb):
                j = idx[s:s + mb]
                _, nl, ent, nv2 = pilot(bo[j], ba[j]); ratio = (nl - bl[j]).exp()
                ad = (badv[j] - badv[j].mean()) / (badv[j].std() + 1e-8)
                pg = torch.max(-ad * ratio, -ad * ratio.clamp(0.8, 1.2)).mean()
                loss = pg + 0.5 * ((nv2 - bret[j]) ** 2).mean() - 0.01 * ent.mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(pilot.parameters(), 0.5); opt.step()
        if len(env.shop_done) >= a.shop_batch:
            data, env.shop_done = env.shop_done, []
            sl, se = update_shop(shop, sopt, data)
            ops = np.bincount([d[2] for d in data], minlength=SHOP_OPS)
            shop_log = {"obchod_loss": round(sl, 4), "obchod_entropie": round(se, 3), "obchod_akce_%": (100 * ops / ops.sum()).round(1).tolist()}
        el = time.time() - t0
        rec = {"it": it, "gstep": gstep, "sps": round(gstep / el), "loss": round(loss.item(), 4),
               "entropie": round(ent.mean().item(), 3), "s": round(el), **shop_log}
        logf.write(json.dumps(rec) + "\n"); logf.flush(); epf.flush(); runf.flush()
        if it % 10 == 0:
            print(json.dumps(rec), flush=True)
            torch.save(pilot.state_dict(), out / "agent.pt"); torch.save(shop.state_dict(), out / "shop.pt")
    torch.save(pilot.state_dict(), out / "agent.pt"); torch.save(shop.state_dict(), out / "shop.pt"); env.close()


if __name__ == "__main__":
    main()
