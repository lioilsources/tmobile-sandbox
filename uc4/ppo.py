#!/usr/bin/env python3
"""UC4: PPO (podle CleanRL ppo.py, diskrétní akce, MLP) nad bezhlavým Kirianem.

  ppo.py --repo /srv/work/uc4/kiran --envs 8 --steps 20000000 --out /srv/work/uc4/runs/ppo1

Každá epizoda začne na náhodném ručně navrženém sektoru s novou lodí. Konce epizod
(sektor, výsledek, HP, místo smrti, zásahy) jdou do episodes.jsonl → evaluate.py.

Kolo 2 („vystřílej všechno“): odměna za sestřely místo za přežití, start z agenta kola 1,
sektory, kde agent kempoval, vylosované dvakrát častěji:
  ppo.py --mode clear --init runs/ppo1/agent.pt --focus 2,4,5,10,11,12,13,17 --out runs/ppo2 ...
"""
import argparse, json, pathlib, time
import numpy as np
import torch, torch.nn as nn
from torch.distributions import Categorical
from kirian_env import N_ACT, OBS, OBS_V2, VecKirian


def layer(i, o, std=np.sqrt(2)):
    l = nn.Linear(i, o); nn.init.orthogonal_(l.weight, std); nn.init.zeros_(l.bias); return l


class Agent(nn.Module):
    def __init__(self, hidden=256, obs=OBS):
        super().__init__()
        self.body = nn.Sequential(layer(obs, hidden), nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.pi = layer(hidden, N_ACT, 0.01); self.v = layer(hidden, 1, 1.0)

    def forward(self, x, a=None):
        h = self.body(x); d = Categorical(logits=self.pi(h))
        if a is None: a = d.sample()
        return a, d.log_prob(a), d.entropy(), self.v(h).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=8); ap.add_argument("--steps", type=int, default=20_000_000)
    ap.add_argument("--rollout", type=int, default=256); ap.add_argument("--sectors", default="0-17")
    ap.add_argument("--lr", type=float, default=2.5e-4); ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=4); ap.add_argument("--flutter", default="flutter")
    ap.add_argument("--base-port", type=int, default=7000)
    ap.add_argument("--device", default="cpu", help="cpu: malá MLP je na CPU rychlejší než sdílená GPU")
    ap.add_argument("--mode", default="survive", choices=["survive", "clear"], help="odměna kola 1 / kola 2 (viz kirian_env)")
    ap.add_argument("--init", help="agent.pt z předchozího kola; nové vstupy dostanou nulové váhy")
    ap.add_argument("--focus", default="", help="sektory losované dvakrát častěji, např. 2,4,5")
    a = ap.parse_args()
    lo, hi = map(int, a.sectors.split("-"))
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device); torch.set_num_threads(2)
    sectors = list(range(lo, hi + 1)) + [int(x) for x in a.focus.split(",") if x]
    env = VecKirian(a.repo, a.envs, sectors, base_port=a.base_port, flutter=a.flutter, log_dir=str(out), mode=a.mode)
    OBS_N = OBS_V2 if a.mode == "clear" else OBS
    agent = Agent(obs=OBS_N)
    if a.init:
        sd = torch.load(a.init, map_location="cpu"); w = sd["body.0.weight"]
        if w.shape[1] < OBS_N:  # rozšířené pozorování: chování z kola 1 zůstane, nové vstupy začnou s nulou
            sd["body.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], OBS_N - w.shape[1])], 1)
        agent.load_state_dict(sd)
    agent = agent.to(dev); opt = torch.optim.Adam(agent.parameters(), lr=a.lr, eps=1e-5)
    T, N = a.rollout, a.envs
    obs_b = torch.zeros((T, N, OBS_N), device=dev); act_b = torch.zeros((T, N), dtype=torch.long, device=dev)
    logp_b, rew_b, done_b, val_b = (torch.zeros((T, N), device=dev) for _ in range(4))
    obs = torch.tensor(env.reset(), device=dev); done = torch.zeros(N, device=dev)
    epf = open(out / "episodes.jsonl", "a"); logf = open(out / "train.jsonl", "a")
    gstep, t0, it = 0, time.time(), 0
    while gstep < a.steps:
        it += 1
        frac = 1 - gstep / a.steps; opt.param_groups[0]["lr"] = frac * a.lr
        t_env, t_it = 0.0, time.time()
        for t in range(T):
            obs_b[t] = obs; done_b[t] = done
            with torch.no_grad():
                act, logp, _, v = agent(obs)
            act_b[t], logp_b[t], val_b[t] = act, logp, v
            te = time.time(); o, r, d, infos = env.step(act.cpu().numpy()); t_env += time.time() - te
            rew_b[t] = torch.tensor(r, device=dev); obs = torch.tensor(o, device=dev); done = torch.tensor(d, device=dev)
            for inf in infos:
                inf["gstep"] = gstep + t * N; epf.write(json.dumps(inf) + "\n")
        gstep += T * N
        with torch.no_grad():  # GAE
            nv = agent(obs)[3]; adv = torch.zeros_like(rew_b); last = 0
            for t in reversed(range(T)):
                nnt = 1.0 - (done if t == T - 1 else done_b[t + 1]); nxt = nv if t == T - 1 else val_b[t + 1]
                delta = rew_b[t] + 0.99 * nxt * nnt - val_b[t]
                last = delta + 0.99 * 0.95 * nnt * last; adv[t] = last
            ret = adv + val_b
        bo, ba, bl, badv, bret = obs_b.reshape(-1, OBS_N), act_b.reshape(-1), logp_b.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        idx = np.arange(T * N); mb = T * N // a.minibatches
        for _ in range(a.epochs):
            np.random.shuffle(idx)
            for s in range(0, T * N, mb):
                j = idx[s:s + mb]
                _, nl, ent, nv2 = agent(bo[j], ba[j]); ratio = (nl - bl[j]).exp()
                ad = (badv[j] - badv[j].mean()) / (badv[j].std() + 1e-8)
                pg = torch.max(-ad * ratio, -ad * ratio.clamp(0.8, 1.2)).mean()
                loss = pg + 0.5 * ((nv2 - bret[j]) ** 2).mean() - 0.01 * ent.mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(agent.parameters(), 0.5); opt.step()
        el = time.time() - t0
        rec = {"it": it, "gstep": gstep, "sps": round(gstep / el), "env_cas_%": round(100 * t_env / (time.time() - t_it), 1),
               "loss": round(loss.item(), 4), "entropie": round(ent.mean().item(), 3), "s": round(el)}
        logf.write(json.dumps(rec) + "\n"); logf.flush(); epf.flush()
        if it % 10 == 0:
            print(json.dumps(rec), flush=True); torch.save(agent.state_dict(), out / "agent.pt")
    torch.save(agent.state_dict(), out / "agent.pt"); env.close()


if __name__ == "__main__":
    main()
