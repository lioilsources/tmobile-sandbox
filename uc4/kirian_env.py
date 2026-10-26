"""UC4: Kirian jako RL prostředí.

Každé prostředí = jeden proces `flutter test test/rl/rl_server_test.dart` (bezhlavá hra, větev
rl-headless v repu lioilsources/Kiran, commit 47e39a7; snímek téže změny je uc4/kiran-rl.patch,
`git am` nad Kiran main bd5c64c), se kterým mluvíme JSON řádky přes TCP. Hra se krokuje synchronně,
takže běží stovkykrát rychleji než reálný čas.

Pozorování (vektor 145): loď (pozice, HP, štít, generátor), 12 nejbližších nepřátel, 16 nejbližších
nepřátelských střel a 4 bonusy relativně k lodi, normalizované velikostí pole; + sektor.
Akce (18): 9 směrů × střelba ano/ne (střelba bere energii generátoru, ze které se dobíjí štít).
Odměna (mode="survive", kolo 1): +Δskóre/100, −Δ(HP+štít)/20, +0,002 za přežitý krok, −5 smrt, +5 dokončený sektor.
Odměna (mode="clear", kolo 2 „vystřílej všechno“): +0,3 za sestřel, −0,3 za nepřítele, který uletěl,
+Δskóre/400 (bonusy), −Δ(HP+štít)/20, −0,003 za krok, −5 smrt, −5 vypršený čas,
dokončení +5 + 10 × podíl sestřelených + 5 navíc, když padli všichni. Pozorování má navíc 3 čísla
(podíl sestřelených, podíl skončených formací, podíl uletělých) → OBS_V2.
"""
import json, os, socket, subprocess, time
import numpy as np

DIRS = [(0, 0), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
N_ACT = 18
NH, NP, NC = 12, 16, 4
OBS = 9 + NH * 5 + NP * 4 + NC * 3
OBS_V2 = OBS + 3
OBS_V3 = OBS_V2 + 6  # kolo 3 (kampaň): + výbava lodi a kredit
SHOP_OPS = 8         # 0 odlet, 1 upgrade přední zbraně, 2 další přední zbraň, 3/4 levá/pravá boční (koupit/upgrade),
                     # 5 upgrade generátoru, 6/7 další boční zbraň vlevo/vpravo; viz rl_server_test.dart
SHOP_OBS = 3 + 4 * 4 + SHOP_OPS


def kstats(o):
    """(sestřeleno, celkem, skončených formací, formací, uletělo) z o["k"] = [kills, count, cleared, finished, fleets, escaped]."""
    k = o.get("k") or [0] * 6
    return k[0], k[1], k[3], k[4], k[5]


class KirianProc:
    """Jeden běžící server hry."""

    def __init__(self, repo, port, flutter="flutter", log_dir="/tmp"):
        self.port = port
        env = {**os.environ, "RL_PORT": str(port)}
        self.log = open(os.path.join(log_dir, f"kirian_{port}.log"), "w")
        self.p = subprocess.Popen([flutter, "test", "test/rl/rl_server_test.dart"], cwd=os.path.join(repo, "tyrian_mobile"),
                                  env=env, stdout=self.log, stderr=subprocess.STDOUT)
        for _ in range(600):
            try:
                self.s = socket.create_connection(("127.0.0.1", port), timeout=600)
                break
            except OSError:
                if self.p.poll() is not None: raise RuntimeError(f"server {port} skončil, viz {self.log.name}")
                time.sleep(0.5)
        self.f = self.s.makefile("rw")

    def send(self, o):
        self.f.write(json.dumps(o) + "\n"); self.f.flush()

    def recv(self):
        return json.loads(self.f.readline())

    def req(self, o):
        self.send(o); return self.recv()

    def close(self):
        try: self.req({"cmd": "close"})
        except Exception: pass
        self.p.terminate()


def loadout(o):
    """Výbava pro pilota: přední zbraň (stupeň, úroveň), boční zbraně, generátor, kredit."""
    s = o.get("shop") or {}; f, l, r, g = (s.get(k, [-1, 0, 0]) for k in ("front", "left", "right", "gen"))
    return [(f[0] + 1) / 4, f[1] / 25, (l[0] + 1) / 4 + l[1] / 100, (r[0] + 1) / 4 + r[1] / 100, g[1] / 25,
            float(np.log1p(s.get("credit", 0))) / 14]


def shop_obs(o, sector, n_sectors):
    """Vstup obchodníka: kredit, odemčené stupně, příští sektor, 4 sloty × (stupeň, úroveň, cena, stačí kredit), maska akcí."""
    s = o["shop"]; c = s["credit"]
    def slot(x, n):
        t, lv, price = x
        return [(t + 1) / n, lv / 25, float(np.log1p(price)) / 14, float(t >= 0 and c >= price)]
    x = [float(np.log1p(c)) / 14, s["unlock"] / 3, sector / max(n_sectors, 1)]
    x += slot(s["front"], 4) + slot(s["left"], 4) + slot(s["right"], 4) + slot(s["gen"], 1) + [float(m) for m in s["mask"]]
    return np.asarray(x, dtype=np.float32), np.asarray(s["mask"], dtype=bool)


def encode(o, n_sectors, v2=False, v3=False):
    vx, vy, hp, hpm, sh, shm, gen, genm = o["v"]; W, H = o["field"]
    v = [vx / W, vy / H, hp / max(hpm, 1), sh / max(shm, 1), gen / max(genm, 1), o["sector"] / max(n_sectors, 1),
         vx / W - 0.5, vy / H - 0.5, min(o["t"] / 60, 3)]

    def nearest(items, k, f):
        items = sorted(items, key=lambda it: (it[0] - vx) ** 2 + (it[1] - vy) ** 2)[:k]
        rows = [f(it) for it in items]
        width = len(f(items[0])) if items else len(f([0] * 5))
        return [x for r in rows for x in r] + [0.0] * (k - len(rows)) * width
    h = nearest(o["h"], NH, lambda it: [(it[0] - vx) / W, (it[1] - vy) / H, min(it[2] / 200, 3), it[3] / W, 1.0])
    p = nearest(o["p"], NP, lambda it: [(it[0] - vx) / W, (it[1] - vy) / H, it[2] / W, 1.0])
    c = nearest(o["c"], NC, lambda it: [(it[0] - vx) / W, (it[1] - vy) / H, 1.0])
    x = v + h + p + c
    if v2:
        kills, count, fin, fleets, esc = kstats(o)
        x += [kills / max(count, 1), fin / max(fleets, 1), esc / max(count, 1)]
    if v3:
        x += loadout(o)
    return np.asarray(x, dtype=np.float32)


def reward(prev, o, mode, max_t):
    """Odměna za krok mezi pozorováními prev → o a konec epizody (None, "smrt", "dokonceno", "cas").
    survive = kolo 1; clear a campaign = kolo 2 a 3 (sestřely, viz docstring modulu)."""
    dmg = (prev["v"][2] + prev["v"][4]) - (o["v"][2] + o["v"][4])
    kills, count, _, _, esc = kstats(o)
    if mode == "survive":
        r = (o["score"] - prev["score"]) / 100 - max(dmg, 0) / 20 + 0.002
    else:
        dk = kills - kstats(prev)[0]; de = esc - kstats(prev)[4]
        r = 0.3 * dk - 0.3 * de + (o["score"] - prev["score"]) / 400 - max(dmg, 0) / 20 - 0.003
    end = None
    if o["over"]: r -= 5; end = "smrt"
    elif o["done"]:
        r += 5; end = "dokonceno"
        if mode != "survive": r += 10 * min(kills / max(count, 1), 1) + (5 if kills >= count else 0)
    elif o["t"] >= max_t:
        end = "cas"
        if mode != "survive": r -= 5
    return r, end, dmg


class VecKirian:
    """N serverů; automatický reset po konci epizody. Sektor epizody: náhodně z `sectors`."""

    def __init__(self, repo, n, sectors, base_port=7000, frame_skip=4, max_t=150.0, seed=0, mode="survive", **kw):
        self.mode, self.v2 = mode, mode == "clear"
        self.envs = []
        for i in range(n):  # postupně, ať se kompilace testů nepřetahují
            self.envs.append(KirianProc(repo, base_port + i, **kw))
        self.sectors, self.k, self.max_t = list(sectors), frame_skip, max_t
        self.rng = np.random.default_rng(seed)
        self.n_sectors = max(self.sectors) + 1
        self.last = [None] * n
        self.ep_info = [{} for _ in range(n)]

    def _reset(self, i, sector=None):
        s = int(self.rng.choice(self.sectors)) if sector is None else sector
        o = self.envs[i].req({"cmd": "reset", "sector": s})
        self.last[i] = o; self.ep_info[i] = {"sector": s, "ret": 0.0, "steps": 0, "hits": []}
        return encode(o, self.n_sectors, self.v2)

    def reset(self):
        return np.stack([self._reset(i) for i in range(len(self.envs))])

    def step(self, actions):
        obs, rews, dones, infos = [], [], [], []
        # nejdřív pošli všem, pak čti — servery krokují souběžně
        for i, a in enumerate(actions):
            d, fire = DIRS[a % 9], a // 9
            self.envs[i].send({"cmd": "step", "a": [d[0], d[1], int(fire)], "k": self.k})
        for i, a in enumerate(actions):
            prev = self.last[i]
            o = self.envs[i].recv()
            self.last[i] = o
            r, end, dmg = reward(prev, o, self.mode, self.max_t)
            if dmg > 0: self.ep_info[i]["hits"].append((o["v"][0] / o["field"][0], o["v"][1] / o["field"][1]))
            kills, count, _, _, esc = kstats(o)
            info = self.ep_info[i]; info["ret"] += r; info["steps"] += 1
            if end:
                info.update(end=end, hp=o["v"][2] / max(o["v"][3], 1), score=o["score"], t=o["t"], kills=kills, count=count, esc=esc,
                            death=(o["v"][0] / o["field"][0], o["v"][1] / o["field"][1]) if end == "smrt" else None)
                infos.append(info); obs.append(self._reset(i)); dones.append(1.0)
            else:
                obs.append(encode(o, self.n_sectors, self.v2)); dones.append(0.0)
            rews.append(r)
        return np.stack(obs), np.asarray(rews, np.float32), np.asarray(dones, np.float32), infos

    def close(self):
        for e in self.envs: e.close()
