#!/usr/bin/env python3
"""UC4: DreamerV3 (danijar/dreamerv3, JAX) na Kirianu — world model místo PPO, GPU-bound protějšek kola 2.

Prostředí = stejný headless server jako u PPO (KirianProc), pozorování = vektor OBS_V2 (148 čísel; headless
`flutter test` nevykresluje, takže obraz nemáme), odměna kola 2 „vystřílej všechno“ (kirian_env.reward, mode
clear), akce = 18 diskrétních (9 směrů × střelba). Epizoda = jeden náhodný sektor 0–17, konec smrtí,
dokončením (terminal) nebo limitem 150 s (truncation). Do logu Dreameru jdou log/ klíče: sestřeleno, dokončeno, sektor.

Spouští se tímto souborem místo dreamerv3/main.py; doplní do make_env sadu `kirian` bez úpravy repa Dreameru:
  PYTHONPATH=<dreamerv3>:<uc4> python kirian_dreamer.py --configs kirian size50m --logdir /srv/work/uc4/dreamer/runs/d1
"""
import json, pathlib, socket, sys, time

import elements
import embodied
import numpy as np

from kirian_env import DIRS, OBS_V2, KirianProc, encode, kstats, reward

REPO = "/srv/work/uc4/kiran"
FLUTTER = "/srv/work/uc4/flutter/bin/flutter"


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class Kirian(embodied.Env):

    def __init__(self, task, sectors=(0, 17), max_t=150.0, frame_skip=4, seed=0, logdir=None):
        self.mode = task  # clear | survive
        self.sectors = list(range(sectors[0], sectors[1] + 1)); self.n_sectors = sectors[1] + 1
        self.max_t, self.k = max_t, frame_skip
        self.rng = np.random.default_rng(seed)
        log_dir = str(logdir) if logdir else "/tmp"
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.proc = KirianProc(REPO, free_port(), flutter=FLUTTER, log_dir=log_dir)
        self.o, self.done, self.ret = None, True, 0.0
        self.eplog = open(pathlib.Path(log_dir) / "episodes.jsonl", "a")  # po sektorech, pro srovnání s PPO

    @property
    def obs_space(self):
        return {
            "vector": elements.Space(np.float32, (OBS_V2,)),
            "reward": elements.Space(np.float32),
            "is_first": elements.Space(bool), "is_last": elements.Space(bool), "is_terminal": elements.Space(bool),
            "log/sestreleno": elements.Space(np.float32), "log/dokonceno": elements.Space(np.float32),
            "log/sektor": elements.Space(np.float32),
        }

    @property
    def act_space(self):
        return {"action": elements.Space(np.int32, (), 0, 18), "reset": elements.Space(bool)}

    def _obs(self, o, r=0.0, first=False, last=False, terminal=False, end=None):
        kills, count, *_ = kstats(o)
        return {"vector": encode(o, self.n_sectors, True), "reward": np.float32(r),
                "is_first": first, "is_last": last, "is_terminal": terminal,
                "log/sestreleno": np.float32(min(kills / max(count, 1), 1)), "log/dokonceno": np.float32(end == "dokonceno"),
                "log/sektor": np.float32(o["sector"])}

    def step(self, action):
        if action["reset"] or self.done:
            self.o = self.proc.req({"cmd": "reset", "sector": int(self.rng.choice(self.sectors))}); self.done = False; self.ret = 0.0
            return self._obs(self.o, first=True)
        a = int(action["action"]); d = DIRS[a % 9]
        prev, o = self.o, self.proc.req({"cmd": "step", "a": [d[0], d[1], int(a // 9)], "k": self.k}); self.o = o
        r, end, _ = reward(prev, o, self.mode, self.max_t)
        self.done = end is not None; self.ret += r
        if self.done:
            kills, count, _, _, esc = kstats(o)
            self.eplog.write(json.dumps({"sector": o["sector"], "end": end, "kills": kills, "count": count, "esc": esc,
                                         "t": o["t"], "ret": round(self.ret, 2), "wall": round(time.time())}) + "\n")
            self.eplog.flush()
        return self._obs(o, r, last=self.done, terminal=end in ("smrt", "dokonceno"), end=end)

    def close(self):
        self.proc.close()


def main():
    from dreamerv3 import main as dm
    orig = dm.make_env

    def make_env(config, index, **overrides):
        suite, task = config.task.split("_", 1)
        if suite != "kirian":
            return orig(config, index, **overrides)
        env = Kirian(task, seed=hash((config.seed, index)) % (2 ** 32 - 1), logdir=elements.Path(config.logdir) / f"env{index}")
        return dm.wrap_env(env, config)

    dm.make_env = make_env
    # vlastní blok configu `kirian` vedle configs.yaml Dreameru
    cfg = pathlib.Path(dm.__file__).parent / "configs.yaml"
    extra = pathlib.Path(__file__).with_name("kirian_dreamer.yaml").read_text()
    if "\nkirian:" not in cfg.read_text():
        cfg.write_text(cfg.read_text().rstrip() + "\n\n" + extra)
    dm.main(sys.argv[1:])


if __name__ == "__main__":
    main()
