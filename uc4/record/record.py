#!/usr/bin/env python3
"""UC4 záznam: N běhů vykresleného Kirianu s RL agentem → 1920×1080, 60 fps, H.264 12 Mbit/s + AAC 48 kHz stereo 192 kbit/s.

Na jeden běh: Xvfb (virtuální displej), PulseAudio s null sinkem (zvuk hry), agent_bridge.py (pilot + obchodník),
hra (build `flutter build linux --release -t lib/main_record.dart`), ffmpeg x11grab + pulse do rychlého
téměř bezztrátového záznamu v --tmp, po běhu převod do finálního formátu (--layout):
  native   okno 1920×1080, desktopová hra na šířku (pole 600×1067; pozorování pro agenta přepočítané)
  native720 / native900  totéž v okně 1280×720 / 1600×900, zvětšené na 1920×1080 (softwarové vykreslování
           v Xvfb stihne 60 fps jen při menším okně)
  portrait okno 540×1080 (pole 600×300 přesně jako při tréninku) uprostřed, pozadí = rozmazaná a ztmavená
           zvětšená kopie téhož obrazu

  record.py --bundle /srv/work/uc4/kiran-rec/tyrian_mobile/build/linux/x64/release/bundle \\
            --pilot /srv/work/uc4/runs/ppo3/agent.pt --shop /srv/work/uc4/runs/ppo3/shop.pt \\
            --runs 3 --sector 0 --out /cold/uc4/trailer/2026-11-04 [--loadout '{"front":[1,5]}'] [--cut]

Výstup: <out>/full/<run>.mp4, <out>/runs/<run>/{events.jsonl,trace.jsonl,run.json,meta.json,game.log}.
--cut potom zavolá cut_clips.py (klipy do <out>/clips/, clips.json).
"""
import argparse, json, os, pathlib, re, shutil, signal, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
LAYOUTS = {"native": (1920, 1080), "native720": (1280, 720), "native900": (1600, 900), "portrait": (540, 1080)}
FMT = ["-r", "60", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", "-movflags", "+faststart"]


def venc(encoder):
    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-profile:v", "high", "-rc", "vbr",
                "-b:v", "12M", "-maxrate", "12M", "-bufsize", "24M", "-g", "120"]
    return ["-c:v", "libx264", "-preset", "medium", "-profile:v", "high", "-b:v", "12M", "-maxrate", "12M",
            "-bufsize", "24M", "-g", "120"]


def vfilter(layout):
    if layout.startswith("native"):
        return "[0:v]fps=60,scale=1920:1080:flags=lanczos,setsar=1,format=yuv420p[v]"
    return ("[0:v]fps=60,split[a][b];[a]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,"
            "boxblur=24:3,eq=brightness=-0.22:saturation=0.8[bg];[b]scale=-2:1080:flags=lanczos[fg];"
            "[bg][fg]overlay=(W-w)/2:0,setsar=1,format=yuv420p[v]")


def wait_for(pred, timeout, what):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred(): return
        time.sleep(0.2)
    raise RuntimeError(f"timeout: {what}")


class Pulse:
    """Vlastní PulseAudio daemon s null sinkem 'rec' (48 kHz stereo); hra do něj hraje, ffmpeg nahrává rec.monitor."""

    def __init__(self, rt):
        self.rt = pathlib.Path(rt); self.rt.mkdir(parents=True, exist_ok=True); os.chmod(self.rt, 0o700)
        self.sock = self.rt / "pulse.sock"; self.server = f"unix:{self.sock}"
        env = {**os.environ, "XDG_RUNTIME_DIR": str(self.rt), "PULSE_RUNTIME_PATH": str(self.rt / "pulse")}
        self.env = env
        if subprocess.run(["pactl", "info"], env={**env, "PULSE_SERVER": self.server}, capture_output=True).returncode != 0:
            subprocess.run(["pulseaudio", "-n", "--daemonize=yes", "--exit-idle-time=-1", "--disallow-exit",
                            "-L", f"module-native-protocol-unix socket={self.sock}",
                            "-L", "module-null-sink sink_name=rec rate=48000 channels=2",
                            "-L", "module-always-sink"], env=env, check=True)
            wait_for(lambda: self.sock.exists(), 10, "pulseaudio")
        subprocess.run(["pactl", "set-default-sink", "rec"], env={**env, "PULSE_SERVER": self.server}, check=True)


def record_run(a, i, run_name, pulse):
    W, H = LAYOUTS[a.layout]
    out = pathlib.Path(a.out); rdir = out / "runs" / run_name; rdir.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    (out / "full").mkdir(parents=True, exist_ok=True)
    raw = tmp / f"{out.name}_{run_name}_raw.mkv"
    disp = f":{a.display}"
    procs = []
    try:
        xvfb = subprocess.Popen(["Xvfb", disp, "-screen", "0", f"{W}x{H}x24", "-nolisten", "tcp", "-nocursor"],
                                stdout=subprocess.DEVNULL, stderr=open(rdir / "xvfb.log", "w")); procs.append(xvfb)
        wait_for(lambda: pathlib.Path(f"/tmp/.X11-unix/X{a.display}").exists(), 10, "Xvfb")
        port = a.port + i
        bridge = subprocess.Popen([PY, str(HERE / "agent_bridge.py"), "--pilot", a.pilot, "--shop", a.shop, "--policy", a.policy,
                                   "--port", str(port), "--out", str(rdir), "--max-t", str(a.max_t),
                                   "--max-sectors", str(a.max_sectors), "--seed", str(a.seed + i), "--obs-map", a.obs_map],
                                  stdout=open(rdir / "bridge.log", "w"), stderr=subprocess.STDOUT); procs.append(bridge)
        wait_for(lambda: "BRIDGE_READY" in (rdir / "bridge.log").read_text(), 60, "bridge")
        cap_v = (["-c:v", "h264_nvenc", "-preset", "p1", "-tune", "lossless"] if a.capture == "nvenc"
                 else ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "4", "-threads", "3"])
        ff = subprocess.Popen(["ffmpeg", "-hide_banner", "-y", "-thread_queue_size", "1024", "-f", "x11grab", "-framerate", "60",
                               "-video_size", f"{W}x{H}", "-draw_mouse", "0", "-i", disp,
                               "-thread_queue_size", "1024", "-f", "pulse", "-sample_rate", "48000", "-channels", "2",
                               "-i", "rec.monitor", *cap_v, "-c:a", "pcm_s16le", str(raw)],
                              stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=open(rdir / "ffmpeg_capture.log", "w"),
                              env={**pulse.env, "PULSE_SERVER": pulse.server}); procs.append(ff)
        time.sleep(1.0)
        home = tmp / f"home_{run_name}"; shutil.rmtree(home, ignore_errors=True); home.mkdir()
        genv = {**pulse.env, "PULSE_SERVER": pulse.server, "DISPLAY": disp, "HOME": str(home),
                "RL_RECORD_PORT": str(port), "REC_SECTOR": str(a.sector), "REC_W": str(W), "REC_H": str(H),
                "REC_LOADOUT": a.loadout or "", "REC_SHOP_DWELL_MS": str(a.shop_dwell_ms), "REC_SKIN": a.skin}
        game = subprocess.Popen([str(pathlib.Path(a.bundle) / "kirian")], env=genv, stdout=open(rdir / "game.log", "w"),
                                stderr=subprocess.STDOUT); procs.append(game)
        limit = a.run_timeout
        try:
            game.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            print(f"[{run_name}] časový limit {limit} s, ukončuji hru", flush=True); game.terminate(); game.wait(10)
        time.sleep(1.0)
        try:
            ff.communicate(b"q", timeout=30)
        except subprocess.TimeoutExpired:
            ff.send_signal(signal.SIGINT); ff.wait(20)
        bridge.wait(timeout=20)
    finally:
        for p in reversed(procs):
            if p.poll() is None:
                p.terminate()
                try: p.wait(10)
                except subprocess.TimeoutExpired: p.kill()
    # začátek videa v čase "wall": x11grab dává časové značky podle hodin systému
    log = (rdir / "ffmpeg_capture.log").read_text()
    m = re.search(r"Input #0, x11grab.*?start: ([0-9.]+)", log, re.S)
    v_t0 = float(m.group(1)) if m else None
    events = [json.loads(l) for l in open(rdir / "events.jsonl")] if (rdir / "events.jsonl").exists() else []
    hello = next((e["wall"] for e in events if e["type"] == "run_start"), None)
    ss = max(0.0, hello - v_t0 - 0.5) if (hello and v_t0) else 0.0
    full = out / "full" / f"{run_name}.mp4"
    enc = ["ffmpeg", "-nostdin", "-hide_banner", "-y", "-ss", f"{ss:.3f}", "-i", str(raw), "-filter_complex", vfilter(a.layout),
           "-map", "[v]", "-map", "0:a", *venc(a.encoder), *FMT, str(full)]
    r = subprocess.run(enc, stdout=subprocess.DEVNULL, stderr=open(rdir / "ffmpeg_encode.log", "w"))
    fps = [float(x) for x in re.findall(r"\[rec\] fps=([0-9.]+) state=GameState.playing", (rdir / "game.log").read_text())]
    meta = {"run": run_name, "video": str(full), "video_t0_wall": (v_t0 + ss) if v_t0 else None, "layout": a.layout,
            "capture_raw": str(raw), "encode_rc": r.returncode, "game_fps_playing": fps,
            "pilot": a.pilot, "shop": a.shop, "policy": a.policy, "start_sector": a.sector, "loadout": a.loadout}
    (rdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    if r.returncode == 0 and not a.keep_raw:
        raw.unlink(missing_ok=True)
    shutil.rmtree(tmp / f"home_{run_name}", ignore_errors=True)
    summ = json.loads((rdir / "run.json").read_text()) if (rdir / "run.json").exists() else {}
    print(json.dumps({"run": run_name, "video": str(full), "encode_rc": r.returncode, "end": summ.get("end"),
                      "sectors": [(s["sector"], s["end"], s["kills"], s["count"]) for s in summ.get("sectors", [])],
                      "fps_med": sorted(fps)[len(fps) // 2] if fps else None}, ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", default="/srv/work/uc4/kiran-rec/tyrian_mobile/build/linux/x64/release/bundle")
    ap.add_argument("--pilot", required=True); ap.add_argument("--shop", default="greedy", help="shop.pt | greedy | none")
    ap.add_argument("--policy", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--runs", type=int, default=1); ap.add_argument("--sector", type=int, default=0)
    ap.add_argument("--loadout", default="", help='JSON jako příkaz loadout: {"front":[tier,level],"left":[..],"right":[..],"gen":lv,"credit":n,"unlock":i}')
    ap.add_argument("--max-sectors", type=int, default=0, help="0 = kampaň až do smrti / času / sektoru 17")
    ap.add_argument("--max-t", type=float, default=150.0)
    ap.add_argument("--layout", default="native", choices=list(LAYOUTS))
    ap.add_argument("--obs-map", default="train", choices=["train", "raw"])
    ap.add_argument("--encoder", default="x264", choices=["x264", "nvenc"], help="finální kódování")
    ap.add_argument("--capture", default="x264", choices=["x264", "nvenc"], help="rychlý mezizáznam během hry")
    ap.add_argument("--out", required=True); ap.add_argument("--tmp", default="/cold/uc4/trailer/tmp")
    ap.add_argument("--name", default="run", help="předpona jmen běhů"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--display", type=int, default=77); ap.add_argument("--port", type=int, default=7400)
    ap.add_argument("--shop-dwell-ms", type=int, default=900); ap.add_argument("--skin", default="default")
    ap.add_argument("--run-timeout", type=float, default=1800); ap.add_argument("--keep-raw", action="store_true")
    ap.add_argument("--cut", action="store_true", help="po záznamu nastříhat klipy (cut_clips.py)")
    a = ap.parse_args()
    # timeout/kill → SystemExit, aby se v record_run uklidily Xvfb, hra, most a ffmpeg (finally)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    pulse = Pulse(pathlib.Path(a.tmp) / "pulse_rt")
    stamp = time.strftime("%H%M%S")
    for i in range(a.runs):
        name = f"{a.name}{i + 1:02d}_s{a.sector}_{stamp}"
        try:
            record_run(a, i, name, pulse)
        except Exception as e:  # jeden pokažený běh nezastaví ostatní
            print(f"[{name}] CHYBA: {e}", flush=True)
    if a.cut:
        subprocess.run([PY, str(HERE / "cut_clips.py"), "--out", a.out, "--encoder", a.encoder])


if __name__ == "__main__":
    main()
