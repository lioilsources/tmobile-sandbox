#!/usr/bin/env python3
"""UC5(a): benchmark ComfyUI workflow na L40S — stejný skript pustíme na Sparku přes sparkctl.

Každý workflow: 1 zahřívací běh + N měřených (medián s/kus). Workflow volíme jménem (--only),
spouští se po jednom, aby měl každý vlastní GPU log (bench/gpu_log.sh v run.sh).

  comfy_bench.py --only flux_schnell --reps 10 --out /srv/work/uc5/flux_schnell.json [--image start.png]
"""
import argparse, json, statistics, time, urllib.request

API = "http://127.0.0.1:8188"
PROMPT = "a small fox knight in shining armor standing on a mossy rock in a misty forest at dawn, storybook illustration"


def call(path, data=None):
    req = urllib.request.Request(API + path, json.dumps(data).encode() if data else None, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def unet(n): return {"class_type": "UNETLoader", "inputs": {"unet_name": n, "weight_dtype": "default"}}
def lora(m, n): return {"class_type": "LoraLoaderModelOnly", "inputs": {"model": m, "lora_name": n, "strength_model": 1.0}}
def enc(clip, text): return {"class_type": "CLIPTextEncode", "inputs": {"clip": clip, "text": text}}


def ks(model, pos, neg, lat, steps, cfg, sampler="euler", sched="simple"):
    return {"class_type": "KSampler", "inputs": {"model": model, "positive": pos, "negative": neg, "latent_image": lat, "seed": 0,
            "steps": steps, "cfg": cfg, "sampler_name": sampler, "scheduler": sched, "denoise": 1.0}}


def flux_schnell(_):
    return {"m": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
            "p": enc(["m", 1], PROMPT), "n": enc(["m", 1], ""),
            "l": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "k": ks(["m", 0], ["p", 0], ["n", 0], ["l", 0], 4, 1.0),
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["m", 2]}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "uc5/flux_schnell"}}}, "k"


def sdxl(_):
    return {"m": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
            "p": enc(["m", 1], PROMPT), "n": enc(["m", 1], "blurry, low quality"),
            "l": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
            "k": ks(["m", 0], ["p", 0], ["n", 0], ["l", 0], 30, 7.0, "dpmpp_2m", "karras"),
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["m", 2]}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "uc5/sdxl"}}}, "k"


def qwen_image(_):
    return {"u": unet("qwen_image_fp8_e4m3fn.safetensors"),
            "c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image"}},
            "v": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
            "ms": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["u", 0], "shift": 3.1}},
            "p": enc(["c", 0], PROMPT), "n": enc(["c", 0], ""),
            "l": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1328, "height": 1328, "batch_size": 1}},
            "k": ks(["ms", 0], ["p", 0], ["n", 0], ["l", 0], 20, 2.5),
            "d": {"class_type": "VAEDecode", "inputs": {"samples": ["k", 0], "vae": ["v", 0]}},
            "s": {"class_type": "SaveImage", "inputs": {"images": ["d", 0], "filename_prefix": "uc5/qwen_image"}}}, "k"


def _wan(kind, image=None):
    hi, lo = (f"wan2.2_{kind}_high_noise_14B_fp8_scaled.safetensors", f"wan2.2_{kind}_low_noise_14B_fp8_scaled.safetensors")
    lv = "v1.1" if kind == "t2v" else "v1"
    wf = {"hi": unet(hi), "lo": unet(lo),
          "hil": lora(["hi", 0], f"wan2.2_{kind}_lightx2v_4steps_lora_{lv}_high_noise.safetensors"),
          "lol": lora(["lo", 0], f"wan2.2_{kind}_lightx2v_4steps_lora_{lv}_low_noise.safetensors"),
          "his": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["hil", 0], "shift": 5.0}},
          "los": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lol", 0], "shift": 5.0}},
          "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
          "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
          "t+": enc(["clip", 0], PROMPT + ", the fox slowly raises its sword, gentle camera push-in"),
          "t-": enc(["clip", 0], "blurry, low quality, distorted")}
    pos, neg = ["t+", 0], ["t-", 0]
    if image:
        wf["img"] = {"class_type": "LoadImage", "inputs": {"image": image}}
        wf["i2v"] = {"class_type": "WanImageToVideo", "inputs": {"positive": pos, "negative": neg, "vae": ["vae", 0], "width": 832,
                     "height": 480, "length": 81, "batch_size": 1, "start_image": ["img", 0]}}
        pos, neg, lat = ["i2v", 0], ["i2v", 1], ["i2v", 2]
    else:
        wf["lat"] = {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": 832, "height": 480, "length": 81, "batch_size": 1}}
        lat = ["lat", 0]

    def adv(m, l, start, end, noise, left):
        return {"class_type": "KSamplerAdvanced", "inputs": {"model": m, "positive": pos, "negative": neg, "latent_image": l, "add_noise": noise,
                "noise_seed": 0, "steps": 4, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple", "start_at_step": start,
                "end_at_step": end, "return_with_leftover_noise": left}}
    wf["k1"] = adv(["his", 0], lat, 0, 2, "enable", "enable"); wf["k2"] = adv(["los", 0], ["k1", 0], 2, 4, "disable", "disable")
    wf["dec"] = {"class_type": "VAEDecode", "inputs": {"samples": ["k2", 0], "vae": ["vae", 0]}}
    wf["save"] = {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["dec", 0], "filename_prefix": f"uc5/wan_{kind}", "fps": 16,
                  "lossless": False, "quality": 85, "method": "default"}}
    return wf, "k1"


def wan_t2v(_): return _wan("t2v")
def wan_i2v(a): return _wan("i2v", a.image)


def ace_step(_):
    return {"m": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "ace_step_v1_3.5b.safetensors"}},
            "t": {"class_type": "TextEncodeAceStepAudio", "inputs": {"clip": ["m", 1], "tags": "chiptune, upbeat, retro arcade, 140 bpm, synth lead",
                  "lyrics": "[instrumental]", "lyrics_strength": 0.99}},
            "n": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["t", 0]}},
            "l": {"class_type": "EmptyAceStepLatentAudio", "inputs": {"seconds": 60.0, "batch_size": 1}},
            "k": ks(["m", 0], ["t", 0], ["n", 0], ["l", 0], 50, 5.0),
            "d": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["k", 0], "vae": ["m", 2]}},
            "s": {"class_type": "SaveAudio", "inputs": {"audio": ["d", 0], "filename_prefix": "uc5/ace_step"}}}, "k"


WORKFLOWS = {"flux_schnell": flux_schnell, "sdxl": sdxl, "qwen_image": qwen_image, "wan_t2v": wan_t2v, "wan_i2v": wan_i2v, "ace_step": ace_step}


def run_once(wf, seed_node, seed):
    key = "noise_seed" if "noise_seed" in wf[seed_node]["inputs"] else "seed"
    wf[seed_node]["inputs"][key] = seed
    t = time.time(); pid = call("/prompt", {"prompt": wf})["prompt_id"]
    while True:
        h = call(f"/history/{pid}")
        if pid in h: break
        time.sleep(0.25)
    ok = h[pid]["status"]["status_str"] == "success"
    if not ok: raise RuntimeError(json.dumps(h[pid]["status"])[-800:])
    return time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", required=True, choices=WORKFLOWS); ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--image", help="vstupní obrázek pro wan_i2v (název v ComfyUI input/)"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wf, seed_node = WORKFLOWS[a.only](a)
    cold = run_once(wf, seed_node, 1)
    times = [run_once(wf, seed_node, 100 + i) for i in range(a.reps)]
    res = {"workflow": a.only, "studeny_s": round(cold, 1), "median_s": round(statistics.median(times), 2),
           "min_s": round(min(times), 2), "max_s": round(max(times), 2), "opakovani": a.reps, "casy": [round(t, 2) for t in times]}
    open(a.out, "w").write(json.dumps(res, ensure_ascii=False, indent=1)); print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
