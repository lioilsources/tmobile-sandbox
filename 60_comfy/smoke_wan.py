#!/usr/bin/env python3
"""Smoke test ComfyUI: Wan2.2 T2V 14B fp8 (high+low noise) + lightx2v 4 kroky, 832×480, 33 snímků → WEBP."""
import json, time, urllib.request
API = "http://127.0.0.1:8188"
def unet(n): return {"class_type": "UNETLoader", "inputs": {"unet_name": n, "weight_dtype": "default"}}
def lora(m, n): return {"class_type": "LoraLoaderModelOnly", "inputs": {"model": m, "lora_name": n, "strength_model": 1.0}}
def shift(m): return {"class_type": "ModelSamplingSD3", "inputs": {"model": m, "shift": 5.0}}
def ks(m, lat, start, end, noise, leftover):
    return {"class_type": "KSamplerAdvanced", "inputs": {"model": m, "positive": ["t+", 0], "negative": ["t-", 0],
            "latent_image": lat, "add_noise": noise, "noise_seed": 42, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
            "scheduler": "simple", "start_at_step": start, "end_at_step": end, "return_with_leftover_noise": leftover}}
wf = {
 "hi": unet("wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"),
 "lo": unet("wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"),
 "hil": lora(["hi", 0], "wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors"),
 "lol": lora(["lo", 0], "wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors"),
 "his": shift(["hil", 0]), "los": shift(["lol", 0]),
 "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
 "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
 "t+": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text":
        "a smiling call center agent with a headset talking to a customer, modern office, soft daylight, cinematic"}},
 "t-": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["clip", 0], "text": "blurry, low quality, distorted"}},
 "lat": {"class_type": "EmptyHunyuanLatentVideo", "inputs": {"width": 832, "height": 480, "length": 33, "batch_size": 1}},
 "k1": ks(["his", 0], ["lat", 0], 0, 2, "enable", "enable"),
 "k2": ks(["los", 0], ["k1", 0], 2, 4, "disable", "disable"),
 "dec": {"class_type": "VAEDecode", "inputs": {"samples": ["k2", 0], "vae": ["vae", 0]}},
 "save": {"class_type": "SaveAnimatedWEBP", "inputs": {"images": ["dec", 0], "filename_prefix": "smoke_wan",
          "fps": 16, "lossless": False, "quality": 85, "method": "default"}},
}
def call(path, data=None):
    req = urllib.request.Request(API + path, json.dumps(data).encode() if data else None, {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
for run in ("cold", "warm"):
    t = time.time(); pid = call("/prompt", {"prompt": wf})["prompt_id"]
    wf["k1"]["inputs"]["noise_seed"] += 1
    while pid not in (h := call(f"/history/{pid}")): time.sleep(1)
    st = h[pid]["status"]
    out = [i["filename"] for o in h[pid]["outputs"].values() for i in o.get("images", [])]
    print(f"{run}: {time.time()-t:.0f}s status={st['status_str']} {out}", flush=True)
    if st["status_str"] != "success": print(json.dumps(st["messages"])[-1500:]); break
