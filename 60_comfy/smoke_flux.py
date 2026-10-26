#!/usr/bin/env python3
"""Smoke test ComfyUI: FLUX.1-schnell fp8, 1024², 4 kroky, přes HTTP API. Vypíše čas a cestu k obrázku."""
import json, time, urllib.request
API = "http://127.0.0.1:8188"
wf = {
 "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "flux1-schnell-fp8.safetensors"}},
 "2": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text":
       "a modern magenta telecom flagship store in Prague at dusk, photorealistic, wide angle"}},
 "3": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["1", 1], "text": ""}},
 "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
 "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
       "latent_image": ["4", 0], "seed": 42, "steps": 4, "cfg": 1.0, "sampler_name": "euler",
       "scheduler": "simple", "denoise": 1.0}},
 "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
 "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "smoke_flux"}},
}
def call(path, data=None):
    req = urllib.request.Request(API + path, json.dumps(data).encode() if data else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))
for run in ("cold", "warm"):
    t = time.time(); pid = call("/prompt", {"prompt": wf})["prompt_id"]; wf["5"]["inputs"]["seed"] += 1
    while pid not in (h := call(f"/history/{pid}")): time.sleep(0.5)
    st = h[pid]["status"]
    imgs = [i["filename"] for o in h[pid]["outputs"].values() for i in o.get("images", [])]
    print(f"{run}: {time.time()-t:.1f}s status={st['status_str']} {imgs}")
