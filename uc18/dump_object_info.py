#!/usr/bin/env python3
"""Snapshot of ComfyUI /object_info WITHOUT a running server and without touching the GPU.

  CUDA_VISIBLE_DEVICES= /srv/comfy/ComfyUI/venv/bin/python dump_object_info.py \
      --comfy /srv/comfy/ComfyUI --out uc18/object_info.json [--nodes CheckpointLoaderSimple,KSampler,...]

Imports ComfyUI core + built-in extra nodes (no custom nodes, no API nodes) with --cpu and builds the
same dict as server.py `node_info()`. Enum inputs (ckpt_name, sampler_name, …) are read from disk, so
the snapshot also says which checkpoint files ComfyUI would offer. Used by comfy.validate() in dry-run
(gen_matrix.py --dry-run) when ComfyUI isn't running.
"""
import argparse, asyncio, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--comfy", default="/srv/comfy/ComfyUI")
ap.add_argument("--out", required=True)
ap.add_argument("--nodes", default="", help="comma list; default = everything loaded")
a = ap.parse_args()
a.out = os.path.abspath(a.out)

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never create a CUDA context (GPU belongs to another job)
sys.argv = ["main.py", "--cpu", "--disable-all-custom-nodes"]
sys.path.insert(0, a.comfy)
os.chdir(a.comfy)
import comfy.options  # noqa: E402
comfy.options.enable_args_parsing()
import nodes  # noqa: E402

asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
try:
    from comfy_api.internal import _ComfyNodeInternal  # noqa: E402
except Exception:  # older ComfyUI
    _ComfyNodeInternal = ()


def node_info(name):
    c = nodes.NODE_CLASS_MAPPINGS[name]
    if _ComfyNodeInternal and issubclass(c, _ComfyNodeInternal):
        return c.GET_NODE_INFO_V1()
    it = c.INPUT_TYPES()
    return {"input": it, "input_order": {k: list(v.keys()) for k, v in it.items()}, "output": list(c.RETURN_TYPES),
            "name": name, "output_node": bool(getattr(c, "OUTPUT_NODE", False))}


want = [n for n in a.nodes.split(",") if n] or sorted(nodes.NODE_CLASS_MAPPINGS)
out, bad = {}, {}
for n in want:
    try:
        out[n] = node_info(n)
    except Exception as e:  # one broken node must not lose the snapshot
        bad[n] = repr(e)[:300]
json.dump(out, open(a.out, "w"), default=str)
print(json.dumps({"nodes": len(out), "failed": bad, "version": getattr(__import__("comfyui_version"), "__version__", "?")}),
      file=sys.stderr)
