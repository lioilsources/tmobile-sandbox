#!/usr/bin/env python3
"""UV probe for UC2 shaders (MirrorBooth's idea, 11. 11.): render with a float texture whose R = u, G = v, B = 0,
nearest filtering and a float framebuffer, so the output .rg is the UV map the shader sampled.

  deform = p90 of |out.rg − pixel uv| over valid pixels      optics: high, colour/light: ~0
  quant  = share of pixels whose out.rg equals the right or lower neighbour   mosaic: high, smooth warp: ~0
  gray   = median |out − 0.5| on a constant 0.5 texture      recolouring independent of geometry
  cell_px = mosaic cell width (face band), for the ≥ 8 cells across the face gate
  valid  = share of pixels with |B| < 0.05 and rg inside [0, 1] (the shader sampled uTexture there)

The probe texture is PROBE_SCALE× the render size, so magnifying warps (fisheye ≤ 4×) do not repeat texels and
read as quantization. Same GL shim and uniforms as MirrorBooth preview/render.py; does not import it.

  python3 probe.py shader.frag [...]      # prints one JSON line per shader
"""
import json, re, sys

import numpy as np

VERTEX = "#version 330 core\nin vec2 in_pos;\nvoid main() { gl_Position = vec4(in_pos, 0.0, 1.0); }\n"
SHIM = "#version 330 core\n#define FlutterFragCoord() vec4(gl_FragCoord.x, uResolution.y - gl_FragCoord.y, 0.0, 1.0)\n"
W, H, PROBE_SCALE, EPS = 384, 512, 4, 1e-4


def context():
    import moderngl
    try:
        return moderngl.create_standalone_context(require=330, backend="egl")
    except Exception:  # noqa: BLE001
        return moderngl.create_standalone_context(require=330)


def _draw(ctx, prog, tex_data, tw, th, t):
    quad = ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], dtype="f4").tobytes())
    vao = ctx.vertex_array(prog, [(quad, "2f", "in_pos")])
    tex = ctx.texture((tw, th), 3, np.ascontiguousarray(tex_data, dtype="f4").tobytes(), dtype="f4")
    tex.filter = (ctx.NEAREST, ctx.NEAREST); tex.repeat_x = tex.repeat_y = False
    col = ctx.texture((W, H), 4, dtype="f4"); fbo = ctx.framebuffer(color_attachments=[col]); fbo.use()
    if "uTexture" in prog: prog["uTexture"].value = 0
    if "uTime" in prog: prog["uTime"].value = float(t)
    tex.use(0); fbo.clear(0.0, 0.0, 0.0, 1.0); vao.render()
    out = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype="f4").reshape(H, W, 4)[::-1, :, :3].copy()
    for o in (vao, quad, tex, col, fbo): o.release()
    return out


def probe(code, ctx, t=0.7):
    body = re.sub(r"^\s*#include\s*<flutter/runtime_effect\.glsl>\s*$", "", code, flags=re.MULTILINE)
    prog = ctx.program(vertex_shader=VERTEX, fragment_shader=SHIM + body)
    for name, val in (("uResolution", (float(W), float(H))), ("uFaceCenter", (0.5, 0.5)), ("uFaceScale", 0.35)):
        if name in prog: prog[name].value = val
    tw, th = W * PROBE_SCALE, H * PROBE_SCALE
    u, v = np.meshgrid((np.arange(tw) + 0.5) / tw, (np.arange(th) + 0.5) / th)
    out = _draw(ctx, prog, np.dstack([u, v, np.zeros_like(u)]), tw, th, t)
    gray = _draw(ctx, prog, np.full((th, tw, 3), 0.5, "f4"), tw, th, t)
    prog.release()
    pu, pv = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)
    rg = out[..., :2]
    # B < 0.05 tolerates light tints/brightness; a heavy recolour leaves too few valid pixels to judge geometry
    ok = (np.abs(out[..., 2]) < 0.05) & (rg >= 0).all(-1) & (rg <= 1).all(-1)
    eq_r = np.zeros((H, W), bool); eq_d = np.zeros((H, W), bool)
    eq_r[:, :-1] = np.abs(rg[:, 1:] - rg[:, :-1]).max(-1) < EPS
    eq_d[:-1] = np.abs(rg[1:] - rg[:-1]).max(-1) < EPS
    okq = ok.copy(); okq[:, -1] = okq[-1] = False
    dev = np.abs(rg - np.dstack([pu, pv])).sum(-1)
    return {"valid": round(float(ok.mean()), 3),
            "deform": round(float(np.percentile(dev[ok], 90)), 4) if ok.any() else None,  # p90: local warps (nose, bulge)
            "quant": round(float((eq_r | eq_d)[okq].mean()), 3) if okq.any() else None,
            "gray": round(float(np.median(np.abs(gray - 0.5))), 4),
            "cell_px": cell_px(rg, ok)}


def cell_px(rg, ok):
    """Mosaic cell size in px: sqrt of the pixel-weighted mean area of equal-UV groups in the face band (rows 30–70 %)."""
    b = slice(int(H * 0.3), int(H * 0.7))
    m = ok[b]
    if m.mean() < 0.5:
        return None
    q = np.round(rg[b][m] * 4096).astype(np.int64)  # probe texels are 1/1536 × 1/2048 → 4096 keeps them apart
    _, cnt = np.unique(q[:, 0] * 8192 + q[:, 1], return_counts=True)
    # pixel-weighted mean area of equal-UV groups: big flat cells dominate, dot shading and borders do not
    return round(float(np.sqrt((cnt.astype(np.float64) ** 2).sum() / cnt.sum())), 1)


if __name__ == "__main__":
    c = context()
    for p in sys.argv[1:]:
        try:
            r = probe(open(p).read(), c)
        except Exception as e:  # noqa: BLE001
            r = {"error": str(e)[:200]}
        print(json.dumps({"file": p, **r}), flush=True)
