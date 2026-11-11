#!/usr/bin/env python3
"""UC12: export AZ sítě (flax, nets/net_*.msgpack) do ONNX pro Flutter (onnxruntime). Jen CPU.

BatchNorm se složí do konvolucí (inference), vstup zůstává NHWC [B, 8, 8, N_PLANES] jako v `observe()`,
výstupy `logits` [B, N_ACTIONS] (kanonický rámec, maska legálních tahů a `to_real` patří aplikaci) a `value` [B] (tanh).
Po exportu se výstup porovná s flax sítí na náhodných vstupech (max. odchylka do --tol).

  JAX_PLATFORMS=cpu python uc12/export.py /cold/w2/uc12/ckpt/nets/net_032070.msgpack /cold/w2/uc12/export/orbitron_az.onnx
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import az  # noqa: E402
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

EPS = 1e-5   # flax nn.BatchNorm výchozí epsilon


class G:
    def __init__(self):
        self.nodes, self.inits, self.k = [], [], 0

    def const(self, name, arr):
        self.inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32) if arr.dtype != np.int64 else arr, name))
        return name

    def op(self, typ, ins, **attrs):
        self.k += 1; out = f"{typ.lower()}_{self.k}"
        self.nodes.append(helper.make_node(typ, ins, [out], **attrs)); return out

    def conv_bn(self, x, conv, bn, relu=True):
        """Conv (bez biasu) + BatchNorm → jedna Conv s biasem. Kernel flax HWIO → ONNX OIHW."""
        w = np.asarray(conv["kernel"], np.float32)
        b = np.asarray(conv.get("bias", np.zeros(w.shape[-1])), np.float32)
        if bn is not None:
            p, s = bn
            scale = np.asarray(p["scale"]) / np.sqrt(np.asarray(s["var"]) + EPS)
            w = w * scale; b = (b - np.asarray(s["mean"])) * scale + np.asarray(p["bias"])
        kh = w.shape[0]; pad = kh // 2
        y = self.op("Conv", [x, self.const(f"w{self.k}", w.transpose(3, 2, 0, 1)), self.const(f"b{self.k}", b)],
                    kernel_shape=[kh, kh], pads=[pad] * 4)
        return self.op("Relu", [y]) if relu else y


def build(variables, meta):
    P, S = variables["params"], variables["batch_stats"]
    bn = lambda pp, ss, name: (pp[name], ss[name])
    g = G()
    x = g.op("Transpose", ["obs"], perm=[0, 3, 1, 2])                             # NHWC → NCHW
    x = g.conv_bn(x, P["Conv_0"], bn(P, S, "BatchNorm_0"))
    for i in range(meta["net"]["blocks"]):
        bp, bs = P[f"Block_{i}"], S[f"Block_{i}"]
        y = g.conv_bn(x, bp["Conv_0"], bn(bp, bs, "BatchNorm_0"))
        y = g.conv_bn(y, bp["Conv_1"], bn(bp, bs, "BatchNorm_1"), relu=False)
        x = g.op("Relu", [g.op("Add", [x, y])])
    p = g.conv_bn(x, P["Conv_1"], bn(P, S, "BatchNorm_1"))
    p = g.conv_bn(p, P["Conv_2"], None, relu=False)
    p = g.op("Transpose", [p], perm=[0, 2, 3, 1])                                 # zpět NHWC, flax reshape je v tomhle pořadí
    flat = g.const("shape_flat", np.array([-1, az.R.N_ACTIONS], np.int64))
    g.nodes.append(helper.make_node("Reshape", [p, flat], ["logits"]))
    v = g.conv_bn(x, P["Conv_3"], bn(P, S, "BatchNorm_2"))
    v = g.op("Transpose", [v], perm=[0, 2, 3, 1])
    v = g.op("Reshape", [v, g.const("shape_v", np.array([0, -1], np.int64))])
    v = g.op("Relu", [g.op("Gemm", [v, g.const("d0w", np.asarray(P["Dense_0"]["kernel"])), g.const("d0b", np.asarray(P["Dense_0"]["bias"]))])])
    v = g.op("Gemm", [v, g.const("d1w", np.asarray(P["Dense_1"]["kernel"])), g.const("d1b", np.asarray(P["Dense_1"]["bias"]))])
    v = g.op("Tanh", [v])
    g.nodes.append(helper.make_node("Reshape", [v, g.const("shape_val", np.array([-1], np.int64))], ["value"]))
    graph = helper.make_graph(
        g.nodes, "orbitron_az",
        [helper.make_tensor_value_info("obs", TensorProto.FLOAT, ["batch", 8, 8, az.N_PLANES])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", az.R.N_ACTIONS]),
         helper.make_tensor_value_info("value", TensorProto.FLOAT, ["batch"])], g.inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], producer_name="tmobile-sandbox uc12")
    m.ir_version = 8
    onnx.checker.check_model(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("out"); ap.add_argument("--tol", type=float, default=1e-3)
    a = ap.parse_args()
    net, variables, meta = az.load_net(a.ckpt)
    m = build(variables, meta)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    onnx.save(m, a.out)
    import onnxruntime as ort
    x = np.random.default_rng(0).random((16, 8, 8, az.N_PLANES), dtype=np.float32)
    ref_l, ref_v = (np.asarray(t) for t in net.apply(variables, x, train=False))
    got_l, got_v = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"obs": x})
    dl, dv = float(np.abs(got_l - ref_l).max()), float(np.abs(got_v - ref_v).max())
    info = {"ckpt": a.ckpt, "out": a.out, "net": meta["net"], "iter": meta.get("iter"), "gpu_hours": meta.get("gpu_hours"),
            "inputs": {"obs": ["batch", 8, 8, az.N_PLANES]}, "outputs": {"logits": ["batch", az.R.N_ACTIONS], "value": ["batch"]},
            "max_abs_diff": {"logits": dl, "value": dv}, "bytes": os.path.getsize(a.out)}
    json.dump(info, open(a.out[:-len(".onnx")] + ".json", "w"), indent=1)
    print(json.dumps(info))
    if max(dl, dv) > a.tol:
        raise SystemExit(f"ONNX se liší od flax: logits {dl:.2e}, value {dv:.2e}")


if __name__ == "__main__":
    main()
