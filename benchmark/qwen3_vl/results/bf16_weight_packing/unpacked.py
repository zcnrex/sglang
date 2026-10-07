import argparse
import glob
import json
import random
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, "/root/qvl/experiments/lossless-kv")
from flashinfer import autotune
from flashinfer.gemm import mm_bf16
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
    default_tactic,
    run_direct_dense,
)

root = "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
d = {}
for p in glob.glob(root + "/*.safetensors"):
    with safe_open(p, framework="pt", device="cpu") as f:
        for key in f.keys():
            if any(".layers." + str(l) + ".mlp." in key for l in [0, 17, 35]) and any(
                key.endswith(t + ".weight") for t in ["gate_proj", "up_proj"]
            ):
                d[key] = f.get_tensor(key).cuda()
ws = [
    torch.cat(
        [
            d[f"model.language_model.layers.{l}.mlp.{t}_proj.weight"]
            for t in ["gate", "up"]
        ]
    )
    for l in [0, 17, 35]
]

from flashinfer.autotuner import AutoTuner

p = argparse.ArgumentParser()
p.add_argument("--m", type=int)
a = p.parse_args()
M = a.m
torch.manual_seed(123)
x = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
outs = [torch.empty(M, 19456, device="cuda", dtype=torch.bfloat16) for w in ws]
tactic = default_tactic(M, 19456, 2560)


def raw():
    for w, o in zip(ws, outs):
        run_direct_dense(x, w.t(), o, False, tactic)


def prod():
    for w, o in zip(ws, outs):
        torch.mm(x, w.t(), out=o)


with autotune(tuning_buckets=(M,), round_up=False):
    mm_bf16(x, ws[0].t(), out=outs[0], backend="cute-dsl")
AutoTuner.get().save_configs(
    "/root/qvl/experiments/weight-pack/unpacked-m%d-cache.json" % M
)


def fi():
    for w, o in zip(ws, outs):
        mm_bf16(x, w.t(), out=o, backend="cute-dsl")


prod()
refs = [o.clone() for o in outs]
report = {"m": M, "checks": {}, "rounds": []}
for name, fn in [("raw", raw), ("fi", fi)]:
    fn()
    report["checks"][name] = [
        {
            "bitwise": torch.equal(o, r),
            "nrms": (
                (o.float() - r.float()).square().mean() / r.float().square().mean()
            )
            .sqrt()
            .item(),
            "max_abs": (o.float() - r.float()).abs().max().item(),
        }
        for o, r in zip(outs, refs)
    ]
graphs = {}
for name, fn in [("raw", raw), ("fi", fi), ("torch", prod)]:
    fn()
    g = torch.cuda.CUDAGraph()
    g.enable_debug_mode()
    with torch.cuda.graph(g):
        fn()
    g.debug_dump("/root/qvl/experiments/weight-pack/unpacked-m%d-%s.dot" % (M, name))
    graphs[name] = g
rng = random.Random(512)
for _ in range(8):
    order = list(graphs)
    rng.shuffle(order)
    r = {}
    for name in order:
        g = graphs[name]
        for z in range(5):
            g.replay()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        for z in range(80):
            g.replay()
        e.record()
        e.synchronize()
        r[name] = s.elapsed_time(e) / 240
    report["rounds"].append(r)
print(json.dumps(report), flush=True)
open("/root/qvl/experiments/weight-pack/unpacked-m%d.json" % M, "w").write(
    json.dumps(report, indent=2) + "\n"
)
