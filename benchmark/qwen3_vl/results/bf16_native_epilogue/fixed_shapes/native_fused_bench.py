import argparse
import json
import pathlib
import random
import sys

sys.path.insert(
    0,
    "/root/qvl/venv-sgl/lib/python3.12/site-packages/flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell",
)
import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils as utils
import dense_gemm_persistent as base
import flashinfer
import native_fused_gemm as fused
import torch
from cutlass.cute.runtime import from_dlpack
from flashinfer.gemm import mm_bf16

from sglang.kernels.ops.activation import silu_and_mul

p = argparse.ArgumentParser()
p.add_argument("--m", type=int)
p.add_argument("--two", action="store_true")
a = p.parse_args()
m = a.m
n = 19456
k = 2560
R = 4
torch.manual_seed(71)
x = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16) * 0.1
weights = [
    torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1 for _ in range(R)
]
t = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
t.record()
inter = [
    torch.stack((w[: n // 2], w[n // 2 :]), 1).reshape(n, k).contiguous()
    for w in weights
]
end.record()
end.synchronize()
transform_ms = t.elapsed_time(end)
outs = [torch.empty(1, m, n, device="cuda", dtype=torch.bfloat16) for _ in range(R)]
fo = [torch.empty_like(o) for o in outs]
act = [torch.empty(m, n // 2, device="cuda", dtype=torch.bfloat16) for _ in range(R)]
stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
tile = (256, 256) if a.two else (128, 128)
cluster = (2, 1) if a.two else (1, 1)
active = utils.HardwareInfo().get_max_active_clusters(cluster[0])
xcu = from_dlpack(x, assumed_align=16)


def setup(mod, ws, os):
    ts = [
        (
            xcu,
            from_dlpack(w.t().unsqueeze(0), assumed_align=16),
            from_dlpack(o, assumed_align=16),
        )
        for w, o in zip(ws, os)
    ]
    fn = mod.compile_bmm(
        (m, n, k, 1),
        *ts[0],
        cutlass.Float32,
        "k",
        "k",
        "n",
        tile,
        cluster,
        active,
        a.two,
        False,
    )
    return fn, ts


bf, bt = setup(base, weights, outs)
ff, ft = setup(fused, inter, fo)
for i in range(R):
    bf(*bt[i], cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    ff(*ft[i], cuda.CUstream(torch.cuda.current_stream().cuda_stream))
torch.cuda.synchronize()
checks = []
for i in range(R):
    ref = silu_and_mul(outs[i].view(m, n))
    cand = fo[i].flatten()[: m * n // 2].view(m, n // 2)
    dif = ref.float() - cand.float()
    checks.append(
        {
            "bitwise": torch.equal(ref, cand),
            "max_abs": dif.abs().max().item(),
            "nrms": (dif.square().mean() / ref.float().square().mean()).sqrt().item(),
        }
    )
assert all(z["bitwise"] for z in checks), checks
if m == 128:
    with flashinfer.autotune(tuning_buckets=(128,), round_up=False):
        mm_bf16(
            x.view(m, k), weights[0].t(), out=outs[0].view(m, n), backend="cublaslt"
        )


def native():
    for i in range(R):
        bf(*bt[i], cuda.CUstream(torch.cuda.current_stream().cuda_stream))
        silu_and_mul(outs[i].view(m, n), out=act[i])


def candidate():
    for i in range(R):
        ff(*ft[i], cuda.CUstream(torch.cuda.current_stream().cuda_stream))


def production():
    for i in range(R):
        if m == 128:
            mm_bf16(
                x.view(m, k), weights[i].t(), out=outs[i].view(m, n), backend="cublaslt"
            )
        else:
            torch.mm(x.view(m, k), weights[i].t(), out=outs[i].view(m, n))
        silu_and_mul(outs[i].view(m, n), out=act[i])


graphs = {}
for name, fn in [
    ("native", native),
    ("candidate", candidate),
    ("production", production),
]:
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    g.enable_debug_mode()
    with torch.cuda.graph(g):
        fn()
    graphs[name] = g
    g.debug_dump(f"/root/qvl/experiments/native-epilogue/m{m}-{name}.dot")
rounds = []
for rnd in range(6):
    order = list(graphs)
    random.Random(rnd).shuffle(order)
    row = {}
    for name in order:
        for _ in range(3):
            graphs[name].replay()
        t.record()
        for _ in range(20):
            graphs[name].replay()
        end.record()
        end.synchronize()
        row[name] = t.elapsed_time(end) * 1000 / (20 * R)
    rounds.append(row)
    print(row, flush=True)
result = {
    "m": m,
    "tile": tile,
    "checks": checks,
    "rounds_us": rounds,
    "weight_bytes": weights[0].numel() * 2 * R,
    "transform_ms": transform_ms,
    "extra_weight_bytes": sum(w.numel() * 2 for w in inter),
    "output_note": "candidate stores contiguous half-width result in first half of full-width allocated buffer; no intermediate store/read",
}
pathlib.Path(
    f"/root/qvl/experiments/native-epilogue/m{m}-two{int(a.two)}.json"
).write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
