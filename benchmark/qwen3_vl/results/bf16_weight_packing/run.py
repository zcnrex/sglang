import glob
import json
import sys

import cuda.bindings.driver as cuda
import cutlass
import torch
from cutlass.cute import experimental as ext
from cutlass.cute.runtime import from_dlpack
from safetensors import safe_open

sys.path.insert(0, "/root/qvl/experiments/lossless-kv")
from flashinfer import autotune
from flashinfer.gemm import mm_bf16
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
    default_tactic,
    run_direct_dense,
)
from lossless import pack, unpack
from packed_direct import DirectDenseGemmKernel

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
packed = []
metas = []
for w in ws:
    p, m = pack(w.reshape(-1, 128))
    assert torch.equal(unpack(p, m).view_as(w).view(torch.int16), w.view(torch.int16))
    packed.append(p.view(torch.bfloat16).view_as(w))
    metas.append(m.reshape(w.shape[0], -1))
print("pack exact", flush=True)
reports = []


def timing(fn):
    for _ in range(3):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    times = []
    for _ in range(5):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        for _ in range(40):
            g.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 120)
    return times


for M in [1, 4, 8]:
    torch.manual_seed(123)
    x = torch.randn(M, 2560, device="cuda", dtype=torch.bfloat16)
    outs = [torch.empty(M, 19456, device="cuda", dtype=torch.bfloat16) for _ in ws]
    tactic = default_tactic(M, 19456, 2560)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = [
        tuple(from_dlpack(t, assumed_align=32) for t in [x, p, o, m])
        for p, o, m in zip(packed, outs, metas)
    ]
    kernel = DirectDenseGemmKernel(
        element_type=cutlass.BFloat16,
        num_rows=M,
        k_extent=2560,
        tactic=tactic,
        use_pdl=False,
    )
    compiled = ext.compile(
        kernel, *args[0], stream, options="--ptxas-options -maxrregcount=64"
    )

    def candidate():
        for ar in args:
            compiled(*ar, cuda.CUstream(torch.cuda.current_stream().cuda_stream))

    def raw():
        for w, o in zip(ws, outs):
            run_direct_dense(x, w.t(), o, False, tactic)

    raw()
    refs = [o.clone() for o in outs]
    candidate()
    torch.cuda.synchronize()
    assert all(torch.equal(o, r) for o, r in zip(outs, refs))
    with autotune(tuning_buckets=(M,), round_up=False):
        mm_bf16(x, ws[0].t(), out=outs[0], backend="cute-dsl")

    def fi():
        for w, o in zip(ws, outs):
            mm_bf16(x, w.t(), out=o, backend="cute-dsl")

    row = {
        "m": M,
        "tactic": str(tactic),
        "bitwise_raw_direct": True,
        "raw_ms": timing(raw),
        "packed_ms": timing(candidate),
        "fi_ms": timing(fi),
        "production_torch_ms": timing(
            lambda: [torch.mm(x, w.t(), out=o) for w, o in zip(ws, outs)]
        ),
    }
    reports.append(row)
    print(row, flush=True)
open("/root/qvl/experiments/weight-pack/result.json", "w").write(
    json.dumps(reports, indent=2) + "\n"
)
