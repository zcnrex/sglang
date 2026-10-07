import argparse
import json

import torch
import triton.language as tl
from flashinfer import autotune
from flashinfer.gemm import mm_bf16

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.fused_moe_triton_kernels import (
    fused_moe_kernel,
    invoke_fused_moe_kernel,
)

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
torch.manual_seed(123)
M = a.m
N = 19456
K = 2560
R = 4


def graph_ms(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        g.replay()
    times = []
    for _ in range(5):
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        for _ in range(20):
            g.replay()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) / 20 / R)
    return times


x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(R)]
idx = torch.stack(
    (torch.arange(N // 2, device="cuda"), torch.arange(N // 2, N, device="cuda")), 1
).flatten()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
wis = [w[idx].unsqueeze(0) for w in ws]
e.record()
e.synchronize()
interleave_ms = s.elapsed_time(e)
full = [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in range(R)]
out = [torch.empty(M, N // 2, device="cuda", dtype=torch.bfloat16) for _ in range(R)]
ref = silu_and_mul(torch.mm(x, ws[0].t()))
report = {
    "m": M,
    "n": N,
    "k": K,
    "extra_weight_bytes": N * K * 2,
    "rotating_weight_bytes": R * N * K * 2,
    "interleave_ms_per_weight": interleave_ms / R,
    "results": [],
}


def base():
    for i in range(R):
        torch.mm(x, ws[i].t(), out=full[i])
        silu_and_mul(full[i], out=out[i])


report["torch_pair_ms"] = graph_ms(base)
with autotune(tuning_buckets=(M,), round_up=False):
    mm_bf16(x, ws[0].t(), out=full[0], backend="cublaslt")


def lt():
    for i in range(R):
        mm_bf16(x, ws[i].t(), out=full[i], backend="cublaslt")
        silu_and_mul(full[i], out=out[i])


report["lt_pair_ms"] = graph_ms(lt)
for bm, bn, bk in [(32, 128, 64), (64, 128, 64), (64, 256, 64)]:
    config = dict(
        BLOCK_SIZE_M=bm,
        BLOCK_SIZE_N=bn,
        BLOCK_SIZE_K=bk,
        GROUP_SIZE_M=8,
        num_warps=4,
        num_stages=3,
    )
    ids = torch.arange(M, device="cuda", dtype=torch.int32)
    experts = torch.zeros((M + bm - 1) // bm, device="cuda", dtype=torch.int32)
    count = torch.tensor([M], device="cuda", dtype=torch.int32)
    topids = torch.zeros(M, 1, device="cuda", dtype=torch.int32)
    topw = torch.ones(M, 1, device="cuda")

    def run(i, fused=True):
        invoke_fused_moe_kernel(
            x,
            wis[i],
            None,
            out[i] if fused else full[i],
            None,
            None,
            None,
            topw,
            topids,
            ids,
            experts,
            count,
            False,
            1,
            config,
            tl.bfloat16,
            False,
            False,
            False,
            False,
            False,
            fuse_swiglu=fused,
        )

    run(0, False)
    causal = silu_and_mul(torch.cat((full[0][:, 0::2], full[0][:, 1::2]), 1))
    run(0)
    err = out[0].float() - ref.float()
    nrms = (err.square().mean() / ref.float().square().mean()).sqrt().item()
    bitwise = torch.equal(out[0], causal)
    assert bitwise, "fused epilogue differs from same GEMM rounded BF16 reference"
    assert nrms < 0.005, nrms

    def fused():
        for i in range(R):
            run(i)

    resources = [
        {"n_regs": k.n_regs, "n_spills": k.n_spills, "shared": k.metadata.shared}
        for cache in fused_moe_kernel.device_caches.values()
        for k in cache[0].values()
    ]
    row = {
        "compiled_resources_so_far": resources,
        "config": config,
        "causal_bitwise": bitwise,
        "torch_nrms": nrms,
        "torch_max_abs": err.abs().max().item(),
        "times_ms": graph_ms(fused),
    }
    report["results"].append(row)
    print(row, flush=True)
open(a.out, "w").write(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
