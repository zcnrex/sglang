import json
import pathlib
import sys

sys.path.insert(
    0,
    "/root/qvl/venv-sgl/lib/python3.12/site-packages/flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell",
)
import hashlib
import os
import types

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils as utils
import flashinfer
import native_base_tail as base
import native_fused_tail as fused
import torch
from cutlass.cute.runtime import from_dlpack
from flashinfer.gemm import mm_bf16

from sglang.kernels.ops.activation import silu_and_mul

a = types.SimpleNamespace(two=True)


def build(m):
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
    out_back = [
        torch.full((1, m + 256, n), float("nan"), device="cuda", dtype=torch.bfloat16)
        for _ in range(R)
    ]
    outs = [o[:, :m, :] for o in out_back]
    fo_back = [torch.full_like(o, float("nan")) for o in out_back]
    fo = [o[:, :m, :] for o in fo_back]
    act = [
        torch.empty(m, n // 2, device="cuda", dtype=torch.bfloat16) for _ in range(R)
    ]
    assert (
        m in (8200, 16331, 16384)
        and n % 256 == 0
        and all(o.data_ptr() % 16 == 0 for o in fo)
    )
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
                "nrms": (dif.square().mean() / ref.float().square().mean())
                .sqrt()
                .item(),
            }
        )
    assert all(z["bitwise"] for z in checks), checks
    assert all(bool(torch.isnan(o[:, m:, :]).all()) for o in out_back + fo_back)
    assert all(bool(torch.isnan(o.flatten()[m * n // 2 :]).all()) for o in fo)
    print("TAIL_CANARY_PASS", m, flush=True)
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

    from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import use_cutedsl_bf16_gemm
    from sglang.srt.layers.quantization import unquant as u

    assert not use_cutedsl_bf16_gemm(m, n, k)
    assert not u.use_bf16_splitk_gemm(m, n, k)
    u._enable_bf16_splitk_gemm = True
    u._use_cutedsl_bf16_gemm = use_cutedsl_bf16_gemm
    linear_calls = []
    linear_orig = u.F.linear

    def linear_proof(*args, **kwargs):
        linear_calls.append(list(args[0].shape))
        return linear_orig(*args, **kwargs)

    u.F.linear = linear_proof
    u._bf16_gemm_dispatch_impl(x.view(m, k), weights[0], None)
    u.F.linear = linear_orig
    assert linear_calls == [[m, k]], linear_calls
    print(
        "PRODUCTION_DISPATCH",
        linear_calls,
        hashlib.sha256(pathlib.Path(u.__file__).read_bytes()).hexdigest(),
        flush=True,
    )

    def production():
        for i in range(R):
            out = u._bf16_gemm_dispatch_impl(x.view(m, k), weights[i], None)
            silu_and_mul(out, out=act[i])

    graphs = {}
    for name, fn in [("candidate", candidate), ("production", production)]:
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        g.enable_debug_mode()
        with torch.cuda.graph(g):
            fn()
        graphs[name] = g
        g.debug_dump(
            f"/root/qvl/experiments/native-epilogue-ragged/m{m}-gpu{os.environ['CUDA_VISIBLE_DEVICES']}-{name}.dot"
        )
    return graphs, checks, dict(locals())


all_graphs = {}
all_checks = {}
keepalive = {}
for m in [16331, 16384]:
    graphs, checks, state = build(m)
    all_checks[m] = checks
    keepalive[m] = state
    for kind, g in graphs.items():
        all_graphs[(m, kind)] = g
for _ in range(40):
    for g in all_graphs.values():
        g.replay()
torch.cuda.synchronize()
rows = []
for rnd in range(8):
    order = list(all_graphs)
    if (rnd + int(os.environ.get("PAIR_OFFSET", "0"))) % 2:
        order.reverse()
    row = {}
    for key in order:
        g = all_graphs[key]
        for _ in range(3):
            g.replay()
        t = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        t.record()
        for _ in range(20):
            g.replay()
        e.record()
        e.synchronize()
        row[str(key)] = t.elapsed_time(e) * 1000 / 80
    rows.append(row)
    print(row, flush=True)
pathlib.Path(
    "/root/qvl/experiments/native-epilogue-ragged/fourway-retained-gpu"
    + os.environ["CUDA_VISIBLE_DEVICES"]
    + ".json"
).write_text(
    json.dumps({"rows": rows, "checks": all_checks, "tail_canaries": True}, indent=2)
)
