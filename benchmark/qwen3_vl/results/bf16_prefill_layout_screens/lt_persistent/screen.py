import json
import pathlib
import random
import statistics
import time

import torch
from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module
from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module
from torch.utils.cpp_extension import load

root = pathlib.Path("/root/qvl/experiments/cublaslt-persistent")
torch.manual_seed(42)
random.seed(42)
print("BUILD_START", flush=True)
mod = load(
    name="qvl_persistent_lt",
    sources=[str(root / "persistent.cpp")],
    extra_cflags=["-O3"],
    extra_ldflags=["-lcublasLt"],
    with_cuda=True,
    verbose=True,
)
print("BUILD_DONE", flush=True)
runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
raw = gen_mm_bf16_cublaslt_module().build_and_load()
workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
handle = torch.cuda.current_blas_handle()
sets = []
checks = []
for n, k, t in [(6144, 2560, 2), (2560, 4096, 2), (19456, 2560, 0), (2560, 9728, 6)]:
    a = torch.randn(8192, k, device="cuda", dtype=torch.bfloat16) * 0.1
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
    out = torch.empty(8192, n, device="cuda", dtype=torch.bfloat16)
    inp = [a, w.T, None, False, out, workspace]
    algos, count = runner._get_algos(inp)
    ref = torch.nn.functional.linear(a, w)
    for cached in [False, True]:
        mod.run(
            a,
            w,
            out,
            workspace,
            algos,
            t,
            handle,
            torch.cuda.current_stream().cuda_stream,
            cached,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
        checks.append(
            {
                "n": n,
                "k": k,
                "cached": cached,
                "bitwise": torch.equal(out, ref),
                "max_abs": (out - ref).abs().max().item(),
            }
        )
    # Different tensors at same cached shape and a nondefault stream test pointer/stream freshness.
    a2 = a.clone()
    a2.mul_(0.5)
    w2 = w.clone()
    out2 = torch.empty_like(out)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mod.run(a2, w2, out2, workspace, algos, t, handle, stream.cuda_stream, True)
    stream.synchronize()
    ref2 = torch.nn.functional.linear(a2, w2)
    torch.cuda.synchronize()
    torch.testing.assert_close(out2, ref2, rtol=0.02, atol=0.02)
    checks.append(
        {
            "n": n,
            "k": k,
            "fresh_pointers_nondefault_stream": True,
            "bitwise": torch.equal(out2, ref2),
        }
    )
    sets.append((inp, t, algos, w))


def wrapper():
    for inp, t, algo, w in sets:
        runner.forward(inp, tactic=t)


def direct():
    for inp, t, algo, w in sets:
        raw.mm_bf16_cublaslt_run_with_algo(
            inp[0], w, None, inp[4], workspace, handle, algo, t
        )


def cached():
    stream = torch.cuda.current_stream().cuda_stream
    for inp, t, algo, w in sets:
        mod.run(inp[0], w, inp[4], workspace, algo, t, handle, stream, True)


def uncached():
    stream = torch.cuda.current_stream().cuda_stream
    for inp, t, algo, w in sets:
        mod.run(inp[0], w, inp[4], workspace, algo, t, handle, stream, False)


def baseline():
    for inp, t, algo, w in sets:
        torch.mm(inp[0], w.T, out=inp[4])


funcs = ["wrapper", "direct", "cached", "uncached", "baseline"]
for _ in range(10):
    for key in funcs:
        globals()[key]()
torch.cuda.synchronize()
rows = []
for rep in range(8):
    order = funcs.copy()
    random.shuffle(order)
    for name in order:
        fn = globals()[name]
        torch.cuda.synchronize()
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        start = time.perf_counter()
        for _ in range(20):
            fn()
        host = (time.perf_counter() - start) * 1e6 / 80
        b.record()
        b.synchronize()
        rows.append(
            {
                "round": rep,
                "kind": name,
                "host_submit_us_per_projection": host,
                "gpu_us_per_projection": a.elapsed_time(b) * 1000 / 80,
            }
        )
summary = {
    name: {
        field: statistics.median(r[field] for r in rows if r["kind"] == name)
        for field in ["host_submit_us_per_projection", "gpu_us_per_projection"]
    }
    for name in funcs
}
(root / "results.json").write_text(
    json.dumps({"checks": checks, "rounds": rows, "summary": summary}, indent=2)
)
print(json.dumps(summary), flush=True)
