import json
import pathlib
import random
import statistics
import time

import torch
from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module

root = pathlib.Path("/root/qvl/experiments/cublaslt-tactics")
torch.manual_seed(42)
random.seed(42)
mod = get_mm_bf16_cublaslt_module()
runner = mod.cublaslt_bf16_gemm_runner()
workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
handle = torch.cuda.current_blas_handle()
sets = []
for n, k, t in [(6144, 2560, 2), (2560, 4096, 2), (19456, 2560, 0), (2560, 9728, 6)]:
    a = torch.randn(8192, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(8192, n, device="cuda", dtype=torch.bfloat16)
    inp = [a, w.T, None, False, out, workspace]
    algos, count = runner._get_algos(inp)
    # Public factory hides module; get same built module without wrapper.
    from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

    raw = gen_mm_bf16_cublaslt_module().build_and_load()
    sets.append((inp, t, algos, w))


def wrapper():
    for inp, t, algo, w in sets:
        runner.forward(inp, tactic=t)


def direct():
    for inp, t, algo, w in sets:
        raw.mm_bf16_cublaslt_run_with_algo(
            inp[0], w, None, inp[4], workspace, handle, algo, t
        )


for _ in range(10):
    wrapper()
    direct()
torch.cuda.synchronize()
rows = []
for rep in range(6):
    order = ["wrapper", "direct"]
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
    for name in ["wrapper", "direct"]
}
(root / "overhead.json").write_text(
    json.dumps({"rounds": rows, "summary": summary}, indent=2)
)
print(json.dumps(summary))
