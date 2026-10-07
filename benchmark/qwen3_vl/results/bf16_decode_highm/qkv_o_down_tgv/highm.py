import argparse
import json

import torch
from triton.testing import do_bench_cudagraph

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
    _pick_tactic,
    _run_tgv,
    use_cutedsl_bf16_gemm,
)

p = argparse.ArgumentParser()
p.add_argument("--shape", type=int)
args = p.parse_args()
name, n, k = [("qkv", 6144, 2560), ("o", 2560, 4096), ("down", 2560, 9728)][args.shape]
torch.manual_seed(8)
for m in [64, 128]:
    xs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(16)]
    ws = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(16)
    ]
    ys = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(16)]
    use = use_cutedsl_bf16_gemm(m, n, k)
    base_tactic = _pick_tactic(m, n, k) if use else None

    def run(t):
        for x, w, y in zip(xs, ws, ys):
            if t is None:
                torch.mm(x, w.T, out=y)
            else:
                _run_tgv(x, w.T, None, y, True, t)

    def baseline():
        run(base_tactic)

    baseline()
    refs = [y.clone() for y in ys]
    results = []
    for t in [1, 7, 8, 9, 10, 14, 15, 22, 23, 24, 27, 28]:
        try:
            run(t)
            torch.cuda.synchronize()
            nrms = max(
                ((y.float() - r.float()).square().mean() / r.float().square().mean())
                .sqrt()
                .item()
                for y, r in zip(ys, refs)
            )
            assert nrms < 0.005
            times = []
            for iteration in range(3):
                if iteration % 2 == 0:
                    b = do_bench_cudagraph(baseline, rep=80)
                    c = do_bench_cudagraph(lambda: run(t), rep=80)
                else:
                    c = do_bench_cudagraph(lambda: run(t), rep=80)
                    b = do_bench_cudagraph(baseline, rep=80)
                times.append(
                    {"baseline_us": b * 1000 / 16, "candidate_us": c * 1000 / 16}
                )
            print(
                json.dumps(
                    {
                        "shape": name,
                        "m": m,
                        "n": n,
                        "k": k,
                        "baseline_tactic": base_tactic,
                        "tactic": t,
                        "nrms": nrms,
                        "rotating_weight_bytes": 16 * n * k * 2,
                        "times": times,
                    }
                ),
                flush=True,
            )
        except Exception as e:
            print(
                json.dumps({"shape": name, "m": m, "tactic": t, "error": str(e)}),
                flush=True,
            )
