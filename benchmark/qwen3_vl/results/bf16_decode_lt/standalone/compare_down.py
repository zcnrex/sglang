import argparse
import json
import pathlib
import random
import statistics

import torch
from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module
from flashinfer.jit.gemm import gen_mm_bf16_cublaslt_module

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import (
    _run_tgv,
    cutedsl_bf16_gemm_out,
    use_cutedsl_bf16_gemm,
)

p = argparse.ArgumentParser()
p.add_argument("--m", type=int, required=True)
args = p.parse_args()
m = args.m
root = pathlib.Path("/root/qvl/experiments/decode-lt")
root.mkdir(exist_ok=True)
torch.manual_seed(42)
random.seed(42)
runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
raw = gen_mm_bf16_cublaslt_module().build_and_load()
workspace = torch.empty(40 * 1024 * 1024, device="cuda", dtype=torch.uint8)
handle = torch.cuda.current_blas_handle()
results = []


def graph(fn):
    fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g


def time_graph(g, count, repeat=120):
    for _ in range(3):
        g.replay()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(repeat):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (repeat * count)


for name, n, k in [("down", 2560, 9728)]:
    count = (256 * 1024 * 1024) // (n * k * 2) + 1
    xs = [
        torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(count)
    ]
    ws = [
        torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(count)
    ]
    ys = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(count)]
    algos, num = runner._get_algos([xs[0], ws[0].T, None, False, ys[0], workspace])
    tgv = use_cutedsl_bf16_gemm(m, n, k)

    def baseline(xs=xs, ws=ws, ys=ys, tgv=tgv):
        for x, w, y in zip(xs, ws, ys):
            if tgv:
                cutedsl_bf16_gemm_out(x, w, y)
            else:
                torch.mm(x, w.T, out=y)

    def candidate(t, xs=xs, ws=ws, ys=ys):
        for x, w, y in zip(xs, ws, ys):
            raw.mm_bf16_cublaslt_run_with_algo(
                x, w, None, y, workspace, handle, algos, t
            )

    baseline()
    ref = ys[0].clone()
    truth = xs[0].float() @ ws[0].float().T
    base_nrms = (
        ((ref.float() - truth).square().mean() / truth.square().mean()).sqrt().item()
    )
    bg = graph(baseline)
    row = {
        "m": m,
        "n": n,
        "k": k,
        "name": name,
        "production": "TGV" if tgv else "Torch",
        "weight_count": count,
        "weight_mib": count * n * k * 2 / 1024**2,
        "algorithms_requested": 100,
        "algorithms_returned": num,
        "workspace_mib": 40,
        "baseline_nrms_fp32": base_nrms,
        "screen": [],
    }
    print(json.dumps({"start": row}), flush=True)
    for t in [4]:
        try:
            candidate(t)
            torch.cuda.synchronize()
            torch.testing.assert_close(ys[0], ref, rtol=0.02, atol=0.02)
            nrms = (
                ((ys[0].float() - truth).square().mean() / truth.square().mean())
                .sqrt()
                .item()
            )
            assert nrms < 0.0025, nrms
            g = graph(lambda t=t: candidate(t))
            us = time_graph(g, count, 40)
            row["screen"].append(
                {
                    "tactic": t,
                    "us": us,
                    "bitwise_vs_production": torch.equal(ys[0], ref),
                    "nrms_fp32": nrms,
                }
            )
        except Exception as e:
            row["screen"].append({"tactic": t, "error": str(e)[:500]})
    top = sorted([r for r in row["screen"] if "us" in r], key=lambda r: r["us"])[:3]
    graphs = {"production": bg} | {
        str(r["tactic"]): graph(lambda t=r["tactic"]: candidate(t)) for r in top
    }
    graphs["tgv23"] = graph(
        lambda xs=xs, ws=ws, ys=ys: [
            _run_tgv(x, w.T, None, y, True, 23) for x, w, y in zip(xs, ws, ys)
        ]
    )
    rounds = {key: [] for key in graphs}
    for rep in range(8):
        order = list(graphs)
        random.shuffle(order)
        for key in order:
            rounds[key].append(time_graph(graphs[key], count))
    row["rounds_us"] = rounds
    row["median_us"] = {key: statistics.median(v) for key, v in rounds.items()}
    best = min(top, key=lambda r: row["median_us"][str(r["tactic"])])
    row["best_tactic"] = best["tactic"]
    row["speedup"] = (
        row["median_us"]["production"] / row["median_us"][str(best["tactic"])]
    )
    results.append(row)
    (root / f"compare-down-m{m}.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(row), flush=True)
    del graphs, bg, g, xs, ws, ys, ref, truth
print("DONE", flush=True)
