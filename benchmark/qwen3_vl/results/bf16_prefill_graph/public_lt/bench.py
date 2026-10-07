import json
import statistics
import sys
from pathlib import Path

import flashinfer
import torch
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm import mm_bf16

torch.manual_seed(20261007)
root = Path("/root/qvl/experiments/prefill-public-autotune")
root.mkdir(exist_ok=True)
gpu = sys.argv[1]
m = int(sys.argv[2])
rows = []
for name, k, n in [
    ("qkv", 2560, 6144),
    ("o", 4096, 2560),
    ("gateup", 2560, 19456),
    ("down", 9728, 2560),
]:
    xs = [torch.randn(m, k, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
    ws = [torch.randn(n, k, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
    outs = [torch.empty(m, n, device="cuda", dtype=torch.bfloat16) for _ in range(8)]
    refs = [x @ w.t() for x, w in zip(xs, ws)]
    with flashinfer.autotune(tuning_buckets=(m,), round_up=False):
        mm_bf16(xs[0], ws[0].t(), out=outs[0], backend="cublaslt")
    AutoTuner.get().save_configs(str(root / f"m{m}-gpu{gpu}-cache.json"))
    graphs = {}
    for mode in ["torch", "public"]:

        def call(mode=mode):
            for x, w, o in zip(xs, ws, outs):
                if mode == "torch":
                    torch.mm(x, w.t(), out=o)
                else:
                    mm_bf16(x, w.t(), out=o, backend="cublaslt")

        call()
        torch.cuda.synchronize()
        if mode == "public":
            maxdiff = max(
                (o.float() - r.float()).abs().max().item() for o, r in zip(outs, refs)
            )
            exact = all(torch.equal(o, r) for o, r in zip(outs, refs))
            for o, r in zip(outs, refs):
                torch.testing.assert_close(o, r, atol=0.125, rtol=0.02)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        graphs[mode] = g
    times = {key: [] for key in graphs}
    for rep in range(4):
        for key in list(graphs)[:: 1 if rep % 2 == 0 else -1]:
            g = graphs[key]
            for _ in range(3):
                g.replay()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(5):
                g.replay()
            b.record()
            b.synchronize()
            times[key].append(a.elapsed_time(b) * 1000 / 40)
    row = dict(
        name=name,
        gpu=gpu,
        m=m,
        k=k,
        n=n,
        rotating_weights=8,
        weight_bytes=8 * k * n * 2,
        bitwise=exact,
        max_abs=maxdiff,
        relative_l2=max(
            ((o.float() - r.float()).norm() / r.float().norm()).item()
            for o, r in zip(outs, refs)
        ),
        us=times,
        median_us={key: statistics.median(v) for key, v in times.items()},
    )
    rows.append(row)
    print(json.dumps(row), flush=True)
    (root / f"m{m}-gpu{gpu}-report.json").write_text(json.dumps(rows, indent=2))
