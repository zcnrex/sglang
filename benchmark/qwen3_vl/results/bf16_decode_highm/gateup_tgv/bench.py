import json
import statistics
from pathlib import Path

import torch
from flashinfer.gemm.kernels.tgv_gemm_cute_ext import run_tgv_cute_ext

torch.manual_seed(20261007)
results = []
for m in [64, 128]:
    xs = [torch.randn(m, 2560, device="cuda", dtype=torch.bfloat16) for _ in range(4)]
    ws = [
        torch.randn(19456, 2560, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    outs = [
        torch.empty(m, 19456, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    refs = [x @ w.t() for x, w in zip(xs, ws)]
    graphs = {}
    errors = {}
    for tactic in ["torch", 1, 9, 10, 14, 15, 28]:

        def call(t=tactic):
            for x, w, o in zip(xs, ws, outs):
                if t == "torch":
                    torch.mm(x, w.t(), out=o)
                else:
                    run_tgv_cute_ext(x, w.t(), None, o, False, t)

        try:
            call()
            torch.cuda.synchronize()
            errors[str(tactic)] = max(
                (o.float() - r.float()).abs().max().item() for o, r in zip(outs, refs)
            )
            for o, r in zip(outs, refs):
                torch.testing.assert_close(o, r, atol=0.125, rtol=0.02)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                call()
            graphs[str(tactic)] = g
            print("compiled", m, tactic, flush=True)
        except Exception as e:
            print("error", m, tactic, repr(e), flush=True)
    times = {t: [] for t in graphs}
    for rep in range(6):
        for t in list(graphs)[:: 1 if rep % 2 == 0 else -1]:
            g = graphs[t]
            for _ in range(3):
                g.replay()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(20):
                g.replay()
            b.record()
            b.synchronize()
            times[t].append(a.elapsed_time(b) * 1000 / 80)
    result = dict(
        m=m,
        n=19456,
        k=2560,
        rotating_weights=4,
        weight_bytes=4 * 19456 * 2560 * 2,
        max_abs=errors,
        us=times,
        median_us={t: statistics.median(v) for t, v in times.items()},
    )
    results.append(result)
    print(json.dumps(result), flush=True)
    Path("/root/qvl/experiments/highm-tgv/report.json").write_text(
        json.dumps(results, indent=2)
    )
