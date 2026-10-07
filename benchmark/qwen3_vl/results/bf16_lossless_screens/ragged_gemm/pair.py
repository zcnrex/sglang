import gc
import json
import statistics
from pathlib import Path

import torch

from sglang.srt.layers.activation import silu_and_mul

torch.manual_seed(20261007)
torch.cuda.set_device(0)
rows = []
for m in [16331, 16330, 8201, 16384]:
    p = (m + 127) // 128 * 128
    k = 9728
    n = 2560
    x = torch.randn(m, 2 * k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    act = torch.empty(m, k, device="cuda", dtype=torch.bfloat16)
    pad = torch.empty(p, k, device="cuda", dtype=torch.bfloat16)
    yr = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    yp = torch.empty(p, n, device="cuda", dtype=torch.bfloat16)

    def raw(x=x, act=act, w=w, yr=yr):
        silu_and_mul(x, act)
        torch.mm(act, w.t(), out=yr)

    def padded(x=x, pad=pad, m=m, p=p, w=w, yp=yp):
        silu_and_mul(x, pad[:m])
        if p > m:
            pad[m:].zero_()
        torch.mm(pad, w.t(), out=yp)

    funcs = {"raw": raw, "padded": padded}
    graphs = {}
    for key, fn in funcs.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(5):
                fn()
        graphs[key] = g
    assert torch.equal(act, pad[:m])
    assert torch.equal(yr, yp[:m])
    times = {key: [] for key in funcs}
    for r in range(8):
        for key in list(funcs)[:: 1 if r % 2 == 0 else -1]:
            g = graphs[key]
            for _ in range(3):
                g.replay()
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(10):
                g.replay()
            b.record()
            b.synchronize()
            times[key].append(a.elapsed_time(b) * 1000 / 50)
    result = dict(
        m=m,
        padded_m=p,
        bitwise_activation=True,
        bitwise_output=True,
        zero_tail_per_iteration=True,
        us=times,
        median_us={key: statistics.median(v) for key, v in times.items()},
    )
    rows.append(result)
    print(json.dumps(result), flush=True)
    Path("/root/qvl/experiments/ragged-gemm/pair_report.json").write_text(
        json.dumps(rows, indent=2)
    )
    del graphs, funcs, g, x, w, act, pad, yr, yp
    gc.collect()
    torch.cuda.empty_cache()
