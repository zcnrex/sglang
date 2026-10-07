import gc
import json
import statistics
from pathlib import Path

import torch

torch.manual_seed(20261007)
torch.cuda.set_device(0)
rows = []
for m in [16331, 16330, 8201]:
    p = (m + 127) // 128 * 128
    for name, k, n in [
        ("qkv", 2560, 6144),
        ("o", 4096, 2560),
        ("gateup", 2560, 19456),
        ("down", 9728, 2560),
    ]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        xp = torch.zeros(p, k, device="cuda", dtype=torch.bfloat16)
        xp[:m].copy_(x)
        yr = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        yp = torch.empty(p, n, device="cuda", dtype=torch.bfloat16)

        def raw(x=x, w=w, yr=yr):
            torch.mm(x, w.t(), out=yr)

        def padded(xp=xp, w=w, yp=yp):
            torch.mm(xp, w.t(), out=yp)

        def copied(xp=xp, m=m, x=x, w=w, yp=yp):
            xp[:m].copy_(x)
            torch.mm(xp, w.t(), out=yp)

        funcs = {"raw": raw, "padded": padded, "copy_padded": copied}
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
        diff = (yr.float() - yp[:m].float()).abs().max().item()
        torch.testing.assert_close(yr, yp[:m], atol=0.125, rtol=0.02)
        times = {key: [] for key in funcs}
        for r in range(6):
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
            name=name,
            k=k,
            n=n,
            max_abs=diff,
            bitwise=torch.equal(yr, yp[:m]),
            us=times,
            median_us={key: statistics.median(v) for key, v in times.items()},
        )
        rows.append(result)
        print(json.dumps(result), flush=True)
        Path("/root/qvl/experiments/ragged-gemm/report.json").write_text(
            json.dumps(rows, indent=2)
        )
        del graphs, funcs, g, x, w, xp, yr, yp
        gc.collect()
        torch.cuda.empty_cache()
