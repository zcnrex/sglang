import json
import pathlib
import random
import statistics

import torch
from flashinfer.gemm.gemm_base import get_mm_bf16_cublaslt_module

root = pathlib.Path("/root/qvl/experiments/cublaslt-tactics")
root.mkdir(exist_ok=True)
torch.manual_seed(42)
random.seed(42)
runner = get_mm_bf16_cublaslt_module().cublaslt_bf16_gemm_runner()
workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def timing(fn, reps=15):
    for _ in range(3):
        fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(4):
            fn()
    g.replay()
    torch.cuda.synchronize()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        g.replay()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) * 1000 / (reps * 4)


results = []
for m in [8192, 16331]:
    for name, n, k in [
        ("qkv", 6144, 2560),
        ("o", 2560, 4096),
        ("gateup", 19456, 2560),
        ("down", 2560, 9728),
    ]:
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16) * 0.1
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
        o = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(a, w)
        inputs = [a, w.T, None, False, o, workspace]
        _, count = runner._get_algos(inputs)
        row = {"m": m, "n": n, "k": k, "name": name, "num_algos": count, "screen": []}
        print(json.dumps({"start": row}), flush=True)
        for tactic in range(count):
            try:
                fn = lambda t=tactic, inputs=inputs: runner.forward(inputs, tactic=t)
                fn()
                torch.cuda.synchronize()
                torch.testing.assert_close(o, ref, rtol=0.02, atol=0.02)
                us = timing(fn)
                row["screen"].append({"tactic": tactic, "us": us})
            except Exception as e:
                row["screen"].append({"tactic": tactic, "error": str(e)[:500]})
        valid = sorted([x for x in row["screen"] if "us" in x], key=lambda x: x["us"])
        top = [x["tactic"] for x in valid[:3]]
        top = list(dict.fromkeys([0] + top))
        measures = {"torch": []} | {str(t): [] for t in top}
        funcs = {"torch": lambda a=a, w=w, o=o: torch.mm(a, w.T, out=o)} | {
            str(t): (lambda t=t, inputs=inputs: runner.forward(inputs, tactic=t))
            for t in top
        }
        for repeat in range(4):
            order = list(funcs)
            random.shuffle(order)
            for key in order:
                measures[key].append(timing(funcs[key], 40))
        row["final_us"] = {key: statistics.median(v) for key, v in measures.items()}
        row["rounds"] = measures
        best = min(top, key=lambda t: row["final_us"][str(t)])
        runner.forward(inputs, tactic=best)
        torch.cuda.synchronize()
        diff = o.float() - ref.float()
        row["best"] = best
        row["nrms"] = (
            diff.square().mean().sqrt() / ref.float().square().mean().sqrt()
        ).item()
        row["max_abs"] = diff.abs().max().item()
        row["bitwise"] = torch.equal(o, ref)
        results.append(row)
        (root / "results.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(row), flush=True)
        del a, w, o, ref, inputs, funcs
print("DONE", flush=True)
