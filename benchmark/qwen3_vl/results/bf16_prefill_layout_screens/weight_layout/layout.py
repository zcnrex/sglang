import argparse
import json

import torch
from triton.testing import do_bench_cudagraph

p = argparse.ArgumentParser()
p.add_argument("--shard", type=int, default=0)
args = p.parse_args()
torch.manual_seed(14)
shapes = [
    ("qkv", 6144, 2560),
    ("o", 2560, 4096),
    ("gateup", 19456, 2560),
    ("down", 2560, 9728),
]
for name, n, k in shapes[args.shard :: 3]:
    for m in [8192, 16331]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
        y = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        a = torch.cuda.Event(enable_timing=True)
        z = torch.cuda.Event(enable_timing=True)
        a.record()
        nn = w.T.contiguous()
        z.record()
        z.synchronize()
        transpose_ms = a.elapsed_time(z)
        padded = torch.empty(n, k + 128, device="cuda", dtype=torch.bfloat16)
        padded[:, :k].copy_(w)
        pw = padded[:, :k]
        candidates = {"NT": w.T, "NN_pretransposed": nn, "NT_stride_plus128": pw.T}
        torch.mm(x, w.T, out=y)
        ref = y.clone()
        checks = {}
        for mode, weight in candidates.items():
            torch.mm(x, weight, out=y)
            d = y.float() - ref.float()
            nrms = (d.square().mean() / ref.float().square().mean()).sqrt().item()
            checks[mode] = {
                "nrms": nrms,
                "max_abs": d.abs().max().item(),
                "exact": torch.equal(y, ref),
            }
            assert nrms < 0.005
        for iteration in range(3):
            order = (
                list(candidates) if iteration % 2 == 0 else list(reversed(candidates))
            )
            results = {}
            for mode in order:
                weight = candidates[mode]
                results[mode] = do_bench_cudagraph(
                    lambda: torch.mm(x, weight, out=y), rep=250
                )
            print(
                json.dumps(
                    {
                        "shape": name,
                        "m": m,
                        "n": n,
                        "k": k,
                        "iteration": iteration,
                        "ms": results,
                        "checks": checks,
                        "transpose_ms": transpose_ms,
                        "nn_weight_bytes": nn.numel() * 2,
                        "padding_extra_bytes": n * 128 * 2,
                    }
                ),
                flush=True,
            )
