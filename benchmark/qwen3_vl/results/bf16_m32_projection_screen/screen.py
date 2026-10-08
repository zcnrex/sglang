import json
import os
import pathlib
import statistics

import torch
from flashinfer.gemm import mm_bf16
from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
    SplitKTactic,
    run_splitk_dense,
)
from safetensors import safe_open

from sglang.kernels.ops.gemm.cutedsl_bf16_gemm import use_cutedsl_bf16_gemm
from sglang.srt.layers.quantization.unquant import use_bf16_splitk_gemm

root = pathlib.Path(os.environ["RUN_ROOT"])
root.mkdir(parents=True, exist_ok=True)
name = os.environ["SHAPE"]
n, k = (6144, 2560) if name == "qkv" else (2560, 9728)
assert not use_cutedsl_bf16_gemm(32, n, k) and not use_bf16_splitk_gemm(32, n, k)
model = pathlib.Path(
    "/root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17"
)
idx = json.load(open(model / "model.safetensors.index.json"))["weight_map"]


def load(key):
    with safe_open(model / idx[key], framework="pt", device="cpu") as f:
        return f.get_tensor(key)


ws = []
keys = []
for layer in range(16):
    prefix = f"model.language_model.layers.{layer}."
    names = (
        [prefix + f"self_attn.{p}_proj.weight" for p in ("q", "k", "v")]
        if name == "qkv"
        else [prefix + "mlp.down_proj.weight"]
    )
    keys.append(names)
    w = torch.cat([load(t) for t in names], dim=0).cuda()
    assert w.shape == (n, k)
    ws.append(w)
torch.manual_seed(511)
xs = [torch.randn(32, k, device="cuda", dtype=torch.bfloat16) for _ in ws]
ys = [torch.empty(32, n, device="cuda", dtype=torch.bfloat16) for _ in ws]
configs = [(128, 32, 2, 6) if name == "qkv" else (128, 32, 4, 5)]
report = {
    "pid": os.getpid(),
    "shape": name,
    "m": 32,
    "n": n,
    "k": k,
    "baseline": "production selector F.linear",
    "model_revision": model.name,
    "weight_keys": keys,
    "weight_bytes": sum(w.numel() * w.element_size() for w in ws),
    "results": [],
}


def run(kind):
    for x, w, y in zip(xs, ws, ys):
        if kind == "torch":
            y.copy_(torch.nn.functional.linear(x, w))
        elif kind == "lt":
            mm_bf16(x, w.T, out=y, backend="cublaslt")
        else:
            run_splitk_dense(x, w.T, None, y, True, SplitKTactic(*kind))


def base():
    for i, (x, w) in enumerate(zip(xs, ws)):
        ys[i] = torch.nn.functional.linear(x, w)


# Baseline uses true F.linear, without the copy required by correctness helper.
keep = []
for kind in configs:
    try:
        run("torch")
        refs = [y.clone() for y in ys]
        run(kind)
        torch.cuda.synchronize()
        diffs = [y.float() - r.float() for y, r in zip(ys, refs)]
        nrms = max(
            (d.square().mean() / r.float().square().mean()).sqrt().item()
            for d, r in zip(diffs, refs)
        )
        assert nrms < 0.005
        rec = {
            "kind": kind,
            "nrms": nrms,
            "max_abs": max(d.abs().max().item() for d in diffs),
            "bitwise": all(torch.equal(y, r) for y, r in zip(ys, refs)),
            "argmax_equal": all(
                torch.equal(y.argmax(-1), r.argmax(-1)) for y, r in zip(ys, refs)
            ),
        }
        graphs = []
        buffers = []
        for f in [base, lambda: run(kind)]:
            for _ in range(3):
                f()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                f()
            graphs.append(g)
            buffers.append(list(ys))
        for x in xs:
            x.add_(0.015625)
        graphs[0].replay()
        expected = [y.clone() for y in buffers[0]]
        graphs[1].replay()
        torch.cuda.synchronize()
        err = max(
            ((y.float() - r.float()).square().mean() / r.float().square().mean())
            .sqrt()
            .item()
            for y, r in zip(buffers[1], expected)
        )
        assert err < 0.005
        rec["changed_input_nrms"] = err
        torch.backends.cuda.matmul.allow_tf32 = False
        fp = xs[0].float() @ ws[0].float().T
        rec["fp32_reference"] = {
            label: {
                "nrms": ((out.float() - fp).square().mean() / fp.square().mean())
                .sqrt()
                .item(),
                "max_abs": (out.float() - fp).abs().max().item(),
            }
            for label, out in [("baseline", expected[0]), ("candidate", buffers[1][0])]
        }
        assert rec["fp32_reference"]["candidate"]["nrms"] < 0.005
        rec["rounds"] = []
        for r in range(6):
            row = {}
            for j in [0, 1] if r % 2 == 0 else [1, 0]:
                for _ in range(3):
                    graphs[j].replay()
                a = torch.cuda.Event(enable_timing=True)
                b = torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(30):
                    graphs[j].replay()
                b.record()
                b.synchronize()
                row[["baseline_us", "candidate_us"][j]] = (
                    a.elapsed_time(b) * 1000 / (30 * 16)
                )
            rec["rounds"].append(row)
        rec["median_speedup_pct"] = statistics.median(
            (r["baseline_us"] / r["candidate_us"] - 1) * 100 for r in rec["rounds"]
        )
        keep.append((graphs, buffers))
    except Exception as e:
        rec = {
            "kind": kind,
            "error": str(e),
            "stage": "admission_or_correctness_before_timing",
        }
    report["results"].append(rec)
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(rec), flush=True)
print("DONE", flush=True)
