import hashlib
import json
import pathlib
import types
from unittest.mock import patch

import torch
from flashinfer.gemm.kernels.dense_bf16_gemm_direct import (
    default_tactic,
    run_direct_dense,
)

import sglang.srt.layers.quantization.unquant as u

r = pathlib.Path("/root/qvl/experiments/m1-production-validation")
b = pathlib.Path("/root/qvl/sglang-public-lt-clean/python")
c = pathlib.Path("/root/qvl/sglang-m1-production/python")
diff = []
for p in b.rglob("*"):
    if p.is_file() and p.suffix in {".py", ".cu", ".cuh", ".h", ".cpp"}:
        q = c / p.relative_to(b)
        if p.read_bytes() != q.read_bytes():
            diff.append(str(p.relative_to(b)))
assert diff == ["sglang/srt/layers/quantization/unquant.py"], diff
out = {
    "differences": diff,
    "candidate_sha256": hashlib.sha256((c / diff[0]).read_bytes()).hexdigest(),
    "cases": [],
}
u._enable_m1_bf16_direct = True
u._run_direct_dense = run_direct_dense
u._direct_default_tactic = default_tactic
u._BF16_GEMM_BACKEND = u.Bf16GemmBackend.CUTEDSL
cfg = types.SimpleNamespace(
    kernel=types.SimpleNamespace(disable_flashinfer_autotune=False),
    deterministic=types.SimpleNamespace(enable_deterministic_inference=False),
)
spec = types.SimpleNamespace(speculative_algorithm=None)
torch.manual_seed(42)
x = torch.randn(1, 2560, device="cuda", dtype=torch.bfloat16)
w = torch.randn(19456, 2560, device="cuda", dtype=torch.bfloat16) * 0.02
y = torch.empty(1, 19456, device="cuda", dtype=torch.bfloat16)
with (
    patch.object(u, "get_exec", return_value=cfg),
    patch.object(u, "get_spec", return_value=spec),
    torch.inference_mode(),
):
    for i in range(3):
        u._try_m1_bf16_direct(x, w, None, out=y)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        u._try_m1_bf16_direct(x, w, None, out=y)
    for seed in [42, 43, 44]:
        torch.manual_seed(seed)
        x.copy_(torch.randn_like(x))
        g.replay()
        ref = torch.nn.functional.linear(x, w)
        direct = torch.empty_like(y)
        run_direct_dense(x, w.T, direct, False, default_tactic(1, 19456, 2560))
        torch.cuda.synchronize()
        assert torch.equal(y, direct)
        nrms = (
            ((y.float() - ref.float()).square().mean() / ref.float().square().mean())
            .sqrt()
            .item()
        )
        assert nrms < 0.005
        out["cases"].append(
            {
                "seed": seed,
                "bitwise_vs_direct": True,
                "nrms_vs_torch": nrms,
                "max_abs_vs_torch": (y - ref).abs().max().item(),
            }
        )
    assert (
        u._try_m1_bf16_direct(
            x, w, torch.zeros(19456, device="cuda", dtype=torch.bfloat16)
        )
        is None
    )
    assert u._try_m1_bf16_direct(x.repeat(2, 1), w, None) is None
(r / "standalone-check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out))
