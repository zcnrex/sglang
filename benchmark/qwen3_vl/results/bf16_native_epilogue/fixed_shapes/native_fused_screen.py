import json
import sys

sys.path.insert(
    0,
    "/root/qvl/venv-sgl/lib/python3.12/site-packages/flashinfer/data/cutlass/examples/python/CuTeDSL/blackwell",
)
import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils as utils
import dense_gemm_persistent as base
import native_fused_gemm as fused
import torch
from cutlass.cute.runtime import from_dlpack

from sglang.kernels.ops.activation import silu_and_mul

m, n, k = 128, 256, 256
torch.manual_seed(17)
x = torch.randn(1, m, k, device="cuda", dtype=torch.bfloat16) * 0.1
w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.1
wi = torch.stack((w[: n // 2], w[n // 2 :]), dim=1).reshape(n, k).contiguous()
b = wi.t().unsqueeze(0)
o = torch.empty(1, m, n, device="cuda", dtype=torch.bfloat16)
f = torch.full_like(o, float("nan"))


def compile_run(mod, out):
    ts = [from_dlpack(t, assumed_align=16) for t in [x, b, out]]
    fn = mod.compile_bmm(
        (m, n, k, 1),
        *ts,
        cutlass.Float32,
        "k",
        "k",
        "n",
        (128, 128),
        (1, 1),
        utils.HardwareInfo().get_max_active_clusters(1),
        False,
        False,
    )
    fn(*ts, cuda.CUstream(torch.cuda.current_stream().cuda_stream))
    torch.cuda.synchronize()
    return fn


compile_run(base, o)
compile_run(fused, f)
gate = o[:, :, ::2]
up = o[:, :, 1::2]
split = torch.cat((gate, up), dim=-1).reshape(m, n).contiguous()
ref = silu_and_mul(split)
candidate = f.flatten()[: m * n // 2].reshape(m, n // 2)
print(
    json.dumps(
        {
            "bitwise": torch.equal(ref, candidate),
            "max_abs": (ref.float() - candidate.float()).abs().max().item(),
            "nrms": (
                (ref.float() - candidate.float()).square().mean()
                / ref.float().square().mean()
            )
            .sqrt()
            .item(),
            "finite": bool(torch.isfinite(candidate).all()),
        }
    ),
    flush=True,
)
torch.testing.assert_close(candidate, ref, atol=0.0001, rtol=0.01)
