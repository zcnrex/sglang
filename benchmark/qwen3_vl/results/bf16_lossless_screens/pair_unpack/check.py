import sys

import torch
import triton
import triton.language as tl

sys.path.insert(0, "/root/qvl/experiments/lossless-kv")
from lossless import pack
from pair_decode import _load_kv


@triton.jit
def unpack(P, M, O, N: tl.constexpr):
    rows = (tl.program_id(0) * 16 + tl.arange(0, 16)).to(tl.int64)
    vals = _load_kv(P, M, rows, tl.arange(0, 128), rows < N, True)
    tl.store(
        O + rows[:, None] * 128 + tl.arange(0, 128)[None, :], vals, rows[:, None] < N
    )


torch.manual_seed(42)
cases = [
    (
        "allbits",
        torch.arange(65536, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(-1, 128),
    ),
    (
        "random",
        torch.randint(
            -32768, 32768, (8192, 128), device="cuda", dtype=torch.int16
        ).view(torch.bfloat16),
    ),
    ("normal", torch.randn(8192, 128, device="cuda").bfloat16()),
]
for layer in [0, 17, 35]:
    for kv in "kv":
        cases.append(
            (
                f"{layer}{kv}",
                torch.load(
                    f"/root/qvl/experiments/real-kv-capture/layer{layer}_{kv}.pt",
                    map_location="cuda",
                    weights_only=True,
                ).reshape(-1, 128),
            )
        )
for name, x in cases:
    p, m = pack(x)
    out = torch.empty_like(x)
    unpack[(triton.cdiv(x.shape[0], 16),)](p, m, out, x.shape[0])
    assert torch.equal(x.view(torch.int16), out.view(torch.int16))
    print(name, "bitwise", flush=True)
