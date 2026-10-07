import json
import sys

import torch
import triton
from triton.experimental import gluon as g
from triton.experimental.gluon import language as l
from triton.experimental.gluon.language import (
    DotOperandLayout,
    NVMMADistributedLayout,
    SliceLayout,
)
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

sys.path.insert(0, "/root/qvl/experiments/lossless-kv")
from lossless import pack


@g.jit
def probe(P, M, O, Q, R, N: l.constexpr, PACK: l.constexpr):
    mm: l.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8]
    )
    op: l.constexpr = DotOperandLayout(operand_index=1, parent=mm, k_width=2)
    ds = l.arange(0, 128, layout=SliceLayout(1, op))
    rows = l.program_id(0) * 32 + l.arange(0, 32, layout=SliceLayout(0, op))
    meta = l.load(M + rows, rows < N, other=0).to(l.uint32)
    comp = (meta & 256) != 0
    sm = l.load(
        P + rows[None, :] * 256 + ds[:, None],
        mask=(rows[None, :] < N) & comp[None, :],
        other=0,
    ).to(l.uint16)
    nib = l.load(
        P + rows[None, :] * 256 + 128 + ds[:, None] // 2,
        mask=(rows[None, :] < N) & comp[None, :],
        other=0,
    ).to(l.uint16)
    ex = (meta[None, :] & 255) + ((nib >> ((ds[:, None] % 2) * 4)) & 15)
    bits = ((sm & 128) << 8) | (ex.to(l.uint16) << 7) | (sm & 127)
    raw = l.load(
        P.to(l.pointer_type(l.uint16)) + rows[None, :] * 128 + ds[:, None],
        mask=(rows[None, :] < N) & (~comp[None, :]),
        other=0,
    )
    kv = l.where(comp[None, :], bits, raw).to(l.uint16).to(l.bfloat16, bitcast=True)
    l.store(O + rows[None, :] * 128 + ds[:, None], kv, rows[None, :] < N)
    qa: l.constexpr = DotOperandLayout(operand_index=0, parent=mm, k_width=2)
    mr = l.arange(0, 16, layout=SliceLayout(1, qa))
    kd = l.arange(0, 128, layout=SliceLayout(0, qa))
    q = l.load(Q + mr[:, None] * 128 + kd[None, :])
    acc = l.full((16, 32), 0, l.float32, mm)
    acc = mma_v2(q, kv, acc)
    rr = l.arange(0, 16, layout=SliceLayout(1, mm))
    cc = l.arange(0, 32, layout=SliceLayout(0, mm))
    l.store(R + l.program_id(0) * 512 + rr[:, None] * 32 + cc[None, :], acc)


x = (
    torch.arange(65536, device="cuda")
    .to(torch.int16)
    .view(torch.bfloat16)
    .reshape(-1, 128)
)
p, m = pack(x)
o = torch.empty_like(x)
q = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
r = torch.empty(triton.cdiv(x.shape[0], 32), 16, 32, device="cuda")
c = probe[(triton.cdiv(x.shape[0], 32),)](p, m, o, q, r, x.shape[0], True, num_warps=4)
assert torch.equal(x.view(torch.int16), o.view(torch.int16))
print(
    json.dumps(dict(regs=c.n_regs, shared=c.metadata.shared, spills=c.n_spills)),
    flush=True,
)
open("/root/qvl/experiments/gluon-lossless/probe.ptx", "w").write(c.asm["ptx"])

for name, x in [
    ("normal", torch.randn(4096, 128, device="cuda").bfloat16()),
    (
        "random",
        torch.randint(
            -32768, 32768, (4096, 128), device="cuda", dtype=torch.int16
        ).view(torch.bfloat16),
    ),
] + [
    (
        f"{a}{b}",
        torch.load(
            f"/root/qvl/experiments/real-kv-capture/layer{a}_{b}.pt",
            map_location="cuda",
            weights_only=True,
        ).reshape(-1, 128),
    )
    for a in [0, 17, 35]
    for b in "kv"
]:
    p, m = pack(x)
    o = torch.empty_like(x)
    r = torch.empty(triton.cdiv(x.shape[0], 32), 16, 32, device="cuda")
    probe[(triton.cdiv(x.shape[0], 32),)](p, m, o, q, r, x.shape[0], True, num_warps=4)
    assert torch.equal(x.view(torch.int16), o.view(torch.int16))
    print(name, "bitwise", flush=True)
