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
def loadkv(P, M, rows, ds, valid, PACK: l.constexpr):
    if PACK:
        meta = l.load(M + rows, valid, other=0).to(l.uint32)
        comp = (meta & 256) != 0
        sm = l.load(P + rows * 256 + ds, mask=valid & comp, other=0).to(l.uint16)
        nib = l.load(P + rows * 256 + 128 + ds // 2, mask=valid & comp, other=0).to(
            l.uint16
        )
        ex = (meta & 255) + ((nib >> ((ds % 2) * 4)) & 15)
        bits = ((sm & 128) << 8) | (ex.to(l.uint16) << 7) | (sm & 127)
        raw = l.load(
            P.to(l.pointer_type(l.uint16)) + rows * 128 + ds,
            mask=valid & (~comp),
            other=0,
        )
        return l.where(comp, bits, raw).to(l.uint16).to(l.bfloat16, bitcast=True)
    else:
        return l.load(P + rows * 128 + ds, mask=valid, other=0)


@g.jit
def kernel(
    Q,
    K,
    V,
    KM,
    VM,
    A,
    M,
    L,
    N: l.constexpr,
    S: l.constexpr,
    BN: l.constexpr,
    PACK: l.constexpr,
):
    mm: l.constexpr = NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[1, 4], instr_shape=[16, 8]
    )
    qa: l.constexpr = DotOperandLayout(operand_index=0, parent=mm, k_width=2)
    kb: l.constexpr = DotOperandLayout(operand_index=1, parent=mm, k_width=2)
    bh = l.program_id(0)
    sp = l.program_id(1)
    batch = bh // 8
    head = bh % 8
    qr = l.arange(0, 16, layout=SliceLayout(1, qa))
    qd = l.arange(0, 128, layout=SliceLayout(0, qa))
    q = l.load(Q + (bh * 4 + qr[:, None]) * 128 + qd[None, :], qr[:, None] < 4, other=0)
    ds = l.arange(0, 128, layout=SliceLayout(1, kb))
    ns = l.arange(0, BN, layout=SliceLayout(0, kb))
    vn = l.arange(0, BN, layout=SliceLayout(1, kb))
    vd = l.arange(0, 128, layout=SliceLayout(0, kb))
    acc = l.full((16, 128), 0, l.float32, mm)
    mx = l.full((16,), float("-inf"), l.float32, SliceLayout(1, mm))
    den = l.full((16,), 0, l.float32, SliceLayout(1, mm))
    for off in range(
        sp * triton.cdiv(N, S), l.minimum((sp + 1) * triton.cdiv(N, S), N), BN
    ):
        pos = off + ns
        rows = (
            ((batch * triton.cdiv(N, 32) + pos // 32) * 8 + head) * 32 + pos % 32
        ).to(l.int64)
        k = loadkv(K, KM, rows[None, :], ds[:, None], pos[None, :] < N, PACK)
        scores = mma_v2(q, k, l.full((16, BN), 0, l.float32, mm)) * 0.127517430824
        sn = l.arange(0, BN, layout=SliceLayout(0, mm)) + off
        scores = l.where(sn[None, :] < N, scores, float("-inf"))
        newmx = l.maximum(mx, l.max(scores, 1))
        alpha = l.exp2(mx - newmx)
        p = l.exp2(scores - newmx[:, None])
        den = den * alpha + l.sum(p, 1)
        acc = acc * alpha[:, None]
        posv = off + vn
        rowv = (
            ((batch * triton.cdiv(N, 32) + posv // 32) * 8 + head) * 32 + posv % 32
        ).to(l.int64)
        v = loadkv(V, VM, rowv[:, None], vd[None, :], posv[:, None] < N, PACK)
        pp = l.convert_layout(p.to(l.bfloat16), qa)
        acc = mma_v2(pp, v, acc)
        mx = newmx
    rr = l.arange(0, 16, layout=SliceLayout(1, mm))
    dd = l.arange(0, 128, layout=SliceLayout(0, mm))
    idx = (bh * S + sp) * 4 + rr
    l.store(A + idx[:, None] * 128 + dd[None, :], acc, rr[:, None] < 4)
    l.store(M + idx, mx, rr < 4)
    l.store(L + idx, den, rr < 4)


def main():
    b = 128
    n = 8192
    s = 8
    bn = 32
    torch.manual_seed(20261007)
    q = torch.randn(b, 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b * n // 32, 8, 32, 128, device="cuda").bfloat16()
    v = torch.randn_like(k)
    pk, km = pack(k)
    pv, vm = pack(v)
    a = torch.empty(b * 8 * s * 4, 128, device="cuda")
    m = torch.empty(b * 8 * s * 4, device="cuda")
    ll = torch.empty_like(m)
    sys.path.insert(0, "/root/qvl/experiments/lossless-attention")
    from decode import _reduce

    outputs = []
    resources = []
    funcs = {}
    for packed in [False, True]:
        o = torch.empty_like(q)

        def fn(packed=packed, o=o):
            kernel[(b * 8, s)](
                q,
                pk if packed else k,
                pv if packed else v,
                km,
                vm,
                a,
                m,
                ll,
                n,
                s,
                bn,
                packed,
                num_warps=4,
            )
            _reduce[(b * 32,)](a, m, ll, o, 4, s, s)
            return o

        c = kernel[(b * 8, s)](
            q,
            pk if packed else k,
            pv if packed else v,
            km,
            vm,
            a,
            m,
            ll,
            n,
            s,
            bn,
            packed,
            num_warps=4,
        )
        resources.append(
            dict(
                packed=packed,
                regs=c.n_regs,
                shared=c.metadata.shared,
                spills=c.n_spills,
            )
        )
        outputs.append(fn().clone())
        funcs[str(packed)] = fn
        open(f"/root/qvl/experiments/gluon-lossless/full_{packed}.ptx", "w").write(
            c.asm["ptx"]
        )
    print(
        json.dumps(
            dict(
                resources=resources,
                bitwise=torch.equal(*outputs),
                max_abs=(outputs[0] - outputs[1]).abs().max().item(),
            )
        ),
        flush=True,
    )
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    import math

    import flashinfer

    ws = torch.empty(128 * 1024**2, device="cuda", dtype=torch.uint8)
    bt = torch.arange(b * n // 32, device="cuda", dtype=torch.int32).view(b, -1)
    lens = torch.full((b,), n, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)

    def trt():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q,
            (k, v),
            ws,
            bt,
            lens,
            n,
            bmm1_scale=1 / math.sqrt(128),
            bmm2_scale=1.0,
            out=out,
        )

    ref = trt()
    torch.testing.assert_close(outputs[1], ref, atol=0.002, rtol=0.02)
    print("trt_max_abs", float((outputs[1] - ref).abs().max()), flush=True)
    if resources[1]["shared"] < 139264 and resources[1]["regs"] < 145:
        funcs["trt"] = trt
        for rep in range(3):
            for name in list(funcs)[:: 1 if rep % 2 == 0 else -1]:
                print(
                    json.dumps(
                        dict(
                            rep=rep,
                            name=name,
                            ms=triton.testing.do_bench_cudagraph(funcs[name], rep=100),
                        )
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
