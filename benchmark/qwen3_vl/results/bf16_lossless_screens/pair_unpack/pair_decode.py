import argparse
import json
import math
import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _load_kv(P, M, rows, ds, valid, PACKED: tl.constexpr):
    if PACKED:
        pairs = tl.arange(0, 64)
        meta = tl.load(M + rows, mask=valid, other=0).to(tl.uint32)
        compressed = (meta & 256) != 0
        sm = tl.load(
            P.to(tl.pointer_type(tl.uint16)) + rows[:, None] * 128 + pairs[None, :],
            mask=valid[:, None] & compressed[:, None],
            other=0,
        ).to(tl.uint32)
        nib = tl.load(
            P + rows[:, None] * 256 + 128 + pairs[None, :],
            mask=valid[:, None] & compressed[:, None],
            other=0,
        ).to(tl.uint32)
        spread = tl.inline_asm_elementwise(
            "prmt.b32 $0, $1, 0, 0x4140;",
            constraints="=r,r",
            args=[sm],
            dtype=tl.uint32,
            is_pure=True,
            pack=1,
        )
        exps = (meta[:, None] & 255) * 65537 + (nib & 15) + ((nib >> 4) << 16)
        bits = (spread & 0x007F007F) | ((spread & 0x00800080) << 8) | (exps << 7)
        raw = tl.load(
            P.to(tl.pointer_type(tl.uint32)) + rows[:, None] * 64 + pairs[None, :],
            mask=valid[:, None] & (~compressed[:, None]),
            other=0,
        )
        bits = tl.where(compressed[:, None], bits, raw)
        lo = (bits & 65535).to(tl.uint16)
        hi = (bits >> 16).to(tl.uint16)
        return tl.reshape(tl.join(lo, hi), (rows.shape[0], 128)).to(
            tl.bfloat16, bitcast=True
        )
    else:
        return tl.load(
            P + rows[:, None] * 128 + ds[None, :], mask=valid[:, None], other=0
        )


@triton.jit
def _decode(
    Q,
    K,
    V,
    KM,
    VM,
    A,
    M,
    L,
    O,
    LENGTH: tl.constexpr,
    HKV: tl.constexpr,
    G: tl.constexpr,
    PAGE: tl.constexpr,
    SPLITS: tl.constexpr,
    BN: tl.constexpr,
    PACKED: tl.constexpr,
):
    bh = tl.program_id(0)
    sp = tl.program_id(1)
    batch = bh // HKV
    kh = bh % HKV
    ds = tl.arange(0, 128)
    ms = tl.arange(0, 16)
    q = tl.load(
        Q + (batch * HKV * G + kh * G + ms[:, None]) * 128 + ds[None, :],
        mask=ms[:, None] < G,
        other=0,
    )
    acc = tl.full((16, 128), 0, tl.float32)
    mx = tl.full((16,), float("-inf"), tl.float32)
    den = tl.full((16,), 0, tl.float32)
    per = triton.cdiv(LENGTH, SPLITS)
    for off in range(sp * per, tl.minimum((sp + 1) * per, LENGTH), BN):
        ns = off + tl.arange(0, BN)
        valid = (ns < LENGTH) & (ns < (sp + 1) * per)
        rows = (
            ((batch * triton.cdiv(LENGTH, PAGE) + ns // PAGE) * HKV + kh) * PAGE
            + ns % PAGE
        ).to(tl.int64)
        k = _load_kv(K, KM, rows, ds, valid, PACKED)
        scores = tl.dot(q, tl.trans(k)) * 0.127517430824
        scores = tl.where(valid[None, :], scores, float("-inf"))
        newmx = tl.maximum(mx, tl.max(scores, 1))
        alpha = tl.exp2(mx - newmx)
        p = tl.exp2(scores - newmx[:, None])
        den = den * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = _load_kv(V, VM, rows, ds, valid, PACKED)
        acc += tl.dot(p.to(tl.bfloat16), v)
        mx = newmx
    if SPLITS == 1:
        tl.store(
            O + (batch * HKV * G + kh * G + ms[:, None]) * 128 + ds[None, :],
            acc / den[:, None],
            mask=ms[:, None] < G,
        )
    else:
        base = ((batch * HKV + kh) * SPLITS + sp) * G + ms
        tl.store(A + base[:, None] * 128 + ds[None, :], acc, mask=ms[:, None] < G)
        tl.store(M + base, mx, mask=ms < G)
        tl.store(L + base, den, mask=ms < G)


@triton.jit
def _reduce(A, M, L, O, G: tl.constexpr, SPLITS: tl.constexpr, S: tl.constexpr):
    qh = tl.program_id(0)
    bh = qh // G
    gh = qh % G
    ss = tl.arange(0, S)
    ds = tl.arange(0, 128)
    idx = (bh * SPLITS + ss) * G + gh
    mx = tl.load(M + idx, mask=ss < SPLITS, other=float("-inf"))
    den = tl.load(L + idx, mask=ss < SPLITS, other=0)
    a = tl.load(
        A + idx[:, None] * 128 + ds[None, :], mask=ss[:, None] < SPLITS, other=0
    )
    scale = tl.exp2(mx - tl.max(mx, 0))
    val = tl.sum(a * scale[:, None], 0) / tl.sum(den * scale, 0)
    tl.store(O + qh * 128 + ds, val)


def make_fn(q, k, v, km, vm, bs, length, split, bn, warps, packed):
    a = torch.empty((bs, 8, split, 4, 128), device="cuda", dtype=torch.float32)
    m = torch.empty((bs, 8, split, 4), device="cuda", dtype=torch.float32)
    l = torch.empty_like(m)
    o = torch.empty_like(q)

    def fn():
        _decode[(bs * 8, split)](
            q,
            k,
            v,
            km,
            vm,
            a,
            m,
            l,
            o,
            length,
            8,
            4,
            32,
            split,
            bn,
            packed,
            num_warps=warps,
            num_stages=2,
        )
        if split > 1:
            _reduce[(bs * 32,)](a, m, l, o, 4, split, triton.next_power_of_2(split))
        return o

    return fn


def bench(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    return triton.testing.do_bench_cudagraph(fn, rep=500)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--length", type=int, default=8192)
    p.add_argument("--packed", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    torch.manual_seed(42)
    b, n = args.batch, args.length
    q = torch.randn((b, 32, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((b * n // 32, 8, 32, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    import flashinfer

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    blocks = torch.arange(b * n // 32, dtype=torch.int32, device="cuda").view(b, -1)
    lens = torch.full((b,), n, dtype=torch.int32, device="cuda")
    out = torch.empty_like(q)

    def trt():
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache(
            q,
            (k, v),
            workspace,
            blocks,
            lens,
            n,
            bmm1_scale=1 / math.sqrt(128),
            bmm2_scale=1.0,
            kv_layout="HND",
            out=out,
        )

    expected = trt().clone()
    print(
        json.dumps({"backend": "trt", "batch": b, "length": n, "ms": bench(trt)}),
        flush=True,
    )
    km = vm = torch.empty((1,), device="cuda", dtype=torch.uint16)
    if args.packed:
        sys.path.insert(0, "/root/qvl/experiments/lossless-kv")
        from lossless import pack, unpack

        pk, km = pack(k)
        pv, vm = pack(v)
        assert torch.equal(
            unpack(pk, km).view(torch.int16), k.reshape(-1, 128).view(torch.int16)
        )
        assert torch.equal(
            unpack(pv, vm).view(torch.int16), v.reshape(-1, 128).view(torch.int16)
        )
        print(
            json.dumps(
                {
                    "k_compressed": ((km.to(torch.int32) & 256) > 0)
                    .float()
                    .mean()
                    .item(),
                    "v_compressed": ((vm.to(torch.int32) & 256) > 0)
                    .float()
                    .mean()
                    .item(),
                }
            ),
            flush=True,
        )
    configs = [
        (4, 128, 4),
        (8, 128, 4),
        (16, 128, 4),
        (4, 64, 4),
        (8, 64, 4),
        (1, 128, 4),
    ]
    if args.quick:
        configs = configs[:3]
    for packed in [False, True] if args.packed else [False]:
        for split, bn, warps in configs:
            try:
                fn = make_fn(
                    q,
                    pk if packed else k,
                    pv if packed else v,
                    km,
                    vm,
                    b,
                    n,
                    split,
                    bn,
                    warps,
                    packed,
                )
                actual = fn()
                torch.cuda.synchronize()
                err = (actual.float() - expected.float()).abs()
                ma = err.max().item()
                rms = err.square().mean().sqrt().item()
                torch.testing.assert_close(actual, expected, atol=0.003, rtol=0.03)
                ms = bench(fn)
                print(
                    json.dumps(
                        {
                            "backend": "triton",
                            "packed": packed,
                            "batch": b,
                            "length": n,
                            "split": split,
                            "block_n": bn,
                            "warps": warps,
                            "max_abs": ma,
                            "rmse": rms,
                            "ms": ms,
                        }
                    ),
                    flush=True,
                )
            except Exception as e:
                print(
                    json.dumps(
                        {
                            "packed": packed,
                            "split": split,
                            "block_n": bn,
                            "error": str(e),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
