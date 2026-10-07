"""Experimental lossless BF16 rows; not a production cache format."""

import torch
import triton
import triton.language as tl


@triton.jit
def pack_kernel(S, P, M, N: tl.constexpr, R: tl.constexpr):
    r = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    d = tl.arange(0, 128)
    bits = tl.load(S + r[:, None] * 128 + d[None, :], r[:, None] < N, other=0).to(
        tl.uint16
    )
    exp = (bits >> 7) & 255
    lo = tl.min(exp, axis=1)
    hi = tl.max(exp, axis=1)
    compressed = (hi - lo <= 15) & (hi < 255)
    sm = ((bits >> 8) & 128) | (bits & 127)
    delta = exp - lo[:, None]
    mate = tl.gather(delta, tl.broadcast_to((d ^ 1)[None, :], (R, 128)), axis=1)
    packed = delta | (mate << 4)
    tl.store(
        P + r[:, None] * 256 + d[None, :], sm, (r[:, None] < N) & compressed[:, None]
    )
    tl.store(
        P + r[:, None] * 256 + 128 + d[None, :] // 2,
        packed,
        (r[:, None] < N) & compressed[:, None] & (d[None, :] % 2 == 0),
    )
    tl.store(
        P + r[:, None] * 256 + 2 * d[None, :],
        bits & 255,
        (r[:, None] < N) & ~compressed[:, None],
    )
    tl.store(
        P + r[:, None] * 256 + 2 * d[None, :] + 1,
        bits >> 8,
        (r[:, None] < N) & ~compressed[:, None],
    )
    tl.store(M + r, tl.where(compressed, lo | 256, 0), r < N)


@triton.jit
def load_bits(P, M, r, d, N: tl.constexpr):
    meta = tl.load(M + r, r < N, other=0).to(tl.uint32)
    comp = (meta & 256) != 0
    sm = tl.load(
        P + r[:, None] * 256 + d[None, :], (r[:, None] < N) & comp[:, None], other=0
    ).to(tl.uint16)
    nib = tl.load(
        P + r[:, None] * 256 + 128 + d[None, :] // 2,
        (r[:, None] < N) & comp[:, None],
        other=0,
    ).to(tl.uint16)
    exponent = (meta[:, None] & 255) + ((nib >> ((d[None, :] % 2) * 4)) & 15)
    decoded = ((sm & 128) << 8) | (exponent << 7) | (sm & 127)
    raw = tl.load(
        P.to(tl.pointer_type(tl.uint16)) + r[:, None] * 128 + d[None, :],
        (r[:, None] < N) & ~comp[:, None],
        other=0,
    )
    return tl.where(comp[:, None], decoded, raw).to(tl.uint16)


@triton.jit
def unpack_kernel(P, M, O, N: tl.constexpr, R: tl.constexpr):
    r = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    d = tl.arange(0, 128)
    bits = load_bits(P, M, r, d, N)
    tl.store(O + r[:, None] * 128 + d[None, :], bits, r[:, None] < N)


@triton.jit
def read_kernel(S, P, M, O, N: tl.constexpr, R: tl.constexpr, PACKED: tl.constexpr):
    r = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    d = tl.arange(0, 128)
    if PACKED:
        bits = load_bits(P, M, r, d, N)
    else:
        bits = tl.load(S + r[:, None] * 128 + d[None, :], r[:, None] < N, other=0).to(
            tl.uint16
        )
    vals = bits.to(tl.bfloat16, bitcast=True).to(tl.float32)
    tl.store(O + r, tl.sum(vals, axis=1), r < N)


def pack(x):
    assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape[-1] == 128
    n = x.numel() // 128
    payload = torch.empty((n, 256), device=x.device, dtype=torch.uint8)
    metadata = torch.empty(n, device=x.device, dtype=torch.uint16)
    pack_kernel[(triton.cdiv(n, 16),)](
        x.view(torch.int16), payload, metadata, n, 16, num_warps=4
    )
    return payload, metadata


def unpack(payload, metadata):
    n = metadata.numel()
    out = torch.empty((n, 128), device=payload.device, dtype=torch.bfloat16)
    unpack_kernel[(triton.cdiv(n, 16),)](
        payload, metadata, out.view(torch.int16), n, 16, num_warps=4
    )
    return out


@triton.jit
def copy_kernel(S, P, M, O, N: tl.constexpr, R: tl.constexpr, PACKED: tl.constexpr):
    r = (tl.program_id(0) * R + tl.arange(0, R)).to(tl.int64)
    d = tl.arange(0, 128)
    if PACKED:
        bits = load_bits(P, M, r, d, N)
    else:
        bits = tl.load(S + r[:, None] * 128 + d[None, :], r[:, None] < N, other=0).to(
            tl.uint16
        )
    tl.store(O + r[:, None] * 128 + d[None, :], bits, r[:, None] < N)
