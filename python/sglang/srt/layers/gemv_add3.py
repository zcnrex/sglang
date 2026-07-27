"""Row-per-CTA bf16 GEMV with a fused two-addend epilogue.

``out[m, n] = a[m, n] + x[m] . weight[n] + c[m, n]``: one CTA per output
element, whole weight row in a masked load, one fixed-order reduction (no
split-K, so bitwise-deterministic).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rowcta_gemv_add3_kernel(
    x_ptr,
    w_ptr,
    a_ptr,
    c_ptr,
    out_ptr,
    x_stride,
    a_stride,
    c_stride,
    out_stride,
    K: tl.constexpr,
    BK: tl.constexpr,
):
    n = tl.program_id(0)
    m = tl.program_id(1).to(tl.int64)
    x_row = x_ptr + m * x_stride
    acc = tl.zeros([BK], tl.float32)
    for kb in tl.static_range(0, K, BK):
        offs = kb + tl.arange(0, BK)
        mask = offs < K
        xv = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        wv = tl.load(w_ptr + n.to(tl.int64) * K + offs, mask=mask, other=0.0).to(
            tl.float32
        )
        acc += wv * xv
    av = tl.load(a_ptr + m * a_stride + n).to(tl.float32)
    cv = tl.load(c_ptr + m * c_stride + n).to(tl.float32)
    tl.store(
        out_ptr + m * out_stride + n,
        (av + tl.sum(acc) + cv).to(out_ptr.dtype.element_ty),
    )


def rowcta_gemv_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``a + x @ weight.T + c`` for a decode-shaped ``x`` of ``[M, K]``."""
    m, k = x.shape
    n = weight.shape[0]
    assert weight.shape[1] == k and x.stride(1) == 1 and weight.stride(1) == 1
    assert a.shape == (m, n) and c.shape == (m, n)
    assert a.stride(1) == 1 and c.stride(1) == 1
    if out is None:
        out = torch.empty(m, n, dtype=x.dtype, device=x.device)
    _rowcta_gemv_add3_kernel[(n, m)](
        x,
        weight,
        a,
        c,
        out,
        x.stride(0),
        a.stride(0),
        c.stride(0),
        out.stride(0),
        K=k,
        BK=512,
        num_warps=4,
    )
    return out
