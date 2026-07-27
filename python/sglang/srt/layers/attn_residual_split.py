# SPDX-License-Identifier: Apache-2.0
"""Source-axis split of the K3 attention-residual aggregation.

`attn_residual._mix_fused` splits on the stage axis (score kernel, then combine
kernel); both stages read `prefix_sum`, so the whole N-candidate sweep sits
behind the attention all-reduce. Here the split is on the *source* axis:

  partial(bank, ...)   -> online-softmax state (m, s, acc) over bank rows only
  combine(prefix, ...) -> folds the single prefix candidate, emits the mixture

`partial` has no dependency on `prefix_sum` and can run under attention;
only `combine` stays on the critical path.

The mixture returned by `combine` is numerically equivalent to `_mix_fused`
(same fp32 accumulation, same bf16 output rounding, reassociated softmax).
"""

import torch
import triton
import triton.language as tl

_BLOCK_H: int = 1024


def _tiles(hidden_size: int) -> int:
    return triton.next_power_of_2(triton.cdiv(hidden_size, _BLOCK_H))


@triton.jit
def _partial_kernel(
    bank_ptr,  # [T, NB, H]
    cw_ptr,  # [H] fp32
    m_ptr,  # [T] fp32
    s_ptr,  # [T] fp32
    acc_ptr,  # [T, H] fp32
    NVB,
    eps,
    stride_bm,
    stride_bb,
    stride_am,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs = tl.arange(0, N_TILES)[:, None] * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    mask = offs < H
    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc = tl.zeros([N_TILES, BLOCK_H], tl.float32)
    m_run = float("-inf")
    s_run = 0.0
    for j in range(0, NVB):
        v = tl.load(
            bank_ptr + pid_t * stride_bm + j * stride_bb + offs, mask=mask, other=0.0
        ).to(tl.float32)
        sumsq = tl.sum(tl.sum(v * v, axis=1), axis=0)
        dotv = tl.sum(tl.sum(v * cw, axis=1), axis=0)
        logit = dotv * (1.0 / tl.sqrt(sumsq / H + eps))
        m_new = tl.maximum(m_run, logit)
        corr = tl.exp(m_run - m_new)
        wgt = tl.exp(logit - m_new)
        acc = acc * corr + wgt * v
        s_run = s_run * corr + wgt
        m_run = m_new

    tl.store(acc_ptr + pid_t * stride_am + offs, acc, mask=mask)
    tl.store(m_ptr + pid_t, m_run)
    tl.store(s_ptr + pid_t, s_run)


@triton.jit
def _partial_dual_kernel(
    bank_ptr,
    cw_a_ptr,
    cw_b_ptr,
    m_a_ptr,
    s_a_ptr,
    acc_a_ptr,
    m_b_ptr,
    s_b_ptr,
    acc_b_ptr,
    NVB,
    eps,
    stride_bm,
    stride_bb,
    stride_am,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs = tl.arange(0, N_TILES)[:, None] * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    mask = offs < H
    cw_a = tl.load(cw_a_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    cw_b = tl.load(cw_b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    acc_a = tl.zeros([N_TILES, BLOCK_H], tl.float32)
    acc_b = tl.zeros([N_TILES, BLOCK_H], tl.float32)
    m_a = float("-inf")
    s_a = 0.0
    m_b = float("-inf")
    s_b = 0.0
    for j in range(0, NVB):
        v = tl.load(
            bank_ptr + pid_t * stride_bm + j * stride_bb + offs, mask=mask, other=0.0
        ).to(tl.float32)
        sumsq = tl.sum(tl.sum(v * v, axis=1), axis=0)
        rrms = 1.0 / tl.sqrt(sumsq / H + eps)

        logit_a = tl.sum(tl.sum(v * cw_a, axis=1), axis=0) * rrms
        m_an = tl.maximum(m_a, logit_a)
        corr_a = tl.exp(m_a - m_an)
        wgt_a = tl.exp(logit_a - m_an)
        acc_a = acc_a * corr_a + wgt_a * v
        s_a = s_a * corr_a + wgt_a
        m_a = m_an

        logit_b = tl.sum(tl.sum(v * cw_b, axis=1), axis=0) * rrms
        m_bn = tl.maximum(m_b, logit_b)
        corr_b = tl.exp(m_b - m_bn)
        wgt_b = tl.exp(logit_b - m_bn)
        acc_b = acc_b * corr_b + wgt_b * v
        s_b = s_b * corr_b + wgt_b
        m_b = m_bn

    tl.store(acc_a_ptr + pid_t * stride_am + offs, acc_a, mask=mask)
    tl.store(m_a_ptr + pid_t, m_a)
    tl.store(s_a_ptr + pid_t, s_a)
    tl.store(acc_b_ptr + pid_t * stride_am + offs, acc_b, mask=mask)
    tl.store(m_b_ptr + pid_t, m_b)
    tl.store(s_b_ptr + pid_t, s_b)


@triton.jit
def _combine_kernel(
    prefix_ptr,  # [T, H]
    cw_ptr,  # [H] fp32
    m_ptr,
    s_ptr,
    acc_ptr,  # [T, H] fp32
    out_ptr,  # [T, H]
    eps,
    stride_pm,
    stride_am,
    stride_om,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    N_TILES: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs = tl.arange(0, N_TILES)[:, None] * BLOCK_H + tl.arange(0, BLOCK_H)[None, :]
    mask = offs < H

    v = tl.load(prefix_ptr + pid_t * stride_pm + offs, mask=mask, other=0.0).to(
        tl.float32
    )
    cw = tl.load(cw_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sumsq = tl.sum(tl.sum(v * v, axis=1), axis=0)
    dotv = tl.sum(tl.sum(v * cw, axis=1), axis=0)
    logit_p = dotv * (1.0 / tl.sqrt(sumsq / H + eps))

    m_bank = tl.load(m_ptr + pid_t)
    s_bank = tl.load(s_ptr + pid_t)
    m = tl.maximum(m_bank, logit_p)
    corr = tl.exp(m_bank - m)
    wgt_p = tl.exp(logit_p - m)
    inv_s = 1.0 / (s_bank * corr + wgt_p)

    acc = tl.load(acc_ptr + pid_t * stride_am + offs, mask=mask, other=0.0)
    mix = (acc * corr + wgt_p * v) * inv_s
    tl.store(
        out_ptr + pid_t * stride_om + offs,
        mix.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def _new_state(num_tokens: int, hidden_size: int, device) -> tuple:
    m = torch.empty(num_tokens, dtype=torch.float32, device=device)
    s = torch.empty(num_tokens, dtype=torch.float32, device=device)
    acc = torch.empty((num_tokens, hidden_size), dtype=torch.float32, device=device)
    return m, s, acc


def partial(
    bank: torch.Tensor,
    nvb: int,
    cw: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Online-softmax state over `bank[:, :nvb, :]` only (no prefix dependency)."""
    T, _, H = bank.shape
    m, s, acc = _new_state(T, H, bank.device)
    _partial_kernel[(T,)](
        bank,
        cw,
        m,
        s,
        acc,
        nvb,
        eps,
        bank.stride(0),
        bank.stride(1),
        acc.stride(0),
        H=H,
        BLOCK_H=_BLOCK_H,
        N_TILES=_tiles(H),
        num_warps=8,
    )
    return m, s, acc


def partial_dual(
    bank: torch.Tensor,
    nvb: int,
    cw_a: torch.Tensor,
    cw_b: torch.Tensor,
    eps: float,
) -> tuple[tuple, tuple]:
    """Two partial states from one bank sweep (this layer's side + the next)."""
    T, _, H = bank.shape
    m_a, s_a, acc_a = _new_state(T, H, bank.device)
    m_b, s_b, acc_b = _new_state(T, H, bank.device)
    _partial_dual_kernel[(T,)](
        bank,
        cw_a,
        cw_b,
        m_a,
        s_a,
        acc_a,
        m_b,
        s_b,
        acc_b,
        nvb,
        eps,
        bank.stride(0),
        bank.stride(1),
        acc_a.stride(0),
        H=H,
        BLOCK_H=_BLOCK_H,
        N_TILES=_tiles(H),
        num_warps=8,
    )
    return (m_a, s_a, acc_a), (m_b, s_b, acc_b)


def combine(
    prefix_sum: torch.Tensor,
    m: torch.Tensor,
    s: torch.Tensor,
    acc: torch.Tensor,
    cw: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Fold the prefix candidate in; returns the pre-norm mixture."""
    T, H = prefix_sum.shape
    out = torch.empty_like(prefix_sum)
    _combine_kernel[(T,)](
        prefix_sum,
        cw,
        m,
        s,
        acc,
        out,
        eps,
        prefix_sum.stride(0),
        acc.stride(0),
        out.stride(0),
        H=H,
        BLOCK_H=_BLOCK_H,
        N_TILES=_tiles(H),
        num_warps=8,
    )
    return out
