"""Packed-key sigmoid top-k router (Kimi K3 noaux_tc, degenerate grouping).

A single bitonic ``tl.topk`` over a uint64 key that concatenates the
order-preserving bit image of the biased fp32 score with the inverted expert
id replaces the ``topk`` sequential argmax rounds of the generic router.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _float32_to_ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    full = tl.full(bits.shape, 0xFFFFFFFF, tl.uint32)
    return bits ^ tl.where((bits & sign) != 0, full, sign)


@triton.jit
def _packed_key_sigmoid_topk_kernel(
    logits_ptr,
    bias_ptr,
    out_weights_ptr,
    out_ids_ptr,
    routed_scaling_factor,
    stride_lm,
    stride_wm,
    stride_im,
    NUM_EXPERTS: tl.constexpr,
    PADDED_EXPERTS: tl.constexpr,
    TOPK: tl.constexpr,
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
):
    row = tl.program_id(0)
    row_ptr = logits_ptr + row * stride_lm
    expert = tl.arange(0, PADDED_EXPERTS)
    valid = expert < NUM_EXPERTS

    logits = tl.load(row_ptr + expert, mask=valid, other=-float("inf")).to(tl.float32)
    scores = tl.sigmoid(logits)
    bias = tl.load(bias_ptr + expert, mask=valid, other=0.0).to(tl.float32)
    choice = tl.where(valid, scores + bias, -float("inf"))
    choice = tl.where(choice == choice, choice, -1e30)

    packed = (_float32_to_ordered_key(choice).to(tl.uint64) << 32) | (
        PADDED_EXPERTS - expert
    ).to(tl.uint64)
    selected = tl.topk(packed, TOPK, dim=0)
    ids = (PADDED_EXPERTS - (selected & 0xFFFFFFFF).to(tl.int32)).to(tl.int32)

    weights = tl.sigmoid(tl.load(row_ptr + ids).to(tl.float32))
    if RENORMALIZE:
        denom = tl.sum(weights, axis=0)
        weights /= tl.where(denom > 0.0, denom, 1.0)
    if APPLY_SCALE:
        weights *= routed_scaling_factor

    slot = tl.arange(0, TOPK)
    tl.store(out_weights_ptr + row * stride_wm + slot, weights)
    tl.store(out_ids_ptr + row * stride_im + slot, ids)


def packed_key_sigmoid_topk(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    topk: int,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Route ``[M, num_experts]`` router logits to ``topk`` experts.

    Selection ranks ``sigmoid(logit) + bias`` descending with the lower expert
    id winning ties; the combine weight is the unbiased ``sigmoid(logit)``.
    Returns ``(weights [M, topk] fp32, ids [M, topk] int32)`` ordered by
    descending selection score.
    """
    assert logits.dim() == 2 and logits.stride(1) == 1
    assert bias.dim() == 1 and bias.numel() == logits.size(1)
    assert bias.dtype == torch.float32

    num_tokens, num_experts = logits.shape
    weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=logits.device)
    ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=logits.device)

    _packed_key_sigmoid_topk_kernel[(num_tokens,)](
        logits,
        bias,
        weights,
        ids,
        float(routed_scaling_factor),
        logits.stride(0),
        weights.stride(0),
        ids.stride(0),
        NUM_EXPERTS=num_experts,
        PADDED_EXPERTS=triton.next_power_of_2(num_experts),
        TOPK=topk,
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        num_warps=8,
        num_stages=1,
    )
    return weights, ids
