"""Fused LoRA MoE alignment kernels.

Replaces the 4-op pipeline used by ``_merged_experts_fused_moe_lora_add_impl``:

    1. ``_fused_virtual_topk_ids_kernel``     (virtual topk + token_lora_mask)
    2. ``_align_block_size_jit``              (2 CUDA kernels: align + scatter)
    3. Python ``tight_padded`` slice          (Python-side cdiv arithmetic)
    4. ``_fused_sanitize_expert_ids_kernel``  (out-of-range expert_id -> -1)

with **2** Triton kernels:

    A. ``_count_align_kernel`` (grid=(2,), single-block per pid):
         pid=0: histogram (atomic-add to global counts) + padded prefix scan +
                 ``expert_ids`` fill via marker+cummax (O(VNE + max_blocks)
                 instead of O(VNE * max_blocks)) + writes
                 ``num_tokens_post_padded`` + writes exclusive offsets back
                 into the counts buffer (scatter cursor).
         pid=1: pads ``sorted_token_ids`` with the ``numel`` placeholder.

    B. ``_scatter_kernel`` (multi-block):
         each program processes a chunk of tokens, atomic-adds to the cursor,
         writes the resolved ``sorted_token_ids`` slot.

Virtual topk is computed inline on every token read in both kernels — the
``[T, top_k]`` ``virtual_topk_ids`` buffer is never materialized. Sentinel
rows (``lora_id < 0`` or ``base < 0``) are dropped at histogram time, so
their padding never contributes to ``num_tokens_post_padded`` — the
``tight_padded`` slice and the sanitize sweep both fall out automatically.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


def _next_pow2(x: int) -> int:
    assert x >= 1
    return 1 << (x - 1).bit_length()


@triton.jit
def _max_combine(a, b):
    return tl.maximum(a, b)


# ---------------------------------------------------------------------------
# Kernel A: histogram + scan + expert_ids + pad sorted_token_ids
# ---------------------------------------------------------------------------
@triton.jit
def _count_align_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    cumsum_buffer_ptr,
    marker_buffer_ptr,  # [MAX_BLOCKS_PAD] scratch, pre-zeroed by host
    numel,
    num_experts_for_weight,
    virtual_num_experts,
    max_padded,
    max_blocks,
    top_k: tl.constexpr,
    block_size: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    VNE_PADDED: tl.constexpr,
    MAX_BLOCKS_PAD: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)

    if pid == 1:
        # ---- pad sorted_token_ids with ``numel`` placeholder ----
        offs = tl.arange(0, BLOCK_M)
        for i in range(0, max_padded, BLOCK_M):
            idxs = i + offs
            mask = idxs < max_padded
            tl.store(sorted_token_ids_ptr + idxs, numel, mask=mask)
        return

    offs = tl.arange(0, BLOCK_M)
    bucket_idx = tl.arange(0, VNE_PADDED)
    real_mask = bucket_idx < virtual_num_experts

    # ---- Phase 0: zero the scratch (saves the host-side torch.zeros) ----
    #
    # cumsum_buffer is accumulated into by atomic-add in Phase A; marker_buffer
    # is read by Phase C's cummax at unwritten positions. Both must start at 0.
    tl.store(cumsum_buffer_ptr + bucket_idx, tl.zeros((VNE_PADDED,), dtype=tl.int32))
    marker_idx_init = tl.arange(0, MAX_BLOCKS_PAD)
    tl.store(
        marker_buffer_ptr + marker_idx_init,
        tl.zeros((MAX_BLOCKS_PAD,), dtype=tl.int32),
    )
    tl.debug_barrier()

    # ---- Phase A: histogram (atomic-add) ----
    for i in range(0, numel, BLOCK_M):
        idxs = i + offs
        valid = idxs < numel
        m = idxs // top_k
        base = tl.load(topk_ids_ptr + idxs, mask=valid, other=-1).to(tl.int32)
        lora = tl.load(token_lora_mapping_ptr + m, mask=valid, other=-1).to(tl.int32)
        if SHARED_OUTER:
            vid = lora
        else:
            vid = base + lora * num_experts_for_weight
        bad = (
            (~valid)
            | (lora < 0)
            | (base < 0)
            | (vid < 0)
            | (vid >= virtual_num_experts)
        )
        safe_vid = tl.where(bad, 0, vid)
        tl.atomic_add(cumsum_buffer_ptr + safe_vid, 1, mask=~bad)

    tl.debug_barrier()

    # ---- Phase B: padded prefix-sum ----
    counts = tl.load(cumsum_buffer_ptr + bucket_idx, mask=real_mask, other=0)
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    padded_counts = tl.where(real_mask, padded_counts, 0)
    inclusive = tl.cumsum(padded_counts, axis=0)
    exclusive = inclusive - padded_counts
    total = tl.sum(padded_counts, axis=0)
    tl.store(num_tokens_post_padded_ptr, total)

    # ---- Phase C: expert_ids via marker scatter + cummax ----
    #
    # For each nonempty real expert e, write ``e + 1`` to
    # ``marker[block_starts[e]]``. Empty experts collide on block_starts with
    # the next nonempty expert; they're masked off so only the next expert
    # writes there. After a cumulative-max scan, every position carries the
    # active expert id + 1; subtract 1 to recover the expert id.
    block_starts = exclusive // block_size  # (VNE_PADDED,)
    block_counts_per_e = padded_counts // block_size
    write_mask = real_mask & (block_counts_per_e > 0)
    tl.store(
        marker_buffer_ptr + block_starts,
        (bucket_idx + 1).to(tl.int32),
        mask=write_mask,
    )

    tl.debug_barrier()

    marker_idx = tl.arange(0, MAX_BLOCKS_PAD)
    num_blocks_actual = total // block_size
    marker = tl.load(
        marker_buffer_ptr + marker_idx,
        mask=marker_idx < num_blocks_actual,
        other=0,
    )
    marker_max = tl.associative_scan(marker, axis=0, combine_fn=_max_combine)
    expert_per_block = marker_max - 1
    tl.store(
        expert_ids_ptr + marker_idx,
        expert_per_block,
        mask=marker_idx < num_blocks_actual,
    )

    # ---- Phase D: overwrite cumsum_buffer with exclusive offsets (scatter cursor) ----
    tl.store(cumsum_buffer_ptr + bucket_idx, exclusive, mask=real_mask)


# ---------------------------------------------------------------------------
# Kernel B: scatter sorted_token_ids
# ---------------------------------------------------------------------------
@triton.jit
def _scatter_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    sorted_token_ids_ptr,
    cumsum_buffer_ptr,  # holds exclusive offsets; mutated as running cursor
    numel,
    num_experts_for_weight,
    virtual_num_experts,
    top_k: tl.constexpr,
    SHARED_OUTER: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    base_i = pid * BLOCK_M
    if base_i >= numel:
        return
    offs = tl.arange(0, BLOCK_M)
    idxs = base_i + offs
    valid = idxs < numel
    m = idxs // top_k
    base = tl.load(topk_ids_ptr + idxs, mask=valid, other=-1).to(tl.int32)
    lora = tl.load(token_lora_mapping_ptr + m, mask=valid, other=-1).to(tl.int32)
    if SHARED_OUTER:
        vid = lora
    else:
        vid = base + lora * num_experts_for_weight
    bad = (~valid) | (lora < 0) | (base < 0) | (vid < 0) | (vid >= virtual_num_experts)
    safe_vid = tl.where(bad, 0, vid)
    slot = tl.atomic_add(cumsum_buffer_ptr + safe_vid, 1, mask=~bad)
    tl.store(sorted_token_ids_ptr + slot, idxs.to(tl.int32), mask=~bad)


def lora_moe_align(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    max_loras: int,
    block_size: int,
    shared_outer: bool = False,
    block_m_count: int = 128,
    block_m_scatter: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Fused virtual-topk + align_block_size + tight-slice + sanitize.

    Returns: sorted_token_ids, expert_ids, num_tokens_post_padded,
             virtual_num_experts.
    """
    assert topk_ids.is_cuda, "lora_moe_align requires CUDA tensors"
    assert token_lora_mapping is not None, "segments-mode not supported"
    assert topk_ids.dim() == 2
    T, top_k = topk_ids.shape
    device = topk_ids.device
    numel = T * top_k

    num_experts_for_weight = 1 if shared_outer else num_experts
    virtual_num_experts = num_experts_for_weight * max_loras

    max_nonempty = min(numel, virtual_num_experts)
    if numel == 0:
        max_padded = 0
    else:
        max_padded = (
            triton.cdiv(numel + max_nonempty * (block_size - 1), block_size)
            * block_size
        )
    max_blocks = max_padded // block_size

    sorted_token_ids = torch.empty(max(max_padded, 1), dtype=torch.int32, device=device)
    expert_ids = torch.empty(max(max_blocks, 1), dtype=torch.int32, device=device)
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device=device)

    if numel == 0:
        sorted_token_ids.fill_(0)
        expert_ids.fill_(-1)
        num_tokens_post_padded.zero_()
        return sorted_token_ids, expert_ids, num_tokens_post_padded, virtual_num_experts

    VNE_PADDED = _next_pow2(max(virtual_num_experts, 1))
    MAX_BLOCKS_PAD = _next_pow2(max(max_blocks, 1))

    # Scratch left uninitialized; Kernel A's Phase 0 zeros both regions
    # before any read.
    scratch = torch.empty(VNE_PADDED + MAX_BLOCKS_PAD, dtype=torch.int32, device=device)
    cumsum_buffer = scratch[:VNE_PADDED]
    marker_buffer = scratch[VNE_PADDED:]

    if token_lora_mapping.dtype != torch.int32:
        token_lora_mapping = token_lora_mapping.to(torch.int32)

    _count_align_kernel[(2,)](
        topk_ids,
        token_lora_mapping,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        marker_buffer,
        numel,
        num_experts_for_weight,
        virtual_num_experts,
        max_padded,
        max_blocks,
        top_k=top_k,
        block_size=block_size,
        SHARED_OUTER=shared_outer,
        VNE_PADDED=VNE_PADDED,
        MAX_BLOCKS_PAD=MAX_BLOCKS_PAD,
        BLOCK_M=block_m_count,
        num_warps=4,
    )

    grid = (triton.cdiv(numel, block_m_scatter),)
    _scatter_kernel[grid](
        topk_ids,
        token_lora_mapping,
        sorted_token_ids,
        cumsum_buffer,
        numel,
        num_experts_for_weight,
        virtual_num_experts,
        top_k=top_k,
        SHARED_OUTER=shared_outer,
        BLOCK_M=block_m_scatter,
        num_warps=2,
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded, virtual_num_experts
