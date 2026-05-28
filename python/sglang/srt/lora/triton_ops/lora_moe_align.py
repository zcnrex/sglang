"""LoRA MoE alignment kernels.

Replaces the 4-op pipeline used by ``_merged_experts_fused_moe_lora_add_impl``
with **3** Triton kernels:

    A. ``_histogram_kernel`` (multi-block, grid=cdiv(numel, BLOCK_M)):
         each program processes a chunk of tokens, computes the virtual
         expert id inline (``base + lora * num_experts_for_weight``), and
         atomic-adds 1 to ``cumsum_buffer[vid]``.

    B. ``_count_align_kernel`` (grid=(2,), single-block per pid):
         pid=0: padded prefix scan over the histogram + ``expert_ids`` fill
                 via marker+cummax (O(VNE + max_blocks) instead of
                 O(VNE * max_blocks)) + writes ``num_tokens_post_padded`` +
                 writes exclusive offsets back into the counts buffer
                 (scatter cursor).
         pid=1: pads ``sorted_token_ids`` with the ``numel`` placeholder.

    C. ``_scatter_kernel`` (multi-block, grid=cdiv(numel, BLOCK_M)):
         each program processes a chunk of tokens, atomic-adds to the cursor,
         writes the resolved ``sorted_token_ids`` slot.

Virtual topk is computed inline on every token read in A and C — the
``[T, top_k]`` ``virtual_topk_ids`` buffer is never materialized. Sentinel
rows (``lora_id < 0`` or ``base < 0``) are dropped at histogram time, so
their padding never contributes to ``num_tokens_post_padded`` — the
``tight_padded`` slice and the sanitize sweep both fall out automatically.

Kernel B is structurally single-block (the prefix scan and cummax both need
a global view); its work is bounded by ``VNE_PADDED + MAX_BLOCKS_PAD`` so
the 2-block grid is fine. Kernels A and C scale with the GPU because their
O(numel) work is partitioned across SMs.
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
# Kernel A: histogram (multi-block)
# ---------------------------------------------------------------------------
@triton.jit
def _histogram_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    cumsum_buffer_ptr,  # pre-zeroed by host
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
    tl.atomic_add(cumsum_buffer_ptr + safe_vid, 1, mask=~bad)


# ---------------------------------------------------------------------------
# Kernel B: scan + expert_ids + pad sorted_token_ids
# ---------------------------------------------------------------------------
@triton.jit
def _count_align_kernel(
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    cumsum_buffer_ptr,  # populated with histogram counts by _histogram_kernel
    marker_buffer_ptr,
    numel,
    virtual_num_experts,
    max_padded,
    block_size: tl.constexpr,
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

    bucket_idx = tl.arange(0, VNE_PADDED)
    real_mask = bucket_idx < virtual_num_experts

    # ---- Phase 0: zero marker_buffer (read by Phase C's cummax) ----
    marker_idx_init = tl.arange(0, MAX_BLOCKS_PAD)
    tl.store(
        marker_buffer_ptr + marker_idx_init,
        tl.zeros((MAX_BLOCKS_PAD,), dtype=tl.int32),
    )
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
    block_m_hist: int = 256,
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

    # cumsum_buffer needs to start zero (histogram atomic-adds into it);
    # marker_buffer is zeroed inside _count_align_kernel's Phase 0.
    scratch = torch.empty(VNE_PADDED + MAX_BLOCKS_PAD, dtype=torch.int32, device=device)
    cumsum_buffer = scratch[:VNE_PADDED]
    marker_buffer = scratch[VNE_PADDED:]
    cumsum_buffer.zero_()

    if token_lora_mapping.dtype != torch.int32:
        token_lora_mapping = token_lora_mapping.to(torch.int32)

    hist_grid = (triton.cdiv(numel, block_m_hist),)
    _histogram_kernel[hist_grid](
        topk_ids,
        token_lora_mapping,
        cumsum_buffer,
        numel,
        num_experts_for_weight,
        virtual_num_experts,
        top_k=top_k,
        SHARED_OUTER=shared_outer,
        BLOCK_M=block_m_hist,
        num_warps=4,
    )

    _count_align_kernel[(2,)](
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        marker_buffer,
        numel,
        virtual_num_experts,
        max_padded,
        block_size=block_size,
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
