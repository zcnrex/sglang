"""
LoRA Virtual Experts Triton Ops.
"""

import functools
from typing import Any

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.moe_align import moe_align_block_size as jit_moe_align_block_size
from sglang.srt.lora.triton_ops.kernel_utils import lora_kernels_v2_enabled


@triton.jit
def _fused_virtual_topk_ids_kernel(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    seg_indptr_ptr,
    req_to_lora_ptr,
    virtual_topk_ids_ptr,
    token_lora_mask_ptr,
    num_experts_for_weight: tl.constexpr,
    shared_outer: tl.constexpr,
    M,
    top_k: tl.constexpr,
    USE_SEGMENTS: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    WRITE_MASK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses _get_virtual_topk_ids: comparison + clamp + arithmetic into one kernel.

    For each (m, k):
        lora_id = token_lora_mapping[m]
        mask[m] = (lora_id >= 0)
        safe_lora = max(lora_id, 0)
        if shared_outer:
            virtual_topk_ids[m, k] = safe_lora * 1  (= safe_lora)
        else:
            virtual_topk_ids[m, k] = topk_ids[m, k] + safe_lora * num_experts_for_weight
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = M * top_k
    valid = offs < total

    m = offs // top_k
    # k = offs % top_k  # not needed directly

    if USE_SEGMENTS:
        lo = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
        hi = tl.full((BLOCK_SIZE,), NUM_SEGMENTS, dtype=tl.int32)
        while tl.max(lo < hi) != 0:
            mid = (lo + hi) // 2
            seg_end = tl.load(seg_indptr_ptr + mid + 1, mask=valid, other=0)
            move_right = seg_end <= m
            lo = tl.where(move_right, mid + 1, lo)
            hi = tl.where(move_right, hi, mid)
        lora_id = tl.load(
            req_to_lora_ptr + lo,
            mask=valid & (lo < NUM_SEGMENTS),
            other=-1,
        )
    else:
        lora_id = tl.load(token_lora_mapping_ptr + m, mask=valid, other=0)
    safe_lora = tl.maximum(lora_id, 0)

    base = tl.load(topk_ids_ptr + offs, mask=valid, other=0)
    if shared_outer:
        base = tl.zeros((BLOCK_SIZE,), dtype=base.dtype)
    # Preserve negative sentinel topk_ids (e.g. -1 for non-local experts after
    # EP dispatch). Without this, `-1 + safe_lora * num_experts` would land on
    # a real virtual-expert slot belonging to another adapter and trigger OOB
    # loads in downstream LoRA kernels.
    shifted = base + safe_lora * num_experts_for_weight
    result = tl.where(base < 0, base, shifted)
    tl.store(virtual_topk_ids_ptr + offs, result, mask=valid)

    if WRITE_MASK:
        # Write mask once per row (at first k position)
        mask_val = lora_id >= 0
        k = offs % top_k
        is_first_k = k == 0
        tl.store(token_lora_mask_ptr + m, mask_val, mask=valid & is_first_k)


def _fused_virtual_topk_ids(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor | None,
    num_experts: int,
    shared_outer: bool,
    max_loras: int,
    seg_indptr: torch.Tensor | None = None,
    req_to_lora: torch.Tensor | None = None,
    write_mask: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, int]:
    """
    Returns virtual topk_ids, token_lora_mask (None when write_mask=False),
    and virtual_num_experts.
    """
    M, top_k = topk_ids.shape
    device = topk_ids.device
    use_segments = token_lora_mapping is None
    if use_segments:
        assert lora_kernels_v2_enabled()
        assert seg_indptr is not None and req_to_lora is not None
        token_lora_mapping_ptr = topk_ids
        seg_indptr_ptr = seg_indptr
        req_to_lora_ptr = req_to_lora
        num_segments = seg_indptr.shape[0] - 1
    else:
        token_lora_mapping_ptr = token_lora_mapping
        seg_indptr_ptr = topk_ids
        req_to_lora_ptr = topk_ids
        num_segments = 0

    num_experts_for_weight = 1 if shared_outer else num_experts

    virtual_topk_ids = torch.empty_like(topk_ids)
    if write_mask:
        token_lora_mask = torch.empty(M, dtype=torch.bool, device=device)
        mask_ptr = token_lora_mask
    else:
        token_lora_mask = None
        # Pass a non-null dummy pointer; the kernel guards the store on WRITE_MASK.
        mask_ptr = topk_ids

    BLOCK_SIZE = 1024
    grid = ((M * top_k + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_virtual_topk_ids_kernel[grid](
        topk_ids,
        token_lora_mapping_ptr,
        seg_indptr_ptr,
        req_to_lora_ptr,
        virtual_topk_ids,
        mask_ptr,
        num_experts_for_weight,
        shared_outer,
        M,
        top_k,
        use_segments,
        num_segments,
        write_mask,
        BLOCK_SIZE,
    )

    virtual_num_experts = num_experts_for_weight * max_loras
    return virtual_topk_ids, token_lora_mask, virtual_num_experts


@triton.jit
def _fused_sanitize_expert_ids_kernel(
    expert_ids_ptr,
    output_ptr,
    num_virtual_experts,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    valid = offs < N

    eid = tl.load(expert_ids_ptr + offs, mask=valid, other=0)
    result = tl.where(eid < num_virtual_experts, eid, -1)
    tl.store(output_ptr + offs, result, mask=valid)


def fused_sanitize_expert_ids(
    expert_ids: torch.Tensor,
    num_virtual_experts: int,
) -> torch.Tensor:
    """
    Sanitize expert_ids by replacing values >= num_virtual_experts with -1.

    Returns a new tensor with expert_ids >= num_virtual_experts replaced by -1.
    """
    N = expert_ids.numel()
    output = torch.empty_like(expert_ids)

    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    _fused_sanitize_expert_ids_kernel[grid](
        expert_ids,
        output,
        num_virtual_experts,
        N,
        BLOCK_SIZE,
    )
    return output


@triton.jit
def _moe_lora_shrink_splitk_kernel(
    # Pointers
    a_ptr,  # type: ignore  # [num_tokens, K]
    b_ptr,  # type: ignore  # [num_virtual_experts, N, K]
    c_ptr,  # type: ignore  # [num_tokens * top_k, N]  (pre-zeroed when SPLIT_K > 1)
    sorted_token_ids_ptr,  # type: ignore
    expert_ids_ptr,  # type: ignore
    num_tokens_post_padded_ptr,  # type: ignore
    # Dimensions
    N,  # type: ignore
    K,  # type: ignore
    num_valid_tokens,  # type: ignore
    # Strides
    stride_am,  # type: ignore
    stride_ak,  # type: ignore
    stride_be,  # type: ignore
    stride_bn,  # type: ignore
    stride_bk,  # type: ignore
    stride_cm,  # type: ignore
    stride_cn,  # type: ignore
    # Constexprs
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """Split-K grouped GEMM for the LoRA A (shrink) stage with few virtual experts."""
    pid = tl.program_id(0)
    pid_sk = pid % SPLIT_K
    pid_mn = pid // SPLIT_K

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid_mn // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid_mn % num_pid_in_group) % group_size_m)
    pid_n = (pid_mn % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    # Token routing (same pattern as fused_moe_triton_kernels)
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return

    # Pointers
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = pid_sk * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    )

    # Accumulate
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    grid_k = tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)
    for k in range(0, grid_k):
        k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
        k_mask = offs_k[:, None] < k_remaining
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_remaining),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)
        accumulator += tl.dot(a, b.to(a.dtype))
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    # Write output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask, sem="relaxed")


def _invoke_moe_lora_shrink_splitk(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: dict[str, Any],
) -> None:
    """Launch split-K shrink kernel for LoRA A with few virtual experts."""
    N = weight.shape[1]
    K = weight.shape[2]
    BLOCK_SIZE_M = config["BLOCK_SIZE_M"]
    BLOCK_SIZE_N = min(config.get("BLOCK_SIZE_N", 64), max(16, N))
    BLOCK_SIZE_K = config.get("BLOCK_SIZE_K", 64)
    GROUP_SIZE_M = config.get("GROUP_SIZE_M", 1)

    num_m_blocks = triton.cdiv(sorted_token_ids.shape[0], BLOCK_SIZE_M)
    num_n_blocks = triton.cdiv(N, BLOCK_SIZE_N)
    base_grid = num_m_blocks * num_n_blocks
    if config.get("ENABLE_SPLIT_K", True):
        max_split_k = max(1, K // BLOCK_SIZE_K)
        SPLIT_K = min(max_split_k, max(1, 128 // base_grid)) if base_grid < 128 else 1
    else:
        SPLIT_K = 1

    if SPLIT_K > 1:
        output.zero_()

    grid = (SPLIT_K * base_grid,)

    _moe_lora_shrink_splitk_kernel[grid](
        hidden_states,
        weight,
        output,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        topk_ids.numel(),
        hidden_states.stride(0),
        hidden_states.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        output.stride(0),
        output.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        SPLIT_K=SPLIT_K,
        num_warps=config.get("num_warps", 4),
        num_stages=config.get("num_stages", 4),
    )


@triton.jit
def _moe_lora_expand_add_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    add_mask_ptr,
    N,
    K,
    num_valid_tokens,
    stride_am,
    stride_ak,
    stride_be,
    stride_bn,
    stride_bk,
    stride_cm,
    stride_cn,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    ROUTER_TOPK: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    EVEN_K: tl.constexpr,
    EVEN_N: tl.constexpr,
    INLINE_MASK: tl.constexpr,
):
    """Dedicated LoRA-B expand+add (one tile per program).

    Drop-in replacement for the generic ``fused_moe_kernel`` with
    ``FUSE_ADD_TO_OUTPUT=True`` on the LoRA-B expand path. Specialized to the
    LoRA shape (tiny K=r, no quant/bias/swap-ab branches) so the only knob that
    matters for the ragged MoE-LoRA grid - ``BLOCK_SIZE_M`` - can be tuned down
    from the generic kernel's 64 to cut the padding waste over near-empty
    virtual experts (avg ~2 real rows/expert)."""
    pid = tl.program_id(axis=0)

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    num_pid_m = tl.cdiv(num_tokens_post_padded, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if pid >= num_pid_m * num_pid_n:
        return

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid - pid_m * num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        return

    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens
    if INLINE_MASK:
        # add_mask_ptr is token_lora_mapping (int32/int64). Recompute the
        # "this token has an adapter" predicate inline instead of consuming a
        # materialized bool buffer written by _fused_virtual_topk_ids_kernel.
        lora_id = tl.load(
            add_mask_ptr + offs_token // ROUTER_TOPK, mask=token_mask, other=-1
        )
        add_mask = lora_id >= 0
    else:
        add_mask = tl.load(
            add_mask_ptr + offs_token // ROUTER_TOPK, mask=token_mask, other=False
        )
    row_mask = token_mask & add_mask

    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # `a` is the shrink intermediate, already laid out one row per (token, slot).
    a_ptrs = a_ptr + offs_token[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (
        b_ptr
        + off_experts * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    if EVEN_K:
        a = tl.load(a_ptrs, mask=row_mask[:, None], other=0.0)
        if EVEN_N:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=offs_bn[None, :] < N, other=0.0)
    else:
        k_mask = offs_k < K
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        b_mask = k_mask[:, None]
        if not EVEN_N:
            b_mask = b_mask & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    accumulator = tl.dot(a, b.to(a.dtype))

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=row_mask, other=0.0)
        accumulator = accumulator * moe_weight[:, None]
    accumulator = accumulator.to(c_ptr.dtype.element_ty)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_mask = row_mask[:, None] & (offs_cn[None, :] < N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    existing = tl.load(c_ptrs, mask=c_mask, other=0.0)
    tl.store(c_ptrs, existing + accumulator, mask=c_mask)


def invoke_moe_lora_expand_add(
    intermediate: torch.Tensor,  # (num_tokens * top_k, K=r)
    lora_b: torch.Tensor,  # (num_virtual_experts, N, K=r)
    output: torch.Tensor,  # (num_tokens * top_k, N) - added in place
    topk_weights: torch.Tensor,  # (num_tokens, top_k)
    topk_ids: torch.Tensor,  # (num_tokens, top_k)
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    add_output_mask: torch.Tensor | None,  # (num_tokens,) bool, or None
    mul_routed_weight: bool,
    router_topk: int,
    block_size_m: int,
    block_size_n: int = 128,
    group_size_m: int = 1,
    num_warps: int = 4,
    num_stages: int = 4,
    token_lora_mapping: torch.Tensor | None = None,  # used when add_output_mask is None
) -> None:
    """Launch the dedicated LoRA-B expand+add kernel (v2).

    Semantically identical to the generic ``invoke_fused_moe_kernel(...,
    fuse_add_to_output=True)`` call it replaces on the LoRA-B expand path, but
    specialized to the LoRA shape. ``block_size_m`` MUST match the block size
    the routing tensors were aligned with (``moe_align_block_size``).

    ``output`` must already be a 2D ``[num_tokens * top_k, N]`` tensor. The
    caller is responsible for the flatten/slice (so gated ``gate_up_proj_moe``
    can pass a non-contiguous half-slice of the underlying 3D buffer).

    The per-token "has adapter" predicate can be supplied either as a
    materialized bool buffer (``add_output_mask``) or computed inline from
    ``token_lora_mapping`` (``add_output_mask`` left None). The inline path
    skips the bool-buffer alloc and the masked store in
    ``_fused_virtual_topk_ids_kernel``."""
    _, K = intermediate.shape
    _, N, K_b = lora_b.shape
    assert K_b == K, (K_b, K)
    assert output.dim() == 2, output.shape

    inline_mask = add_output_mask is None
    if inline_mask:
        assert token_lora_mapping is not None
        mask_ptr = token_lora_mapping
    else:
        mask_ptr = add_output_mask

    out2d = output
    block_size_n = min(block_size_n, max(16, triton.next_power_of_2(N)))
    block_size_k = max(16, triton.next_power_of_2(K))
    even_k = K % block_size_k == 0
    even_n = N % block_size_n == 0

    grid = lambda META: (
        triton.cdiv(sorted_token_ids.shape[0], META["BLOCK_SIZE_M"])
        * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    _moe_lora_expand_add_kernel[grid](
        intermediate,
        lora_b,
        out2d,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mask_ptr,
        N,
        K,
        topk_ids.numel(),
        intermediate.stride(0),
        intermediate.stride(1),
        lora_b.stride(0),
        lora_b.stride(1),
        lora_b.stride(2),
        out2d.stride(0),
        out2d.stride(1),
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        ROUTER_TOPK=router_topk,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        GROUP_SIZE_M=group_size_m,
        EVEN_K=even_k,
        EVEN_N=even_n,
        INLINE_MASK=inline_mask,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _align_block_size_jit(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CUDA JIT align_block_size for num_experts > 1024 (up to 8191).

    Uses the v2 kernel from moe_align_kernel.cu which supports large expert
    counts via per-thread multi-expert processing and a two-level warp scan,
    replacing the previous pure-PyTorch fallback that had excessive CPU overhead
    from 15+ individual kernel launches and torch.argsort.

    The JIT kernel uses a +1 offset convention: topk_ids are shifted by +1 so
    that the EP sentinel value (-1) maps to bucket 0. The kernel internally
    handles histogram, padded prefix-sum, expert_ids assignment, and token
    scattering in just 2–3 CUDA kernel launches.
    """
    assert num_experts <= 8191, (
        f"_align_block_size_jit supports at most 8191 experts "
        f"(num_moe_experts * max_loras), got {num_experts}"
    )

    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1)
    if flat_topk_ids.dtype == torch.int64:
        flat_topk_ids = flat_topk_ids.to(torch.int32)
    num_total_tokens = flat_topk_ids.numel()

    if num_total_tokens == 0:
        empty = torch.empty(0, dtype=torch.int32, device=device)
        return empty, empty, torch.zeros(1, dtype=torch.int32, device=device)

    # JIT kernel uses +1 offset convention: -1 -> bucket 0 (sentinel),
    # expert i -> bucket i+1. So pass num_experts + 1 as the bucket count.
    jit_num_experts = num_experts + 1

    if num_total_tokens < jit_num_experts:
        max_num_tokens_padded = num_total_tokens * block_size
    else:
        max_num_tokens_padded = num_total_tokens + jit_num_experts * (block_size - 1)

    # Align every sub-buffer offset to a multiple of 4 (VEC_SIZE). The CUDA
    # kernel fills sorted_token_ids with vectorized int4 writes whose last
    # store can spill up to 3 int32s past the logical end. With a fused
    # allocation the spill would corrupt the adjacent sub-buffer.
    _A4 = lambda n: (n + 3) & ~3  # noqa: E731
    max_num_tokens_padded = _A4(max_num_tokens_padded)
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size
    max_num_m_blocks_padded = _A4(max_num_m_blocks)
    num_post_pad_size = _A4(1)  # 1 element, padded to 4
    cumsum_size = _A4(jit_num_experts + 1)

    # Single allocation sliced into 4 views (zero-copy) to avoid
    # per-call Python overhead of 4 separate torch.empty calls.
    total_buf = (
        max_num_tokens_padded
        + max_num_m_blocks_padded
        + num_post_pad_size
        + cumsum_size
    )
    buf = torch.empty(total_buf, dtype=torch.int32, device=device)
    off = 0
    sorted_token_ids = buf[off : off + max_num_tokens_padded]
    off += max_num_tokens_padded
    expert_ids = buf[off : off + max_num_m_blocks]
    off += max_num_m_blocks_padded
    num_tokens_post_padded = buf[off : off + 1]
    off += num_post_pad_size
    cumsum_buffer = buf[off : off + jit_num_experts + 1]

    jit_moe_align_block_size(
        flat_topk_ids,
        jit_num_experts,
        block_size,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        cumsum_buffer,
        True,  # pad_sorted_token_ids
    )

    return sorted_token_ids, expert_ids, num_tokens_post_padded


@torch.compile(dynamic=True)
def _align_block_size_torch(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch align_block_size for num_experts > 1024, compiled via torch.compile.

    Fallback for platforms where the CUDA JIT kernel is unavailable (e.g. AMD/ROCm).

    Out-of-range topk_ids (negative sentinels left by EP dispatch, or virtual-
    expert IDs >= num_experts produced when those sentinels are combined with
    a per-adapter offset) are routed into a dedicated sentinel bucket. Without
    this, indexing ``padded_offsets[sorted_expert_ids]`` would wrap (-1) or
    OOB-read, and the bad expert ids would propagate into the downstream LoRA
    GEMM as real expert slots.
    """
    device = topk_ids.device
    flat_topk_ids = topk_ids.reshape(-1).to(torch.int64)
    num_total_tokens = flat_topk_ids.numel()

    sentinel = num_experts
    valid_mask = (flat_topk_ids >= 0) & (flat_topk_ids < num_experts)
    safe_topk_ids = torch.where(
        valid_mask,
        flat_topk_ids,
        torch.full_like(flat_topk_ids, sentinel),
    )

    bucket_count = num_experts + 1
    max_total_padded_tokens = (
        (num_total_tokens + bucket_count * (block_size - 1) + block_size - 1)
        // block_size
    ) * block_size
    max_num_blocks = max_total_padded_tokens // block_size

    sorted_token_ids = torch.full(
        (max_total_padded_tokens,),
        num_total_tokens,
        dtype=torch.int32,
        device=device,
    )
    expert_ids = torch.full(
        (max_num_blocks,),
        -1,
        dtype=torch.int32,
        device=device,
    )

    if num_total_tokens == 0:
        num_tokens_post_padded = torch.zeros((1,), dtype=torch.int32, device=device)
        return sorted_token_ids, expert_ids, num_tokens_post_padded

    sorted_order = torch.argsort(safe_topk_ids)
    sorted_expert_ids = safe_topk_ids[sorted_order]
    expert_range = torch.arange(bucket_count, device=device, dtype=torch.int64)
    counts_offsets = torch.searchsorted(sorted_expert_ids, expert_range, right=False)
    counts_end = torch.searchsorted(sorted_expert_ids, expert_range, right=True)
    counts = counts_end - counts_offsets
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded_tokens = padded_counts.sum().to(torch.int32).reshape(1)
    padded_offsets = torch.cumsum(padded_counts, dim=0) - padded_counts

    token_ranks = (
        torch.arange(num_total_tokens, device=device, dtype=torch.int64)
        - counts_offsets[sorted_expert_ids]
    )
    output_positions = padded_offsets[sorted_expert_ids] + token_ranks
    sorted_token_ids.scatter_(
        0,
        output_positions.to(torch.int64),
        sorted_order.to(torch.int32),
    )

    block_counts = padded_counts // block_size
    real_block_counts = block_counts.clone()
    real_block_counts[sentinel] = 0
    actual_num_blocks = real_block_counts.sum()

    if max_num_blocks <= 0:
        return sorted_token_ids, expert_ids, total_padded_tokens

    block_offsets = torch.cumsum(real_block_counts, dim=0)
    all_block_positions = torch.arange(max_num_blocks, device=device, dtype=torch.int64)
    assigned_experts = torch.searchsorted(
        block_offsets, all_block_positions, right=True
    ).to(torch.int32)
    expert_ids.copy_(
        torch.where(
            all_block_positions < actual_num_blocks,
            assigned_experts,
            torch.full_like(assigned_experts, -1),
        )
    )

    return sorted_token_ids, expert_ids, total_padded_tokens


def _align_block_size_large(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to the CUDA JIT kernel when available, otherwise fall back to
    the pure-PyTorch torch.compile path (needed on AMD/ROCm or when the JIT
    module fails to load)."""
    try:
        return _align_block_size_jit(topk_ids, block_size, num_experts)
    except Exception:
        return _align_block_size_torch(topk_ids, block_size, num_experts)


def _merged_experts_fused_moe_lora_add_fake(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    return


def _merged_experts_fused_moe_lora_add_impl(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor | None,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
    seg_indptr: torch.Tensor | None = None,
    req_to_lora: torch.Tensor | None = None,
) -> None:
    """
    1. Prepare virtual expert routing metadata from topk_ids + token_lora_mapping * num_experts.
    2. Flatten LoRA weights from [max_loras, num_experts, ...] to [max_loras * num_experts, ...].
    3. Run regular SGLang fused-MoE kernels for LoRA A and LoRA B.
    4. Mask out tokens with token_lora_mapping == -1 on the add path.
    """
    max_loras, _, max_lora_rank, _ = lora_a.shape
    use_v2 = lora_kernels_v2_enabled()
    num_tokens = topk_ids.shape[0]
    input_top_k = 1 if hidden_states.shape[0] == topk_ids.numel() else topk_ids.shape[1]

    def _merge_lora_expert_weight(t: torch.Tensor) -> torch.Tensor:
        # [max_loras, num_experts, x, y] -> [max_loras * num_experts, x, y]
        return t.reshape(t.shape[0] * t.shape[1], t.shape[2], t.shape[3])

    def _get_stage_config(
        weight: torch.Tensor,
        stage_top_k: int,
    ) -> dict[str, Any]:
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_config import (
            get_config_dtype_str,
            try_get_optimal_moe_config,
        )

        config_dtype = get_config_dtype_str(dtype=hidden_states.dtype)
        get_config_func = functools.partial(
            try_get_optimal_moe_config,
            weight.shape,
            weight.shape,
            stage_top_k,
            config_dtype,
        )
        try:
            cfg = get_config_func(num_tokens)
        except ValueError:
            K_dim = weight.shape[2]
            N_dim = weight.shape[1]
            if K_dim >= 1024:
                default_block_k = 256
            elif K_dim >= 64:
                default_block_k = 64
            else:
                default_block_k = max(16, K_dim)
            cfg = {
                "BLOCK_SIZE_M": 64,
                "BLOCK_SIZE_N": min(64, max(16, N_dim)),
                "BLOCK_SIZE_K": min(default_block_k, max(16, K_dim)),
                "GROUP_SIZE_M": 1,
                "num_warps": 4,
                "num_stages": 4,
            }
        return cfg

    def _align_block_size(
        topk_ids: torch.Tensor,
        block_size: int,
        num_experts: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The native align kernel consumes num_experts + 1 internally for its
        # sentinel bucket, so the 1024-expert boundary must use the fallback path.
        if num_experts < 1024:
            from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
                moe_align_block_size as native_moe_align_block_size,
            )

            return native_moe_align_block_size(topk_ids, block_size, num_experts)
        return _align_block_size_large(topk_ids, block_size, num_experts)

    # On v2 the LoRA-B expand+add kernel can recompute the per-token "has
    # adapter" predicate inline from token_lora_mapping, so the bool buffer
    # write in _fused_virtual_topk_ids_kernel is dead work. The segments path
    # (token_lora_mapping is None) still needs materialization because a
    # consumer-side binary search would be much more expensive.
    inline_mask = use_v2 and token_lora_mapping is not None

    def _get_routing(
        topk_ids: torch.Tensor,
        token_lora_mapping: torch.Tensor | None,
        num_experts: int,
        shared_outer: bool,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        # Check routing_cache for cross-call reuse (gate_up and down share routing)
        cache_key = (num_experts, shared_outer, block_size)
        if routing_cache is not None:
            cached = routing_cache.get(cache_key)
            if cached is not None:
                return cached

        virtual_topk_ids, token_lora_mask, virtual_num_experts = (
            _fused_virtual_topk_ids(
                topk_ids,
                token_lora_mapping,
                num_experts,
                shared_outer,
                max_loras,
                seg_indptr,
                req_to_lora,
                write_mask=not inline_mask,
            )
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = _align_block_size(
            virtual_topk_ids,
            block_size=block_size,
            num_experts=virtual_num_experts,
        )
        # _align_block_size uses a worst-case padded allocation. Trim the routing buffers
        # to a tighter upper bound so we keep the real routed work but drop unused padding
        num_tokens = topk_ids.numel()
        max_nonempty = min(num_tokens, virtual_num_experts)
        tight_padded = (
            triton.cdiv(num_tokens + max_nonempty * (block_size - 1), block_size)
            * block_size
        )
        sorted_token_ids = sorted_token_ids[:tight_padded]
        expert_ids = expert_ids[: tight_padded // block_size]
        if not use_v2:
            expert_ids = fused_sanitize_expert_ids(expert_ids, virtual_num_experts)
        result = (
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            token_lora_mask,
        )

        if routing_cache is not None:
            routing_cache[cache_key] = result

        return result

    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
        invoke_fused_moe_kernel,
    )

    lora_a_virtual = _merge_lora_expert_weight(lora_a)
    lora_b_virtual = _merge_lora_expert_weight(lora_b)
    num_experts_a = lora_a.shape[1]
    num_experts_b = lora_b.shape[1]

    # Gated gate_up_proj_moe: lora_a is stacked along its rank dim to produce
    # `2*r` outputs (gate || up halves), while lora_b stays per-half (K_b = r).
    # The shrink writes a single `2*r`-wide intermediate; the expand must then
    # be issued twice on disjoint half-slices of intermediate / lora_b / output
    # so the inner-product K matches lora_b's per-half rank.
    lora_b_rank = lora_b.shape[3]
    is_gated = max_lora_rank != lora_b_rank
    if is_gated:
        assert max_lora_rank == 2 * lora_b_rank, (max_lora_rank, lora_b_rank)

    intermediate = (torch.empty if use_v2 else torch.zeros)(
        [num_tokens, topk_ids.shape[1], max_lora_rank],
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    a_stage_config = _get_stage_config(lora_a_virtual, input_top_k)
    if use_v2:
        # LoRA-A shrink is ragged over virtual experts.  Smaller M tiles reduce
        # padding waste for the bs64/r16 decode shape called out in the kernel
        # analysis while keeping the old config available through the shared
        # SGLANG_LORA_KERNELS_V2=0 switch.
        a_stage_config = {
            **a_stage_config,
            "BLOCK_SIZE_M": 16,
            # Avoid the standalone output.zero_() launch and split-K atomics
            # on the v2 shrink path.
            "ENABLE_SPLIT_K": False,
        }
    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_a,
        experts_shared_outer_loras_a,
        a_stage_config["BLOCK_SIZE_M"],
    )

    _invoke_moe_lora_shrink_splitk(
        hidden_states,
        lora_a_virtual,
        intermediate.view(-1, max_lora_rank),
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        input_top_k,
        a_stage_config,
    )

    b_stage_config = _get_stage_config(lora_b_virtual, 1)
    if use_v2:
        # Match the LoRA-A shrink tile size so _get_routing's cache key
        # (num_experts, shared_outer, block_size) can hit on the B call when
        # the other two keys match (i.e. the standard non-shared-outer case),
        # skipping the second virtual-topk + moe_align launch sequence.
        b_stage_config = {
            **b_stage_config,
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 128,
        }

    (
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        token_lora_mask,
    ) = _get_routing(
        topk_ids,
        token_lora_mapping,
        num_experts_b,
        experts_shared_outer_loras_b,
        b_stage_config["BLOCK_SIZE_M"],
    )

    intermediate_2d = intermediate.view(-1, max_lora_rank)
    # Flatten output to [M*top_k, gate_up_dim] once. invoke_fused_moe_kernel
    # and invoke_moe_lora_expand_add both index via C.stride(-2)/stride(-1),
    # so the kept-stride 2D half-slices below work without copy.
    output_2d = output.view(-1, output.shape[-1])
    if is_gated:
        out_half = lora_b_virtual.shape[1] // 2
        expand_halves = [
            (
                intermediate_2d[:, :lora_b_rank],
                lora_b_virtual[:, :out_half, :],
                output_2d[:, :out_half],
            ),
            (
                intermediate_2d[:, lora_b_rank:],
                lora_b_virtual[:, out_half:, :],
                output_2d[:, out_half:],
            ),
        ]
    else:
        expand_halves = [(intermediate_2d, lora_b_virtual, output_2d)]

    for inter_h, lora_b_h, out_h in expand_halves:
        if use_v2:
            invoke_moe_lora_expand_add(
                inter_h,
                lora_b_h,
                out_h,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                token_lora_mask,
                mul_routed_weight,
                topk_ids.shape[1],
                # Keep the routing alignment and kernel M tile in lock-step.
                block_size_m=b_stage_config["BLOCK_SIZE_M"],
                block_size_n=b_stage_config["BLOCK_SIZE_N"],
                group_size_m=b_stage_config.get("GROUP_SIZE_M", 1),
                num_warps=b_stage_config.get("num_warps", 4),
                num_stages=b_stage_config.get("num_stages", 4),
                token_lora_mapping=token_lora_mapping,
            )
        else:
            invoke_fused_moe_kernel(
                inter_h,
                lora_b_h,
                None,
                out_h,
                None,
                None,
                None,
                topk_weights,
                topk_ids,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                mul_routed_weight,
                1,
                b_stage_config,
                tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16,
                False,
                False,
                False,
                False,
                False,
                None,
                fuse_add_to_output=True,
                add_output_mask=token_lora_mask,
                router_topk=topk_ids.shape[1],
            )


def _merged_experts_fused_moe_lora_add_op(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
) -> None:
    _merged_experts_fused_moe_lora_add_impl(
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
    )


from sglang.srt.utils.common import direct_register_custom_op

direct_register_custom_op(
    op_name="merged_experts_fused_moe_lora_add",
    op_func=_merged_experts_fused_moe_lora_add_op,
    mutates_args=["output"],
    fake_impl=_merged_experts_fused_moe_lora_add_fake,
)


def merged_experts_fused_moe_lora_add(
    output: torch.Tensor,
    hidden_states: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    token_lora_mapping: torch.Tensor | None,
    mul_routed_weight: bool,
    experts_shared_outer_loras_a: bool,
    experts_shared_outer_loras_b: bool,
    routing_cache: dict | None = None,
    seg_indptr: torch.Tensor | None = None,
    req_to_lora: torch.Tensor | None = None,
) -> None:
    """Public API: wraps the registered op with routing_cache support."""
    _merged_experts_fused_moe_lora_add_impl(
        output,
        hidden_states,
        lora_a,
        lora_b,
        topk_ids,
        topk_weights,
        token_lora_mapping,
        mul_routed_weight,
        experts_shared_outer_loras_a,
        experts_shared_outer_loras_b,
        routing_cache,
        seg_indptr,
        req_to_lora,
    )
