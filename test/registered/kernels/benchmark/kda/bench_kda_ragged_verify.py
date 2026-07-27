"""KDA ragged-verify dense scatter/gather cost probe.

!! NOT RUN — NO MEASURED RESULTS, AND THE PREMISE IS REFUTED. !!

This bench was written to quantify a predicted "~4 extra full-tensor passes per
KDA layer per verify step". A real TP8 bs=1 DSPARK decode trace
(`k3-profiles/07-24-replay-ssm/nv_cutedsl-bs1-car/...TP-0-DECODE.trace.json.gz`)
refutes that: the per-layer loop contains **exactly one** copy
(`at::native::elementwise_kernel` direct_copy, 68/step, 3.00 us => 204 us/step =
1.1% of a 13.5 ms step), with no `index_copy_` / `where` / `arange` /
`searchsorted` / transpose churn inside it. Those ops appear once per step on the
eager stream, not per layer. And in that config the CuTe path was ACCEPTED —
`cutlass_kda_decode_mtp_kernel` runs 69x/step — so the dense fallback this bench
models is not the executed path there.

It may still be the executed path for a genuinely ragged layout, which is why it
is kept. But it has never been run, so there are no numbers below, and the
"time x 69" framing in the header overstates the cost relative to what the trace
shows. Verify against a trace of the config you care about BEFORE acting on
anything this measures.


Reproduces the data-movement sequence `KDAAttnBackend.forward_target_verify`
performs when `_can_run_dspark_cutedsl_mtp` declines (it always declines for a
ragged layout: `kda_backend.py:876` `if ragged_layout is not None ... return
False`), i.e. `kda_backend.py:756-804` plus the output-zeroing pass at
`:845-850`. The `causal_conv1d_update` call itself is excluded; only the extra
passes the dense layout forces are measured.

Impls:
  dense_scatter  index recompute + new_zeros/index_copy_ scatter + post-conv
                 transpose/reshape copy + new_zeros/copy/gather + torch.where
  indices_only   only `ragged_verify_dense_scatter_indices` (arange +
                 searchsorted + clamp), recomputed per KDA layer today
  fused_triton   one Triton kernel doing scatter + gather + output mask in a
                 single pass, writing the [bs, D, N] conv layout directly

Kimi-K3 has 69 KDA layers and the sequence runs once per layer per verify step,
so the number that matters is `time x 69`:
  per-decode-step dense_scatter cost = dense_scatter(us) * 69
  per-decode-step index recompute    = indices_only(us) * 69
(`indices_only` x 69 is what caching the indices in forward metadata would
save; `dense_scatter - fused_triton` x 69 is the scatter/gather headroom.)

Shapes: D = q_dim + k_dim + v_dim = 3 * num_heads_per_tp * head_dim.
K3 linear_attn_config num_heads=96, head_dim=128, at TP8 -> 12 heads/rank ->
D = 3 * 12 * 128 = 4608. Output tensor is [1, seq_len, 12, head_dim] (DV=1536).

head_dim=128 is the combination the CuTe gate ACCEPTS on shape grounds
(`kda_backend.py:884-889`); head_dim=64 is REJECTED by that same gate. Both
still take the dense path here because the ragged-layout gate fires first.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.attention.linear.kda_backend import (
    ragged_verify_dense_scatter_indices,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=10, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

NUM_HEADS_PER_TP = 12
NUM_KDA_LAYERS = 69


def dense_scatter(mixed_qkv, conv_out, attn_in, query_start_loc, draft_token_num):
    seq_len = mixed_qkv.shape[0]
    batch_size = query_start_loc.shape[0] - 1
    num_dense_tokens = batch_size * draft_token_num
    idx = ragged_verify_dense_scatter_indices(
        query_start_loc=query_start_loc,
        seq_len=seq_len,
        draft_token_num=draft_token_num,
    )
    dense = mixed_qkv.new_zeros(num_dense_tokens + 1, mixed_qkv.shape[-1])
    dense.index_copy_(0, idx, mixed_qkv)
    dense_view = dense[:num_dense_tokens].view(batch_size, draft_token_num, -1)
    dense_bdn = dense_view.transpose(1, 2)
    flat = conv_out.transpose(1, 2).reshape(num_dense_tokens, -1)
    padded_flat = flat.new_zeros(num_dense_tokens + 1, flat.shape[-1])
    padded_flat[:num_dense_tokens] = flat
    gathered = padded_flat[idx]
    covered = idx < num_dense_tokens
    masked = torch.where(covered.view(1, -1, 1, 1), attn_in, 0.0)
    return dense_bdn, gathered, masked


def indices_only(mixed_qkv, conv_out, attn_in, query_start_loc, draft_token_num):
    return ragged_verify_dense_scatter_indices(
        query_start_loc=query_start_loc,
        seq_len=mixed_qkv.shape[0],
        draft_token_num=draft_token_num,
    )


@triton.jit
def _fused_scatter_gather_kernel(
    mixed_qkv_ptr,
    conv_out_ptr,
    attn_in_ptr,
    dense_ptr,
    gathered_ptr,
    attn_out_ptr,
    qsl_ptr,
    batch_size,
    draft_token_num,
    D: tl.constexpr,
    DV: tl.constexpr,
    stride_b,
    stride_d,
    stride_n,
    BLOCK_BS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    t = tl.program_id(0)
    offs_b = tl.arange(0, BLOCK_BS)
    in_range = offs_b < batch_size
    ends = tl.load(qsl_ptr + 1 + offs_b, mask=in_range, other=0)
    slot = tl.sum(((ends <= t) & in_range).to(tl.int32), axis=0)
    step = t - tl.load(qsl_ptr + slot)
    covered = slot * draft_token_num + step < batch_size * draft_token_num
    base = slot * stride_b + step * stride_n

    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        m = cols < D
        x = tl.load(mixed_qkv_ptr + t * D + cols, mask=m, other=0.0)
        tl.store(dense_ptr + base + cols * stride_d, x, mask=m & covered)
        y = tl.load(conv_out_ptr + base + cols * stride_d, mask=m & covered, other=0.0)
        tl.store(gathered_ptr + t * D + cols, y, mask=m)

    for off in range(0, DV, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        m = cols < DV
        v = tl.load(attn_in_ptr + t * DV + cols, mask=m & covered, other=0.0)
        tl.store(attn_out_ptr + t * DV + cols, v, mask=m)


def fused_triton(mixed_qkv, conv_out, attn_in, query_start_loc, draft_token_num):
    seq_len, D = mixed_qkv.shape
    batch_size = query_start_loc.shape[0] - 1
    DV = attn_in.shape[-1] * attn_in.shape[-2]
    dense_bdn = torch.empty_like(conv_out)
    gathered = torch.empty_like(mixed_qkv)
    attn_out = torch.empty_like(attn_in)
    _fused_scatter_gather_kernel[(seq_len,)](
        mixed_qkv,
        conv_out,
        attn_in,
        dense_bdn,
        gathered,
        attn_out,
        query_start_loc,
        batch_size,
        draft_token_num,
        D,
        DV,
        conv_out.stride(0),
        conv_out.stride(1),
        conv_out.stride(2),
        BLOCK_BS=triton.next_power_of_2(batch_size),
        BLOCK_D=512,
        num_warps=4,
    )
    return dense_bdn, gathered, attn_out


FN_MAP = {
    "dense_scatter": dense_scatter,
    "indices_only": indices_only,
    "fused_triton": fused_triton,
}


def make_inputs(bs: int, draft_token_num: int, head_dim: int):
    device = torch.device("cuda")
    D = 3 * NUM_HEADS_PER_TP * head_dim
    seq_len = bs * draft_token_num
    query_start_loc = torch.arange(
        0, seq_len + 1, draft_token_num, device=device, dtype=torch.int32
    )
    mixed_qkv = torch.randn(seq_len, D, device=device, dtype=torch.bfloat16)
    conv_out = torch.randn(bs, D, draft_token_num, device=device, dtype=torch.bfloat16)
    attn_in = torch.randn(
        1, seq_len, NUM_HEADS_PER_TP, head_dim, device=device, dtype=torch.bfloat16
    )
    return mixed_qkv, conv_out, attn_in, query_start_loc


@marker.parametrize("head_dim", [128, 64], [128])
@marker.parametrize("draft_token_num", [2, 3, 5, 7, 8], [7])
@marker.parametrize("bs", [1, 4, 16, 32, 64], [32])
@marker.benchmark("impl", ["dense_scatter", "indices_only", "fused_triton"])
def benchmark(bs: int, draft_token_num: int, head_dim: int, impl: str):
    mixed_qkv, conv_out, attn_in, query_start_loc = make_inputs(
        bs, draft_token_num, head_dim
    )
    return marker.do_bench(
        FN_MAP[impl],
        input_args=(mixed_qkv, conv_out, attn_in, query_start_loc, draft_token_num),
    )


def check_fused_matches_dense():
    mixed_qkv, conv_out, attn_in, query_start_loc = make_inputs(32, 7, 128)
    ref = dense_scatter(mixed_qkv, conv_out, attn_in, query_start_loc, 7)
    got = fused_triton(mixed_qkv, conv_out, attn_in, query_start_loc, 7)
    for name, r, g in zip(("dense", "gathered", "attn_out"), ref, got):
        assert torch.equal(r, g), f"fused_triton mismatch on {name}"


if __name__ == "__main__":
    check_fused_matches_dense()
    benchmark.run()
