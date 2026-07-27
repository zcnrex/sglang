"""A/B the MLA q_a / kv_a RMSNorm pair: SGLang's two launches vs one grid-y fused kernel.

SGLang today is TWO launches (`deepseek_common/attention_forward_methods/forward_mla.py:296-366`):
under CUDA-graph capture `q_a_layernorm` runs on the main stream and `kv_a_layernorm` on
`alt_stream` plus two stream waits; otherwise two serial `RMSNorm.forward_cuda` calls into
flashinfer `rmsnorm`. Fused single-launch variants exist only on ROCm (aiter).

SGLang's existing fused CUDA qknorm (`jit/csrc/elementwise/qknorm.cuh`) is unusable here:
it needs ONE shared `kHeadDim` from {64,128,256,512,1024}, but MLA needs 1536 (q) + 512 (kv).
So `fused_grid_y` is new -- vLLM's trick of putting the token on grid-x and the *task id*
on grid-y (`vllm/models/common/ops/fused_qk_rmsnorm.py`), two independent norms per launch.

Measured GB300 (SM103), bf16, q=1536 kv=512, marker CUDA-graph, median us:

   num_tokens |   sglang_current(us)  two_launch(us)  fused_grid_y(us) |  sglang_current(GB/s)  two_launch(GB/s)  fused_grid_y(GB/s)
0           1 |               3.5338          3.5542            2.8275 |                  3.24              3.22                4.05
1           8 |               4.1686          4.1894            2.6534 |                 15.56             15.48               24.44
2          32 |               4.2304          4.1894            2.5920 |                 58.61             59.19               95.66
3         128 |               4.3939          4.3734            2.6672 |                223.12            224.17              367.57
4         512 |               4.6912          4.6810            2.8579 |                833.49            835.31             1368.15
5        2048 |               6.1009          6.1011            5.2892 |               2561.73           2561.65            2954.87
6        8192 |              12.2796         12.7823           14.4023 |               5090.07           4889.89            4339.84

Correctness vs an fp32 reference: exact at 1 token; max_abs 6.1e-5 @512; max_abs 0.031 /
max_rel 7.6e-3 @8192 -- same order as `two_launch` (0.016 / 7.8e-3), so the fusion is
numerically equivalent to the current kernel.

Reading it:
  - `fused_grid_y` wins 1-2048 tokens (1.25x @1, 1.63x @32, 1.64x @512, 1.15x @2048) and
    LOSES 0.85x @8192 -- it is one CTA per row with no vectorized multi-row streaming.
  - `sglang_current` ~= `two_launch` everywhere, i.e. the nn.Module wrapper costs nothing
    under graph replay and the gap is pure launch/tail overhead, not Python.
  - CAVEAT: `sglang_current` here is the SERIAL two-norm path. The real decode path under
    capture overlaps both norms on `alt_stream` (`get_is_capture_mode()` is not set in this
    bench), so the in-situ edge is smaller than 1.6x.

int32 stride overflow does NOT transfer from vLLM. vLLM casts strides to int64 because its
post-`q_b_proj` per-head `q_in_stride ~= 24K` (128 heads x 192) wraps int32 past ~87K tokens.
SGLang feeds the *latent* strides (1536 / 512) to flashinfer `RMSNormKernel`, which uses
`uint32_t stride_input` -- wrap threshold 2^32/1536 = 2.79M tokens, unreachable. Safe by
SHAPE, not by type: any future reuse with a >=64K-element row stride wraps. The port keeps
vLLM's `.to(tl.int64)` cast anyway.

Relevance caveat: on a real TP8 bs=1 decode trace the in-graph MLA norms are 48 calls,
~103 us/step summed and ~0 us *exposed*, so this is worth ~0% of the critical path today.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.benchmark import marker
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=6, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

EPS = 1e-6
Q_LORA_RANK = 1536
KV_LORA_RANK = 512


@triton.jit
def _fused_q_kv_rmsnorm_kernel(
    q_ptr,
    q_out_ptr,
    q_weight_ptr,
    q_in_stride,
    q_out_stride,
    kv_ptr,
    kv_out_ptr,
    kv_weight_ptr,
    kv_in_stride,
    kv_out_stride,
    eps,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    pid_task = tl.program_id(1)

    if pid_task == 0:
        SIZE = Q_SIZE
        row_in = q_ptr + token_idx * q_in_stride
        weight_ptr = q_weight_ptr
        row_out = q_out_ptr + token_idx * q_out_stride
    else:
        SIZE = KV_SIZE
        row_in = kv_ptr + token_idx * kv_in_stride
        weight_ptr = kv_weight_ptr
        row_out = kv_out_ptr + token_idx * kv_out_stride

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < SIZE
    x = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / SIZE
    rrms = tl.rsqrt(variance + eps)
    w = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
    y = x * rrms * w
    tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)


def fused_grid_y_qknorm(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert q.ndim == 2 and kv.ndim == 2 and q.shape[0] == kv.shape[0]
    assert q.stride(-1) == 1 and kv.stride(-1) == 1
    q_out = torch.empty_like(q)
    kv_out = torch.empty_like(kv)
    num_tokens = q.shape[0]
    _fused_q_kv_rmsnorm_kernel[(num_tokens, 2)](
        q,
        q_out,
        q_weight,
        q.stride(0),
        q_out.stride(0),
        kv,
        kv_out,
        kv_weight,
        kv.stride(0),
        kv_out.stride(0),
        eps,
        Q_SIZE=q.shape[1],
        KV_SIZE=kv.shape[1],
        BLOCK_SIZE=triton.next_power_of_2(max(q.shape[1], kv.shape[1])),
    )
    return q_out, kv_out


def two_launch_qknorm(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    from sgl_kernel import rmsnorm

    return rmsnorm(q, q_weight, eps), rmsnorm(kv, kv_weight, eps)


def reference_qknorm(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    kv_weight: torch.Tensor,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    def _norm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        var = xf.pow(2).mean(dim=-1, keepdim=True)
        return (xf * torch.rsqrt(var + eps) * w.float()).to(x.dtype)

    return _norm(q, q_weight), _norm(kv, kv_weight)


@marker.parametrize("num_tokens", [1, 8, 32, 128, 512, 2048, 8192], [512])
@marker.benchmark("impl", ["sglang_current", "two_launch", "fused_grid_y"])
def benchmark(num_tokens: int, impl: str):
    dtype = torch.bfloat16
    q = torch.randn(num_tokens, Q_LORA_RANK, device="cuda", dtype=dtype)
    kv = torch.randn(num_tokens, KV_LORA_RANK, device="cuda", dtype=dtype)
    q_norm = RMSNorm(Q_LORA_RANK, eps=EPS).to(device="cuda", dtype=dtype)
    kv_norm = RMSNorm(KV_LORA_RANK, eps=EPS).to(device="cuda", dtype=dtype)
    with torch.no_grad():
        q_norm.weight.copy_(torch.randn(Q_LORA_RANK, device="cuda", dtype=dtype))
        kv_norm.weight.copy_(torch.randn(KV_LORA_RANK, device="cuda", dtype=dtype))
    q_weight = q_norm.weight.data
    kv_weight = kv_norm.weight.data

    expected = reference_qknorm(q, kv, q_weight, kv_weight)
    actual = fused_grid_y_qknorm(q, kv, q_weight, kv_weight)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got.float(), want.float(), rtol=1e-2, atol=1e-3)

    def sglang_current(
        q: torch.Tensor,
        kv: torch.Tensor,
        q_weight: torch.Tensor,
        kv_weight: torch.Tensor,
        eps: float = EPS,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return q_norm(q), kv_norm(kv)

    fn_map = {
        "sglang_current": sglang_current,
        "two_launch": two_launch_qknorm,
        "fused_grid_y": fused_grid_y_qknorm,
    }
    return marker.do_bench(
        fn_map[impl],
        input_args=(q, kv, q_weight, kv_weight),
        memory_args=(q, kv, q_weight, kv_weight),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
