"""A/B the K3 latent up-projection tail: mm+add3 vs one fused row-CTA GEMV+add3.

`rowcta_gemv_add3` ports tokenspeed's `rowcta_gemv_add3`
(`ops/gemm/triton_gemv.py:199-231`): `out[n] = a[n] + x@w[n] + c[n]` in ONE launch, one CTA
per output row, whole weight row in a masked load, dotted against an L2-resident activation.
Deterministic by construction -- one fixed-order reduction per output, NO split-K.

SGLang today is 2 launches (`models/kimi_k3.py:1128-1131`): a cuBLASLt GEMM through
`RowParallelLinear`, then the JIT CUDA `add3` (semantics `bf16(bf16(a+b)+c)`, i.e. double
rounding). Same pair at :972 for the non-fused-norm branch.

Measured GB300 (SM103), bf16, marker CUDA-graph, median us:

      N      K    M |  torch_mm_add3(us)  rowcta_fused(us)  cublas_then_add(us) | torch_mm_add3(GB/s) rowcta_fused(GB/s) cublas_then_add(GB/s)
   7168   3584    1 |            10.8364           11.1092              15.0296 |            4420.12           4311.60               3186.92
   7168   3584    2 |            10.9047           16.6522              15.0494 |            4396.74           2879.20               3185.83
   7168   3584    4 |            10.7971           27.8150              15.0693 |            4449.20           1727.07               3187.85
   7168   3584    8 |            10.7194           50.0706              14.9903 |            4498.89            963.15               3217.10
   7168   3584   16 |            10.7779           94.8078              15.0882 |            4509.15            512.61               3221.02
   3584   7168    1 |            12.5330           11.2363              16.0244 |            3820.70           4261.64               2988.26
   3584   7168    2 |            12.6114           15.9269              15.9653 |            3799.62           3008.65               3001.41
   3584   7168    4 |            12.5918           25.3090              15.7519 |            3810.80           1895.97               3046.30
   3584   7168    8 |            12.6897           43.7800              15.8296 |            3791.95           1099.10               3039.78
   3584   7168   16 |            12.8262           80.8289              16.0049 |            3772.40            598.62               3023.18
   6288   7168    1 |            18.5473           17.4190              22.0456 |            4529.10           4822.47               3810.39
   6288   7168    2 |            18.5662           26.9102              21.9227 |            4527.08           3123.38               3833.97
   6288   7168    4 |            18.6797           44.7825              22.1787 |            4504.77           1879.03               3794.09
   6288   7168    8 |            18.1585           80.3760              21.4770 |            4644.75           1049.34               3927.07
   6288   7168   16 |            18.4240          151.0921              21.7707 |            4598.87            560.78               3891.92

Reading it:
  - At M=1 `rowcta_fused` wins the two shapes tokenspeed claims: 3584x7168 +10.4%
    (11.24 vs 12.53) and 6288x7168 +6.1% (17.42 vs 18.55). It LOSES on 7168x3584
    (-2.5%), matching tokenspeed's own registry note that N=7168 with small K loses
    to cuBLASLt.
  - M>=2 degrades linearly -- one CTA per output *element* -- 8.8x worse at M=16. The M
    axis exists only so the sweep runs; this is not a viable M>1 path.
  - `cublas_then_add` (eager `+`) is 20-40% worse than the add3 fusion, so SGLang's
    existing `add3` is already earning its keep.
  - ACCURACY is the better reason to take this: `rowcta_fused` is BIT-EXACT vs an fp32
    reference at M=1 (max_abs 0.0) on all shapes; the current `torch_mm_add3` path is
    max_abs 0.031 everywhere because add3 double-rounds `bf16(bf16(a+b)+c)`.
  - tokenspeed's "+13-14% over cuBLASLt" is measured against a BARE GEMV. Against
    SGLang's fused mm+add3 the win is 6-10%; against cuBLASLt+eager-adds it looks like
    +30%. Pick the comparison deliberately.

Relevance caveat: production does NOT take the mm+add3 path when the fused tail is
eligible. A real TP8 bs=1 decode trace shows `gemm_ag::gemm_ag_gemv_kernel<3584,7168,6,8>`
(8.15 us) + `gemm_ag::spin_add3_kernel<7168>` (2.45 us), 92 each -- i.e.
`k3_ar_fusion.gemm_ag_up_proj` (column-parallel GEMV + multicast all-gather + add3) is
already live. This bench targets the non-eligible / single-GPU case only.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.ops.elementwise.add3 import add3
from sglang.srt.layers.gemv_add3 import rowcta_gemv_add3
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=8, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)


def torch_mm_add3(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return add3(torch.mm(x, weight.t()), a, c, prefetch_bc=True)


def cublas_then_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.linear(x, weight) + a + c


def reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    a: torch.Tensor,
    c: torch.Tensor,
) -> torch.Tensor:
    return (x.float() @ weight.float().t() + a.float() + c.float()).to(x.dtype)


FN_MAP = {
    "torch_mm_add3": torch_mm_add3,
    "rowcta_fused": rowcta_gemv_add3,
    "cublas_then_add": cublas_then_add,
}


@marker.parametrize("N,K", [(7168, 3584), (3584, 7168), (6288, 7168)], [(3584, 7168)])
@marker.parametrize("M", [1, 2, 4, 8, 16], [1])
@marker.benchmark("impl", ["torch_mm_add3", "rowcta_fused", "cublas_then_add"])
def benchmark(M: int, N: int, K: int, impl: str):
    dtype = torch.bfloat16
    x = torch.randn(M, K, device="cuda", dtype=dtype) / K**0.5
    weight = torch.randn(N, K, device="cuda", dtype=dtype) / K**0.5
    a = torch.randn(M, N, device="cuda", dtype=dtype)
    c = torch.randn(M, N, device="cuda", dtype=dtype)

    expected = reference(x, weight, a, c)
    actual = rowcta_gemv_add3(x, weight, a, c)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-2, atol=1e-3)

    return marker.do_bench(
        FN_MAP[impl],
        input_args=(x, weight, a, c),
        memory_args=(x, weight, a, c),
        memory_output="out",
    )


if __name__ == "__main__":
    benchmark.run()
