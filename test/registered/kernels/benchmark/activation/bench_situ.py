"""Race SiTU (SoftCap-GLU) activation: eager torch vs fused JIT CUDA vs torch.compile.

K3's SwiGLU replacement, `[T, 2*3072] bf16 -> [T, 3072] bf16`:
    gate_out = beta * tanh(gate/beta) * sigmoid(gate)      beta = 4.0
    up_out   = linear_beta * tanh(up/linear_beta)          linear_beta = 25.0
    out      = gate_out * up_out

Measured GB300 (SM103), GPU 1, torch 2.11.0+cu130, marker CUDA-graph, median us:

          d  num_tokens |   torch_eager(us)  sglang_fused(us)  torch_compile(us) |   torch_eager(GB/s)  sglang_fused(GB/s)  torch_compile(GB/s)
0      3072           1 |           23.2362            2.8682             1.5434 |                0.74                5.99                11.12
1      3072           8 |           24.4544            2.8538             1.6573 |                5.62               48.12                82.86
2      3072          32 |           24.8022            2.9709             1.7795 |               22.15              184.90               308.69
3      3072         128 |           26.8093            3.1808             3.1827 |               81.96              690.79               690.37
4      3072         512 |           38.8822            5.2035             7.7600 |              226.04             1689.06              1132.61
5      3072        2048 |           99.8960           12.9989            26.2380 |              351.93             2704.56              1339.90
6      3072        8192 |          462.5243           43.6846           100.3502 |              304.04             3219.10              1401.36

check: max_abs_err 7.812e-3 = 1 bf16 ULP (fp32->bf16 rounding, not algorithmic).

WARNING the `torch_compile` column at T>=512 is a BENCHMARK ARTIFACT. `torch.compile`
defaults to `dynamic=None`, so after the 2nd shape it recompiles DYNAMIC and every later
row times the dynamic kernel. With `dynamic=False`, T=8192 is **36.5 us**, not 100.35 --
i.e. static Inductor still BEATS the fused kernel (36.5 vs 43.7) at the largest shape.
Fix the bench before quoting the right-hand rows.

Reading it:
  - fused beats eager 7.5-10.6x, no crossover.
  - `torch_compile` wins at T<=32 (1.7-1.9x over fused), ties at 128, and per the note
    above also wins at 8192 once compiled statically. The fused kernel's advantage in
    the middle is real; its large-T deficit is a tanh-precision cost, see below.

WHY compile wins at low T -- measured, not inferred:

`torch.compile` emits ONE Inductor pointwise Triton kernel, fp32 intermediates,
**1 element per thread**, no explicit vectorization, PDL off. At T=1: `xnumel=3072`,
grid=(24), block=(128) (XBLOCK=128, 4 warps), 16 regs. The math is IDENTICAL to ours
(accurate `libdevice.tanh`, no fast-math); the only difference is work decomposition.

Ours uses `kMaxVecBytes=32` on Blackwell => `kVecSize=16`, `kBlockSize=256`, so the grid
is 1 / 6 / 24 CTAs at T = 1 / 8 / 32. At T=1 that is **one CTA, 192 active threads
(6 warps)** doing 3072 elements of transcendental math at 32 regs -- no warp- or
instruction-level parallelism to hide MUFU/polynomial latency.

Excess over a real null-kernel floor (0.458 us through the identical `LaunchKernel` +
`__grid_constant__` + tvm-ffi + graph-node path) is **exactly linear in elements per
thread**, ~0.1 us per element per thread:

  variant (T=1)                          bench us   grid  regs
  null kernel (floor, PDL on)               0.458      1     -
  production, 32B vec = 16 el/thr           2.690      1    32
  16B vec (8 el/thr)                        1.810      2    26
  8B vec (4 el/thr)                         1.439      3    21
  4B vec (2 el/thr)                         1.240      6    18
  32B vec, block 128 (grid 2, same el/thr)  2.558      2    32
  32B vec, tanh.approx                      2.296      1    32
  32B vec, no tanh at all                   2.231      1    32
  32B vec, --use_fast_math                  1.588      1    32
  torch_compile (1 el/thr)                  1.336     24    16

Doubling CTAs at constant elements-per-thread changes nothing (2.690 -> 2.558), so this
is TOTAL THREAD COUNT, not CTA or SM count.

  - PDL is NOT the cause -- it is a 0.5 us WIN (2.708 on vs 3.198 off at T=1); the
    profiler shows negative inter-kernel gaps (-1.86 us), i.e. graph nodes genuinely
    overlap. Do not disable it. (Caveat: for a 6144-block null kernel PDL was slower,
    6.7 vs 3.9 us -- PDL is not free at very large grids.)
  - Not fast-math: `_fast_math_flags()` returns [] on sm_100+ ("Blackwell needs
    bit-exact expf"), so we build with accurate `tanhf`. Enabling fast-math alone gives
    2.690 -> 1.588 at T=1 and 44.5 -> 25.3 at T=8192.
  - At T=8192 the kernel is COMPUTE-bound on tanh, not bandwidth-bound: accurate tanhf
    costs 18 of the 44.5 us (tanh.approx 26.8, no-tanh 25.0), against a ~19 us memory
    roofline. So the earlier "~40% of roofline, headroom remains" reading was right about
    the headroom and wrong about the mechanism.

FIX (not yet applied): token-count-adaptive vector width for the act/situ family. Best per
T: <=128 -> 4B vec, 512 -> 8B, 2048 -> 16B, 8192 -> 32B. Heuristic reproducing the optimum:
shrink vec width until `num_tokens * hidden / kVecSize` >~ 152*2048 threads (one
full-occupancy wave). Bit-identical, PDL keeps helping at every width, ~1.4 us/call at
decode shapes. This is SYSTEMIC for `jit/csrc/elementwise/activation.cuh`
(`act_and_mul_kernel`, silu/gelu/gelu_tanh) which has the identical 16-el/thread structure;
the penalty scales with math per element, so cheap-math kernels (`add3`,
`fused_add_rmsnorm`, `qknorm_across_heads`) are not affected the same way. NOTE the fixed
launch cost is only 0.46 us, so "tiny JIT kernels pay ~1 us of launch overhead" is the
WRONG framing -- this is an occupancy/ILP problem.

Relevance caveat: on a real TP8 bs=1 decode trace `situ_and_mul` has ~1.6 us of
*exposed* time (fully hidden on the MoE combine stream), and on Blackwell the
deep_gemm contiguous path already fuses act+fp8-quant
(`DEEPGEMM_SCALE_UE8M0 = DEEPGEMM_BLACKWELL`) -- the eager fallback only fires on
Hopper. The remaining genuinely-eager site is the Triton MoE runner
(`moe_runner/triton_utils/fused_moe.py:663-673`).
"""

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.kernels.ops.kimi_k3.activation import situ_and_mul
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=15, stage="base-b-kernel-benchmark", runner_config="1-gpu-large"
)

SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def situ_eager(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    gate = x[..., :d].float()
    up = x[..., d:].float()
    gate = SITU_BETA * torch.tanh(gate / SITU_BETA) * torch.sigmoid(gate)
    up = SITU_LINEAR_BETA * torch.tanh(up / SITU_LINEAR_BETA)
    return (gate * up).to(x.dtype)


situ_compiled = torch.compile(situ_eager)


def situ_fused(x: torch.Tensor) -> torch.Tensor:
    return situ_and_mul(x, None, SITU_BETA, SITU_LINEAR_BETA)


FN_MAP = {
    "torch_eager": situ_eager,
    "sglang_fused": situ_fused,
    "torch_compile": situ_compiled,
}


@marker.parametrize("d", [3072], [3072])
@marker.parametrize("num_tokens", [1, 8, 32, 128, 512, 2048, 8192], [512])
@marker.benchmark("impl", ["torch_eager", "sglang_fused", "torch_compile"])
def benchmark(d: int, num_tokens: int, impl: str):
    x = create_random(num_tokens, 2 * d, dtype=torch.bfloat16)
    return marker.do_bench(FN_MAP[impl], input_args=(x,))


def check():
    torch.manual_seed(0)
    for num_tokens in (1, 512, 8192):
        x = create_random(num_tokens, 2 * 3072, dtype=torch.bfloat16) * 8.0
        ref = situ_eager(x)
        out = situ_fused(x)
        err = (out.float() - ref.float()).abs().max().item()
        assert torch.allclose(out, ref, atol=8e-3, rtol=1e-2), (num_tokens, err)
        print(f"num_tokens={num_tokens} max_abs_err={err:.3e}")


if __name__ == "__main__":
    check()
    benchmark.run()
