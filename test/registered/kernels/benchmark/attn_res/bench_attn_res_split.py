"""A/B the K3 attention-residual PRODUCTION kernel against a source-axis split.

Production dispatch is `attn_residual._aggregate`: on SM100+ with H=7168 it takes
`_aggregate_fast`, the warp-specialized CUDA/TMA kernel `attn_res_fused_tma`,
which fuses score -> online softmax -> mix -> **output RMSNorm** into one launch
and can also snapshot the prefix row into `bank[:, nvb, :]` for free. Everything
else falls back to `out_norm(_mix_fused(...))`, the Triton 2-kernel pipeline.

`attn_residual_split` splits on the *source* axis:

  partial(bank, ...)   -> online-softmax state over bank rows only (no
                          `prefix_sum` dependency, hideable under attention)
  combine(prefix, ...) -> folds the single prefix candidate, emits the mixture

The split does not reduce total work, it moves work off the critical path, so a
plain total-kernel-time microbench cannot show its benefit. Hence the impls
below. Both `_mix_fused` and `split.combine` return the **pre-norm** mixture, and
neither writes the bank, so any comparison against production has to add the
`out_norm` launch and (on block-write layers) the standalone bank-row copy.

  attn_res_tma        -- `_aggregate`: PRODUCTION. CUDA/TMA, out_norm fused.
  attn_res_tma_write  -- `_aggregate(write_bank_row=True)`: production on every
                         `layer_idx % attn_res_block_size == 0` layer; the
                         snapshot rides the score pass.
  sglang_fused_norm   -- `out_norm(_mix_fused(...))`: the Triton fallback. Same
                         function as attn_res_tma.
  split_total_norm    -- `out_norm(partial + combine)`: the split's total work.
                         Same function as attn_res_tma.
  split_critical_norm -- `out_norm(combine)` with `(m, s, acc)` precomputed
                         outside the timed region: what the split leaves on the
                         critical path. Same function as attn_res_tma.
  split_crit_norm_wr  -- `out_norm(combine)` + `bank[:, nvb, :].copy_(prefix)`:
                         same function as attn_res_tma_write.
  *_prenorm           -- the four pre-norm lines of the superseded table below
                         (`sglang_fused`, `split_total`, `split_critical`,
                         `split_dual_total`), kept so the decomposition stays
                         visible. They compute LESS than production: no output
                         norm, no bank write.

Measured GB300 (SM103), GPU 1, bf16, H=7168, marker CUDA-graph, median us.
Bandwidth columns omitted here for width; `_verify_tma_kernel()` profiled one
`_aggregate` call (T=32, nvb=8) and the trace contained exactly one kernel,

  void sglang::attn_res_fused_tma_kernel<
      sglang::KimiK3AttnResTrait<7168l, 8u, 5u, 200u>, 1u>(AttnResTMAParams)

the same symbol the TP8 bs=1 decode trace shows -- so this line really is the
production CUDA/TMA path, and the single launch confirms out_norm is fused in.

check T=1   nvb=1: tma vs out_norm(_mix_fused) rel 2.841e-03 | vs out_norm(split) 2.841e-03
check T=4   nvb=1: tma vs out_norm(_mix_fused) rel 4.608e-03 | vs out_norm(split) 4.608e-03
check T=32  nvb=8: tma vs out_norm(_mix_fused) rel 7.246e-03 | vs out_norm(split) 7.246e-03
check T=128 nvb=4: tma vs out_norm(_mix_fused) rel 6.803e-03 | vs out_norm(split) 6.803e-03

=================================================================================================================================================================================================================================================================================
         nvb  num_tokens |   attn_res_tma(us)  attn_res_tma_write(us)  sglang_fused_norm(us)  split_total_norm(us)  split_critical_norm(us)  split_crit_norm_wr(us)  sglang_fused_prenorm(us)  split_total_prenorm(us)  split_critical_prenorm(us)  split_dual_total_prenorm(us)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
0          1           1 |             2.6230                  2.7046                11.5414               10.2986                   7.0054                  8.4390                    8.7264                   7.7530                      4.4006                       11.6749
1          1           2 |             2.6230                  2.7050                11.6134               10.3642                   7.0771                  9.9341                    8.7872                   7.8147                      4.4867                       11.7360
2          1           4 |             2.6227                  2.7046                11.7981               10.3648                   7.1184                 10.0266                    9.0122                   7.8102                      4.5274                       12.0029
3          1           8 |             2.6227                  2.7251                11.9210               10.4051                   7.1392                 10.1594                    9.0845                   7.8554                      4.5683                       12.0435
4          1          16 |             2.6842                  2.7971                12.0438               10.4362                   7.2208                 10.3030                    9.2483                   7.8557                      4.6464                       12.0947
5          1          32 |             2.7661                  2.8486                12.1664               10.4666                   7.3027                 10.3846                    9.3667                   7.9376                      4.7120                       12.0230
6          1          64 |             2.8074                  2.9302                12.3411               10.6000                   7.3638                 10.4358                    9.5139                   8.0394                      4.7731                       12.2790
7          1         128 |             3.3808                  3.6266                12.4739               10.6202                   7.5789                 11.0125                    9.7398                   8.1114                      4.8963                       12.3712
8          2           1 |             2.8486                  2.9098                12.3101               11.3475                   7.0160                  8.4598                    9.4938                   8.7978                      4.3843                       13.2518
9          2           2 |             2.8275                  2.9098                12.4941               11.3882                   7.0573                  9.9344                    9.6883                   8.7981                      4.4867                       13.2931
10         2           4 |             2.8074                  2.9094                12.6790               11.3885                   7.1386                 10.0163                    9.8621                   8.8486                      4.5478                       13.5693
11         2           8 |             2.8480                  2.9402                12.7808               11.3680                   7.1798                 10.2006                    9.9651                   8.8182                      4.5686                       13.5488
12         2          16 |             2.8890                  2.9917                12.9347               11.4496                   7.2416                 10.2822                   10.1293                   8.9206                      4.6298                       13.6822
13         2          32 |             2.9712                  3.0528                13.0778               11.5216                   7.3027                 10.3645                   10.2925                   8.9616                      4.7347                       13.6093
14         2          64 |             2.9914                  3.1552                13.2314               11.6032                   7.4045                 10.4563                   10.4771                   9.0637                      4.7632                       13.8765
15         2         128 |             3.4013                  3.5740                13.6714               11.7261                   7.5581                 11.0375                   10.8416                   9.2275                      4.8963                       14.0403
16         4           1 |             3.4214                  3.5034                12.8422               14.2960                   7.0058                  8.4189                   10.0154                  11.7261                      4.3843                       16.9590
17         4           2 |             3.3808                  3.4630                13.0480               14.3376                   7.0566                  9.9549                   10.2522                  11.7875                      4.4659                       16.9997
18         4           4 |             3.4010                  3.4832                13.1907               14.3648                   7.1184                 10.0163                   10.3546                  11.8080                      4.5274                       17.3238
19         4           8 |             3.4218                  3.5136                13.4259               14.4397                   7.1549                 10.1392                   10.5795                  11.8899                      4.5482                       17.3376
20         4          16 |             3.4627                  3.5648                13.5798               14.5014                   7.2410                 10.2621                   10.7331                  11.9578                      4.6298                       17.5117
21         4          32 |             3.5034                  3.6058                13.6819               14.4806                   7.3235                 10.3642                   10.8864                  11.9517                      4.7117                       17.3485
22         4          64 |             3.5341                  3.7082                13.8666               14.4704                   7.3949                 10.5113                   11.1837                  12.0128                      4.7734                       17.6141
23         4         128 |             4.2470                  4.5034                14.9308               14.8693                   7.5530                 11.0018                   12.0979                  12.2274                      4.8960                       17.8598
24         8           1 |             3.3600                  3.4013                13.9584               20.1536                   7.0058                  8.4394                   11.1530                  17.5894                      4.4250                       24.2602
25         8           2 |             3.3194                  3.3600                14.1942               20.1744                   7.0566                  9.9347                   11.3782                  17.6240                      4.4864                       24.3110
26         8           4 |             3.3194                  3.3603                14.3779               20.2768                   7.1386                 10.0362                   11.5622                  17.7472                      4.5277                       24.6694
27         8           8 |             3.3395                  3.3907                14.5424               20.4714                   7.1594                 10.1802                   11.7062                  17.9523                      4.5683                       24.8029
28         8          16 |             3.3805                  3.4317                14.6342               20.4198                   7.2307                 10.2720                   11.8285                  17.8701                      4.6502                       24.7821
29         8          32 |             3.4422                  3.5242                14.6854               20.2765                   7.2982                 10.3642                   12.0538                  17.8291                      4.7325                       24.6070
30         8          64 |             3.6067                  3.7626                15.5497               20.4322                   7.3638                 10.5252                   12.7400                  17.8520                      4.7533                       24.8145
31         8         128 |             5.0257                  5.3549                18.4332               20.9096                   7.5891                 11.0410                   15.6071                  18.2901                      4.8960                       25.2676
=================================================================================================================================================================================================================================================================================

VERDICT: the source-axis split retains NO critical-path value. `split_critical_norm`
is flat at 7.01-7.58 us -- the split's own signature, one candidate folded
regardless of bank depth -- but the ENTIRE production kernel costs 2.62-5.03 us.
The split's residue is slower than the whole thing it was meant to shorten, at
every (nvb, T) in the sweep:

  split_critical_norm / attn_res_tma   (>1 means the split loses)
    nvb=1: 2.67x (T=1) ... 2.24x (T=128)
    nvb=2: 2.46x (T=1) ... 2.22x (T=128)
    nvb=4: 2.05x (T=1) ... 1.78x (T=128)
    nvb=8: 2.09x (T=1) ... 1.51x (T=128)   <- best case for the split, still 1.5x behind

  split_crit_norm_wr / attn_res_tma_write  (block-write layers)
    2.06x - 3.73x across the sweep; worst at nvb=1, best at nvb=8/T=128.

The `2.54x win` in the superseded table was an artifact of comparing a pre-norm
Triton residue against a pre-norm Triton monolith. Adding the two omitted costs
kills it twice over:
  - out_norm as a separate launch: +2.5 us on the split residue
    (split_critical_prenorm 4.40-4.90 -> split_critical_norm 7.01-7.58), a cost
    the TMA kernel pays zero for.
  - the bank snapshot: fused into the TMA score pass for +0.04-0.33 us, versus
    +1.41-3.48 us for the standalone `bank[:, nvb, :].copy_(prefix)` the split
    would need. A 10-40x difference on that one row.

CONFOUND, stated plainly: `split_critical` is Triton, `attn_res_tma` is CUDA/TMA
compiled with --use_fast_math, so the ratios above mix "split vs monolithic" with
"Triton vs TMA". Separating the axes does not rescue the split:
  - Within Triton the split still works: split_critical_norm beats
    sglang_fused_norm by 1.6-2.4x. The idea is not numerically wrong.
  - But the whole Triton family is 3.5-4.6x off the production kernel
    (sglang_fused_norm / attn_res_tma), and a hypothetical TMA-implemented split
    has no headroom to close a 1.5-2.7x gap: `combine` must read `acc` as fp32
    [T, H], i.e. two bf16 rows' worth of bytes, so at nvb=1 its critical path
    moves MORE bytes than the entire fused aggregation (prefix + 1 bank row) and
    exactly as many at nvb=2. Only nvb>=4 reads fewer bytes, and there the whole
    family is latency-bound anyway (both lines are flat in T up to 64).
  - Even a real win would be immaterial: the TP8 trace has AttnRes ~84% hidden
    behind the layer pipeline (~225 us exposed per 13.5 ms step), so the ceiling
    on ALL AttnRes work is ~1.7% of the step.

The split is DEAD. Do not carry `attn_residual_split` onto the critical path.

Ways these numbers could still mislead:
  - `split_critical*` gets `(m, s, acc)` free, which flatters the split: it
    assumes `partial` hides perfectly under attention. It still loses.
  - `split_dual_total_prenorm` is pre-norm only; its post-norm variant was not
    measured, but it already loses to 2x `sglang_fused_prenorm` at nvb=8.
  - GB/s columns are not comparable across lines (different byte accounting for
    the write variants); read the us columns.
  - Single GPU, no concurrent kernels: the marker CUDA-graph loop measures each
    line in isolation, so the PDL overlap the TMA kernel is tuned for (see
    `_TMA_BEST_CONFIG`) has no neighbor to overlap with here.

--------------------------------------------------------------------------------
SUPERSEDED, kept as the Triton-only comparison. This is the same sweep with only
the four pre-norm Triton lines, whose column names map to the `*_prenorm` columns
above. Its `split_critical` 2.54x headline is what the out_norm + bank-write
accounting above overturns.

check T=1 nvb=1: rel err 0.000e+00
check T=4 nvb=1: rel err 2.101e-03
check T=32 nvb=8: rel err 7.102e-04
check T=128 nvb=4: rel err 2.222e-03
         nvb  num_tokens |   sglang_fused(us)  split_total(us)  split_critical(us)  split_dual_total(us) |   sglang_fused(GB/s)  split_total(GB/s)  split_critical(GB/s)  split_dual_total(GB/s)
0          1           1 |             8.6848           7.3744              4.3946               11.5315 |                 7.69               9.05                 18.23                    9.26
1          1           2 |             8.7872           7.4768              4.4659               11.6749 |                12.16              14.29                 29.90                   13.72
2          1           4 |             9.0019           7.4563              4.4970               11.3472 |                20.76              25.07                 53.45                   23.53
3          1           8 |             9.0845           7.7536              4.5478               11.8349 |                38.21              44.77                 99.83                   40.61
4          1          16 |             9.2586           7.7731              4.6301               11.8182 |                72.10              85.88                190.35                   76.82
5          1          32 |             9.3606           7.8454              4.7117               12.0029 |               139.78             166.78                368.43                  146.83
6          1          64 |             9.5104           7.9370              4.7590               12.0643 |               272.35             326.34                723.92                  287.74
7          1         128 |             9.7501           8.0915              4.8960               12.2688 |               528.58             636.92               1401.88                  561.53
8          2           1 |             9.5651           8.4042              4.3843               13.1491 |                 8.38               9.53                 18.27                    9.14
9          2           2 |             9.8726           8.4803              4.4762               13.2112 |                13.52              15.74                 29.83                   14.15
10         2           4 |             9.7907           8.5213              4.5072               12.9139 |                24.55              28.20                 53.33                   24.81
11         2           8 |             9.9344           8.7360              4.5379               13.4262 |                45.69              51.96                100.05                   43.75
12         2          16 |            10.1286           8.8080              4.6298               13.4570 |                87.00             100.04                190.36                   83.34
13         2          32 |            10.2822           8.9165              4.7120               13.6205 |               168.80             194.66                368.41                  160.76
14         2          64 |            10.4870           8.9715              4.7629               13.6518 |               328.47             383.96                723.33                  316.87
15         2         128 |            10.9274           9.1763              4.8963               13.9075 |               628.02             747.86               1401.79                  618.25
16         4           1 |            10.1802          11.3677              4.3840               16.8768 |                10.49               9.40                 18.27                    8.70
17         4           2 |            10.2826          11.4499              4.4758               16.9590 |                18.18              16.33                 29.83                   14.17
18         4           4 |            10.4464          11.4662              4.5171               16.7133 |                33.23              30.27                 53.21                   25.56
19         4           8 |            10.5690          11.7981              4.5482               17.2048 |                63.16              56.58                 99.82                   46.56
20         4          16 |            10.7123          11.8595              4.6093               17.2352 |               122.14             110.33                191.20                   89.86
21         4          32 |            10.9171          11.9347              4.7325               17.3853 |               237.26             217.03                366.81                  175.10
22         4          64 |            11.1837          12.0227              4.7734               17.5939 |               460.82             428.66                721.73                  343.01
23         4         128 |            12.2519          12.1660              4.8963               17.7685 |               839.10             845.03               1401.79                  676.27
24         8           1 |            11.0813          17.1434              4.3734               24.1168 |                14.46               9.35                 18.32                    8.30
25         8           2 |            11.3776          17.2454              4.4864               24.2598 |                25.82              17.03                 29.76                   14.31
26         8           4 |            11.5728          17.3683              4.5078               24.0038 |                48.46              32.29                 53.32                   26.70
27         8           8 |            11.7565          17.8394              4.5338               24.6080 |                93.12              61.37                100.14                   49.92
28         8          16 |            11.7568          17.8390              4.6093               24.6493 |               183.97             121.25                191.20                   97.50
29         8          32 |            11.9821          17.8192              4.7120               24.6080 |               358.80             241.27                368.41                  193.15
30         8          64 |            12.6910          17.8232              4.7632               24.7779 |               675.41             480.92                723.28                  381.50
31         8         128 |            15.6788          18.2633              4.8957               25.1882 |              1091.70             937.21               1401.97                  748.45

Reading the superseded table (pre-norm Triton only, all four lines):
  - `split_critical` is FLAT at 4.37-4.90 us across every nvb and T -- exactly the
    predicted signature, since `combine` folds one candidate regardless of bank depth.
    2.54x under the pre-norm Triton baseline at nvb=8/T=32 (4.71 vs 11.98). That
    baseline is not production.
  - `split_total` BEATS the pre-norm baseline at nvb=1-2 (7.37 vs 8.68 at nvb=1/T=1) and
    only loses at nvb>=4 (17.82 vs 11.98 at nvb=8), i.e. the overhead of splitting is
    nvb-dependent, not a fixed tax.
  - `split_dual_total` LOSES to 2 x baseline at nvb=8 (24.61 vs 23.96): the dual variant
    doubles the online-softmax accumulators (~160 regs/thread) and spills. The
    amortisation of one bank sweep does not pay for the spill.
  - Correctness max rel err 2.4e-3, split vs `_mix_fused` (both pre-norm, Triton).
"""

import types

import torch

from sglang.kernels.jit.benchmark import marker
from sglang.kernels.jit.benchmark.utils import create_random
from sglang.srt.layers import attn_residual_split as split
from sglang.srt.layers.attn_residual import _aggregate, _mix_fused, _use_fast, get_cw
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=20,
    stage="base-b-kernel-benchmark",
    runner_config="1-gpu-large",
)

_H = 7168
_EPS = 1e-6

_IMPLS = [
    "attn_res_tma",
    "attn_res_tma_write",
    "sglang_fused_norm",
    "split_total_norm",
    "split_critical_norm",
    "split_crit_norm_wr",
    "sglang_fused_prenorm",
    "split_total_prenorm",
    "split_critical_prenorm",
    "split_dual_total_prenorm",
]

_POSTNORM = {
    "attn_res_tma",
    "attn_res_tma_write",
    "sglang_fused_norm",
    "split_total_norm",
    "split_critical_norm",
    "split_crit_norm_wr",
}


def _weights(hidden_size: int):
    """Stand-ins for (score_proj, score_norm) as consumed by get_cw/_mix_fused."""
    proj = types.SimpleNamespace(weight=create_random(1, hidden_size))
    norm = types.SimpleNamespace(
        weight=create_random(hidden_size) * hidden_size**-0.5,
        variance_epsilon=_EPS,
    )
    return proj, norm


def _out_norm(hidden_size: int) -> RMSNorm:
    """Real sglang RMSNorm, as production hands to _aggregate."""
    norm = RMSNorm(hidden_size, eps=_EPS, weight_dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        norm.weight.copy_(create_random(hidden_size) * hidden_size**-0.5)
    return norm


def _inputs(num_tokens: int, nvb: int, bank_rows: int | None = None):
    prefix = create_random(num_tokens, _H)
    bank = create_random(num_tokens, bank_rows or nvb, _H)
    proj, norm = _weights(_H)
    cw = get_cw(proj, norm)
    return prefix, bank, proj, norm, cw


def _split_total(prefix, bank, nvb, cw):
    m, s, acc = split.partial(bank, nvb, cw, _EPS)
    return split.combine(prefix, m, s, acc, cw, _EPS)


def _split_dual_total(prefix, bank, nvb, cw_a, cw_b):
    state_a, state_b = split.partial_dual(bank, nvb, cw_a, cw_b, _EPS)
    out_a = split.combine(prefix, *state_a, cw_a, _EPS)
    out_b = split.combine(prefix, *state_b, cw_b, _EPS)
    return out_a, out_b


def _combine_norm_write(prefix, m, s, acc, cw, out_norm, bank, nvb):
    """What the split must launch to match attn_res_tma_write: combine, then a
    separate out_norm launch, then a separate bank-row copy."""
    normed = out_norm(split.combine(prefix, m, s, acc, cw, _EPS))
    bank[:, nvb, :].copy_(prefix)
    return normed


def _rel_err(got, ref) -> float:
    scale = max(ref.float().abs().max().item(), 1e-6)
    return (got.float() - ref.float()).abs().max().item() / scale


def _assert_close(got, ref, what: str, tol: float) -> float:
    rel = _rel_err(got, ref)
    assert rel <= tol, f"{what}: rel err {rel:.3e} > {tol:.3e}"
    return rel


def _check(num_tokens: int, nvb: int) -> tuple[float, float]:
    """attn_res_tma vs out_norm(_mix_fused) and vs out_norm(partial+combine).

    Tolerance is set from measurement, not guessed: the observed max rel err
    over these shapes is exactly 1/128 = 7.812e-3, one bf16 ULP relative to the
    largest output element (the TMA kernel rounds the mixture to bf16 before the
    output norm for bit-parity with the unfused path, so a single-ULP
    disagreement in the mixture survives the norm; --use_fast_math on the TMA
    module feeds that ULP). tol = 1/64 leaves exactly 2x headroom -- tight
    enough that a real 2-ULP drift trips it.
    """
    prefix, bank, proj, norm, cw = _inputs(num_tokens, nvb)
    out_norm = _out_norm(_H)
    ref_triton = out_norm(_mix_fused(prefix, bank, nvb, proj, norm))
    ref_split = out_norm(_split_total(prefix, bank, nvb, cw))
    got = _aggregate(prefix, bank, nvb, proj, norm, out_norm)
    tag = f"T={num_tokens} nvb={nvb}"
    tol = 1.0 / 64
    return (
        _assert_close(got, ref_triton, f"tma vs out_norm(_mix_fused) {tag}", tol),
        _assert_close(got, ref_split, f"tma vs out_norm(split) {tag}", tol),
    )


def _verify_tma_kernel() -> None:
    """Confirm the CUDA/TMA kernel (not Triton) is what _aggregate launches."""
    assert _use_fast(_H), "TMA path ineligible on this device"
    prefix, bank, proj, norm, cw = _inputs(32, 8)
    out_norm = _out_norm(_H)
    _aggregate(prefix, bank, 8, proj, norm, out_norm)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        _aggregate(prefix, bank, 8, proj, norm, out_norm)
        torch.cuda.synchronize()
    names = [
        e.key
        for e in prof.key_averages()
        if e.device_type == torch.autograd.DeviceType.CUDA and e.self_device_time_total
    ]
    print("kernels launched by _aggregate:")
    for n in names:
        print("   ", n)
    assert any("attn_res_fused_tma" in n for n in names), names


@marker.parametrize("nvb", [1, 2, 4, 8], [8])
@marker.parametrize("num_tokens", [1, 2, 4, 8, 16, 32, 64, 128], [32])
@marker.benchmark("impl", _IMPLS)
def benchmark(num_tokens: int, nvb: int, impl: str):
    if impl in _POSTNORM and not _use_fast(_H):
        # Never silently time a different function than production runs.
        marker.skip("TMA path ineligible (needs SM100+ and H=7168)")

    # The bank-write lines need row nvb to exist; it is written but never read,
    # so the target stays inert across replays.
    bank_rows = nvb + 1 if impl in ("attn_res_tma_write", "split_crit_norm_wr") else nvb
    prefix, bank, proj, norm, cw = _inputs(num_tokens, nvb, bank_rows)
    out_norm = _out_norm(_H)
    # Bytes the kernel actually touches: nvb bank rows, not the padded bank.
    read_bank = bank[:, :nvb, :]

    if impl == "attn_res_tma":
        return marker.do_bench(
            lambda p, b: _aggregate(p, b, nvb, proj, norm, out_norm),
            input_args=(prefix, bank),
            memory_args=(prefix, read_bank, cw, out_norm.weight),
        )

    if impl == "attn_res_tma_write":
        return marker.do_bench(
            lambda p, b: _aggregate(
                p, b, nvb, proj, norm, out_norm, write_bank_row=True
            ),
            input_args=(prefix, bank),
            memory_args=(prefix, read_bank, cw, out_norm.weight),
            extra_memory_args=(prefix,),  # the snapshot store
        )

    if impl == "sglang_fused_norm":
        return marker.do_bench(
            lambda p, b: out_norm(_mix_fused(p, b, nvb, proj, norm)),
            input_args=(prefix, bank),
            memory_args=(prefix, read_bank, cw, out_norm.weight),
        )

    if impl == "split_total_norm":
        return marker.do_bench(
            lambda p, b: out_norm(_split_total(p, b, nvb, cw)),
            input_args=(prefix, bank),
            memory_args=(prefix, read_bank, cw, out_norm.weight),
        )

    if impl in ("split_critical_norm", "split_crit_norm_wr"):
        m, s, acc = split.partial(read_bank, nvb, cw, _EPS)
        if impl == "split_critical_norm":
            return marker.do_bench(
                lambda p, m_, s_, a_: out_norm(split.combine(p, m_, s_, a_, cw, _EPS)),
                input_args=(prefix, m, s, acc),
                memory_args=(prefix, m, s, acc, cw, out_norm.weight),
            )
        return marker.do_bench(
            lambda p, m_, s_, a_, bk: _combine_norm_write(
                p, m_, s_, a_, cw, out_norm, bk, nvb
            ),
            input_args=(prefix, m, s, acc, bank),
            memory_args=(prefix, m, s, acc, cw, out_norm.weight),
            extra_memory_args=(prefix,),  # the standalone snapshot copy
        )

    if impl == "sglang_fused_prenorm":
        return marker.do_bench(
            lambda p, b: _mix_fused(p, b, nvb, proj, norm),
            input_args=(prefix, bank),
            memory_args=(prefix, bank, cw),
        )

    if impl == "split_total_prenorm":
        return marker.do_bench(
            lambda p, b: _split_total(p, b, nvb, cw),
            input_args=(prefix, bank),
            memory_args=(prefix, bank, cw),
        )

    if impl == "split_critical_prenorm":
        m, s, acc = split.partial(bank, nvb, cw, _EPS)
        return marker.do_bench(
            lambda p, m_, s_, a_: split.combine(p, m_, s_, a_, cw, _EPS),
            input_args=(prefix, m, s, acc),
            memory_args=(prefix, m, s, acc, cw),
        )

    cw_b = get_cw(*_weights(_H))
    return marker.do_bench(
        lambda p, b: _split_dual_total(p, b, nvb, cw, cw_b),
        input_args=(prefix, bank),
        memory_args=(prefix, bank, cw, cw_b),
    )


if __name__ == "__main__":
    _verify_tma_kernel()
    for _t, _n in ((1, 1), (4, 1), (32, 8), (128, 4)):
        _a, _b = _check(_t, _n)
        print(
            f"check T={_t} nvb={_n}: tma vs out_norm(_mix_fused) rel {_a:.3e}"
            f" | vs out_norm(split) {_b:.3e}"
        )
    benchmark.run()
