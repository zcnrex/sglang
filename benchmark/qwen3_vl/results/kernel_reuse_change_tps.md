# Code changes and measured output throughput

Historical measurements audited against code PR head `d7b4556d6b0cb1ffe3421924bbd86f33d81f1cd8`; file ownership below reflects consolidation commit `b1774d9935`. TPS means output tokens per second. Rows use different preceding implementations, concurrency levels and experiment protocols; gains must not be added or multiplied into an overall claim. This table records historical measurements, not a new benchmark of subsequent file consolidation.

All production paths below are relative to the repository root.

| Change | Production files | Measured effect on TPS | Evidence and limits |
|---|---|---|---|
| Reuse the existing normalization warp kernel; remove the standalone MRoPE kernel/wrapper | `python/sglang/kernels/jit/csrc/elementwise/qknorm.cuh`; `python/sglang/kernels/jit/csrc/elementwise/fused_qknorm_rope.cuh`; `python/sglang/kernels/ops/attention/fused_qknorm_rope.py`; registry and Qwen3 imports | Observed C16 **+0.089%**, C128 **−0.087%** (two same-GPU pairs). C1 **−0.443%**, only N3 smoke | [Serving screen](mrope_reuse_serving/README.md), [model gate](mrope_reuse_model/README.md). C16 pairs have opposite signs. One GPU pair has different M128 down tactics; the other matches. No general speedup or statistical noninferiority claim. |
| Route one-token tails in mixed prefill through decode attention | `python/sglang/srt/layers/attention/trtllm_mha_backend.py` | C128:3201.14→3777.27, **+17.998%** | [Experiment history](../README.md). Single serving sweep; applies to the measured mixed-chunk configuration. |
| Fuse QK normalization and multimodal rotation | `python/sglang/kernels/ops/attention/fused_qknorm_rope.py`; `python/sglang/kernels/jit/csrc/elementwise/fused_qknorm_rope.cuh`; `python/sglang/kernels/jit/csrc/elementwise/qknorm.cuh`; `python/sglang/kernels/ops/attention/__init__.py`; `python/sglang/srt/models/qwen3.py` | C1:375.15→383.81, **+2.308%**; C128:3777.27→3801.87, **+0.651%** | [Experiment history](../README.md). Single sweeps; later cache/QKV fusion rows use different baselines. |
| Extend normalization/rotation fusion with HND cache writes | Same attention files above; `python/sglang/srt/mem_cache/memory_pool.py` | Short same-device screens: C1/C4/C8 average **+1.41%/+1.73%/+1.37%**. Full-sweep ablation: C1 **+0.993%**, C8 **+0.838%**, C128 **+0.183%** | [Cache-fusion artifacts](bf16_norm_rope_cache_fusion/), [experiment history](../README.md). Short C1 screen uses only3 measured requests. Full-sweep rates:379.67→383.44,1870.41→1886.09,3790.59→3797.51. |
| Fuse small-batch QKV GEMM epilogue with norm/rotation/cache write | `python/sglang/kernels/ops/attention/qkv_norm_mrope.py`; `python/sglang/kernels/ops/gemm/dense_bf16_gemm_sm100_splitk_epilogue.py`; `python/sglang/kernels/ops/attention/__init__.py`; `python/sglang/srt/models/qwen3.py` | Geometric paired gains: C1 **+4.856%**, C2 **+4.185%**, C4 **+3.781%**, C8 **+1.071%** | [C1/C2](qkv_small_production_serving/README.md), [C4](qkv_m4_production_serving/README.md), [C8](qkv_integrated_serving/README.md). Two same-GPU pairs per concurrency. Each batch extension has its own preceding baseline. C8 used graph cap8; later C1/C2/C4 use cap128. |
| Use direct BF16 GEMM for single-row gate/up | `python/sglang/srt/layers/quantization/unquant.py` | C1 geometric gain **+1.692%**, four same-GPU pairs | [Promotion/evidence history](../README.md). Candidate390.93–393.82 TPS. Numerical/accuracy limitations remain recorded separately; this is not an equivalence claim. |
| Public M128 gate/up and down GEMM autotuning | `python/sglang/srt/layers/quantization/unquant.py`; `python/sglang/srt/model_executor/runner/flashinfer_autotune.py` | C128 geometric gain **+0.336%**, eight same-GPU pairs; mean3809.82→3822.62 TPS | [Promotion/evidence history](../README.md). Fresh isolated tuning caches; variation across other days or configurations is not established. |
| Public batch4/batch8 vocabulary-projection autotuning | `python/sglang/srt/layers/logits_processor.py`; `python/sglang/srt/model_executor/runner/flashinfer_autotune.py` | C4 **+0.191%**; C8 **+0.318%**, geometric gains from two same-GPU pairs each | [Promotion/evidence history](../README.md). These are separate incremental changes, not the QKV-fusion gains above. |
| Add small-batch split-K tactics and batch16 down-projection tactic | `python/sglang/srt/layers/quantization/unquant.py` | Initial12 tactics: **no isolated serving TPS attribution**. Additional batch16 down tactic: C16 **+0.443%**, two same-GPU pairs | [Promotion/evidence history](../README.md). Initial evidence includes1.26–1.30× selected cold-weight GEMM speed and short model latency; neither is serving TPS. |
| Pack cached-prefix/fresh KV and use eligible FA4 prefill | `python/sglang/srt/layers/attention/trtllm_mha_backend.py`; `python/sglang/kernels/ops/attention/dllm_kv_pack.py`; `python/sglang/kernels/ops/attention/flash_attn/cute/interface.py` | C128 mean paired gain **+0.446%**, four same-GPU pairs, all positive | [Promotion/evidence history](../README.md). Existing GEMM tactics matched within pairs. Mixed median TTFT changes prevent a general latency-improvement claim. |
| Support native HND cache-writer strides | `python/sglang/kernels/ops/kvcache/cache_ops.py`; `python/sglang/srt/mem_cache/memory_pool.py` | **No isolated serving TPS attribution.** Writer64tokens:10.2→1.45µs;8192tokens:86.64→10.18µs | [Experiment history](../README.md). Microkernel timings cannot be converted directly into serving TPS. |
| Tune BF16 SiLU-and-multiply launch shape | `python/sglang/kernels/ops/activation/activation.py`; `python/sglang/kernels/jit/csrc/elementwise/activation.cuh` | Observed short-screen geometric changes: C1 **+0.797%**, C8 **+0.266%**, C128 **+0.209%** | [Activation serving evidence](bf16_activation_serving/README.md). These are not isolated causal gains: public M128 down tactics differed, and lower-concurrency measured M128 execution was not separately instrumented. Two swapped pairs per concurrency; reduced request/warmup counts. |
| Initialize BF16 backend in the one-batch runner | `python/sglang/benchmark/one_batch.py` | **No serving TPS attribution** | [Experiment history](../README.md). Makes benchmark backend initialization match serving; it is benchmark correctness, not an independent serving optimization. |

## Reading the measurements

Kernel speed, fixed-state graph replay, output TPS and TTFT are different metrics. The table uses TPS only where serving measurements exist. It does not reinterpret kernel savings as throughput gains or infer that fewer launches necessarily improve all concurrency levels.

Activation latency fields named `mean_ttft` in raw serving summaries are means. Earlier prose mislabeled some of them as medians; this table deliberately makes no activation median-TTFT claim. The raw field names and arrays take precedence over earlier prose labels.

The final pinned seven-concurrency snapshot in [final sweep evidence](final_qkv_full_sweep/README.md) describes a combined configuration and cannot isolate the contribution of individual rows. Subsequent replacement/consolidation measurements require their own row and source identity.

## Standalone reuse comparison

The shared Triton adaptation is excluded: B128 failed strict equality, and larger batches were slower. The selected CUDA adaptation reuses the existing `fused_qknorm_warp` engine with a default identity epilogue. MRoPE supplies rotation and a cache-write hook after the common store, retaining alias ordering. Both newly added standalone files are deleted; `qkv_norm_mrope.py` remains the distinct projection-fused path.

Selected CUDA adaptation, HND page-32 cache writes, microseconds per layer:

| Batch | Former standalone | Adapted existing kernel | Latency change |
|---:|---:|---:|---:|
| 1 | 1.7078 | 2.1624 | +26.6% |
| 8 | 1.8413 | 2.2831 | +24.0% |
| 16 | 1.9409 | 2.3431 | +20.7% |
| 128 | 3.4325 | 3.7083 | +8.0% |

These timings use 36 disjoint layer buffers and eight opposite-order rounds. No-cache B128 improves 3.2481→3.1367 microseconds; other no-cache points regress. [Complete standalone results](bf16_qk_norm_rotary_cuda_v2/README.md) retain both modes and all batch sizes. All 32 standard and eight alias cases pass bitwise. Existing normalization tests pass 560/560; namespace/dispatch tests pass 122 with one skipped. Matched-tactic model logits pass bitwise at B1/B16/B128. This replacement removes duplicate normalization execution code; it does not improve standalone cache-writing speed.

Normal-serving comparison uses C1 N3/warm1, C16 N40/warm16 and C128 N128/warm128 with nominal 8192-input/1024-output prompts. C16 paired changes are +0.531%/−0.352%; C128 −0.0755%/−0.0986%. Public startup tactics match on GPU7; GPU6 selects M128 down tactic 4 for control and 1 for candidate. All counts are exact and all four short sanity runs score 19/20. This bounded screen suggests a small overall effect, but does not establish equivalence or a general speed improvement.
