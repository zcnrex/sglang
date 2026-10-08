# BF16 Qwen3-VL trace comparison

The traces show fewer SGLang decode kernels and more overlapping kernel intervals, while total summed kernel durations are close. At B128, attention dominates both implementations; fewer preparation kernels alone do not imply a large remaining throughput gain. These are five instrumented decode steps, not an unprofiled serving benchmark, and they do not measure TTFT directly.

Raw traces are losslessly compressed; oversized files use [split archive reconstruction](TRACE_ARCHIVES.md).

[Complete per-step timings, operation counts, exact-name metrics and all unchanged three-table skill reports](report-metrics.md) accompany this report. Timings below are median kernel-duration sums across all five steps, in microseconds. They are attribution, not additive latency savings.

| B | SGLang kernels | vLLM kernels | Attention SG / vLLM | GEMM and reduction SG / vLLM | Preparation SG / vLLM | SiLU × up SG / vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 294 | 510 | 1776 / 1664 | 1391 + 421 fused QKV / 1799 | included in fused QKV / 249 | 118 / 54 |
| 16 | 330 | 510 | 3057 / 3039 | 1779 / 1815 | 149 / 246 | 122 / 57 |
| 128 | 366 | 474 | 22740 / 22781 | 2124 / 2151 | 210 / 296 | 256 / 88 |

Preparation includes the 35 repeated compiled QK/MRoPE kernels, their 35 coefficient-selection kernels, and 36 cache writes on vLLM. First-layer compiled variants remain separately labeled in the detailed table. SG B8 has 36 fused QKV projection/norm/MRoPE/cache kernels; its preparation cannot be timed separately. Both have 36 attention and 36 activation calls. SG has 73 residual/final RMSNorm calls; vLLM has 72 repeated residual RMSNorm calls plus a first-layer/final compiled variant in Other.

## Capture contract and limits

SGLang is exact PR head `c76cfaba1a1154a4a24b8694184cb8d3318fc358`. vLLM is official `0.30.1rc1.dev648+g92044241a`, installed in a reconstructed environment. Both use BF16 model/query/KV, TP1, the same nested 8192-token fixture, and five actual full-batch decode steps after warmup near context 8704. See framework collector metadata for full package locks, arguments, source admission, telemetry and sanity results. This environment comparison is not a single-variable experiment.

Strict CPU `user_annotation` scope → same-thread CUDA launch → device kernel correlation passes for all six decode traces. Each step has graph-launch and graph-ID proof. vLLM runtime descriptors show FULL for B8/B16/B128; its startup PIECEWISE capture messages do not describe these measured steps. SG captured graph keys match actual B8/B16/B128.

B8 and B16 have matching sorted per-request context distributions across frameworks at each step. Initial means are 8704.125 and 8704.0625, respectively; each step increments by one. B128 starts at mean 8704.0078125 on SG versus 8704.5078125 on vLLM, with ranges 8672–8736 versus 8673–8737. Retain this half-token mean mismatch rather than claiming exactly identical contexts.

SG's boundary is `ModelRunner.forward`, including attention/graph metadata, gathers/copies and vocabulary projection. vLLM's selected boundary combines nested decoder and vocabulary projection; its worker bookkeeping and sampling are recorded separately. The decoder-to-projection span includes intervening work/gaps, so it is not pure model latency. The trace records all five vLLM sample phases, while SG's forward-only profiler can include the first four intervening sample phases and omit final sampling. Do not compare raw whole-trace totals as identical cycles.

| vLLM B | Decoder sum | Projection sum | Worker-only sum (11 kernels) | Sample outside projection (9 kernels) |
|---:|---:|---:|---:|---:|
| 8 | 3919 | 128 | 32 | 35 |
| 16 | 5298 | 129 | 38 | 37 |
| 128 | 25469 | 136 | 40 | 65 |

Sample outside projection includes one pre-projection gather plus remaining sample/state work. Component subgroups overlap by definition and must not be summed twice. CPU overhead and copies/memsets are not included in kernel-duration tables.

The first SG step has a larger first-to-last span gap than subsequent steps in all three captures (B8 span4168µs versus3634µs median; B16 5927 versus5194; B128 25960 versus25375). All five remain in the report; the gap is not silently discarded or assigned a cause. Profiler overhead and concurrently occupied GPUs are captured context, not controlled-away variables.

Exact-name source-backed annotations are saved in [kernel-annotations.json](source_evidence/kernel-annotations.json); names without sufficient semantic evidence remain unverified in raw metrics.

## Source-backed interpretation

| Observed family | Source evidence | Correct interpretation |
|---|---|---|
| `nvjet_sm103_*`, BF16 split-K and cuBLAS reduction | BF16 unquantized dispatch in `python/sglang/srt/layers/quantization/unquant.py`; vLLM compiled graph BF16 arguments and reduction template `__nv_bfloat16` | BF16 GEMM. The skill's FP8 replacement recommendation is a false match for this capture. |
| `fmhaSm100fKernel_QkvBfloat16OBfloat16H128PagedKvCausalP*` | Both pinned TRTLLM decode backend calls; name explicitly BF16 Q/K/V/output | Attention, not RoPE. SG dispatch uses P32, vLLM P16. At B8 both use MultiCtas; B16/B128 use persistent variants. |
| SG `splitk_epilogue...SplitKDenseGemmKernel` at B8 | `qkv_norm_mrope.py` and shared `dense_bf16_gemm_sm100_splitk_epilogue.py`, exact PR head | QKV GEMM plus norm/MRoPE/cache epilogue,36 launches. B16/B128 instead show separate `fused_qk_norm_mrope_kernel`,36 launches. |
| SG `FusedAddRMSNormKernel` | FlashInfer norm kernel named in trace and SG residual norm call | Residual RMSNorm, not GEMM despite the raw skill category. |
| vLLM `triton_red_fused_fused_add_rms_norm_0/2` and `triton_poi_fused_mul_silu_slice_1` | [Saved generated source](source_evidence/vllm-generated.py.txt), source-node comments19/107/164 | Compiled residual/RMSNorm and slice+SiLU+multiply. Both frameworks already fuse SiLU and multiply; this is an implementation-speed comparison, not an absent-fusion finding. |
| vLLM `triton_red_fused_4` | Saved generated source around418–623: BF16 pointer signature, per-head128 reduction, rsqrt, norm weights and cos/sin products; sequential combo branches for Q and K | Compiled Q/K norm plus rotary application. Coefficient indexing/selection remains a separate `triton_poi_fused_arange...` call; cache write also remains separate. |

The saved generated file comes from the pinned AOT compiler cache; its example-input size is16384 and its kernels accept dynamic row counts. It supports family semantics and BF16 signatures, not proof of an exact launch-specialization/cubin identity. The trace's recorded grid/block and exact names remain authoritative for each measured launch. No cross-framework bitwise numerical equivalence is asserted.

Both cache layouts are head-major at the per-layer kernel interface. vLLM LBHNC drops the outer layer dimension to BHNC; its FlashInfer backend explicitly forms HND views, with BF16 K/V packed in the last content dimension and then split into zero-copy views. SG describes HND with separate K/V buffers. Runtime stride magnitudes/packing equivalence has not been independently recorded here, so labels alone are not used as an explanation. The observed page-size dispatch difference P32/P16 is concrete. See [source audit](analysis_tools/v2_scope_audit.md).

## Overlap and fusion tables

This is single-trace triage with source follow-up, not a mapping/formal two-trace pair. Raw skill tables are retained unchanged; corrections here take precedence over their name-based FP8/GEMM misclassifications.

| Overlap evidence | Interpretation | Supported opportunity |
|---|---|---|
| SG B8 step1 contains283 intersecting kernel-interval pairs; representative fused QKV→fmha overlap5.792µs on the same stream13 | Measured interval overlap, well above timestamp rounding; compatible with PDL, not conventional multi-stream overlap | No quantified transferable speedup without dependency/resource evidence |
| SG median sum−union413/272/327µs at B8/16/128; vLLM47/45/29µs | Sum and union measure different things; neither is request latency | Do not add overlaps to fusion savings or interpret all as removable overhead |
| SG QKV wrapper sets `use_pdl=True`; shared GEMM enables dependent launch; installed FlashInfer decode defaults PDL from device support | Source corroborates PDL-compatible same-stream observation on B300 | Preserve this when evaluating kernels; no toggle experiment was performed |
| Saved vLLM compiled Triton source has `launch_pdl=False` | Applies to those compiled kernels only, not all vLLM kernels | A source-backed research lead; no claim that enabling it is safe or sufficient |

| Fusion pattern | Current observation | Remaining lead |
|---|---|---|
| Projection + QK norm + MRoPE + cache | SG B8 already fused; vLLM has separate GEMM/reduction, norm/rotary, coefficient selection, cache writes | Explains a major kernel-count difference; does not independently quantify serving gain |
| QK norm + MRoPE/cache at larger B | SG B16/B128 one preparation kernel per layer; vLLM compiled norm/rotary plus coefficient selection plus cache write | SG already captures this fusion; larger-B projection fusion requires separate feasibility/correctness work |
| Residual add + RMSNorm | Both already fused; SG repeated/final norm sums319/332/314µs vs vLLM repeated norms226/227/242µs, with small category-boundary difference | Compare standalone BF16 norm implementations at actual shapes before serving experiments |
| SiLU × up | Both already fused; SG118/122/256µs vs vLLM54/57/88µs across36 calls | Strong bounded standalone implementation lead, especially B128; preserve precision and assess PDL interaction |
| Attention | About22.7ms of25.6–25.7ms summed B128 work;36 calls on both | Largest B128 cost. P16/P32 and cache-view/backend details are leads, not proven causes or recommendations to change defaults |

Standalone controls must use BF16 gate/up `[B,19456]` with row stride19456 and contiguous output `[B,9728]`, plus residual/norm rows `[B,2560]`. SG Qwen3 imports Qwen2MLP, which calls gate_up then SiluAndMul; activation.py allocates contiguous half-width output and the JIT kernel indexes gate/up halves explicitly. The observed SG template has PDL=true, rounding=false, input-reuse=false. Saved vLLM generated source107–160 uses the same19456/9728 offsets and contiguous output, with FP32 activation arithmetic and BF16 stores; its residual norm graph22–35 uses2560-wide contiguous BF16 rows. This establishes shape/stride controls, not identical rounding semantics: numerical checks must precede timing any replacement. The B128 activation difference is168µs across36 calls, under0.7% of the25.6ms summed budget; it is not a10% opportunity by itself. Norm totals also include different first/final variants and are an eligibility lead rather than an exact ratio.

## Prefill is separate

Each saved prefill trace has its own unchanged three-table report. These captures use chunked/mixed prefill scheduling, not a single pure-prefill forward or TTFT benchmark. SG B8/B16/B128 audits prove65536/131072/1048576 prompt tokens plus7/15/127 provisional decode-tail tokens under mixed overlap. Full per-forward extend/prefix lengths are retained. vLLM prefill nested sample/logits labels contain the requested capture batch and must not be read as actual per-forward batch geometry; use summary query/sequence metadata. Consequently aggregate prefill kernel sums are not used for a TTFT ratio.

The defensible next step is short standalone BF16 activation/norm benchmarking at the traced shapes, while treating B128 attention as the dominant remaining cost. The traces support these leads and the observed fusion/launch differences; they do not establish a new 10% serving advantage or causal percentages for existing speedups.

The most material prefill dispatch difference is SG CuTe `FlashAttentionForwardSm100` versus vLLM TRTLLM `fmhaSm103a...P16...PersistentContext`. At B128 each trace contains2304 long-prefill attention calls; SG additionally records a final mixed/tail forward (2340 activation calls versus2304). SG `_pack_kv` also appears in long-prefill preparation. Common nvjet BF16 GEMM dominates both raw tables (roughly60%SG/57%vLLM of summed durations), while the two attention families contribute about22%/25%. These are descriptive shares of differently scheduled whole traces, not a throughput or TTFT comparison. The raw skill labels the BF16 vLLM cache-write template as quantize; template type0 with BF16 input/output is a cache write, not FP8 quantization.

### First B128 prefill chunk with matched geometry

The first saved chunk on each framework is actual B2,16384 input tokens, sequence/query lengths `[8192,8192]`; SG prefix lengths are zero. This is one observed warmed chunk, not an aggregate of different mixed/tail geometry. [SG exact membership](prefill-first-chunk-sglang.json) and [vLLM exact membership](prefill-first-chunk-vllm.json) use unique launch correlations. SG forward includes its projection; vLLM execute includes input bookkeeping and excludes subsequent logits/sampling. Thus whole-scope spans are retained as evidence but not compared as identical latency.

| First-chunk family | SG calls / summed ms | vLLM calls / summed ms |
|---|---:|---:|
| Main BF16 GEMMs | 144 /72.125 |141 /68.813 plus3 bias-add variants /1.387 |
| Long-prefill attention |36 CuTe /25.659 |36 TRTLLM context /28.795 |
| SiLU × up |36 /6.923 |36 /6.655 |
| Residual RMSNorm |72 /4.356 |72 /4.646 |
| QK norm and rotary |36 norm +36 MRoPE /5.923 |35 compiled norm/rotary +35 coefficient-selection /5.715, plus first-layer variants |
| Cache write / preparation |36 cache +36 pack /2.193 |36 cache /2.484 |

The shared geometry makes these operation-family differences useful leads: SG's CuTe attention is lower in this single chunk, whereas vLLM's main GEMM sum is lower; activation is much closer here than at small decode batches. Backend/compiler implementations and first-layer variants differ; the BF16 query/KV contract is matched, while intermediate rounding equivalence is not asserted. SG scope has449 kernels, sum117.418ms, union116.821ms, span118.863ms; vLLM execute has456 kernels, sum118.913ms, union118.845ms, span126.304ms. No TTFT ratio follows from these spans, and one chunk does not establish a stable performance delta.
