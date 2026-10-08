# Qwen3-VL-4B B300 BF16 experiments

## Current status

The target is at least 10% more output throughput than the recorded handoff baseline, with **BF16 weights, BF16 KV cache and BF16 attention queries**. The target is **not met across the seven concurrency points**. Only concurrency 1 clears 10% in the completed native-NHD combined sweep below. Earlier FP8 results are historical and do not satisfy this precision-matched goal. No speculative decoding or draft-model configuration is used in the current candidate.

The handoff reference used one B300, vLLM `0.30.1rc1.dev648+g92044241a`, BF16 model dtype and KV `auto` resolving to BF16. Its server log was checked at `/root/qvl/out-baseline-20261007/vllm/server.log`. Candidate environment: Torch `2.14.1+cu130`, FlashInfer `0.7.0.post1`, Transformers `5.17.0`, CUDA 13.0. Baseline numbers were supplied in the handoff; they were not remeasured here.

## Latest committed increments

The tables below retain the earlier full-sweep results and are not a fresh
sweep of the latest head. Subsequent controlled increments are:

| Change | Production commit | Measured serving gain | Accuracy limitation |
| --- | --- | --- | --- |
| Public M128 BF16 GEMM autotuning | `5f60fb8b67` | +0.336% at c128, eight N640 pairs | Equal aggregate GSM8K scores across two GPU-swapped pairs; individual scores and outputs differ |
| Direct M1 BF16 gate/up GEMM | `bedf8a4c15` | +1.692% at c1, four pairs | Repeated 1217/1314 versus 1219/1314 control; reproducible two-question loss |
| Bounded packed-prefix FA4 prefill | `91042a055f` | +0.446% at c128, four N640 pairs | Matched-tactic diagnostic 1216/1314 versus 1217/1314; only one context batch used packing |
| Batch-4 BF16 vocabulary projection | `5e1601731b` | +0.191% at c4, two N40 pairs | Full C4 GSM8K 1218/1314 versus 1216/1314; one paired check, not equivalence |

The packed-prefill change retains BF16 weights, K/V and queries, and uses an
explicit CLC scheduler override only for its eligible attention calls.
Existing packing and decode fallbacks remain available. Its six short
lower-concurrency pairs improved throughput by 0.10–0.54%, but median TTFT
regressed by 7.54%, 12.54% and 21.97% at c4, c32 and c64. One c128 pair also
regressed TTFT by 21.94%. This is not a consistent latency improvement.
Lower-concurrency checks used warmup equal to concurrency, unlike the
original full-sweep protocol.

An initial full packed-prefill accuracy pair scored 1207 versus 1218, but
selected different M128 down-projection tactics. The matched-tactic result
above came from a separate external startup-policy diagnostic that verified
both optimized paths during graph capture and held both tactics fixed.
It does not establish numerical equivalence or broad cached-prefix accuracy.
The external diagnostic is not included in production code.

Evidence: `results/bf16_decode_lt/production_validation/`,
`results/bf16_weight_packing/m1_production_validation/`, and
`results/bf16_prefill_fa4_audit/` (production validation, matched-tactic
diagnostic and lower-concurrency regression subdirectories). The code PR
#42913 is at `a49b059185`; #42914 retains the optional deployment recipe.
None of these increments meets the remaining high-concurrency target.

The latest committed increment is batch-4 BF16 vocabulary-projection tuning
(`5e1601731b`). Its external-hook serving crossover improved c4 throughput
by 0.298% and 0.132%, with exact request/token counts. Clean production
source checks verified actual tuned graph dispatch and bitwise prefill and
decode logits. Image requests matched greedy tokens but had different
prefill batching and log probabilities across servers; a separate same-input
check matched original and tuned BF16 outputs on 24 observed image decode
steps. This is limited numerical evidence, not full accuracy equivalence.
Production serving improved by 0.160% and 0.221% in two same-GPU pairs,
reaching 1240.50 and 1246.50 output tokens/s. Both exceed the C4 target of
1237.5, but this is not a new seven-point sweep. Median TTFT was 215.67 and
209.95 ms, above the handoff C4 value of 200 ms. Full C4 GSM8K scored
1218/1314 versus 1216/1314 control (9 control-only and 11 candidate-only
correct). This is a single empirical accuracy comparison. See
`results/bf16_lmhead_screen/` for standalone, model, serving-gate and
production evidence. This does not establish a new completed full sweep or
achievement of the overall target.

The latest experimental BF16 gate/up plus SiLU fusion remains outside the
production PR. Reducing the native CUDA epilogue subtile removed register
spills and improved the dominant M16331 standalone operation by 5.48–5.82%.
The direct tensor-call version improved the exact mixed-model forward by
only 0.40–0.64% across two GPUs, with six of eight adjacent pairs positive.
All 36 layers exercised the candidate, and bounded prefill plus three-step
decode logits matched bitwise. No serving improvement is established; the
model measurements do not establish an additional gain from the faster host
wrapper. Raw evidence is in `results/bf16_native_epilogue/subtile/`,
`subtile_model/`, `ffi_wrapper/` and `ffi_model/` under that same directory.

A subsequent single-compiled dynamic-row version passed standalone checks
and the short model numerical gate. Its matched-tactic c128/N128 serving
screen then measured throughput changes of −0.032% and −0.013% in two
GPU-swapped pairs, with median TTFT changes of +0.203% and +0.109%.
All request/token counts matched, both variants used the same 1.4M-token
KV capacity, and more than 99.6% of measured prefill token rows were
eligible for fusion. This bounded screen establishes no serving gain;
the experimental kernel is not being promoted. See `dynamic_m/`,
`dynamic_model/` and `dynamic_serving/` under `results/bf16_native_epilogue/`.

Two further bounded screens were rejected: reusing device text-position
metadata produced opposite-sign serving changes (−0.185% and +0.157%)
despite full dispatch coverage, and public FA2 paged decode at the recorded
39-request mixed suffix was 27.03% slower than TRT. Their evidence is in
`results/bf16_native_epilogue/mrope_metadata_serving/` and
`results/mixed_decode_screen/`. Neither changes production behavior.

The batch-16 down-projection split-K candidate remains experimental. Its
C16/N80 serving crossover improved throughput by 0.494% and 0.312%, with
exact request/token counts and no observed M128 forwards. Median TTFT
improved on one GPU and regressed on the other. A 64-question GSM sanity
scored 60/64 versus 61/64 control. Full normal-startup accuracy scored
1222/1314 versus 1213/1314, but both runs exercised M128 operations with
different startup tactics. The matched-tactic diagnostic scored
1220/1314 versus 1221/1314 with identical validated startup tactics; this is
a finite, schedule-sensitive comparison, not numerical equivalence.
The one-entry production implementation is committed as `06d1a12172` and
is undergoing production validation before PR promotion. The separate batch-16 gate/up public-Lt
screen saved only 1.2–2.1 microseconds across 36 layers, so it was not advanced
to model testing. See `results/bf16_m16_projection_screen/down_serving_sanity/`
and `results/bf16_m16_gateup_screen/`.

The batch-8 vocabulary extension is committed as `5babd15f4f`, with production
validation pending before PR promotion. Its external-hook C8 serving pairs
improved throughput by 0.284% and 0.379%, reaching 1890.39 and 1912.20
output tokens/s, still below the C8 target of 1951.4. TTFT changed by −1.93%
and +6.73%. Production B8 checks on two input seeds matched prefill/decode
logits and greedy tokens bitwise. See `results/bf16_lmhead_screen/` under
`m8_serving_gate/`, `m8_local_guards/` and `m8_production_model/`.
Larger-batch vocabulary kernels passed standalone and short-context model
screens, but have no serving result yet. They are not enabled in production.

## Configuration and reproducibility

```bash
HF_HOME=/root/qvl/hf CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/python \
  /root/qvl/venv-sgl/bin/sglang serve \
  --model-path Qwen/Qwen3-VL-4B-Instruct \
  --dtype bfloat16 --kv-cache-dtype auto \
  --attention-backend trtllm_mha --page-size 32 \
  --chunked-prefill-size 16384 --enable-mixed-chunk \
  --enforce-disable-flashinfer-allreduce-fusion \
  --host 127.0.0.1 --port 30000
```

This uses the default NHD cache storage. Native HND is a separate opt-in experiment using `SGLANG_USE_HND_KVCACHE=1`; its measurements must not be substituted into the NHD comparison.

`scheduler_bench.py` reproduces text-only random prompts of exactly 8192 input and 1024 output tokens. Concurrency is 1/4/8/16/32/64/128 with 30/40/80/80/160/320/640 measured requests. Each point has `max(64, concurrency)` warmup requests followed by a cache flush. Every completed full sweep has 1,350 requests, 11,059,200 input tokens, and 1,382,400 output tokens. The client is sgl-bench `a9da34ad1f997ca05878d858d3d01970e4a49af9`.

```bash
HF_HOME=/root/qvl/hf PYTHONPATH=$PWD/python \
  python benchmark/qwen3_vl/scheduler_bench.py \
  --gpu 0 --port 32000 --output /root/qvl/experiments/example \
  --python /root/qvl/venv-sgl/bin/python \
  --bench /root/qvl/venv-bench/bin/sgl-bench -- \
  --dtype bfloat16 --kv-cache-dtype auto \
  --attention-backend trtllm_mha --page-size 32 \
  --chunked-prefill-size 16384 --enable-mixed-chunk \
  --enforce-disable-flashinfer-allreduce-fusion
```

Add `--screen` before the final `--` for a short screen (c1: 3 requests, 1 warmup; c128: 256 requests, 128 warmups). Screening results are provisional. Raw JSONL, commands, installed packages, GPU process monitoring and source provenance are in each `results/bf16_*` directory.

## Exact BF16 comparison

These columns compare the same fusion-enabled source with mixed chunking disabled versus enabled. Throughput counts output tokens only; TTFT and TPOT are medians. Results are single full sweeps, not confidence intervals.

| Concurrency | Handoff output tok/s | Mixed output tok/s | vs handoff | Normal TTFT ms | Mixed TTFT ms | Mixed TPOT ms |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 334 | 383.81 | +14.91% | 75.08 | 75.50 | 2.53 |
| 4 | 1125 | 1234.49 | +9.73% | 187.29 | 158.94 | 3.08 |
| 8 | 1774 | 1888.61 | +6.46% | 384.52 | 262.65 | 3.96 |
| 16 | 2499 | 2558.44 | +2.38% | 566.15 | 383.80 | 5.87 |
| 32 | 3084 | 3136.65 | +1.71% | 1080.92 | 522.48 | 9.73 |
| 64 | 3494 | 3518.27 | +0.69% | 2158.99 | 543.47 | 17.60 |
| 128 | 3815 | 3801.87 | -0.34% | 4120.29 | 577.62 | 32.95 |

At concurrency 128, mixed chunking cuts TTFT from 4120.29 to 577.62 ms (85.98%) while output throughput changes from 3763.54 to 3801.87 tok/s (+1.02%). TPOT increases from 29.97 to 32.95 ms (+9.96%). This is a latency tradeoff, not a uniform speedup.

## Optional native HND layout

The completed exact HND sweep is in `results/bf16_combined_hnd/`. At c128 it reaches 3841.08 output tok/s, only 0.68% above the handoff baseline and below the 4196.5 target. It improves about 1.03% over the combined NHD full sweep, with smaller or noisy changes at other concurrencies. Short c128 crossover runs on each GPU separately found HND gains of 1.30% on GPU5 and 1.64% on GPU7; see `hnd_crossover.json`.

| Concurrency | HND output tok/s | vs handoff | TTFT ms | TPOT ms |
|---|---:|---:|---:|---:|
| 1 | 382.58 | +14.54% | 75.70 | 2.54 |
| 4 | 1230.06 | +9.34% | 189.14 | 3.07 |
| 8 | 1891.34 | +6.61% | 317.76 | 3.93 |
| 16 | 2553.93 | +2.20% | 386.67 | 5.87 |
| 32 | 3142.30 | +1.89% | 526.60 | 9.70 |
| 64 | 3533.87 | +1.14% | 545.69 | 17.54 |
| 128 | 3841.08 | +0.68% | 582.68 | 32.59 |

## Where the improvements come from

1. Small-batch BF16 GEMM tuning (`cd4657fd5b`): 12 split-K tactics for QKV, output and down projections at batches 1/2/4/8. The one-batch runner initialization fix (`a2ab66d8f8`) makes its backend match serving.
2. Mixed attention routing (`d434f638ec`): trailing one-token requests use decode attention while the prefill prefix uses context attention. Before this change, enabling mixed chunking at c128 reduced throughput from 3764.67 to 3201.14 tok/s; routing the tails correctly recovers it to 3777.27 tok/s. A representative standalone mixed batch fell from about 3.79 to 1.00 ms.
3. BF16 QK RMSNorm plus multimodal RoPE fusion (`7f966e810c`): one kernel replaces separate normalization and rotation for small batches on Blackwell, retaining BF16 rounding. With mixed chunking, c1 throughput changes from 375.15 to 383.81 tok/s and c128 from 3777.27 to 3801.87 tok/s. Full serving changes are small at large batches because attention dominates.
4. Native HND cache writer (`f81b0bfa51`): an optional layout gets a fused BF16 writer instead of multiple indexing kernels. Standalone writes improved from about 10.2 to 1.45 microseconds at 64 tokens and 86.64 to 10.18 microseconds at 8192 tokens. The default cache layout remains unchanged. Full HND results are recorded separately above.

`bf16_mixed_comparison.json` contains all five controlled sweeps: original normal, original mixed, corrected mixed routing, corrected mixed routing plus fusion, and fusion with mixed chunking disabled. Original here includes the small-batch GEMM tuning, so this is not a clean-main GEMM ablation. All runs were isolated to one GPU; independent runs used different GPUs, and small differences may include GPU/run variation.

## Accuracy and implementation validation

Full text GSM8K: reference 1215/1314 (92.47%); mixed routing plus fusion 1216/1314 (92.54%); optional HND combined candidate 1220/1314 (92.85%). See `bf16_combined_accuracy.json` and `bf16_combined_hnd_accuracy.json`. The first five dataset rows are few-shot examples, all remaining 1314 are evaluated with temperature 0, top-p 1, max output 2048 and concurrency 32. These single-run checks do not establish statistical or multimodal accuracy equivalence.

Standalone validation: 24 distinct QK norm/MRoPE cases, repeated after formatting, were bitwise equal to the original operations. Shapes cover tokens 1/8/128, Q heads 4/32, KV heads 1/8 and contiguous/interleaved multimodal axes. Eighteen mixed-attention cases cover heterogeneous lengths and shared-output-buffer aliasing (maximum absolute difference 0.001953125). Twenty cache-write cases cover layouts, negative slots, sliding-window destinations and noncontiguous inputs; six BF16/FP16/FP8-scale regressions match original NHD outputs. Existing CPU namespace/fusion/dispatch tests: 123 passed. No standalone benchmark or added test files are included in the code PR.

One image smoke test using `examples/assets/example_image.png` produced identical 32-token greedy descriptions on original main and the combined BF16 candidate (234 prompt tokens, including 216 image tokens). A trace confirmed 144 fused QK norm/MRoPE kernel executions during image decode. Selected-token logprobs were not bitwise identical (maximum absolute difference 0.04090); this comparison includes GEMM tactic changes. Raw responses and source/fixture hashes are in `results/bf16_multimodal_smoke/`. This is a smoke test, not a multimodal accuracy evaluation.

The serving checkouts recorded Git base `b13cd34649` with patches applied, rather than the later clean commits. The saved diffs and SHA256 manifests identify the actual tested files, including untracked JIT source. `f81b0bfa51` is the combined committed implementation; PR #42913 has equivalent cherry-picked commits on its code-only branch.

## Other BF16 screens

`bf16_scheduler_results.json` records short screening runs. Chunk sizes, prefill/decode scheduling interval, context limits, alternative attention backends, attention split counts and expanded GEMM tactics did not produce a further substantial throughput improvement. High-concurrency decode was dominated by BF16 KV reads; the measured attention kernels were near the device's sustainable memory bandwidth. Expanded GEMM tactics yielded less than 1% and were not promoted. Historical FP8 artifacts remain in Git history and old result folders, but are excluded from current candidate claims.

## Devbox recovery and further BF16 screening

Experiments resumed on `chunan-b300-8`, using the same model revision, Torch,
FlashInfer and benchmark client as the earlier host. A short recovery screen
reached 376.85 output tok/s at c1 and 3783.68 at c128 for the combined HND
candidate. These short runs do not replace the full sweep above. Exact environment
and raw evidence are in `results/rx_devbox_recovery/`.

Correcting the single-expert fused-MLP prototype to FlashInfer's Up/Gate weight
order produced bitwise-equal outputs after tuning, but was 2.6% slower at 8192
tokens and 9.4% slower at 16384 tokens. Packed BF16 normalization and tighter
attention context bounds also failed screening; none was promoted.

A standalone QK normalization/MRoPE/cache-write fusion passed 20 bitwise kernel
cases and three exact greedy token/logprob comparisons. Its first short serving
pair improved output throughput by 0.4–0.8%. Same-GPU crossovers on all eight
GPUs then showed mean gains of 1.41%, 1.73% and 1.37% at concurrency 1, 4 and 8;
every pair was positive. An additional 24 head-shape cases passed bitwise checks.
All 24 crossover greedy responses retained identical tokens, with maximum selected
logprob difference 0.0000224. The guarded production implementation was subsequently committed as `466c9e0f40`; final validation follows.

## BF16 normalization/rotary/cache-write fusion

Commit `466c9e0f40` combines the small-batch QK normalization/rotation kernel
with HND cache writes during ordinary decode. It preserves the cache pool's
physical-slot validation, OOB checks and transfer synchronization, and is gated
to plain BF16 HND page-32 pools with the tested TRT backend. Default NHD is
unchanged. Raw production validation is in `results/bf16_norm_rope_cache_fusion/`.

| Concurrency | Prior HND output tok/s | Fused output tok/s | Gain | Fused TTFT ms |
|---|---:|---:|---:|---:|
| 1 | 379.67 | 383.44 | +0.99% | 87.68 |
| 4 | 1218.89 | 1230.54 | +0.96% | 222.50 |
| 8 | 1870.41 | 1886.09 | +0.84% | 392.77 |
| 16 | 2534.82 | 2557.31 | +0.89% | 512.43 |
| 32 | 3112.32 | 3125.67 | +0.43% | 561.62 |
| 64 | 3491.64 | 3504.64 | +0.37% | 718.22 |
| 128 | 3790.59 | 3797.51 | +0.18% | 610.25 |

These are same-host full sweeps on separate GPUs with exact request/token counts.
The eight-GPU same-device crossover above provides stronger evidence for the
small low-concurrency gain. At c128, throughput improves only 0.18%; the
4196.5 output tok/s target is still unmet.

The production operator passed all 44 bitwise standalone cases. Existing CPU
regressions: 122 passed, one skipped. Two full GSM8K runs scored 1217 and 1222
out of 1314 for the candidate, versus 1222 and 1221 for the preceding HND
implementation with GPUs swapped. These runs demonstrate variation, not
statistical accuracy equivalence.

A fresh standalone page-size check found BF16 page16/32/64 outputs bitwise equal
and throughput differences below 0.35%; the page-16 reference-source lead was
not promoted. All weights, KV storage and attention queries remain BF16.

## Further memory-traffic investigations

Exact prefix reuse offers little opportunity in this workload: reconstructed
prompts have at most about 0.315% reusable page-32 prefix tokens, and inspected
measured-phase logs show zero nonzero prefix-cache hits after flushing.

A standalone lossless BF16 packing prototype reconstructed all 65536 BF16 bit
patterns, a 4 GiB cache, and captured real K/V exactly. It saved logical reads
but was slower: best packed attention after bounded tuning took 1.686 ms versus
0.586 ms for the existing TRT kernel. Register/shared-memory pressure and
byte-unpacking/layout operations outweighed the memory savings. No packed
cache format was integrated.

Transparent CUDA memory compression was also tested with actual compressed
allocation properties verified. It accelerated a zero-filled control from
588.54 to 225.80 microseconds, but real BF16 K/V from layers 0/17/35 took
590.62–591.69 microseconds versus 588.11–589.08 in uncompressed VMM storage.
Caches and outputs were bitwise equal. This option was not promoted.

Scripts, numerical checks, captured-value statistics and timing reports are in
`results/bf16_lossless_screens/`; large captured tensors remain on the devbox.

A 12-configuration uncompressed Triton attention screen also failed to beat
TRT: best 616.40 versus 588.50 microseconds on the same GPU. The prototype
included runtime sequence lengths and page-table lookup, and numerical checks
passed. No serving tests were launched for these rejected kernels.

TRT supports padded KV page strides, but a final four-layout screen found no
benefit: padding 0/128/512/4096 BF16 elements per page took respectively
588.86/590.47/588.95/597.79 microseconds, with bitwise-identical outputs.
The padded layouts increase cache capacity requirements and were rejected.

Nsight Compute measured the warmed BF16 TRT decode at B128, L8192, Q32/KV8,
head dimension 128 and HND page32. Actual DRAM reads were 4,296,317,440 bytes,
only 0.0314% above the 4,294,967,296-byte KV payload. DRAM throughput reached
94.15% of sustained peak; SM throughput was 46.48% and achieved occupancy
24.89%. Kernel time under profiling was 595.488 microseconds; a separate
process without the profiler measured 586.609 microseconds. These counters
confirm near-minimal traffic and near-saturated bandwidth for this case.
Raw counters and the standalone script are in
`results/bf16_lossless_screens/ncu_decode/`.

A separate two-stream CUDA-graph screen overlapped this decode with independent
BF16 prefill GEMMs. Five alternating measurement orders gave median paired
speedups of 0.9996x at 8192 prefill rows and 1.0080x at 4096 rows; outputs
matched bitwise. The negligible benefit did not justify a whole-model overlap
experiment. See `results/bf16_lossless_screens/overlap/overlap_report.json`; this result
applies to the tested GEMM/kernel pairing, not all possible overlap schedules.

Explicit green-context partitioning also failed the standalone screen.
Attention/GEMM partitions of 112/32 and 96/48 SMs took 2.109 and 1.509 ms,
respectively, versus full-device serial controls of 1.045 and 1.057 ms.
Outputs matched bitwise; the GEMM slowdown outweighed any overlap.
Reports and scripts are in `results/bf16_lossless_screens/overlap/`.

## Current whole-workload timing budget

An instrumented c128/N256 run at production source `466c9e0f40` completed
all requests with nominal 8192-input/1024-output tokens. CUDA events around
ModelRunner.forward measured 68.750 s against 69.049 s benchmark wall time:
72.36% of forward time was DECODE, 27.55% MIXED, and 0.09% EXTEND.
These are diagnostic intervals, not a new throughput claim or rigorous CPU
overhead attribution. Chat formatting adds tokens beyond nominal input length.

A separate late mixed-batch trace attributed approximately 49.3% of kernel
time to BF16 GEMMs, 21.1% to context attention, 16.5% to decode attention,
and 4.8% to SiLU. Later-context pure decode attributed 87.9% to attention.
Raw automated triage contains known classification errors; use the corrected
interpretation in `results/current_c128_profile/`. Large traces remain on the
devbox, with paths and reproduction scripts recorded in that directory.

The current full-sweep c128 result of 3797.51 output tok/s still requires
about 9.5% less wall time to reach 4196.5. If decode time stayed fixed,
prefill-containing forwards would need approximately 34% less time.

Ragged mixed-batch GEMM padding was also screened. Padding alone improved
the down projection by about 6%, but input-copy cost erased the benefit.
Writing SiLU directly into a padded buffer avoided that copy but produced
less than 1% pair-level change, no better than the aligned control, with
timing drift. The padded producer still used an explicit tail-zero launch.
Neither variant was promoted to production. Scripts and reports are in
`results/bf16_lossless_screens/ragged_gemm/`.

## BF16 prefill layout and algorithm screens

Large mixed-batch Q retains the QKV projection stride of 6144 BF16 elements
per token, versus 4096 for compact Q. Synthetic two-prefill geometries
matching the observed total rows showed 1.4–2.7% faster context attention
with compact Q, but a separate copy lost about 10%. An external rotary
producer writing compact Q directly preserved bitwise outputs but did not
improve the normalization/rotary/attention chain. It was not promoted.

Context-attention hardware counters showed 81.69% SM throughput, 78.34%
tensor-pipe activity, and only 3.42% DRAM throughput. This is a compute-heavy
path, unlike decode. Fresh and steady timing samples differed substantially;
the evidence records measurement-history differences without attributing an
unmeasured cause or claiming a speedup from them.

Once-transposed contiguous weights and padded weight-row strides were
compared against native linear operations at M8192 and M16331 for all four
projection shapes. Every output was bitwise equal; timing differences were
below 0.5%. Neither layout was promoted. Reports and reproducible scripts
are in `results/bf16_prefill_layout_screens/`.

Explicit cuBLASLt heuristic enumeration returned eight algorithms for each
of eight projection/token-count cases (the API requested up to 100).
Selected algorithms improved several M8192 kernels, including down projection
from 330.86 to 298.78 microseconds; M16331 aggregate benefit was much smaller.
Two short-model passes with GPUs swapped did not retain those gains:
candidate B1 prefill medians were 66.04 and 65.38 ms, versus 64.09 and
63.54 ms controls on the respective GPUs. Captured logits and checked
real-weight projections were bitwise equal. The existing dispatch path was
not promoted. Evidence is in `results/bf16_prefill_layout_screens/lt_tactics/`.

Bypassing Python runner bookkeeping reduced eager CPU submission from
20.91 to 11.80 microseconds per projection, but GPU time was unchanged
(299.50 versus 300.88 microseconds). The lower-level API still constructs
cuBLASLt descriptors per call. This bounded follow-up did not demonstrate
a model-level improvement, so no further serving run was launched.

## Persistent GEMM descriptors and dispatch follow-up

An external C++ prototype compared cached and uncached cuBLASLt descriptors
through the same binding. It passed 12 bitwise checks, including fresh
tensor pointers and a non-default stream. Descriptor reuse saved only about
0.22 microseconds of CPU submission per projection and did not improve GPU
time. Descriptors were keyed by device, shape and dtype; data pointers and
streams were supplied on every call. No production cache was added.

A direct-module path was then tested within one loaded model. An initial
six-pair run suggested 0.7% lower prefill latency, but a counterbalanced
12-pair run reversed it: control median 66.729 ms versus candidate 67.225 ms,
with six candidate wins. Both runs produced bitwise-identical checked logits.
The apparent gain was not robust to order, so this path was rejected before
serving tests. Evidence is in
`results/bf16_prefill_layout_screens/lt_persistent/`.

A source audit also verified that production already passes persistent
self-resetting decode counters: the extra fill in the earlier standalone
NCU script was not a production optimization opportunity. Full c128 decode
uses an exact graph bucket, and inspected synchronization paths are debug
gated. Source references and limitations are retained in
`results/bf16_descriptor_screens/`.

## High-batch decode GEMM investigation

M64/M128 screens used independent weight sets larger than L2 and CUDA-graph
replay. The direct and split-K implementations support only M<=32 and were
not forced onto larger batches. Most bounded TGV configurations lost to the
existing selector. TGV down-projection at M128 improved its kernel by about
5%, but its short-model combination was slower and changed five of 128
first-decode argmax results; that combination was not promoted.

Explicit cuBLASLt screening found roughly 12% gate/up and 6–7% down-projection
kernel gains at M128. Pure-Lt model logits were bitwise equal, and traces
confirmed both changed kernels ran inside decode CUDA graphs. A same-GPU
short-context model run showed lower decode latency, but this did not
translate into a comparably large long-context serving gain.

Eight-GPU c128/N128 serving crossovers estimated +0.258% geometric-mean
throughput, with six of eight paired gains positive. A descriptive paired
bootstrap 95% interval was [-0.100%, +0.593%]; the t interval also included
zero. This is an uncertain positive signal, not an established speedup. All
16 runs completed their exact request/token counts. Two-token greedy probes
used 128 identical short prompts per run and matched tokens/logprobs; they
are not a diverse accuracy evaluation.

The supported public BF16 autotuner independently selected the same winning
algorithms on two GPUs and retained the cold-weight gains with bitwise-equal
outputs. Its default measurement policy already uses cold L2 and CUDA graphs.
TGV/public-tuner evidence is in `results/bf16_decode_highm/`; Lt/model/serving
evidence is in `results/bf16_decode_lt/`. The initial screen required stronger serving confirmation; the later
production result and committed integration are summarized at the top.

The subsequent public-autotuning c128/N640 crossover completed all 16 runs
with exact counts. All eight GPU pairs improved: geometric mean +0.3156%,
with descriptive paired bootstrap 95% interval [+0.2651%, +0.3621%].
The candidate averaged roughly 3821 output tok/s and remains below 4196.5.
Autotuning selected different down-projection algorithms across starts.
Probe token IDs matched, while logprobs differed by up to 0.0426643; this
requires accuracy validation before production promotion. These intervals
describe eight GPU pairs across two periods, not repeated-day uncertainty.

A separate lossless-cache follow-up reconstructed BF16 pairs using uint32
operations. All 65536 bit patterns and captured K/V checks passed, but
register usage barely changed (146 to 145) and shared memory increased
from 139264 to 155648 bytes. It failed the resource screen, so no full
attention timing or production integration followed. Evidence is in
`results/bf16_lossless_screens/pair_unpack/`.
