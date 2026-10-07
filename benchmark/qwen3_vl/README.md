# Qwen3-VL-4B B300 BF16 experiments

## Current status

The target is at least 10% more output throughput than the recorded handoff baseline, with **BF16 weights, BF16 KV cache and BF16 attention queries**. The target is **not met across the seven concurrency points**. Only concurrency 1 clears 10% in the completed native-NHD combined sweep below. Earlier FP8 results are historical and do not satisfy this precision-matched goal. No speculative decoding or draft-model configuration is used in the current candidate.

The handoff reference used one B300, vLLM `0.30.1rc1.dev648+g92044241a`, BF16 model dtype and KV `auto` resolving to BF16. Its server log was checked at `/root/qvl/out-baseline-20261007/vllm/server.log`. Candidate environment: Torch `2.14.1+cu130`, FlashInfer `0.7.0.post1`, Transformers `5.17.0`, CUDA 13.0. Baseline numbers were supplied in the handoff; they were not remeasured here.

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
