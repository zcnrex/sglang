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
