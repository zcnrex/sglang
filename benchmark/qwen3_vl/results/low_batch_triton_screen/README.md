# Existing Triton low-batch decode: rejected

All six batch/split configurations lose to TRT in all eight paired rounds. No model/serving follow-up or production edit is warranted.

| Batch | Splits | TRT us/layer | Triton us/layer | Slowdown |
| --- | ---: | ---: | ---: | ---: |
| 16 | 8 | 80.402 | 91.936 | 14.34% |
| 16 | 16 | 80.409 | 102.321 | 27.25% |
| 16 | 32 | 80.407 | 109.791 | 36.54% |
| 8 | 8 | 43.214 | 69.675 | 61.23% |
| 8 | 16 | 43.226 | 54.515 | 26.12% |
| 8 | 32 | 43.216 | 64.684 | 49.67% |

The exact existing callable is `sglang.kernels.ops.attention.decode_attention.decode_attention_fwd` from frozen `/root/qvl/sglang-m16-down-production`. Its standard grouped kernel uses BLOCK_N 32, BLOCK_H 16, four warps, two stages; PDL is false as in the callable default. The only screened tuning dimension is explicit splits 8/16/32. This is distinct from the prior B128 raw Triton prototype and the rejected public FA2 backend.

## External stride adapter

The unchanged public wrapper accepts only affine 3D token-major cache views. A benchmark-only replacement for `_extract_kv_strides` admits a zero-copy logical `[page, token, head, dim]` view of the same physical HND allocation and returns exact token/head/page/token strides `(128,4096,32768,128)`. No storage is copied or repacked. Existing jitted stage1 independently computes `page_id=kv_loc//32`, `token=kv_loc%32` and uses the supplied page/token/head strides. All kernel arithmetic and reduction code remain unchanged. The adapter is not presented as an unchanged public-wrapper feature or a production implementation.

BF16 Q/KV/output, Q32/KV8/D128, page 32, Q token stride 6144, lengths 8192+i. Random physical pages are shared identically by both paths. Each graph traverses 36 independent layer states: approximately 9.70 GB / 19.40 GB of physical KV at B8/B16, far beyond L2. TRT uses production max-sequence bound 262144, persistent multi-CTA counters and 512 MiB workspace. Triton partials/LSE are FP32 as in its ordinary backend; reference matmuls disable TF32.

## Timing and correctness

Eight opposite-order pairs per candidate, each 12 replays of a 36-layer retained graph. Both graphs receive post-capture warmup before timing. No planning, compilation, reference work or input copies occurs in the timed intervals. These are seeded finite-input standalone measurements, not real-model or serving claims.

All 36 layer states pass cross-backend normalized RMS error below 0.005. Every request in state 0 is checked against explicit FP32 softmax attention both after an eager input change and after a second input change through retained graph replay. Maximum reference normalized RMS error is 0.002851 at B8 and 0.002337 at B16. The checks validate the stride adapter and ordinary BF16 numerical tolerance; they do not establish bitwise/full-model accuracy equivalence.

Exact script and imported kernel source hashes are in result/launch JSON. Raw original scripts, logs, graph dumps and results are archived; readable copies use `.py.txt`. Remote root `/root/qvl/experiments/low-batch-triton-screen`. PIDs 616248/616249 are terminal and GPUs 2/3 released. No subsequent GPU work is queued.
