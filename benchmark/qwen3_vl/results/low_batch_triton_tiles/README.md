# Larger Triton KV tiles: rejected

Both larger tiles lose to TRT in every paired round, and also regress against the previous standard tile 32 results. The hypothesis that fewer KV loop iterations would improve this shape did not survive the bounded screen. No model/serving follow-up or production edit is warranted.

| Batch | Splits | BLOCK_N | TRT us/layer | Triton us/layer | Slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16 | 8 | 64 | 80.399 | 137.398 | 70.90% |
| 16 | 8 | 128 | 80.408 | 102.425 | 27.38% |
| 8 | 16 | 64 | 43.252 | 76.199 | 76.18% |
| 8 | 16 | 128 | 43.261 | 62.753 | 45.06% |

The prior tile 32 baselines were 54.515 us at B8/split16 and 91.936 us at B16/split8. These are separate prior runs, not simultaneously measured controls; TRT is the paired control above. No compiler resource failure occurred.

## Exact bounded change

An external clone of frozen `_decode_grouped_att_m_fwd` changes only `BLOCK = 32` to 64 or 128. The actual jitted stage1/reduction kernels and all math remain unchanged, with the same four warps/two stages and PDL false default. No other split or tile was tested. Original and generated wrapper source copies are retained. This reuses the external metadata-only stride adapter documented in `../low_batch_triton_screen/README.md`: logical NHD views preserve identical physical HND page 32 storage and pass exact independent page/token/head strides without repacking.

Both arms retain BF16 Q/KV/output, Q32/KV8/D128, query token stride 6144, lengths 8192+i, 36 independent disjoint layer caches totaling approximately 9.70 GB / 19.40 GB. TRT uses production max-sequence bound 262144, persistent counters and 512 MiB workspace. Inputs are seeded finite synthetic values. Each candidate receives eight opposite-order CUDA-event pairs, each 12 full 36-layer graph replays after post-capture warmup. No tuning/compilation/reference work is timed.

## Correctness and evidence

All 36 state comparisons pass normalized RMS error below 0.005. Every request in state 0 passes explicit FP32 softmax reference with TF32 disabled after changed input, including a second changed-input retained-graph replay. Maximum reference NRMS is 0.002851 for B8 and 0.002337 for B16. This is a standalone tolerance check, not a full-model accuracy claim.

Exact script and imported source hashes are in launch/result JSON. Raw originals, logs, graph dumps and generated wrapper variants are archived; readable copies use `.py.txt`. Remote root `/root/qvl/experiments/low-batch-triton-tiles`. PIDs 616635/616636 completed and GPUs 2/3 are released. No additional tuning or GPU work is queued.
