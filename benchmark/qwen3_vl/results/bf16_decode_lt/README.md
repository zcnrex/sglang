# BF16 decode GEMM tactic investigation

These are external experiments on clean source `466c9e0f4073bcad6e4403c6f54cfc8c1821b867`, Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, B300 and FlashInfer 0.7.0.post1. No production changes are included. Weights, queries and KV remain BF16; Lt computation uses FP32 accumulation. No quantization or speculative decoding is used.

## Cold-weight standalone screen

Earlier recovered Lt screens covered M 8192/16331, not M 64/128. `standalone/screen.py` enumerates all eight returned cuBLASLt heuristics (requests up to 100) using 40MiB workspace. CUDA graphs rotate independent weight matrices totaling 260–285MiB, exceeding L2. The baseline uses the actual production selector: TGV for O projection, Torch for the other three. After screening, eight shuffled-order rounds compare the best three candidates against production on the same GPU. Outputs are checked against production and an FP32 reference.

| M | Projection | Production us | Selected Lt us | Tactic |
| --- | --- | ---: | ---: | ---: |
| 64 | QKV | 7.616 | 7.624 | 0 |
| 64 | O (TGV baseline) | 6.803 | 7.927 | 1 |
| 64 | gate/up | 19.105 | 18.652 | 4 |
| 64 | down | 13.093 | 13.026 | 3 |
| 128 | QKV | 7.964 | 7.965 | 0 |
| 128 | O (TGV baseline) | 7.781 | 8.316 | 0 |
| 128 | gate/up | 22.688 | 20.208 | 1 |
| 128 | down | 15.357 | 14.349 | 4 |

The selected M 128 gate/up and down Lt outputs are bitwise equal to production in these checks. Their combined saving is about 3.49us/layer, or 0.126ms across 36layers; the short-context model percentage must not be extrapolated to a long-context serving gain.

An independent TGV down candidate was compared on GPU 2: Torch 15.279us, Lt 4 14.325us, TGV 23 14.411us. The first model experiment preferred the existing TGV API because it was within 1% of Lt, but numerical/model evidence below ruled out promoting that choice. Explicit Lt heuristic indices are experimental and may change with the driver/library; they are not a proposed production interface.

## Short model checks

The initial candidate combined Lt gate/up with TGV down. On GPU 2 its median decode latency was 4.300ms versus 4.106ms for control on GPU 3; unchanged prefill also differed between GPUs. More importantly, its first-decode logits differed (NRMS 0.00519, maximum absolute difference 0.3164, five of 128 argmax values changed). It is a negative result. Its raw `checks.json` down tactic field erroneously retained the Lt index 4; the actual `QVL_TGV_DOWN=1` branch executed TGV 23. `first_pair_numerics.json` documents this explicitly.

The follow-up uses pure Lt gate/up 1/down 4 and runs baseline then candidate on the same GPU 2. Each loaded model receives the same standard warmup followed by six B128, input128, output32 repetitions. Median decode latency is 4.832ms baseline versus 4.400ms candidate. Unchanged prefill also shifts by about 3%, so the full 8.95% decode reduction cannot be attributed solely to the patch.

All 217 decode forwards in each variant report CUDA-graph replay. A warmup-only torch trace confirms the gate/up kernel changed from `nvjet...320x64...` to `nvjet...144x128...`, and down from `nvjet...128x64...splitK` to `nvjet...64x128...splitK`, with 36 launches each. The trace is excluded from measured repetitions. Prefill and first-decode logits match bitwise, with zero argmax differences. Raw trace paths/hashes are retained; large traces and logits tensors remain remote.

## Eight-GPU short serving crossover

`serving/run.py` launches all eight GPUs, with controls on even GPUs and candidates on odd GPUs in phase A, then swaps each GPU's variant in phase B. Every phase waits for all servers to be ready. Each run uses c128, N128, warm128, nominal 8192 input and 1024 output tokens, HND, page32, mixed16k, disabled prefill graphs, explicit BF16 model/KV. The candidate is an external hook selecting only the two M 128 Lt shapes. No production file is modified.

All 16 runs complete 128 requests,1,048,576 nominal input tokens and 131,072 output tokens each. A separate greedy probe contains 128 identical synthetic 32-token prompts and two output tokens per row; its token IDs **and logprobs** match across all 16 runs. These are repeated probes, not 2048 distinct prompts. The separate short model check uses 128 random synthetic prompts.

| GPU | Control output tok/s | Candidate output tok/s | Candidate change |
| --- | ---: | ---: | ---: |
| 0 | 3781.14 | 3800.21 | +0.50% |
| 1 | 3752.63 | 3794.57 | +1.12% |
| 2 | 3797.38 | 3782.88 | -0.38% |
| 3 | 3795.79 | 3810.93 | +0.40% |
| 4 | 3793.65 | 3808.75 | +0.40% |
| 5 | 3813.42 | 3828.96 | +0.41% |
| 6 | 3789.94 | 3799.54 | +0.25% |
| 7 | 3793.54 | 3769.79 | -0.63% |

The paired geometric-mean estimate is+0.258%, median+0.398%, with six of eight positive pairs. Descriptive 95% intervals on paired log-ratios are bootstrap[-0.100%,+0.593%] (100,000 resamples, seed 42) and t[-0.194%,+0.712%] (7 degrees of freedom). **Both include zero: this short crossover does not establish a serving gain.** They quantify variation across eight GPU pairs/two periods, not independent repeated-day uncertainty. The result does not meet the 10%-over-vLLM goal.

Raw benchmark outputs, counts, metadata, server logs, probes, source hooks, GPU telemetry and analysis script are retained. The serving summary is `serving/crossover_summary.json`; standalone and model results have separate subdirectories. Remote roots are `/root/qvl/experiments/decode-lt` and `/root/qvl/experiments/decode-lt-serving`. All jobs owned by this short crossover were stopped after completion.

## Full serving crossover using public autotuning

`public_serving/` repeats the eight-GPU crossover with N640 and warm128 per run. The external hook tunes public `flashinfer.gemm.mm_bf16(..., backend="cublaslt")` on actual loaded layer-0 weights before decode graph capture, then dispatches only the same two M128 shapes through that API. The tuning context uses `tuning_buckets=(128,), round_up=False`; cache metadata and startup numerical checks are saved per candidate. Other settings and source revision are unchanged. No production code is changed.

All 16 runs complete 640 requests, 5,242,880 nominal input tokens and 655,360 output tokens each. All eight paired throughput changes are positive. The geometric-mean gain is **0.3156%**, with median 0.3319%. Descriptive paired 95% intervals are bootstrap **[0.2651%, 0.3621%]** and t **[0.2522%, 0.3790%]**. These intervals describe eight GPU pairs across two periods, not independent repeated-day validation. The roughly 3,821 output tok/s candidate remains below the 4,196.5 target.

All candidates select gate/up tactic 1. Down selects tactic 1 on six GPUs, tactic 4 on GPU4 and tactic 3 on GPU6. This is a public-autotuning experiment, distinct from the earlier fixed-index gate/up1/down4 experiment. Actual-weight down tactic1 differs from Torch by normalized RMS 0.00010335 and max absolute 0.00390625; gate/up matches bitwise. The repeated synthetic probe's greedy token IDs match across all runs, but logprobs do not: maximum absolute difference is 0.0426643. Full accuracy validation is required before promotion; this experiment alone does not establish output equivalence.

`public_serving/crossover_summary.json` contains paired results and intervals; each run retains its exact cache, numerical checks, benchmark command, logs, counts and probe. Remote root: `/root/qvl/experiments/decode-lt-public-serving`. All owned servers and telemetry processes finished, and the compute-process inventory was empty afterward.

Historical cache limitation: the external-hook public crossover exported per-run tuner choices, but its servers shared the default runtime cache directory. Those exports do not establish eight independent tuner selections. The subsequent production-path validation uses fresh isolated `SGLANG_CACHE_DIR` directories per worker.
