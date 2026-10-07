# Explicit BF16 cuBLASLt tactic screen

No production changes were made. FlashInfer 0.7.0.post1 was inspected on `chunan-b300-8`; its BF16 runner defaults to tactic 0 without autotuning. The installed API requests up to 100 heuristic algorithms. It returned 8 for every tested shape, and this screen tested all 8. The C++ descriptor uses `CUBLAS_COMPUTE_32F`, BF16 inputs and BF16 outputs. No precision flags were changed. Workspace was 32MiB (the high-level FlashInfer default is 40MiB).

The shapes come from the pinned Qwen3-VL-4B-Instruct checkpoint: QKV N6144/K2560, O N2560/K4096, gate/up N19456/K2560, down N2560/K9728. M8192 is a one-request prefill; M16331 was the most frequent mixed-forward row count in the current c128 profile.

`screen.py` on GPU 2 uses seeded BF16 inputs/weights, checks every tactic against Torch with rtol/atol 0.02, screens under CUDA graphs, then measures the top three plus tactic 0 and Torch in four shuffled-order rounds. Full tensors are larger than L2. Selected winners are bitwise equal to Torch in every tested shape. All raw timings and numerical results are in `results.json`.

| M | Projection | Best selected tactic | Torch us | Candidate us |
| --- | --- | ---: | ---: | ---: |
| 8192 | QKV | 2 | 187.50 | 175.11 |
| 8192 | O | 2 | 127.76 | 118.82 |
| 8192 | gate/up | 0 | 580.19 | 582.01 |
| 8192 | down | 6 | 330.86 | 298.78 |
| 16331 | QKV | 0 | 356.42 | 354.40 |
| 16331 | O | 4 | 251.73 | 252.18 |
| 16331 | gate/up | 0 | 1129.65 | 1142.97 |
| 16331 | down | 2 | 598.20 | 579.05 |

## Short model validation

The external `model.py` monkeypatches only the BF16 dispatch for M8192 QKV/O/down and M16331 down. It retains Torch/existing dispatch for gate/up and every untested shape. The source is the clean detached `466c9e0f4073bcad6e4403c6f54cfc8c1821b867` checkout. Weights, queries and KV remain BF16, with TRTLLM attention, HND, page 32, mixed16k, no speculative decoding, and disabled prefill graphs.

Each run uses one_batch with B1,B1,B1,B2,B2,B2, input8192, output2, and the standard initial warmup. The first real-weight projection comparisons occur during warmup and are bitwise equal. Saved warmup B1 and first B2 last-token logits are bitwise equal across all four runs. Their initial GPU clones add a small diagnostic cost shared by both variants; no full-serving throughput claim is made.

| Run | GPU | B1 median prefill ms | B2 median prefill ms |
| --- | ---: | ---: | ---: |
| Candidate | 2 | 66.041 | 129.762 |
| Control | 6 | 63.542 | 124.674 |
| Candidate swapped | 6 | 65.377 | 130.491 |
| Control swapped | 2 | 64.085 | 129.198 |

B2 has M16384, so it is deliberately an unchanged-path control. Its variability cautions against interpreting small differences in isolated timings. Candidate B1 is slower than control on both GPUs. The selected standalone gains did **not** survive this short model check; no serving or GSM8K run was launched.

Telemetry starts late in the initial pair but covers the entire swapped pair. `model-telemetry.csv` records SM/memory clocks, power and temperature. It does not establish the cause of all timing variation. Logs/results and `model_summary.json` preserve each run.

A direct module call could remove Python runner bookkeeping, but the existing C++ `run_with_algo` still constructs `GemmDescriptors` every invocation. There is no exposed persistent-descriptor API in this wrapper. Source evidence: installed `flashinfer/gemm/gemm_base.py:1329-1452` and `flashinfer/data/include/flashinfer/gemm/mm_bf16_cublaslt.cuh:58-69,85-103,114-129`. No custom wrapper/kernel was added on the strength of the negative model screen.

## Read-only runtime audit

Normal prefill/mixed logits are already pruned to `cumsum(extend_seq_lens)-1` before the LM head (`python/sglang/srt/layers/logits_processor.py:679-685`, then `_get_logits` near 604). There is no redundant full-sequence vocabulary projection in this workload.

Observed Torch defaults were BF16 reduced-precision reduction enabled (including split-K), FP16 accumulation disabled, TF32 disabled. Toggling reduction policies changes numerical behavior, so this investigation did not use them as performance knobs. The FlashInfer BF16 Lt wrapper exposes no math-SM-count argument. No precision change, environment installation or production edit occurred.

Remote evidence: `/root/qvl/experiments/cublaslt-tactics`. The telemetry process and all owned model jobs have stopped. Reproduction command/environment are in `commands.json`.

## Lower-level wrapper overhead

`overhead.py` compares eager submissions of the four M8192 projections using persistent tensors, outputs and a 32MiB workspace. It runs six shuffled-order rounds of 20 four-projection groups after warmup. This is a stream-ordered submission group, not a model data-dependency simulation. Both paths use identical selected algorithms; the direct path caches the BLAS handle and serialized algorithm buffers and calls `mm_bf16_cublaslt_run_with_algo` directly. It still recreates the C++ descriptors internally.

Median CPU submission time per projection is 20.91us through the runner versus 11.80us directly, a 9.11us saving. GPU-event time per projection is 299.50us versus 300.88us: no GPU improvement. This is below the agreed 20us/projection threshold for another model run. Persistent descriptors remain a possible future implementation idea, but their benefit is unmeasured; no change is proposed here. `overhead.json` retains all rounds.
