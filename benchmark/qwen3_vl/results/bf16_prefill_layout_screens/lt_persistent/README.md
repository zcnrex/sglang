# Persistent descriptors and direct BF16 cuBLASLt calls

This is an external prototype and bounded model screen, not a production change or throughput claim. Source: clean `466c9e0f4073bcad6e4403c6f54cfc8c1821b867`; model: Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`; host: `chunan-b300-8`, GPU 2. Compilation used MAX_JOBS=8.

## Descriptor prototype

`persistent.cpp` caches cuBLASLt matrix and operation descriptors per thread, keyed by device, M/N/K and dtype. It admits only contiguous row-major BF16 input/weight/output matrices, which fixes the layout component of the cache key. Tensor pointers, output/workspace pointers, handles and streams are supplied per invocation; no tensor pointers or streams are stored in the cache. Computation remains `CUBLAS_COMPUTE_32F` with BF16 inputs/outputs and 32MiB workspace. The selected algorithms match the preceding `lt_tactics` screen.

`screen.py` builds the extension and checks four actual projection shapes at M8192. Cached and uncached paths, and a fresh-pointer/non-default-stream case for every shape, are all bitwise equal to Torch (12 checks). Eight shuffled timing rounds compare four-projection eager submission groups with persistent outputs/workspace; these are stream-ordered groups, not a complete model dependency simulation.

| Path | CPU submission us/projection | GPU-event us/projection |
| --- | ---: | ---: |
| FlashInfer runner | 20.528 | 305.296 |
| Direct FlashInfer module | 11.223 | 304.158 |
| Prototype cached descriptors | 11.564 | 305.467 |
| Prototype uncached descriptors | 11.787 | 302.460 |
| Torch mm with persistent output | 14.240 | 316.498 |

The controlled same-binding comparison shows only 0.223us CPU savings from descriptor caching and no GPU improvement. This rejects descriptor caching as a material optimization for these shapes. The direct module path separately removes Python runner bookkeeping, so it received a short model check.

## Alternating model screen

The direct-call patch changes only M8192 QKV/O/down projections, with tactic indices 2/2/6. Gate/up and every other shape use the existing dispatch. It caches algorithm bytes, workspace and BLAS handle, then invokes the existing FlashInfer module directly; it does **not** use the persistent-descriptor extension. Output allocation still occurs each call, so this includes model dispatch/allocation overhead.

`model_direct.py` loads one model on GPU 2, performs six common warmups alternating both implementations, then alternates six AB pairs of B1,input8192,output2. Both numerical captures use identical warmup prompts. Projection outputs and full last-token logits are bitwise equal; no correctness check or logits clone occurs in measured passes. Configuration remains BF16 weights/queries/KV, TRTLLM attention, HND, page32, mixed16k, disabled prefill graphs, no quantization or speculative decoding.

Initial medians: control 66.849ms, direct 66.366ms (0.72% lower). Four of six pairs favor direct, but individual differences vary by 1–4ms and order is always AB. This is insufficient evidence of a robust benefit.

`model_direct_balanced.py` repeats the common warmup and measures twelve pairs with pair order alternating AB and BA. Its separate logs and results retain this counterbalanced follow-up. Counterbalanced medians are control 66.729ms versus direct 67.225ms (0.74% slower); direct wins 6/12 pairs. Median candidate-minus-control difference is -0.730ms in AB pairs and +1.070ms in BA pairs. Identical-prompt logits are again bitwise equal. The initial small apparent gain is not robust to pair order, so no model-level improvement is established. Final summary is in `model_direct_balanced/summary.json`.

Remote evidence: `/root/qvl/experiments/cublaslt-persistent`. Compiled binaries and model logits tensors remain remote; compact numerical comparison results, raw timings, scripts, source, logs and GPU telemetry are archived here. No serving or GSM8K run was launched.
