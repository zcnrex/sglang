# Read-only public Lt integration design

Installed FlashInfer provides `flashinfer.gemm.mm_bf16(a, b, bias=None,
pdl=False, out=None, out_dtype=torch.bfloat16, backend="cublaslt")`.
Here b is column-major weight.T. No heuristic index need be embedded in runtime.

The installed `_BF16_GEMM_SM100_TUNING_CONFIG` in gemm_base.py sets
`use_cuda_graph=True`, `use_cold_l2_graph_replay=True`, and
`use_cold_l2_cache=True`. Thus ordinary public `flashinfer.autotune(...)`
already uses cold-L2 profiling for this operator. Earlier concern about a
necessarily warm-cache default does not apply to this pinned implementation.
The separate experimental public API also permits explicit policy:
`flashinfer.autotune_v2(measurement_policy=flashinfer.MeasurementPolicy(
execution_mode="cuda_graph", cold_l2=True))`. Adopting v2 is unnecessary for
this change and would introduce another persistence mechanism.

FlashInfer obtains device L2 capacity through
`torch.cuda.get_device_properties(device_id).L2_cache_size`. Torch device
properties are cached metadata; querying this does not require a benchmark.
No new device query or GPU work was performed for this design audit.

SGLang `disable_flashinfer_autotune` defaultsFalse; there is no required
additional enable flag. `should_run_flashinfer_autotune` currently admits
selected MoE/FP4/FP8 cases, excluding ordinary dense BF16. `BaseRunner.warmup`
executes autotuning before graph capture. `initialize_bf16_gemm_config` runs
earlier, before loaded-weight-aware tuning is possible.

A narrow integration would retain the existing backend selection, restrict
SM103/BF16/M128 and the validated gate-up/down shapes, and add a preparation
helper at pre-capture warmup. The helper should tune representative loaded
weights with public mm_bf16 under the existing autotune context, then freeze
configuration for live execution. Existing cache load/save coordination should
be reused; independently entering autotune(cache=...) can clear already-loaded
records. Cache fingerprints cover GPU and library/runtime versions. Include
the BF16 backend policy in the SGLang cache identity if the new path uses it.

Simply adding the dense eligibility predicate is insufficient: graph-runner
`_autotune_buffers()` supplies only max_bs, which may be256/512 rather than128.
A wrapper restricted to M128 then never executes. Explicit selected-shape
preparation avoids a new model hook and avoids changing dummy-forward buffer
contracts. No autotuning may occur in CUDA capture or on a first live request.
Disabled tuning, skipped bf16_gemm, unsupported tensors, missing preparation,
and deterministic inference should retain the previous implementation.

This is a proposal, not implemented code. Validate the actual public-autotuner
selection, numerics and serving performance before promotion. Do not reuse
external experimental heuristic indices as stable algorithm identifiers.
