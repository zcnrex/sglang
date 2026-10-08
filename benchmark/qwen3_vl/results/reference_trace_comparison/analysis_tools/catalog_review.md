# Catalog review before new captures

No exact kernel annotations are populated until new trace names and pinned runtime sources are available. The prior parser fixture already used293 real kernels joined to existing cudaGraphLaunch correlation18/graph209, replicated into five synthetic marked scopes; it is a parser test, not a benchmark or measured five-forward trace.

Relevant source-backed catalog families to check:

- Residual-add RMSNorm: distinguish already fused residual norm from standalone first-layer/final norm; a standalone norm does not by itself prove missed fusion.
- Q/K RMSNorm, full-width MRoPE, paged cache write: current SGL small-batch projection epilogue includes all three plus QKV GEMM. Larger batches use different paths. Inspect exact eligibility and launch membership before crediting/removing standalone kernels.
- vLLM QKNormRoPEFusionPass and RopeKVCacheFusionPass: catalog entries name possible compiler passes, not evidence that pinned92044241a exposes/enables them for this model/shape. Check pinned source/config and trace.
- SiLU-and-mul is already an activation/multiply fusion; it is not automatically a GEMM epilogue. BF16 rounding between GEMM and activation is part of the numerical contract.
- Generic nvjet/CUTLASS names do not establish FP8. Verify weights, activation/output dtypes, dispatch and correlated CPU op; FP32 split-K partial reduction is compatible with BF16 inputs/output.
- fmha names identify attention computation only after source confirmation; treat RoPE/cache preparation as a different operation. QkvBfloat16/OBfloat16 naming can support but does not replace configuration/source dtype proof.
- FP8/FP4 norm/activation/attention quantization catalog entries are outside this BF16 comparison's precision contract.
- TP overlap/allreduce fusions require communication. TP1 cannot gain from removing absent collectives. Observed inter-stream concurrency is not proof of additional safe overlap; attention/GEMM dependencies and memory bandwidth matter.

Preserve raw triage tables; document corrected classifications separately with exact kernel names and source commit/path/line. Record version/backend differences and graph-on versus mapping-only evidence explicitly. Model scopes must exclude sampler kernels, while outer worker scopes and unassigned GPU activity retain any first-four inter-forward sampling. Last-step sampling absent from capture must not be estimated as measured.
