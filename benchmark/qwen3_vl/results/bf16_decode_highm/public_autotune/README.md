# Public BF16 Lt autotuning

Independent fresh processes on physicalGPU0 andGPU1. Public
`flashinfer.gemm.mm_bf16(..., backend="cublaslt")` tuned under
`flashinfer.autotune(tuning_buckets=(128,), round_up=False)`. Installed
BF16 configuration uses cold-L2 CUDA-graph profiling. No heuristic index was
specified by the experiment. Library-produced cache snapshots retain the
selected runner/tactic and environment metadata as evidence, not production
constants or manually constructed cache records.

Both devices independently selected CublasltBf16GemmRunner tactic1 for gate-up
and tactic4 for downprojection. Eight rotating distinct weights exceed L2
(796917760bytes gate-up;398458880bytes down). Eight alternating graph timings
measure160calls per sample. Every output matches Torch bitwise, maxabs0.

| GPU | Operation | Torch us | Public autotuned us |
|---:|---|---:|---:|
|0|M128 gate-up|21.962|19.546|
|0|M128 down|15.258|14.325|
|1|M128 gate-up|21.914|19.526|
|1|M128 down|15.145|14.292|

The automatically selected configurations reproduce the cold-weight gains of
the earlier external index-based prototype. No production changes. Run the
script with a GPU label argument and CUDA_VISIBLE_DEVICES selecting that GPU.
Reports are written under `/root/qvl/experiments/public-autotune/`.
