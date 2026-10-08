# One startup-compiled dynamic-M callable

The external DynamicGateUp wrapper compiles the unchanged FFI producer once using symbolic M in fake A/C descriptors. Weight dimensions, the 256×256 MMA tile, two-CTA configuration, 128×64 epilogue, BF16 rounding and helper remain unchanged. Runtime calls accept Torch tensors directly and enforce 8192 ≤ M ≤ 16384. No per-row-count compilation, DLPack conversion or tile selection occurs in this runtime wrapper. The native BF16 reference and static comparator compile independently for their test shapes; they are not part of candidate dispatch.

GPU 0 worker PID 417025 passed bitwise comparisons against native BF16 materialization followed by existing SiLU for M = 8192, 8193, 8200, 16321, 16331, 16383 and 16384. Tests retain a 256-row output-tail canary and the unused second half of the full-width scratch. Both remain NaN. Changed-input CUDA graph replay also matches the static same-kernel result at M = 8192 and 16331; graph states, tensors and callables remain retained. The bounds guard separately rejects M = 8191 and 16385 in an external CPU check. This is not a claim of bitwise equality with cuBLAS/F.linear or full-model logits.

The initial attempt compiled successfully but rejected the first runtime call before launch because the fake weight descriptor had the wrong stride rank ordering. The corrected weight stride_order is (2, 0, 1), matching Torch shape (1, K, N) with transposed contiguous weights. The original rejection log is retained. No device crash occurred.

| M | Dynamic median µs | Static median µs | Median paired latency change |
|---|---:|---:|---:|
| 8192 | 596.109 | 604.529 | +0.041% |
| 16331 | 1311.966 | 1312.030 | −0.050% |

Eight alternating rounds after common warmup; clocks were not locked and visible drift affects the absolute medians at 8192. The per-round paired ratios show no material dynamic scheduling regression. These use one fixed weight and are a dispatch regression gate, not a new rotating-weight performance claim.

Frozen API: instantiate DynamicGateUp() once during startup, then call op(x3d, interleaved_weight_transposed3d, output3d). Shapes are (1, M, 2560), (1, 2560, 19456), (1, M, 19456); the first M×9728 output elements contain the activation. Runtime type/layout checks are enforced by the compiled TVM-FFI descriptor in addition to the Python row-range guard. One callable is shared across layers/weights and row counts.

Module: /root/qvl/experiments/native-epilogue-dynamic-m/native_dynamic_m.py, SHA256 952741f3d6239e5c2c2dbd1037b39828ad4416f9a8461d3af2fce85384e5d924. Frozen dependencies are adjacent. Exact sources are archived as .py.txt and unmodified in raw_evidence.tar.gz with graph DOTs, reports and logs. The API and results were handed to the profile agent for a separate real-model check before serving. No production edits or serving runs were made here; the worker finished.
