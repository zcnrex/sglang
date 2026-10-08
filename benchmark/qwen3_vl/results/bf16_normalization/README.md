# BF16 normalization standalone gate

The installed residual RMSNorm path is the baseline. The candidate uses an independent FP32 row reduction, BF16 residual/output stores, and explicit PDL wait/launch. Same precision does not imply bitwise normalized outputs: residual stores are bitwise, while rare output differences reflect reduction order.

| Batch | Baseline us | Candidate us | Reduction |
|---:|---:|---:|---:|
| 1 | 1.667 | 1.285 | 22.94% |
| 2 | 1.795 | 1.411 | 21.39% |
| 4 | 2.189 | 1.411 | 35.53% |
| 8 | 2.208 | 1.283 | 41.88% |
| 16 | 2.226 | 1.283 | 42.36% |
| 32 | 2.189 | 1.286 | 41.23% |
| 64 | 2.243 | 1.347 | 39.94% |
| 128 | 2.246 | 1.411 | 37.18% |

The production eligibility list excludes B1 and prefill, where the initial screen was flat. It includes only tested batches 2/4/8/16/32/64/128, contiguous nonoverlapping BF16 tensors, hidden size 2560, epsilon 1e-6 and SM103. Unsupported layouts, gradient inputs, outer compilation, and deterministic mode retain the existing path.

Numerical validation covered 35 zero, small, normal, large and cancellation cases plus 14 changed-input CUDA-graph replays. Residuals were bitwise. Against the FP32 reference, each candidate error was bounded by baseline error plus one BF16 output ULP and 1e-6. This is an explicit error envelope, not a claim of output equivalence or full-model accuracy.

Timing uses retained graphs containing 32 sequential in-place calls and 20 replays per event interval, eight opposite-order rounds. This deliberately measures a dependent chain; it is not serving throughput. Prefill 16384 was approximately 51 us for both paths. GPU 6 ran initial/numerical gates and GPU 7 the PDL confirmation.

Existing CPU namespace/fused-op/dispatch tests: 123 passed. External fallback probes: 24 passed with fake tensor metadata and a stub launch, not GPU integration. Precommit passed on the three production files. Actual production-model and serving gates remain separate.

## Rejected after model validation

No normalization implementation is accepted. The production-shaped model gate failed strict numerical equality despite matched startup tactics and bitwise prefill. Reported decode checkpoints reached maximum absolute logit differences of 0.125 at B8 and 0.1875 at B16, maximum KL 0.00178, and 75% top-1 agreement at one B8 checkpoint. B1 remained unchanged and bitwise. Standalone error bounds did not establish acceptable full-model behavior. No accuracy gate passed for promotion. The rejected three-file patch is archived here; only that patch was removed from production.

Exact-arithmetic staging and early-signal follow-ups are archived in the neighboring `bf16_normalization_staging` directory. They were bitwise but provided no meaningful accepted speedup. No further normalization GPU work is planned.
