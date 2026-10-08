# Shared CUDA QK normalization epilogue: second validation

All 32 standard cases pass strict bitwise comparison, including initial and changed-input/position graphs, both multimodal axis maps, stride128 positions, disjoint HND page32 caches, negative slots and actual normalization weights across 36 layers. Eight additional cases, one per batch, pass with value aliasing K; both implementations store the rotated K into the value cache. Complete Q/K/V and cache buffers are compared.

This version adds a nonpersistent schedule only for the rotary epilogue and places cache writes after the common Q/K store. Existing default normalization behavior is retained; its separate regression suite is owned by the implementation agent. Exact frozen source hashes and source copies are included.

Performance still regresses in most cases. Numerical acceptance does not imply latency acceptance. No model or serving run was launched by this benchmark. Timing uses eight opposite-order rounds, 100 retained-graph replays per arm, 36 disjoint layers and GPU events. Reset occurs outside timing; Q/K values evolve in place across replays. This does not measure dependent attention or end-to-end serving.

Interleaved medians, microseconds per layer:

| Batch | Original no store | Shared no store | Original cache store | Shared cache store |
|---:|---:|---:|---:|---:|
| 1 | 1.3104 | 1.7122 | 1.7078 | 2.1624 |
| 2 | 1.3105 | 1.7643 | 1.8229 | 2.2766 |
| 4 | 1.3128 | 1.7675 | 1.8236 | 2.2770 |
| 8 | 1.3716 | 1.7799 | 1.8413 | 2.2831 |
| 16 | 1.4856 | 1.8275 | 1.9409 | 2.3431 |
| 32 | 1.5688 | 1.9407 | 2.0015 | 2.4538 |
| 64 | 1.9966 | 2.2271 | 2.4319 | 2.7259 |
| 128 | 3.2481 | 3.1367 | 3.4325 | 3.7083 |

Exact pre-normalization originals are retained in `raw-evidence.tar.gz` and remotely at `/root/qvl/experiments/bf16-qk-norm-rotary-cuda/v2-frozen`. No large model tensors were exported.
