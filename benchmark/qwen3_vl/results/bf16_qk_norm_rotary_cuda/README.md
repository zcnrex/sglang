# Shared CUDA QK normalization and rotary validation

All 32 cases passed strict bitwise comparison: batches 1–128, contiguous/interleaved multimodal mappings, with and without HND page32 cache writes. Each case covers 36 disjoint layers with actual norm weights, adversarial dynamic-range inputs in two layers, distinct position axes with stride128, and changed-input/position graph replay including a negative slot. Timing restores valid cache slots.

The candidate reuses the existing CUDA normalization kernel through an epilogue policy. Source hashes and archived headers/wrapper are in `source.json`. Reference CUDA wrapper/header hashes match immutable d7b4556d6b. No model gate was launched from these results. The candidate is slower in most cases; numerical success does not establish performance acceptance. Alias semantics for value=K require the separately planned correction and regression check; this run uses disjoint packed Q/K/V slices.

Eight counterbalanced CUDA-event rounds each replay a graph of 36 operators 100 times. Input reset occurs outside timing; in-place Q/K evolves across repetitions. This is operator-only timing, not serving throughput or a producer/consumer dependency test.

Interleaved medians, microseconds per layer:

| Batch | Original no store | Shared no store | Original cache store | Shared cache store |
|---:|---:|---:|---:|---:|
| 1 | 1.3107 | 1.6813 | 1.7077 | 2.1623 |
| 2 | 1.3105 | 1.7076 | 1.8244 | 2.2468 |
| 4 | 1.3126 | 1.7102 | 1.8239 | 2.2456 |
| 8 | 1.3721 | 1.7364 | 1.8671 | 2.2828 |
| 16 | 1.4858 | 1.7707 | 1.9417 | 2.3432 |
| 32 | 1.5505 | 1.8820 | 2.0019 | 2.4552 |
| 64 | 1.9967 | 2.1334 | 2.4355 | 2.7070 |
| 128 | 3.2463 | 3.1342 | 3.4297 | 3.7516 |

Complete original evidence is retained in `raw-evidence.tar.gz` before local whitespace normalization and remotely at `/root/qvl/experiments/bf16-qk-norm-rotary-cuda`. No large model tensors were exported.
