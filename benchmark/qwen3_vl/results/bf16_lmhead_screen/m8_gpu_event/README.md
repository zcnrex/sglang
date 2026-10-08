# M8 LM-head GPU-event model diagnostic

Baseline retained graph median 3.531907ms versus candidate 3.522798ms; median paired throughput improvement **0.2573%**, all six pairs positive. This is approximately 9.1 us per fixed-state decode graph, not a serving gain.

GPU 2 worker 481484 completed. Both arms use current `/root/qvl/sglang-lmhead-production`, including production M4 optimization. The 4229-file Python source manifest matches the local production tree exactly at retrieval. Only an external exact M8 BF16 LM-head hook differs; other shapes including M4 delegate to original dispatch. Public startup autotune bucket 8 uses actual tied weight and selects tactic 2 (cache retained), with no timed tuning or hardcoded tactic. One actual M8 candidate graph-capture dispatch is recorded.

After matching fresh 8K prefills, seed 123 first/fifteenth decode logits are bitwise equal and all 128 generated tokens match. A second actual model sequence with changed seed 124 also has bitwise final logits and matching final tokens. These bounded numerical checks are not full accuracy equivalence.

Each graph pair replays the retained baseline/candidate graphs 256 times under CUDA events, after 40 alternating common warmup pairs. Six pairs alternate order. Both graphs use the same final real-decode inputs, overwrite the same KV positions and do not simulate growing-context generation. All 12 snapshotted shared tensor buffers compare unchanged after timing, including token IDs, lengths, positions and output logits. Actual 15-step event times remain in the report but include host gaps; profiled traces are diagnostic and excluded from paired timing.

Raw archive preserves scripts, logs, source manifest, cache, reports and compressed traces. Large numerical tensors remain remote. No production edits or serving run form part of this result.
