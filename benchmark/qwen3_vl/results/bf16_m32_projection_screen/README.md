# M32 QKV and down split-K screen

Both admission-corrected configurations are slower than production in all six alternating timing pairs. No model or serving experiment follows; no production files changed.

| Projection | Tactic (mma_m, mma_n, split_k, stages) | Approximate production / candidate µs | Median paired speedup |
|---|---|---|---|
| QKV | (128,32,2,5) | 7.493 / 8.451 | −11.34% |
| Down | (128,32,4,4) | 12.075 / 13.533 | −10.73% |

Original QKV stages 6 requires 262280 B shared memory and down stages 5 requires 254088 B, exceeding 232448 B. These were rejected before CUDA launch/timing; original evidence remains under rejected_admission. Exactly one compiler-admitted stage reduction per shape was authorized; no further sweep.

Source /root/qvl/sglang-lmhead-production is the previously audited 5e1601731b overlay. Exact shapes are (32,6144,2560) and (32,2560,9728). Both selectors are asserted to choose F.linear. Sixteen real checkpoint layer weights rotate inside each CUDA graph, totaling 503316480 B for QKV and 796917760 B for down, exceeding L2. Pinned model revision ebb281ec70b05090aa6165b016eac8ec08e71b17; random input seed 511. BF16 inputs/weights/outputs and FP32 accumulation are preserved. Baseline uses actual F.linear; compilation is excluded. Graphs, captured tensors and inputs stay owned through replay. Three warmup replays precede 30 timed replays per pair, normalized over 16 GEMMs, with opposite order each round.

Changed-input graph checks pass. Candidate versus baseline maximum normalized RMS is .0001074 QKV/.0001172 down, max absolute difference .03125; outputs are not bitwise equal, but all checked projection argmax rows agree. First-layer FP32-reference normalized RMS is approximately .00166 for both methods. These diagnostics do not establish model equivalence. Raw reports retain full numbers.

Original workers 481933/481934 and corrected workers 482597/482598 completed; GPUs 4/5 released. Remote roots /root/qvl/experiments/m32-projection and /root/qvl/experiments/m32-projection-admitted preserve raw scripts/logs. No unrelated processes were stopped.
