# M16 BF16 gate/up public Lt screen

Public cuBLASLt yields only a tiny standalone difference, insufficient to justify model/serving integration.

| GPU | Torch median us | Lt median us | Throughput difference | Positive pairs |
| --- | ---: | ---: | ---: | ---: |
| 4 | 17.960668 | 17.903133 | +0.321% | 8/8 |
| 5 | 17.945267 | 17.911134 | +0.191% | 7/8 |

The absolute saving is 0.034–0.058 us per layer, or approximately 1.2–2.1 us over36 layers before any dispatch overhead. This arithmetic is not a measured model gain. No model or serving test was launched.

Only M16/N19456/K2560 was tested. Production reaches F.linear for this exact shape; standalone Torch mm with caller-owned BF16 output uses its same dense matrix operation under graph replay. Candidate is public `flashinfer.gemm.mm_bf16(..., backend="cublaslt")`, tuned once before capture with bucket16 and round_up=False. Both GPUs independently selected public Lt tactic0; no tactic was hardcoded. Numerical checks and all graph timing occur after tuning exits.

Actual BF16 gate/up weights from pinned checkpoint layers0/17/35 are concatenated in original gate-then-up order. The three independent weights total298,844,160 bytes, exceeding B300 L2. Each graph cycles all three weights and distinct seeded BF16 inputs;80 replays give240 calls per event interval. Eight timing rounds alternate order, with GPU5 starting opposite GPU4. Warmup replays precede measurement. Graphs, tensors and callables are retained; DOT files prove graph kernels exist.

Initial and changed-input graph outputs are bitwise equal across Torch and Lt for all three layers. Both are compared against explicit FP32 matrix multiplication with TF32 disabled; maximum normalized RMS error is0.0016615, below0.005. No model accuracy equivalence is claimed. Weights, activations and outputs remain BF16.

Remote `/root/qvl/experiments/m16-gateup-screen/gpu4` and `gpu5`; workers478258 and478376 completed and disappeared from process listings. Raw evidence archive preserves original scripts and logs; copies use `.py.txt`. No production source changed. Package/cache metadata is retained separately per GPU.
