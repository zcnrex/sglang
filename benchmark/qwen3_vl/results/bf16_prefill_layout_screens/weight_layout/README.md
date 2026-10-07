# BF16 weight-layout screen

Rejected: once-pretransposed NN weights and NT weights with row stride K+128 did not materially improve any tested projection. No activation copies, runtime changes, or serving tests were introduced. All output comparisons were bitwise equal.

Tests cover QKV (N6144/K2560), O (N2560/K4096), gate/up (N19456/K2560), and down (N2560/K9728), with M8192 and M16331. Three alternating CUDA-graph timing rounds per shape used the same GPU and BF16 tensors for all variants.

| Projection, M16331 | Native NT (µs) | Pretransposed NN (µs) | Padded NT stride (µs) |
| --- | ---: | ---: | ---: |
| QKV | 366.95 | 366.89 | 366.68 |
| O | 252.67 | 253.71 | 253.35 |
| Gate/up | 1122.51 | 1122.00 | 1122.64 |
| Down | 626.47 | 626.74 | 626.93 |

One-time warmed transpose costs were approximately 0.17–0.49 ms per weight. Initial transpose events include first-use initialization and are not steady-state costs. Keeping both layouts would add one full weight copy (20–100 MB per tested projection); stride padding adds 0.66–4.98 MB. Reports retain exact byte counts and raw checks.

Source inspection found FlashInfer BF16 cuBLASLt hardcodes the native NT layout, with up to 100 heuristic algorithms and default tactic zero. Its algorithm tuning was investigated separately. Existing SGLang weight-row padding targets XPU MoE L3 aliasing and does not establish a B300 benefit.

Reproduce with the experiment's pinned PyTorch environment on three idle B300 GPUs:

```sh
CUDA_VISIBLE_DEVICES=3 /root/qvl/venv-sgl/bin/python layout.py --shard 0
CUDA_VISIBLE_DEVICES=4 /root/qvl/venv-sgl/bin/python layout.py --shard 1
CUDA_VISIBLE_DEVICES=5 /root/qvl/venv-sgl/bin/python layout.py --shard 2
```

The scripts are standalone research probes using synthetic fixed-seed inputs. Each variant excludes one-time weight preparation from the timed GEMM.
