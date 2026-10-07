# QKV / O / down BF16 high-M TGV screen

Bounded standalone screen: 12 SGLang CuTe TGV tactics for each of three projections at M64 and M128 (72 configurations). FlashInfer direct, cluster Split-K and warp Split-K kernels require M<=32 and were excluded. No production changes were made.

Each test rotates 16 independent BF16 weight tensors, totaling 335–797 MB per shape, to exceed L2 capacity. A CUDA graph executes all 16 GEMMs; reported times are normalized per GEMM. Three timing rounds alternate candidate/baseline order. The baseline uses the production CuTe selector and chosen tactic where enabled, otherwise torch.mm. Compilation is outside timing, with MAX_JOBS=4.

| Shape | M | Best tested tactic | Production baseline (µs) | Candidate (µs) |
| --- | ---: | ---: | ---: | ---: |
| QKV | 64 | 23 | 7.336 | 7.336 |
| QKV | 128 | 24 | 7.766 | 8.469 |
| O | 64 | 8 | 6.810 | 7.252 |
| O | 128 | 23 | 7.615 | 7.612 |
| Down | 64 | 8 | 12.695 | 13.436 |
| Down | 128 | 23 | 14.893 | 14.162 |

Only down M128 showed a useful standalone gain (5.2% throughput improvement). It was passed to the separate cuBLASLt comparison before any model-level selection. These measurements do not establish a serving improvement.

Correctness checks compare each candidate against the actual production baseline on all 16 input/weight pairs. The gate is normalized RMS error below 0.005, computed in FP32 from BF16 outputs. QKV and O best candidates recorded zero normalized error on these inputs. Down best candidates recorded approximately 0.000228 at M64 and 0.000221 at M128; they are not bitwise equal. No separate exhaustive exceptional-value or bitwise-equality test was performed. The raw reports preserve every candidate's measured error.

Reproduce in the pinned experiment environment, using idle B300 GPUs:

```sh
CUDA_VISIBLE_DEVICES=4 MAX_JOBS=4 PYTHONPATH=/root/qvl/sglang-perf/python /root/qvl/venv-sgl/bin/python highm.py --shape 0
CUDA_VISIBLE_DEVICES=5 MAX_JOBS=4 PYTHONPATH=/root/qvl/sglang-perf/python /root/qvl/venv-sgl/bin/python highm.py --shape 1
CUDA_VISIBLE_DEVICES=6 MAX_JOBS=4 PYTHONPATH=/root/qvl/sglang-perf/python /root/qvl/venv-sgl/bin/python highm.py --shape 2
```

Tactic IDs refer to the experiment checkout's `sglang.kernels.ops.gemm.cutedsl_bf16_gemm` implementation, not FlashInfer's smaller tactic list. Tested IDs: 1, 7, 8, 9, 10, 14, 15, 22, 23, 24, 27, 28. The script is a research probe using private kernel entry points.
