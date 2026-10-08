# Rejected M16 QKV fusion screen

The bounded two-token-tile fusion is rejected. Across all eight paired rounds on GPU 6, it is slower than both actual production and the unfused identical-GEMM control. No second-GPU confirmation, model, serving, or profiling was run. Production B8 code is unchanged.

| Full QKV + norm/MRoPE/cache + TRT attention sequence | Median µs per layer |
| --- | ---: |
| Production F.linear | 88.919837 |
| Identical split-K GEMM, separate preparation | 92.494845 |
| Fused two-token-tile variant | 126.599248 |

The fused sequence increases latency by **42.37%** versus production. Aggregate sequence timing does not identify the cause; no scheduling, bandwidth, or resource explanation is asserted.

## Why this was a new bounded test

The earlier M16 QKV candidate `(128,16,2,6)` improved isolated GEMM by approximately 8.9%, but regressed model timing by 0.451% and changed 56/256 free-running synthetic tokens. That unfused candidate remains rejected; this screen did not repeat it as a proposed optimization.

Adding the new 128×16 BF16 epilogue tile to that tactic would require approximately 233,728 shared-memory bytes, exceeding the 227 KiB budget. Instead, this screen reuses the proven B8 `(128,8,2,6)` tactic for two token tiles. Positions and cache slots use `global_token = n_idx * 8 + local_token`; shared tile and output-tile addressing remain local. The 128×8 scratch is 2 KiB. No tactic search or alternative tile sweep was performed.

## Correctness and numerical distinctions

All 36 layers pass **bitwise** complete QKV, full K/V cache, and dependent TRT attention comparisons against the unfused **identical split-K GEMM** control, both initially and after changed-input graph replay. Caller-owned output identity is asserted. Three position axes differ, and the last slot is negative, testing the second token tile and skip-write behavior.

The split-K reduction differs from production F.linear. Neither precision labels nor the small reported errors establish numerical equivalence to production. Across 72 layer/check cases:

| Compared output versus production | Largest max absolute error | Largest NRMS |
| --- | ---: | ---: |
| Q after norm/MRoPE | 0.0625 | 0.000216184 |
| K after norm/MRoPE | 0.125 | 0.000388344 |
| V | 0.015625 | 0.000159702 |
| Attention output | 0.015625 | 0.000990795 |
| Written K-cache rows | 0.125 | 0.000401585 |
| Written V-cache rows | 0.015625 | 0.000164696 |

None of these production comparison cases is bitwise equal. `gpu6/report.json` retains every layer's max absolute error, NRMS, changed-element count and element count. Cache error statistics cover valid written rows; separate full-buffer equality flags are also recorded. No model-token or accuracy equivalence is claimed.

## Protocol and provenance

- GPU 6 and GPU 7 were verified empty before launch. Only GPU 6 was used. Worker PID 644718 is terminal; both GPUs were empty afterward.
- Source imports come from frozen `/root/qvl/sglang-qkv-integrated`; only the external `/root/qvl/experiments/qkv-m16-screen/m16_epilogue.py` extends rows and token addressing.
- Actual 36-layer QKV/norm weights, model revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`, M16/N6144/K2560, Q32/KV8/D128, BF16 tensors and HND page32 cache.
- The harness asserts M16 QKV is absent from both the production split-K tuning map and TGV selection. Its production arm calls actual `torch.nn.functional.linear`, followed by unchanged production preparation and TRT attention.
- Distinct projection weights total 1,132,462,080 bytes. All arms retain disjoint full-length cache buffers initialized identically. Weights and cache buffers rotate across all 36 layers; no explicit per-kernel L2 flush is claimed.
- Compilation and warmup precede timing. Eight counterbalanced rounds replay each 36-layer retained CUDA graph 60 times using CUDA events. Graphs, outputs, metadata, inputs and weights remain alive throughout. The GPU 7 opposite-order confirmation was intentionally not launched after the clear negative.
- `source_manifest.json` records local source and harness hashes; `remote_provenance.json` records actual imported remote source hashes, installed GEMM hash, package versions and post-run GPU snapshot. The snapshot is not time-series clock telemetry.

## Reproduction

Copy the two `.py.txt` scripts to an isolated directory as `screen.py` and `m16_epilogue.py`, then run:

```sh
CUDA_VISIBLE_DEVICES=6 MAX_JOBS=4 ORDER=0 RUN_ROOT=/root/qvl/experiments/qkv-m16-screen/gpu6 PYTHONPATH=/root/qvl/sglang-qkv-integrated/python /root/qvl/venv-sgl/bin/python -u screen.py
```

The pinned checkpoint path is explicit in the script. No production source, benchmark workload, or precision setting was changed.
