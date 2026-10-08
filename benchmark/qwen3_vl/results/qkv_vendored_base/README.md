# Vendored split-K prerequisite gate

The existing in-tree engine is a viable plain-GEMM base for the exact B8 QKV tactic `(128, 8, 2, 6)`. No fusion was implemented or benchmarked in this gate.

| Implementation | Median GPU time per layer |
| --- | --- |
| Installed FlashInfer | 6.124018 µs |
| In-tree vendored engine, plain epilogue | 6.124525 µs |

The difference is −0.0083% speedup, effectively identical at this measurement resolution. All 36 layer outputs were bitwise identical, both initially and after changing every input and replaying retained CUDA graphs. Output buffer identity assertions passed. Both use BF16 input, weight, and output with PDL enabled.

## Protocol

- GPU 4, NVIDIA B300; verified empty before launch and after completion.
- Worker PID 619121; terminal result in `run.log` and `report.json`.
- Qwen3-VL-4B-Instruct revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`.
- All 36 real layer Q/K/V weights concatenated in production Q/K/V order; M8, N6144, K2560.
- Distinct weights total 1,132,462,080 bytes, exceeding L2 capacity. Sequential full-model graph replay provides rotating cold weights, not an explicit cache flush before every individual GEMM.
- Exact single tactic only; no tuning or alternative search. The in-tree validator accepted it before allocation/compilation.
- Startup compilation and warmup excluded. Each retained graph contains 36 layer GEMMs; graph DOT files are included. Eight counterbalanced rounds use 100 complete graph replays per arm and CUDA-event elapsed time, divided by 3,600 GEMMs.
- Inputs and output buffers remain alive throughout capture and replay. Changed-input correctness precedes timing. No serving, model-forward, or dependent-attention integration claim is made.

The frozen in-tree source is `vendored.py.txt`; its SHA256 and the installed FlashInfer source SHA256 are in `report.json`. The wrapper uses `run_splitk_dense` with its default plain `none` epilogue and does not edit either implementation. `environment.txt` records the post-run environment; clock values are a snapshot, not time-series telemetry.

## Reproduction

Copy `run.py.txt` and `vendored.py.txt` to an isolated directory as `run.py` and `vendored.py`, then run:

```sh
CUDA_VISIBLE_DEVICES=4 MAX_JOBS=4 PYTHONPATH=/root/qvl/sglang-m16-down-production/python /root/qvl/venv-sgl/bin/python -u run.py
```

The pinned model path is explicit in the script. Remote artifacts were generated under `/root/qvl/experiments/qkv-vendored-base`. No production source files were modified.
