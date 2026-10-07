# BF16 prefill GEMM / decode attention overlap screen

Standalone B300 tests found no useful critical-path reduction from overlapping a prefill gate/up projection with decode attention. No serving or production source changes were made.

Inputs: BF16 gate/up GEMM `[M, 2560] @ [2560, 19456]`, and independent BF16 TRT attention with batch 128, context 8192, 32 query heads, 8 KV heads, head dimension 128, HND cache and page size 32. Outputs matched the serial references bitwise. Each measurement uses 30 CUDA-graph replays, repeated five times with alternating measurement order.

| Streams / SM allocation | GEMM M | Serial median (ms) | Concurrent median (ms) | Median paired speedup |
| --- | ---: | ---: | ---: | ---: |
| Ordinary streams | 8192 | 1.0723 | 1.0761 | 0.9996x |
| Ordinary streams | 4096 | 0.8327 | 0.8250 | 1.0080x |
| Attention 112 / GEMM 32 SMs | 8192 | 1.0447 | 2.1095 | 0.4953x |
| Attention 96 / GEMM 48 SMs | 8192 | 1.0567 | 1.5094 | 0.7001x |

The green-context API works on SM 10.3 through `sgl_kernel.spatial.create_greenctx_stream_by_value`. Both explicit partitions leave four of the device's 148 SMs unused. Serial controls use the full device. Partitioning worsened the critical path; ordinary stream overlap gained less than 1% at the smaller size. The bounded experiment stopped without full-layer or serving tests.

Reproduce on an idle B300 with the pinned experiment environment (PyTorch, FlashInfer and sgl-kernel installed):

```sh
CUDA_VISIBLE_DEVICES=3 /root/qvl/venv-sgl/bin/python overlap.py
CUDA_VISIBLE_DEVICES=3 /root/qvl/venv-sgl/bin/python green_overlap.py
```

The scripts allocate synthetic inputs with a fixed seed. They capture fork/join stream dependencies in CUDA graphs and check output equality before timings. The JSON reports summarize the accompanying raw logs. These are research probes, not production kernel implementations.
