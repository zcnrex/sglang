# Graph-buffer positions stride fix

The public QKV fusion previously rejected positions with shape `(3, 8)` and stride `(128, 1)`, as produced by graph capacity 128. The fix exempts only positions from the contiguous-input requirement and requires column stride 1 and row stride at least 8. All other tensors retain contiguous requirements. Alias checks now include the full positive-stride bounding span, including row gaps. The compile key already includes strides; an uncompiled layout still falls back during capture.

GPU 1 on `chunan-b300-8` ran the isolated revised wrapper over `/root/qvl/sglang-qkv-integrated/python`; no live source was changed. Driver PID 655136 and both sequence children completed. The source SHA256 is `4c9d946f0b826b8607760c3d4a86c498efaf720812214938666fcab2ceb584ec`.

Both stride 128 and stride 8 passed 36 actual model weights, initial execution and changed-input/changed-position retained graph replay. QKV, full K/V caches and dependent TRT attention outputs were bitwise identical to the existing unfused identical-GEMM reference. Positions used distinct axes, and one cache slot was negative. Caller output identity was asserted. These are correctness checks, not performance measurements.

Twenty additional actual GPU wrapper checks passed. They cover unsupported shapes/dtypes/layouts, current-device and grad/compile guards, separate stride-key warmup and reuse before capture, unseen-stride cold-capture fallback, column stride 2, overlapping/broadcast rows, and malicious output or cache aliases at positions row 2 beyond the old numel-based extent.

Run `sequence.py.txt` as `sequence.py` with `POSITION_STRIDE=128` or `8`, and `guards.py.txt` as `guards.py`, beside the archived wrapper restored as `qkv_norm_mrope.py`. Environment: `CUDA_VISIBLE_DEVICES=1`, `PYTHONPATH=/root/qvl/sglang-qkv-integrated/python`, `MAX_JOBS=4`; interpreter `/root/qvl/venv-sgl/bin/python`. Model snapshot is pinned inside the sequence script. Full runtime admission/model validation is separate.

Pre-commit subsequently normalized one generator-expression line. The final production file hash is `fbf528dfcd39f1b0e31a79017fd70a40f1c88628a43d9453a41f38dd16d9e52c`; AST identity with the tested source was checked explicitly. All targeted pre-commit hooks passed.
