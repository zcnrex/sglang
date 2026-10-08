# Shared normalization engine compatibility

The final default specialization passed all 560 existing `test_qknorm.py` cases on GPU 6 in 124.27 seconds. The initial shared-engine revision also passed 560 cases. These existing tests use their original tolerances; exact MRoPE comparisons and alias checks are recorded independently.

Command: `CUDA_VISIBLE_DEVICES=6 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/root/qvl/test-deps:/root/qvl/sglang-mrope-reuse-cuda-v2/python /root/qvl/venv-sgl/bin/python -m pytest -p no:cacheprovider -q /root/qvl/sglang-mrope-reuse-cuda-v2/test/registered/kernels/ops/layernorm/test_qknorm.py`.

The final source hashes and lossless snapshots are included. `precommit.log` records a clean check with no source modifications. The parent folder's three-test/five-subtest Gemma regression applies to the rejected Triton adaptation, not the final implementation.

A read-only `cuobjdump --dump-sass` inspection of the final cache-writing module found one device call, no local-memory loads/stores. The call target starts signed 64-bit division machinery (`I2F.U64.RP`, `MUFU.RCP`), consistent with runtime page addressing. It is not evidence of an out-of-line epilogue function. No force-inline benchmark was run and no such modification was adopted.

The disassembled module was `/root/.cache/sglang/jit/sm103a/sgl_kernel_jit_fused_qk_norm_mrope_true_True/build-497d86cea84d9732/deps-03908839aac55030/sgl_kernel_jit_fused_qk_norm_mrope_true_True.so`. The compressed disassembly retains all instructions, including the unrelated existing rotary entry compiled in the same module.

Readable test logs have hook-normalized trailing whitespace. The `.raw.log.gz` copies retain exact original bytes, verified by `raw-log-hashes.json`.
