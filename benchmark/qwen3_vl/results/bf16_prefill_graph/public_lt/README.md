# Public Lt GEMMs for graph buckets: reject before model

This tests the distinct hypothesis that graph capture removes eager Python overhead from faster prefill Lt GEMMs. Public FlashInfer0.7.0.post1 `mm_bf16(backend="cublaslt")` tunes each of QKV/O/gateup/down at M16384 and8704. Both weights and activations remain BF16. Eight rotating weights exceed L2, and four alternating CUDA graph timing rounds compare Torch with public Lt. The existing default compiled caches are reused. No production files change.

| M | Torch sum of four median GEMMs (us) | Public Lt sum (us) | Interpretation |
| --- | ---: | ---: | --- |
| 16384 | 2239.199 | 2338.896 | 4.45% slower |
| 8704, corrected lookup | 1241.886 | 1227.931 | 1.12% faster, drift caveat |

All eight final shape/projection checks are bitwise equal to Torch. At M16384 only O is meaningfully faster (243.458→224.884us); switching only O saves about0.83% of these four GEMMs. Applied to the measured mixed GEMM share49.3% and mixed forward share27.55%, the optimistic whole-workload bound is about0.11%, below the preceding graph-only0.13% regression.

The corrected8704 down projection appears faster, but Torch drifts265→354us and Lt269→340us over the short run. QKV/O/gateup regress.8704 buckets cover few observed mixed batches. This does not justify a model experiment or broader sweep.

## Public bucket lookup pitfall

The initial8704 experiment tuned successfully inside `autotune(tuning_buckets=(8704,), round_up=False)`, but the default lookup outside that context did not cover the non-default bucket and emitted four fallback warnings. Its timing report is retained as invalid evidence of *tuned* performance.16384 uses a default bucket and emitted no warnings.

`corrected.py` retains the same bucket override while capturing the public calls using `autotune(False, tuning_buckets=(m,), round_up=False)`. It performs no tuning inside capture and emits zero fallback warnings. No heuristic index is manually selected. The production M128 path uses an exact default bucket and is unaffected by this specific pitfall.

Raw alternating timings, automatic cache selections and numerical results are retained. GPU3 ran M16384; GPU4 ran both8704 attempts. All owned processes exited. No graph+Lt model/serving run was launched because the aggregate kernel gate failed.
