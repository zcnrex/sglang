# Ragged native epilogue screen

The dominant mixed-prefill shape M16331 did not show a reproducible gain. No model or serving experiment follows this screen. Inputs, weights, intermediate rounding and outputs remain BF16; comparisons use actual production F.linear plus existing silu_and_mul. The same native producer materializing BF16 output plus activation supplies the bitwise numerical reference, which does not establish equivalence to cuBLAS accumulation.

| GPU | M | Production median, us | Candidate median, us | Mean paired latency reduction | Positive pairs |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 16331 | 1332.620 | 1342.016 | −0.447% | 3/8 |
| 5 | 16331 | 1326.021 | 1315.775 | +0.564% | 4/8 |
| 4 | 16384 | 1313.830 | 1290.702 | +2.122% | 5/8 |
| 5 | 16384 | 1320.709 | 1274.567 | +3.750% | 7/8 |

The confirmation builds all four paths in one process, retains their tensors, descriptors and compiled callables, then alternates forward/reversed path order over eight blocks. GPUs use opposite initial order. Forty common warmup rounds precede timing. Each timing interval contains 20 graph replays, each rotating four weights (398,458,880 bytes per shape, beyond L2). DOT files show four candidate kernel nodes. Telemetry records clocks, power and temperature. Source dispatch proof identifies F.linear and unquant.py SHA 0fa01b10b34bec46f4cca14b6dfb96fa67a07d94e68c1d08035b8c851740f431. No timing logging is active inside the captured kernels.

The earlier separate-shape screen showed M8200 ~8.25% lower latency, M16331 ~1.14% higher latency and M16384 ~3.53% lower latency. Its logs remain, but the same-process four-way confirmation governs the conclusion. The aligned result does not justify optimizing only an unrepresentative row count or advancing the dominant ragged shape. No additional tile sweep was performed.

## Correctness and failed attempts

The producer and fused epilogue use the same 256×256 tile, two CTAs and 2×1 cluster as the fixed-shape screen. Identity-coordinate predicates mask invalid output rows, while existing TMA input bounds handle the input tail. The previous admission check rejected non-multiple M before launching; old-no-predicate-admission logs preserve this failure. Only the M divisibility rejection was relaxed after adding row predicates; N/alignment guards remain. All four weights compare bitwise against the same native materialized BF16 output plus existing activation. Both shapes additionally retain 256 canary rows, and the candidate's unused half-buffer remains NaN.

The first four-way harness returned graphs but dropped ownership of external tensor allocations, DLPack descriptors and compiled functions. Both workers failed at the first common replay synchronization after eager correctness/canaries passed. Those fourway-gpu logs and invalid_lifetime_fourway.py are invalid timing evidence. The retained-state diagnostic synchronizes each individual replay, then checks bitwise outputs and canaries again; all pass.

Compute-sanitizer initially reported 32 CUDA_ERROR_INVALID_VALUE API lookup errors from cuGetProcAddress_v2 at HardwareInfo.__init__, hardware_info.py:34. No memory-access errors were reported. Repeating with `--report-api-errors no` retained memory checking and reported zero errors; this suppresses API lookup reports, not memory or launch errors. Both logs are preserved. These bounded checks support the lifetime correction, not an exhaustive kernel safety claim.

## Reproduction and provenance

Remote root: /root/qvl/experiments/native-epilogue-ragged. Python: /root/qvl/venv-sgl/bin/python. Source: /root/qvl/sglang-prefix-production/python. Environment: CUDA_VISIBLE_DEVICES=4 or 5, PAIR_OFFSET=0 or 1, MAX_JOBS=8, PYTHONPATH set to that source. Run native_tail_fourway_retained.py after reconstructing its external modules in /tmp. No production source was edited. Timing PIDs 410650/410651 and telemetry 410649 completed/stopped.

Producer patches apply to the original installed dense_gemm_persistent.py described in ../fixed_shapes/README.md. The helper patch applies to the reconstructed native_epi_vector.py from that archive. Compressed patches preserve exact bytes; source_hashes.json records original experiment bytes before evidence formatting. Installed NVIDIA source is not duplicated here. Interleaving weights is a startup transform, with extra weight storage and setup costs documented in the fixed-shape archive; it is excluded from steady-state timings.

Memory diagnostic command: compute-sanitizer --tool memcheck --error-exitcode 99 /root/qvl/venv-sgl/bin/python -u /tmp/native_tail_lifetime_debug.py. The second invocation additionally uses --report-api-errors no. Both use SGLANG_KERNEL_API_LOGLEVEL=3 and separate log paths; logging is absent from timing runs.

`original-artifacts.tar.gz` preserves archived files before further repository formatting; raw experiment hashes remain documented in source_hashes.json.
