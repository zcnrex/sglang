# Guarded public cuBLASLt integration validation

Production changes are limited to `unquant.py` and `runner/flashinfer_autotune.py`. The candidate uses the public BF16 cuBLASLt API for two explicitly warmed M128 shapes, after exact autotuner cache-hit verification. Existing dispatch remains the fallback for unsupported contracts.

## Source provenance and excluded preliminary run

The first validation copy, `/root/qvl/sglang-public-lt`, was incorrectly derived from an older working checkout. `source-audit.json` records five unintended Python source differences in addition to the two intended files. Its startup, guard and unit-test logs are retained for diagnosis, but **its serving results cannot establish matched-source accuracy or performance**. Candidate GSM was stopped before evaluation. Those preliminary logs are `server128.log`, `server256.log`, `reload-result.json`, `tests-final.log` and `autotune-cache.json`.

The replacement `/root/qvl/sglang-public-lt-clean` was created by archiving commit `466c9e0f4073bcad6e4403c6f54cfc8c1821b867`, then replacing only the two reviewed files. Every one of 5406 tracked Python paths, including symlinks, was compared against that commit. `clean-source-audit.json` confirms exactly the intended two differences. No live source files were overwritten during the correction.

Frozen intended diff SHA256: `593c46e1f628ea133ac5b7fd650aece3883807dc13ea6b5012b5597978ebc094`.

File SHA256 values, identical in the local reviewed files and candidate:

- `unquant.py`: `9db197749dd28b84f8af16c70187556d035fdaeab148675e043858c565be47f3`
- `flashinfer_autotune.py`: `7f57b1688bb7a53a2b34aa13465392a187b42b8c774ed22851a2399a0726711e`

## Checks

The three existing test files cover autotune synchronization/cache behavior, unquantized addend dispatch and BF16 Split-K dispatch. Preliminary runs passed 33 tests and 9 subtests. External guard checks covered output identity, bias/addend rejection, dtype/layout/output-device rejection, disabled/deterministic paths, absent readiness, speculative/draft exclusion and reinitialization invalidation.

The standalone wrapper asserts two verified readiness entries after the actual runner warmup, then records each selected projection in and outside CUDA graph capture. Its initial attempts used an obsolete CLI spelling and a premature initialization marker; those were wrapper errors, not production failures. Clean-source startup/replay and final test results are recorded separately in files prefixed `clean-`.

Both startup validation and subsequent model evaluations use BF16 weights/KV/queries, TRT attention page size 32, HND layout and mixed chunk size 16384. The clean validation uses an isolated `SGLANG_CACHE_DIR`. These correctness/startup checks do not establish a throughput improvement; paired production performance and GSM evaluation are separate evidence.

## Clean-source result

Clean-source startup at maximum graph batch 256 passed, with both M128 projections observed inside CUDA graph capture. A real batch of 128 requests completed exactly 16 tokens each (2048 total). The isolated fresh autotune cache contains verified entries for both shapes. The corrected source then passed all 33 existing tests and 9 subtests in 58.07 seconds, plus the external guard checks. GPU2 validation server PID 193390 was stopped afterward. Paired model accuracy and throughput are evaluated separately.

Cache hashes in the replay reports identify the remote original bytes; archived JSON may have a terminal newline added by pre-commit.
