# Cached/fresh context partition screen

A bounded standalone alternative routes the cached continuation to TRT and the
fresh request subgroup to FA4 CLC. The unchanged TRT decode suffix executes in
both variants. All comparisons below are within the same GPU. No production
source or serving configuration was changed.

| GPU | Context prefix / extend | Decode suffix | Cold baseline us | Cold partition us | Saved us |
| --- | --- | --- | --- | --- | --- |
| 6 | [7968,0,0] / [232,8200,7872] | 17 | 937.25 | 917.78 | 19.47 |
| 7 | [7968,0,0] / [232,8200,7872] | 17 | 956.42 | 922.62 | 33.80 |
| 6 | [7872,0,0] / [317,8200,7808] | 19 | 1002.81 | 940.10 | 62.71 |
| 7 | [7872,0,0] / [317,8200,7808] | 19 | 998.89 | 955.46 | 43.42 |

Both tuples come directly from the measured serving prefix histogram. Warm and
cold measurements each have six counterbalanced rounds; every paired round is
positive. Warm samples average twenty graph replays; cold samples take the
median of six replays, each preceded by a 256 MiB flush outside event timing.
Initial samples show clock/timing drift; raw samples are retained. This is a
kernel screen, not a serving-throughput estimate.

## Timing scope and correctness

- Baseline: one TRT context call plus the unchanged TRT decode suffix.
- Candidate: one TRT continuation call, one FA4 fresh-context call, plus the same
  TRT decode suffix. Thus all reported times include unchanged decode work.
- The observed continuation is first and fresh requests form a contiguous
  suffix. Q/K/V/output are sliced views; no data gathering or scatter is needed.
  Subgroup page-table selection, cumulative lengths, and metadata preparation
  occur outside per-layer timing. The script has a general noncontiguous branch,
  but that branch is not exercised or claimed validated by these results.
- Inputs are synthetic BF16 with actual Q/K/V token stride 6144, Q32/KV8/D128.
  Cache layout is native HND page32 with shuffled physical page indices. Cached
  prefix rows are present; fresh cache rows exactly copy the supplied K/V.
  Decode suffix counts match the histogram; their synthetic KV lengths are 8201.
  Both paths use the runtime max-context bound 262144.
- Every output row is initialized to NaN and checked finite after execution.
  Partition versus baseline passes atol=rtol=0.02; maximum absolute error is
  0.0078125 and normalized RMS is at most 0.0022001. Decode-tail outputs match
  bitwise. Original output row ordering and shared-output slices are preserved.

CLC is explicitly requested in the external harness. The compile-cache observer
asserts selected key field 39 is true (two hits per case: eager and graph capture).
Constructor records are empty because persistent compiled-cache entries were
reused; this run does not add an independent constructor-level scheduler proof.

Full cached-plus-fresh packing, measured separately on GPUs 4/5, saved roughly
102–120 us of context time and was selected for the subsequent model gate.
Those measurements exclude unchanged decode, unlike this table. Saved time can
be compared qualitatively because unchanged decode cancels within each variant,
but there is no direct same-GPU comparison establishing packing's superiority.
No further partition GPU runs were made.

## Provenance and reproduction

The script verifies all 4,229 Python files against the audited 5f60fb8b67 source
manifest before allocating CUDA tensors. `hashes.txt` records the exact executed
script, manifest, TRT backend and FA4 interface hashes. `bench.py.txt` preserves
the executed script bytes. Reports/logs and exact observed tuples are archived.

Remote root: `/root/qvl/experiments/context-partition`. GPUs 6/7 were verified free
before launch. Both workers exited successfully before the first process-ID
poll; actual worker PIDs were not persisted. This is a provenance limitation,
not evidence of an additional run. No owned workers remain.

```bash
CUDA_VISIBLE_DEVICES=6 PYTHONPATH=/root/qvl/sglang-public-lt-clean/python \
MAX_JOBS=4 TRITON_CACHE_DIR=/root/.cache/sglang/triton \
FLASHINFER_WORKSPACE_BASE=/root/.cache/sglang \
SGLANG_CUTE_AOT_CACHE_DIR=/root/.cache/sglang/cute_aot \
/root/qvl/venv-sgl/bin/python /root/qvl/experiments/context-partition/bench.py
```

The second worker uses GPU 7; ordering is counterbalanced by GPU index and round.
