# Production M1 gate/up validation

Candidate is an isolated copy of audited5f60fb8b67 plus only `unquant.py`, SHA256 `0fa01b10b34bec46f4cca14b6dfb96fa67a07d94e68c1d08035b8c851740f431`. The control copy is unchanged. Existing targeted tests pass33 tests and9 subtests. Three changed-input CUDA-graph replays match the direct kernel bitwise, with normalized RMS error7.15e-5–1.20e-4 versus Torch. Source audit, numerical checks and test output are preserved.

Serving uses the actual production path with no external dispatch hook. Four paired GPUs, swapped phases, c1/N30/warm64 and nominal8192/1024 produce exactly30 requests,245760 input tokens and30720 output tokens in every run. Geometric throughput gain is1.6918%, all four positive. Descriptive95% intervals: t[1.5455%,1.8384%], bootstrap[1.6125%,1.7713%]. These quantify four devices/two periods, not repeated-day validation.

TTFT changes are mixed: +2.17%,-10.50%,+3.95%,+3.50%. Mean of run-median TTFT is85.7121ms control and85.4118ms candidate. The external hook's consistent TTFT regression is not reproduced uniformly; no general TTFT improvement is established. Full paired values, medians, means, TPOT and counts are in `serving/paired-summary.json`.

Full GSM8K repeats the two657-question shards with GPUs swapped relative to the external-hook run. Each retains the original five few-shot rows and runs concurrency1, temperature0, top_p1, max2048. All1314 global row IDs occur exactly once per variant. Control scores1219/1314; production candidate1217/1314. The same14 control-only and12 candidate-only correct rows repeat. This is a reproducible net loss of two questions in this workload, not numerical equivalence or noise. Candidate outputs match the earlier external run's completion-token total249291; control uses241589 tokens. Accuracy observation wraps the existing helper only to record invocation; it does not replace dispatch or kernels. Graph1 replay records36 captured calls.

Fresh isolated tuning caches and warmed shared compilation caches are used. Four independent accuracy GPUs run concurrently with four serving GPUs; no GPU is shared. Commands, source/hook hashes, shard mapping, outcomes, caches, logs and telemetry are preserved. Raw source patches are gzip-compressed to preserve exact bytes. Other-concurrency regression results are archived separately when complete.

Other-concurrency same-GPU paired checks are complete:

| Concurrency | Requests | Control tok/s | Candidate tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 4 | 40 | 1236.664 | 1238.610 | +0.157% |
| 8 | 80 | 1873.986 | 1870.901 | -0.165% |
| 16 | 80 | 2554.519 | 2557.891 | +0.132% |
| 32 | 160 | 3137.068 | 3135.382 | -0.054% |
| 64 | 320 | 3500.916 | 3499.490 | -0.041% |
| 128 | 128 | 3800.605 | 3790.391 | -0.269% |

All request and nominal input/output token counts match the protocol. Each case uses warmup `max(64, concurrency)`, fresh isolated tuner cache and warmed compilation caches. No material regression is observed in these single pairs; statistical equivalence is not established. Raw low/high wave evidence and source patches are under `regressions/`. The first low-wave launcher had a wrong script path and failed before launching servers; its invalid log is preserved. Only that failed stage was restarted after correcting the path. Both completed regression drivers and their owned servers stopped.
