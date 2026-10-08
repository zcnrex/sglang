# High-M vocabulary normal-tuning serving screen

C32 produces a small positive signal across two short pairs; C128 is mixed/negative and has an active projection-tactic confound. No full accuracy or promotion follows from these runs.

| Concurrency / GPU | Throughput change | Median TTFT change | Median TPOT change |
|---|---:|---:|---:|
|32 / 4|+0.11961%|−10.81759%|+0.10416%|
|32 / 5|+0.14352%|+23.56748%|−1.36776%|
|128 / 6|+0.04167%|−0.62810%|+0.09707%|
|128 / 7|−0.42752%|+0.32079%|+0.39668%|

C32 uses N160/warm64; C128 uses N128/warm128. Each request has8192 input/1024 output tokens, with flush after warmup. All eight runs completed exact counts. Same-GPU phases swap candidate/control. Both use frozen lmhead-production (including M4), BF16 weights/KV/query, HND page32, mixed16384, prefill graphs disabled. All source manifests match the4229-file expected manifest before startup. Candidate hook changes only its exact vocabulary M; normal public startup autotuning, fresh isolated tuner caches and shared compilation caches remain. Source/commands/telemetry and per-run absolute metrics are retained. Default graph capture maximum was retained, with52 buckets; no cap changed during live runs.

Actual target-M capture/replay is recorded for every candidate; graph128 replay is recorded for all C128 runs. C128 projection gate/up/down tactics differ: A GPU6 control1/4, A GPU7 candidate1/4, B GPU6 candidate3/6, B GPU7 control1/1. Therefore its paired result cannot isolate vocabulary performance. Vocabulary128 selected6 in both candidates. M4 selected2 throughout. Normal cache selections are data-dependent; no claim of a forced common policy.

C32 projection down choices also differ across phases, but a complete measured-window prefill audit excludes M128. Each log has one benchmark-flush boundary; only following prefill records are audited, excluding readiness probes/warmup. All85/87/86/85 intervals [new-token,new-token+running-req] exclude128. New-token logging is raw extend length; running requests provide a conservative decode-tail upper bound. Single TP1 logging rank logs each prefill, no distributed MLP padding, prefill graphs disabled; decode graphs cannot exceed32. Raw rows and source interpretation follow the prior production_serving audit. Thus M128 startup differences are irrelevant to this measured C32 window, not universally irrelevant to C32 workloads.

Drivers499972/499973 and all owned servers completed; GPUs4–7 released. Remote root /root/qvl/experiments/lmhead-highm-serving. Original compact archive /tmp/lmhead-highm-serving.tar.gz retained locally/remotely; compiled caches excluded, selected tuner records are in analysis.json and raw candidate exports. Production files unchanged.
