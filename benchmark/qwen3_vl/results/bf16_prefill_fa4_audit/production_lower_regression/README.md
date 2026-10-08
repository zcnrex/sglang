# Lower-concurrency production regression pairs

Control /root/qvl/sglang-m1-production versus candidate /root/qvl/sglang-prefix-production. Both 4,229-file source manifests matched exactly before launch; complete audit records and original per-worker source hashes are in the raw archive. Same-GPU A/B pairs swap variants, alternating starting order by GPU parity. GPUs0/1/2/3 ran C1/4/8/16, followed by GPUs0/1 running C32/64. All workers completed and cleaned up their owned servers; parent378353 reported DONE.

These are bounded regression checks, not replacements for the original baseline sweep or evidence that the overall target is met. Warmup was explicitly C requests, identical within each pair, rather than the original max(64,C). Each concurrency has only one pair. BF16 weights/KV/query, mixed16k, HNDpage32 and all serving flags remain the established production recipe. Fresh isolated tuning caches and shared compiled caches were used; no algorithm hook was installed.

| C | Requests each | Control tok/s | Candidate tok/s | Change | TTFT control→candidate ms | TPOT control→candidate ms |
|---|---:|---:|---:|---:|---:|---:|
|1|30|393.213|393.612|+0.101%|85.717→79.515|2.4612→2.4634|
|4|40|1226.498|1228.815|+0.189%|227.680→244.843|3.0356→3.0236|
|8|80|1880.510|1884.969|+0.237%|372.092→357.560|3.8959→3.9050|
|16|80|2563.102|2572.189|+0.355%|592.751→590.455|5.6482→5.6474|
|32|160|3122.868|3139.658|+0.538%|716.534→806.408|9.4661→9.3803|
|64|320|3500.489|3515.637|+0.433%|579.020→706.241|17.6003→17.4978|

All six throughput pairs were positive, but C4/C32/C64 median TTFT increased by approximately7.54%/12.54%/21.97%; do not claim uniform latency improvement. Every worker completed exactly N requests with N×8192 input and N×1024 output tokens, verified by harness assertions and independently rechecked in the summary.

Startup M128 public BF16 tactic choices are preserved verbatim in tactics.json and the original tuner files. They differed across several pairs; lower-concurrency decode does not use the M128 bucket, but the harness's pre-benchmark128-request probe does. These runs do not pin tactics or prove every possible eager128-row dispatch absent. Startup/probe outcomes therefore should not be used as matched-tactic accuracy evidence.

The compressed raw archive preserves original logs, benchmark JSONL, source patches, source hashes, startup tuner JSON, probes, commands, PIDs and telemetry without normalization. No bytecode or compiled-cache artifacts are included. External launcher and summarizer copies accompany it; the unchanged production harness is also preserved. The overall10% target remains unmet.
