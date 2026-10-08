# C64 TTFT audit from existing evidence

The median regression is real in this pair, but the aggregate distribution does not show a uniform slowdown. Candidate/control mean TTFT is992.469/985.203ms (+0.74%), while median is706.241/579.020ms (+21.97%). Candidate p90/p95/p99 improve to2308.658/3409.972/4252.588ms from2374.600/3498.452/4358.431ms. Mean end-to-end latency improves18690.290→18606.544ms. These measurements cannot establish individual-request causality.

The archived JSONL contains aggregate percentiles, not per-request submission/admission/first-token timestamps. HTTP server log lines likewise lack request identifiers and GPU intervals. After the successful cache-flush marker, both server logs contain exactly166 prefill records and2,615,309 logged input tokens, with identical new-sequence histograms:144 calls with3,13 with2,7 with1,2 with4. These logged tokens are a scheduler metric and are not the benchmark's2,621,440 input-token counter.

Candidate/control logged waiting-queue medians are3/2 and pending-token medians25476.5/16919.0. Dividing logged new tokens by logged input throughput reconstructs the interval between prefill-stat reports. Intervals below1second have median141.820ms candidate versus144.371ms control. Source metrics_reporter.py:706–715 computes this rate using time.perf_counter and last_prefill_stats_tic: it is an asynchronous scheduler inter-log interval, **not GPU kernel latency**.

Splitting records at inter-log gaps greater than1second gives five refill waves in each run. Counts are34/33/33/33/33. Sum of within-wave inter-log intervals (excluding the gap before each wave):

|Wave|Control seconds|Candidate seconds|Control median queue|Candidate median queue|
|---|---:|---:|---:|---:|
|1|4.451|4.331|17|21|
|2|4.575|4.466|2|4|
|3|4.584|4.490|1|1|
|4|4.583|4.476|3|3|
|5|4.580|4.473|1|3|

Thus the existing logs do not support blaming slower context compute: prefill-report waves are slightly shorter, with identical batch-count distributions. They do show changed queue occupancy/refill timing. A concrete unproven lead is the phase relationship between completed decode requests, newly submitted closed-loop requests, and admission into the next mixed-prefill batch. A small shift can change which refill wave or chunk supplies the median request's first token without worsening upper tails. Independent startup M128 tactics also differ, so this is not an isolated per-kernel causal study.

To distinguish that lead from host/client timing would require correlated request submission, scheduler admission, first model execution and first-token emission timestamps plus context GPU spans; those are absent here. No new serving run was launched for this audit. Do not claim a scheduling fix or dismiss the median regression based on these aggregates.

Sources: original c64/A/gpu1 (candidate) and c64/B/gpu1 (control) bench.jsonl/server.log in adjacent prefix-lower-evidence.tar.gz; current scheduler metrics reporter's unchanged timing formula. The successful flush marker bounds measured records; startup and warmup records are excluded. Wave partition is a descriptive1second threshold, not an independently observed scheduler phase label.
