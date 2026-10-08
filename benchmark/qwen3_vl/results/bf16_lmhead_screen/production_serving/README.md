# Production M4 LM-head C4 crossover

The reviewed three-file production path retains a small positive serving signal: +0.15997% and +0.22122% same-GPU throughput, geometric mean about +0.1906%. Two short pairs are not a precise effect estimate or achievement of the overall target. No external algorithm replacement or private tuning policy was used; the observer only records admission, public cache state, first dispatch/replay events.

Control remains `/root/qvl/sglang-prefix-production`; candidate `/root/qvl/sglang-lmhead-production` contains the exact audited three-file overlay committed as 5e1601731b. All 4229 Python source hashes are checked before each run. Both use BF16 weights/query/KV, TRT HND page 32, mixed chunk 16384, prefill graphs disabled. Fresh isolated tuner caches and shared compilation caches are recorded. Public normal autotuning independently selected LM-head tactic 2 in both candidate runs, and READY/captured M4/runtime graph4 proof is retained.

Each GPU runs control and candidate in opposite phase order, C4/N40/warm64, 8192 input/1024 output, flush before measurement. Every run completed 40 requests, 327680 input tokens, 40960 output tokens. Full paired metrics are in analysis.json.

| GPU | Control / candidate tok/s | Throughput change | Median TTFT change | Median TPOT change |
|---|---|---|---|---|
|4|1238.522519 /1240.503771|+0.15997%|−10.44455%|+0.17609%|
|5|1243.747305 /1246.498738|+0.22122%|−1.44935%|−0.52659%|

M128 public gate/up tactic 1 is common; down tactics differ (control 4, candidate phase A 1 / phase B 0). These choices cannot be assumed irrelevant solely from C4. The explicit readiness probe has 4×32 input tokens and does execute M128 before measurement. A separate audit of every logged prefill after the single `Cache flushed successfully!` boundary excludes M128 in measured requests. The table preserves 33/34/36/36 batches and raw log lines. `PrefillAdder._update_prefill_budget` adds raw_extend_input_len to log_input_tokens (`schedule_policy.py`); `PrefillStats.from_adder` carries it, and MetricsReporter.log_prefill_stats logs every prefill on this single logging rank without sampling. Mixed decode tails are conservatively bounded by #running-req. No interval [new-token, new-token + #running-req] contains 128: small-context upper bounds are at most 75, all other contexts at least 8045. TP1 has no distributed MLP row padding, and prefill CUDA graphs are disabled; decode batches are at most 4. Thus recorded measured geometry excludes M128, despite pre-measurement probe markers. Exact intervals and flush window are in measured_m128_audit.json.

Driver 453833 completed both phases; worker cleanup terminated owned servers. Full GSM ran concurrently on separate GPUs 6–7 with identical public M128 tactics and is reported separately. Raw pre-format archive `/tmp/lmhead-production-serving.tar.gz` is retained as `original-artifacts.tar.gz`, with the same remote archive path. Raw source.patch files are gzipped to preserve diff bytes.

| GPU | Control / candidate median TTFT (ms) | Control / candidate median TPOT (ms) |
|---|---|---|
|4|240.828271 / 215.674839|3.000260 / 3.005543|
|5|213.042463 / 209.954736|3.016027 / 3.000145|
