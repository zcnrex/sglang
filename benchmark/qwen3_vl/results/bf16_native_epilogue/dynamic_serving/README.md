# Dynamic native epilogue: bounded serving screen

The standalone and component benefits did not produce an end-to-end gain in this two-GPU crossover. Do not promote this candidate from these results; they do not establish exact zero benefit or a general regression.

Both variants used the audited current packed-prefix production source, BF16 weights/query/KV, TRT HND page 32, mixed chunk 16384, and an explicitly equal 1,400,000-token KV pool. Candidate startup interleaved all 36 gate/up weights, compiled one symbolic-M callable, and allocated one shared output buffer. Only text prefill rows 8192–16384 dispatched the fused path; all other shapes used the original implementation. No production files changed.

Each GPU ran both variants in opposite phase order, concurrency 128,128 measured requests,128 warmup requests, random 8192 input/1024 output, flush after warmup. Each run completed 128 requests,1,048,576 input tokens,131,072 output tokens. Exact commands, source manifests, startup allocation/compile counters, cache configs, telemetry and raw outputs are retained. These are diagnostic runs: both variants use synchronous per-prefill JSONL instrumentation and the same scoped private startup tuner policy. The policy forces identical recorded gate/up 1/down 1 Lt choices only during the two startup searches and is restored before requests; startup and terminal configs and captured optimized-dispatch proofs passed.

| GPU | Control output tok/s | Candidate output tok/s | Throughput change | Median TTFT change | Median TPOT change |
|---|---:|---:|---:|---:|---:|
|4|3819.312802|3818.095895|−0.031862%|+0.203157%|−0.044905%|
|5|3845.436459|3844.940387|−0.012900%|+0.108992%|−0.004788%|

Mean paired throughput change was −0.022381%; only two pairs, so no strong statistical conclusion is warranted. Four-run absolute latency/throughput metrics are in analysis.json and each summary.json.

Coverage derives from completed layer 0-to-layer 35 prefill records, grouped by flush epoch. Epoch 0 includes readiness/probe/warmup; epoch 1 is measured post-flush. Both candidate warmups executed 72 fused calls. Each measured candidate executed 2,340 calls (65 forwards×36 layers). GPU 5 phase A covered 1,044,927/1,049,099 mixed-forward token rows (99.6023%); GPU 4 phase B covered 1,049,032/1,049,032 (100%). These rows include decode tails inside mixed forwards and are not the benchmark's input-token accounting. Warmup reused prefixes, so its row count is much smaller than 128×8192; it nevertheless exercised the dynamic path before measurement. Full shape histograms and fallback records are retained.

Driver 417563 and both phase B servers 420721/420722 were absent at terminal cleanup; nvidia-smi reported no active compute processes. The raw pre-format artifact archive is retained as original-artifacts.tar.gz, with a remote copy at /tmp/native-dynamic-serving-results.tar.gz. Original-artifact SHA manifest identifies bytes before repository formatting; no further runs were launched.
