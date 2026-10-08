# Current-source mixed-chunk TTFT/TPOT comparison at C8/C16

Mixed chunking lowers mean TTFT in all four same-GPU pairs, with a small mean TPOT cost. Median/tail behavior is less uniform, and throughput has no consistent C16 improvement. This is a scheduling tradeoff measurement, not a new kernel gain or target achievement.

| C / GPU | Throughput on/off change | Mean TTFT | Median TTFT | P95 TTFT | P99 TTFT | Mean TPOT |
|---|---:|---:|---:|---:|---:|---:|
|8 / 4|+0.002%|−4.750%|+6.367%|+0.893%|+3.240%|+0.508%|
|8 / 5|+0.563%|−11.338%|+0.359%|−3.526%|−9.406%|+0.559%|
|16 / 6|−0.570%|−8.015%|−5.034%|−13.886%|−1.710%|+0.981%|
|16 / 7|+0.474%|−7.090%|−2.272%|−14.903%|+3.422%|+0.319%|

Each concurrency has two phases swapping on/off across the same GPUs. All eight runs complete 80 requests, 655360 input and 81920 output tokens. Warm64, nominal 8192/1024, benchmark flush. Both arms use frozen /root/qvl/sglang-m16-down-production, including packed FA4 and current small-batch production, without M32/M64/cast experiments. BF16 weights/query/KV, TRT HND page 32, chunk 16384, prefill graphs disabled. Graph caps 8/16 and pinned 1600000 KV tokens are identical within each concurrency and asserted before requests. The only serving-argument difference is --enable-mixed-chunk. Public normal startup tuning, isolated tuner caches/shared compiled caches; observer records only startup, flush and first M128 dispatch per epoch/shape, never overrides algorithms.

Full distributions and per-request TTFT/ITL data are gzip-preserved in bench.jsonl with uncompressed hashes. Analysis contains scalar mean/std/median/p90/p95/p99 TTFT/TPOT. All measured postflush prefill intervals exclude 128 and all per-epoch eager M128 dispatch counts are zero; graph caps also exclude 128. Thus independently selected M128 tactics are not active during the measured windows. Complete rows, flush markers and startup tactics are retained.

Historical bf16_mixed_comparison.json already contains older controlled low-C comparisons: combined-source C8 normal/mixed median TTFT 384.519/262.654ms, C16 566.149/383.803ms, but independent GPUs, older source, and no retained full distributions. Current-source rerun was authorized because packed FA4 and later small-batch changes affect mixed execution. It does not reproduce the older approximately 32% median TTFT reduction; do not substitute historical values for this current result.

Initial drivers 589680/589681 failed source preflight before launching any server: expected 4229 runtime files versus 4231 total Python files. The only additions were two known inherited .claude benchmark skill scripts with matching audited hashes. Corrected harness verifies the full explicit 4231 mapping, preserving exact runtime hashes. Failed artifacts remain in mixed-current, valid results in mixed-current-fixed. No source files were edited.

Valid drivers 591888/591889 completed both phases and owned servers cleaned up. GPUs 4–7 released; other jobs were untouched. Remote roots /root/qvl/experiments/mixed-current-fixed and mixed-current; raw archive /tmp/mixed-current-fixed.tar.gz retained locally/remotely. No further repeats were launched.
