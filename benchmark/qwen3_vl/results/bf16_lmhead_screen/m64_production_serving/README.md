# Production M64 C64/N320 crossover — not promoted

The longer production crossover retains only negligible throughput differences and inconsistent latency changes. M64 public promotion is rejected; no full GSM or further serving runs follow. These two pairs do not prove exact zero effect or a general regression.

| GPU | Control / candidate tok/s | Throughput change | Mean / median TTFT change | P95 TTFT change | Median TPOT change |
|---|---|---|---|---|---|
|4|3521.854605 / 3524.843235|+0.08486%|+5.700% / +19.258%|−1.876%|−0.813%|
|5|3541.966982 / 3542.549144|+0.01644%|−3.157% / −12.726%|+2.037%|+0.463%|

Each run completed 320 requests, 2621440 input and 327680 output tokens. C64/N320/warm64, 8192 input/1024 output, benchmark flush. Both variants use BF16 weights/query/KV, TRT HND page 32/mixed 16384, prefill graphs disabled, cap 64 decode graphs and identical asserted 1600000-token KV capacity. Normal public startup tuning with isolated caches/shared compiled caches; observer only, no algorithm hook or private tactic policy. Exact candidate M64 READY/capture and graph 64 replay are proven. Candidate vocabulary 64 chooses tactic 7 in both runs.

Control is immutable /root/qvl/sglang-lmhead-m32-production; candidate /root/qvl/sglang-lmhead-m64-production overlays exactly two files from commit 96ef1b658e. Manifests retain all 4229 runtime Python files and two inherited non-runtime skill scripts, per the separate model audit. Both source trees remain frozen even if the working branch later reverts extensions.

All 166/166/165/165 postflush prefill intervals [new-token,new-token+running-req] exclude 128; TP1, complete prefill logging, no prefill graph or distributed row padding, decodecap 64. Thus differing M128 down tactics (A4control1/A5candidate4/B4candidate1/B5control0) do not explain the measured difference. Gate/up 1 and vocabulary M4/M8/M32 tactic 2 are common. Both sources include M32 support. Startup optimized M32 captures are observed; sampled measured decode logs contain no batch 32 entry, but this observer does not count each replay. Do not infer no transient measured M32 execution. This result does not validate a hypothetical M64-without-M32 source.

Full mean/std/median/p90/p95/p99 TTFT/TPOT and per-request ttfts/itls are retained in gzip-preserved bench.jsonl, with original SHA256 hashes. Analysis retains scalar distributions and every measured prefill interval. TTFT variation is disclosed rather than dismissed. Detailed benchmark output is written after measurement.

Driver 574050 completed both phases. Phase A servers 574055/574056 and phase B 578386/578387 cleaned up; GPUs 4/5 released. Remote root /root/qvl/experiments/lmhead-m64-production/serving. Original compact archive /tmp/lmhead-m64-production-serving.tar.gz remains local/remote. No production edits or PR updates were made by the validation worker.

Large bench.jsonl.gz blobs are stored as ordered ≤1MiB .partNNN files. Concatenate the parts listed in gzip_parts_manifest.json, verify its full gzip SHA256, then decompress. Original byte-identical gzip files remain under /tmp/m64-production-serving-original-gzip; uncompressed SHA256 values remain in compressed_original_hashes.json.
