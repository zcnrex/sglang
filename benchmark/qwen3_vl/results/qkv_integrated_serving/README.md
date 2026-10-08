# Integrated production C8 crossover

Geometric throughput gain **1.071404%**, with two positive same-GPU pairs. Best1934.007 tokens/s remains below1951.4 target. No accuracy or goal-completion claim is made.

| GPU / arm | tokens/s | TTFT mean ms | median ms | p95 ms | p99 ms | TPOT mean ms | median ms | p95 ms | p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 control | 1895.146220 | 356.056012 | 345.155310 | 533.241001 | 556.730626 | 3.872894 | 3.890470 | 4.114955 | 4.121711 |
| 2 candidate | 1914.497649 | 355.397221 | 379.221991 | 537.553521 | 554.317974 | 3.831578 | 3.812034 | 4.056767 | 4.070698 |
| 3 control | 1912.552968 | 353.458304 | 370.674909 | 531.372956 | 550.558594 | 3.836969 | 3.839385 | 4.064603 | 4.070554 |
| 3 candidate | 1934.006623 | 339.365846 | 366.008916 | 524.257903 | 536.159915 | 3.805350 | 3.786359 | 4.001100 | 4.036975 |

## TTFT observations

GPU2 throughput improves1.0211%, but overall TTFT median rises9.870% (345.155→379.222 ms). Mean decreases0.185%, p95 increases0.809%, p99 decreases0.433%. Request-order groups of eight have similar, mixed median differences; those are descriptive groups, not asserted scheduler batches. Counts below200 ms remain19/80 in each arm; counts200–300 ms rise3→10, while500–600 ms fall22→20. Thus the measured distribution changes are not a uniform latency increase. Grouping/order or scheduling causality is not established. All raw80 request TTFTs and sorted values are retained for each run. GPU3 throughput improves1.1217% and mean/median/p95/p99 TTFT all decrease.

## Protocol

Frozen M16 control versus exact four-file integrated production overlay; public model dispatch, no algorithm replacement. Observer-only helper instrumentation asserts all36 fused layers during capture. Source manifests include separately admitted AppleDouble metadata. BF16 weights/query/KV/output; TRT HND page32; mixed chunk16384; graph cap8; verified1,600,000 KV capacity. C8/N80, warm64, flush, nominal8192/1024. Every run completes80 requests/655360 input/81920 output. Normal public startup tuning with fresh isolated caches. No measured M128 forward occurs in any run; startup tactics are retained in raw cache files. This diagnostic protocol must not be silently equated with older graph/capacity settings.

## Evidence

`original.tar.gz` preserves raw scripts, observer, source/cache provenance, logs and per-request JSONL (`--output-details` enabled). Readable Python copies end .py.txt. `qkv-integrated-serving/serving/analysis.json` includes exact aggregate percentile summaries plus full saved TTFT arrays and descriptive groups. Driver628886, phaseA628891/628892 and phaseB632475/632474 are terminal. Full task accuracy is a separate gate.

The raw archive is split byte-for-byte into `original.tar.gz.part*` to meet repository file limits. Concatenate parts in lexical order; `archive-parts.json` records original and part hashes. Readable `bench.jsonl` copies are losslessly gzip-compressed as `bench.jsonl.gz`.
