# Final M32 production performance confirmation

The final two-phase crossover again shows only negligible throughput differences: **+0.0863% / +0.0043%**, geometric mean **+0.0453%**. Across the four production same-GPU pairs including the preceding screen, every point estimate is positive but all are at most0.102%. This does not establish a practically meaningful serving gain. TTFT remains inconsistent, with a sizable mean/median increase in one confirmation pair. No additional repeats were run.

| GPU | Throughput control / candidate (tok/s) | TTFT mean (ms) | TTFT median (ms) | TTFT p95 (ms) | TTFT p99 (ms) | TPOT mean (ms) | TPOT median (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3151.518 /3154.238 | 733.566 /815.990 | 672.538 /803.317 | 1669.478 /1668.667 | 2096.698 /2096.707 | 9.433 /9.343 | 9.504 /9.346 |
| 3 | 3152.527 /3152.662 | 820.214 /789.121 | 799.746 /800.922 | 1655.481 /1646.704 | 2081.402 /2069.049 | 9.344 /9.374 | 9.350 /9.384 |

GPU2 TTFT mean/median increase11.24%/19.45%, while p95/p99 are essentially unchanged. GPU3 mean improves3.79%, median increases0.15%, and p95/p99 improve0.53%/0.59%. These observations do not establish that the kernel change caused a scheduling regression, but they do not support a TTFT improvement claim. TPOT distributions and all percentage changes are in analysis.json.

Exact frozen M8+M16 production control and M32 two-file overlay d0d528db4f are unchanged. Both use BF16 weights/query/KV, TRT HND page32, mixed chunk16384, graph max batch32, capacity1,600,000 and disabled prefill graphs. Capacity assertions pass for all four runs. C32/N160, warm64, flush, nominal8192/1024: every run completes exactly160 requests,1,310,720 input and163,840 output tokens. Expected full source manifests are checked before launch.

This is an external matched-startup diagnostic, not production policy. The four common cache records (M128 gate/up1/down1, M4/M8 vocabulary2) are verified as loaded hits and valid registered tactics under the tuner lock. Original search_cache is restored after initialization in all four runs. Candidate M32 remains normal public autotuning and selects2; graph32 replay is recorded. No inference algorithm replacement, new source edit or further GSM evaluation occurs.

Driver565997 and servers566002/566003,569147/569146 completed and disappeared after PHASE B DONE. GPUs2–3 are free. Raw archive preserves scripts, cache JSON, source verification metadata, logs and full benchmark records. This completes the requested final confirmation; retention/promotion remains a parent decision.
