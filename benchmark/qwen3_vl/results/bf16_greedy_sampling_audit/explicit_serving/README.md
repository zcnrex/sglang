# Production cast short serving gate

Frozen M16 control versus the same source with only committed `f9a704d70a` cast changes. Recursive audit proves two changed files plus new `elementwise/cast.py`; later M32/M64 vocabulary choices are excluded. Both sources include two non-runtime hidden skill scripts beyond the historical 4,229-file inventory. The external hook only observes production cast calls and establishes the same startup policy; it does not implement or replace the cast/GEMM algorithms.

Identical isolated seeded caches validate common public tactics M128 gate/up 1/down 1 and M4/M8 vocabulary 2. Cache hits and valid registry membership are recorded, then the temporary policy is restored before evaluation. Candidate startup warms the actual explicit kernel before graph capture. Both 64/128 captured calls preserve caller FP32 buffer identity, and measured-epoch B128 decode executes in all workers.

C128/N128, warmup 128, fixed 8192/1024 tokens, flush before measurement, cap 128 and 1,600,000 KV tokens. BF16 weights/queries/KV, TRT HND page 32, mixed chunk 16384. Two swapped same-GPU phases; all four runs complete 128 requests, 1,048,576 input and 131,072 output tokens exactly.

| GPU | Control output tok/s | Candidate output tok/s | Gain | Median TTFT control/candidate ms | Median TPOT control/candidate ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3812.101 | 3820.655 | 0.22439% | 4543.487 / 4508.201 | 28.84184 / 28.81478 |
| 1 | 3811.977 | 3819.084 | 0.18642% | 4559.187 / 4555.010 | 28.82498 / 28.76696 |

Geometric mean throughput gain: 0.20540%. Mean TTFT changes 4936.235→4925.494 ms on GPU 0 and 4957.942→4964.319 ms on GPU 1; thus not every latency statistic improves. P99 TTFT changes 9035.683→8990.797 and 9114.911→9084.942 ms. Complete mean/median/std/p90/p95/p99 TTFT, TPOT and E2E values are preserved in summary/raw benchmark JSONL. This is a two-pair short gate, not full-workload confidence evidence.

Parent 582822 and servers 582827/582828/586273/586274 terminated normally. Raw tar preserves original data, hooks and launcher are exact snapshots, and `summarize.py.txt` verifies policy/capture/counts and regenerates summary after copying it to `.py` beside the results. No full GSM run; exact conversion and prior full-logit model checks support advancing to the separately recorded full-serving gate.
