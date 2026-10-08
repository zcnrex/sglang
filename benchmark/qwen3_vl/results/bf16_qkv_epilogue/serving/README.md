# External B8 fused QKV serving crossover

Throughput improved by 1.134035% on GPU2 and 1.237855% on GPU3; geometric mean gain **1.185931%**. This validates the external prototype, not the different integrated production kernel. Best observed throughput 1928.099 tokens/s remains below the C8 target 1951.4.

| GPU / arm | Output tokens/s | TTFT mean ms | TTFT median ms | TTFT p95 ms | TTFT p99 ms | TPOT mean ms | TPOT median ms | TPOT p95 ms | TPOT p99 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 control A | 1893.497429 | 348.696345 | 315.953833 | 536.979825 | 611.953141 | 3.884494 | 3.894496 | 4.085693 | 4.134244 |
| 2 candidate B | 1914.970344 | 357.417526 | 383.202565 | 558.652079 | 561.016849 | 3.827991 | 3.832183 | 4.046859 | 4.068237 |
| 3 control B | 1904.523432 | 362.549502 | 378.680193 | 554.373536 | 555.890537 | 3.845909 | 3.840941 | 4.060775 | 4.085222 |
| 3 candidate A | 1928.098664 | 358.263162 | 376.653394 | 536.652778 | 551.807661 | 3.799344 | 3.790270 | 4.017355 | 4.034693 |

GPU2 TTFT median increased 21.284% (315.954 to 383.203 ms), while mean increased 2.501%, p95 increased 4.036% and p99 decreased 8.324%. Standard deviation was 149.725 versus 150.391 ms. These aggregate statistics do not establish a uniform shift. The benchmark omitted `--output-details`, so per-request TTFT/order was not persisted; grouping/order cannot be distinguished from a broader distribution change. GPU3 mean/median TTFT decreased 1.182%/0.535%. No extra run was made to explain the difference.

These are aggregate percentile summaries, not full per-request distributions.

## Protocol and provenance

Two phases on GPUs2/3, with opposite arm assignments. Both use frozen `/root/qvl/sglang-m16-down-production`, full source manifests verified. Only the candidate external hook replaces exact B8 decode QKV plus preparation; prefill and other batch sizes use the original. All 36 fused capture dispatches are asserted. Detached parameter views assert identical pointers and strides. BF16 weights/query/KV/output, HND page32, existing PDL and TRT attention are preserved.

C8/N80, warm64, cache flush, nominal8192/1024, random range1, mixed chunk16384, prefill graphs disabled, decode graph cap8, and server-verified KV capacity1,600,000. All four runs completed exactly80 requests,655360 input tokens and81920 output tokens. Fresh isolated startup caches use normal public tuning. M4/M8 vocabulary tactic2 and M128 gate/up tactic1 match. GPU2 M128 down differs1 versus4; measured epoch observers show no M128 forward in any run. GPU3 M128 down is4 in both arms. This graph/capacity protocol is diagnostic and absolute rates should not silently be substituted for older protocol results.

## Frozen artifacts

`../serving-original.tar.gz` retains the unmodified raw archive. This directory has readable Python renamed `.py.txt`; `serving/` contains four run logs, JSONL results, source manifests, startup caches, observer records and `analysis.json`. Driver620127, phaseA620133/620132 and phaseB623309/623310 are terminal. No further external serving repetition is planned.
