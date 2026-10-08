# Cast-only short serving crossover

Same frozen `/root/qvl/sglang-m16-down-production` in both arms. The external candidate changes only contiguous BF16-to-FP32 logits copies at 64/128 rows with vocabulary 151936 and an existing matching contiguous FP32 buffer. It asserts caller buffer identity and leaves all unsupported, no-buffer, softcap and truncation cases on the original implementation. No vocabulary GEMM or sampling override. Compilation/warmup occurs before decode capture; serving uses retained captured graphs. The accompanying model gate records generated code/configuration and bitwise full-logit checks.

This is an external matched-startup diagnostic: each isolated cache is seeded from the successful model cache. Both variants validate public cache hits and current registry membership for M128 gate/up 1, down 1, M4 vocabulary 2 and M8 vocabulary 2. The temporary search policy is restored before evaluation. All four validation records and restoration markers are archived for every worker; this policy is not proposed production code.

C128/N128, warmup 128, fixed 8192/1024 input/output, flush before measurement, max-running 128 and pinned 1,600,000 KV tokens. Ordinary BF16 weights/queries/KV, TRT HND page 32, mixed chunk 16384, shared compiled caches. Two swapped phases pair variants on the same GPUs. Every run completed 128 requests, 1,048,576 input tokens and 131,072 output tokens exactly.

| GPU | Control output tok/s | Candidate output tok/s | Gain | TTFT control/candidate ms | TPOT control/candidate ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 3812.952 | 3820.312 | 0.19303% | 4518.266 / 4517.949 | 28.85907 / 28.80497 |
| 1 | 3811.961 | 3819.325 | 0.19317% | 4561.253 / 4557.577 | 28.82630 / 28.76559 |

Geometric mean throughput gain is 0.19310%. Both pairs are positive, but only two short pairs were measured; no full-sweep or statistical-confidence claim is made. No full GSM evaluation was run for this cast variant.

Candidate observers record both 64/128 graph captures with identical caller buffers. All workers record measured B128 decode after the cache-flush epoch. The compile functions are warmed before capture, rather than invoked for the first time during measurement; no blanket assertion about all compiler internals is inferred from timing. Unsupported shapes continue to use the original path.

Parent PID 555214 and servers 555219/555220/560681/560682 terminated normally. Raw tar preserves original files; compact summary retains exact metrics, startup policy and observer events. Generated cast code and chosen configuration live in the sibling `cast_model/compiled` directory. This experiment establishes a small serving gain for this external hook, not a production API decision.
