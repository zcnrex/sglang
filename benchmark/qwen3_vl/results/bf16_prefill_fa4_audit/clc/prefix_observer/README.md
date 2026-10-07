# Baseline measured-window prefix audit

Baseline5f60fb8b67-equivalent source, ordinary TRT BF16 attention, HND/page32/mixed16k, no M1 or FA4 replacement. GPU0 runs c128/N128/warm128 with nominal8192/1024 and the normal post-warmup cache flush. All128 requests,1048576 nominal input tokens and131072 output tokens complete. Timing is an instrumentation diagnostic, not a throughput claim.

The observer appends per-PID JSONL at layer0 context calls. It records only CPU metadata and leaves the original context function intact. A probe phase marker distinguishes startup/probe work; an explicit Scheduler.flush_cache event separates warmup epoch0 from measured epoch1. Epoch0 has2 probe contexts and4 warmup contexts. Exactly one flush occurs, followed by66 measured contexts. Decode suffix prefixes are recorded separately and excluded from the context histogram.

Measured epoch1 contains189 context request-chunks:128 fresh (prefix0) and61 continuation chunks. Only two nonzero prefixes are tiny (32 and64); the other59 range3840–8128. Only5 entire context calls are fresh, containing65382 query tokens. Total context query tokens are1044997:932032 fresh and112965 cached continuation tokens. Of the fresh tokens,866650 occur in calls containing another request with a cached prefix. The sum of cached-prefix tokens over continuation chunks is384896.

Representative context prefixes `[7968,0,0]` and extensions `[232,8200,7872]` have17 decode-tail requests. The next has prefixes `[7872,0,0]`, extensions `[317,8200,7808]` and19 decode tails. These are context request prefixes, not the much larger mixed decode history arrays. The last context is prefix3840/extension4045 with127 decode tails.

Thus the all-context-fresh guard excludes most fresh-token work because a chunk continuation shares its call. Tiny-prefix support alone cannot cover most excluded calls. Partitioning or packing requires separate kernel/correctness validation; this observation makes no claim about their speed.

`prefix-summary.json` contains exact histograms, query-token-weighted prefix counts, all66 measured context tuples and flush evidence. Raw per-PID JSONL, exact commands, scripts, source hash and benchmark count checks are retained. Nominal benchmark input counts differ from context query totals because requests undergo chat formatting/retokenization and cached-prefix handling. The owned server and driver stopped after completion.
