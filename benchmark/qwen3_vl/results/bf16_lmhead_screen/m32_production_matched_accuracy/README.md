# Production M32 matched-startup GSM8K C32 diagnostic

Both control and candidate score **1221/1314**. Eleven questions are correct only in each arm; exact rows are retained in comparison.json. Equal aggregate score is not output or accuracy equivalence. This is one paired full evaluation, with potentially different dynamic batching schedules.

Control is frozen M8+M16 production; candidate is exactly the two-file M32 extension d0d528db4f. Complete source manifests are checked before loading. Both retain BF16 weights/query/KV, TRT HND page32, mixed chunk16384, graph max batch32 and explicit KV capacity1,600,000. Server-reported capacities are asserted. Sampling uses temperature0, top_p1, max output2048, concurrency32; all1314 evaluation rows share the same dataset hash and five-shot selection. Evaluator source_revision466c9e is stale; verified runtime manifests and paths are authoritative.

An **external startup-only diagnostic policy** holds only common M128 gate/up1, M128 down1 and M4/M8 vocabulary2. Seed records come from valid normal-control serving startup. Each held lookup proves a cache hit and registered valid tactic, temporarily suppresses retuning under the tuner lock, and immediately restores the prior tuning flag. Original search_cache is restored after init_cuda_graphs with all four validations asserted. Candidate M32 remains outside this policy and normal public autotuning selects2. No inference algorithm or production source is replaced.

Actual forward instrumentation records **18 control and12 candidate MIXED forwards with exactly128 input rows**. Candidate READY includes M32, optimized capture is observed and graph32 replay is proven. Thus common-tactic matching covers real M128 execution, while normal dynamic batching remains unconstrained.

Drivers551131/551132 and servers551135/551136 completed successfully and were absent before archival; GPUs2–3 are released. Raw archive retains scripts, manifests, public cache JSON, logs, metrics and gzip HTML reports. This diagnostic does not strengthen the weak production serving magnitude reported separately, and no further performance repeat or PR promotion was performed by this worker.
