# Production M8 LM-head full GSM8K C8 pair

Control scores **1213/1314**, candidate **1219/1314**. Eight questions are correct only in control and fourteen only in candidate; exact dataset rows are in comparison.json. This pair does not establish numerical equivalence or isolate LM-head accuracy effects.

Both use normal public startup tuning, BF16 weights/query/KV, TRT HND page32 and mixed chunk16384, temperature0, top_p1, max output2048, concurrency8. The identical dataset hash and first-five-few-shot/remaining1314 selection are retained in metrics. Control is frozen M4 production; candidate is exact M8 two-file commit5babd15f4f. Full source manifests are checked before launch. The evaluator's hardcoded source_revision466c9e field is stale and does not identify the runtime source; command paths and verified source manifests are authoritative.

**Confound:** public M128 down projection chooses tactic1 in control versus4 in candidate. Gate/up1 and M4 LM-head2 match; candidate M8 chooses2. Logged prefill intervals [new-token, new-token + running-requests] include128 in48 control and47 candidate cases. These conservative intervals do not prove exact M128 execution, but cannot exclude the unrelated numerical path. First-occurrence dispatch observations are insufficient to resolve later execution. Do not attribute the six-question aggregate increase to M8 or claim a matched-tactic accuracy gate.

Candidate startup READY, actual M8 capture and graph8 replay are recorded by an observer that changes no algorithm. Prior two-seed production B8 checks had bitwise matching logits/tokens; this full evaluation is complementary evidence with the stated confound.

Drivers507970/507971 and servers507973/507972 completed normally and were absent before evidence retrieval. Both GPUs are free. Raw archive preserves scripts, logs, cache JSON, metrics and gzip HTML reports; compiled cache binaries are excluded. Reports were compressed after evaluation without changing their contents. No additional rerun or PR promotion was performed by this worker.
