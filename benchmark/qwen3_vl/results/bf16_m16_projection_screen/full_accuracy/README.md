# Full accuracy with normal public startup tuning

Control scored **1213/1314**, down-only candidate **1222/1314**. Paired outcomes contain 7 control-only and 16 candidate-only correct rows, listed in summary.json. All 1314 rows are accounted for, exact indices 5–1318 after five examples, dataset SHA256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`. Greedy chat evaluation, concurrency 16, max 2048 tokens; scorer and per-question reports are retained.

This score difference is **confounded by public startup autotuning**. Control selected M128 gate/up tactic 1 and down tactic 4; candidate selected gate/up tactic 2 and down tactic 1. Both selected M4 LM-head tactic 2. Runtime observers recorded 9 control and 12 candidate M128 mixed forwards, so these choices cannot be dismissed as unused startup differences. No numerical equivalence or isolated accuracy benefit is claimed.

Both use frozen `/root/qvl/sglang-lmhead-production` (committed 5e1601731b equivalent, all 4229 source hashes matched), BF16 weights/queries/KV/output and the same standard recipe. Only the external candidate down hook adds split-K(128,16,4,5) at M16. READY, eager/captured down dispatch and actual decode 16 forward markers establish candidate execution. The observer records the first decode 16 marker, not the total count of such forwards. No QKV optimization, production edit or serving run was added.

GPU 0 control driver 478573 / server 478576 and GPU 1 candidate driver 478574 / server 478575 exited normally. Raw-original.tar.gz preserves unnormalized reports/logs/cache records. No retry was automatically launched after this result. A separately authorized matched-startup diagnostic is needed to isolate the down numerical effect.
