# Isolated M16 projection model gates

Frozen prefix-production source for both variants; no LM-head M16 optimization. QKV-only and down-only hooks were tested separately on GPUs 2 and 3. Existing paired graph protocol retains both graph backends and all tensors. Each candidate recorded 36 actual layer captures and 72 eager calls. Fresh identical synthetic 8K prompts precede each numerical run. Free-running 16-token sequences differ: QKV 56/256, down 24/256. First-decode logits compare matching input histories; later logits do not, so later NRMS is not an isolated numerical-error estimate.

QKV first-decode NRMS 0.03862, max absolute difference 2.28125; retained fixed-state graph timing regressed from approximately 4.9588 to 4.9811 ms, −0.451% across six pairs. Rejected; no further QKV tests.

Down first-decode NRMS 0.00753, max absolute difference 0.34375; retained fixed-state graph timing improved from approximately 4.9532 to 4.9257 ms, +0.561% median paired gain. This is not numerical equivalence or serving validation. A separate teacher-forced real-text numerical check is required before advancement. No combined variant, production changes or serving run.

The six alternating 256-replay blocks are explicitly synthetic fixed-state timing, not 256 growing-context decodes. Fifteen real-step event times are separate in report.json. One warmed profiler trace per variant is diagnostic only. Files named lmhead-trace retain the previous harness name but profile the selected QKV/down projection, not the vocabulary projection. Large numerical tensors remain in the remote run directories. Worker PIDs 464910 and 464911 exited normally.
