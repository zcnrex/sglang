# Matched-startup full accuracy diagnostic

Control scored **1221/1314**, down-only candidate **1220/1314**, with 12 control-only and 11 candidate-only correct rows (listed in summary.json). Exact rows 5–1318, five-shot examples 0–4, concurrency 16, greedy max 2048 tokens and the same scorer as the normal public-tuning run. Dataset SHA256 is `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`.

This is an **external matched-startup diagnostic**, not a production tuning policy. Both isolated caches were seeded from the earlier normal control cache. During startup only, the prior validated diagnostic pattern temporarily sets the tuner's internal tuning flag false under its lock for exactly three BF16 records, requires a cache hit and checks the selected tactic against each worker's actual public runner registry. Both verified M128 gate/up tactic 1, M128 down tactic 4 and M4 LM-head tactic 2; each registry listed valid tactics 0–7. The original search_cache method was restored after original graph initialization and before evaluation. Exact policy records, restoration events and final serialized caches are retained.

Both use identical frozen lmhead-production source (5e1601731b equivalent, 4229 hashes matched), BF16 precision and unchanged standard configuration. Only the candidate external hook changes M16 down to split-K(128,16,4,5). Candidate readiness, eager/captured dispatch and actual decode-16 forward markers are recorded. Actual M128 mixed forward events numbered 4 control and 5 candidate, now with identical tactics. No QKV change or production edit.

The matching removes the identified startup-tactic confound. It does not make this finite, asynchronously scheduled evaluation deterministic or prove distributional equivalence. The earlier normal run's +9 score is not an isolated accuracy benefit; this matched run is down by one. The small subset and teacher-forced results remain archived rather than replaced.

The first attempt stopped before any evaluation because the external assertion checked validation records at entry to init_cuda_graphs; startup autotuning occurs inside that call. Failed drivers 488312/488313 and servers 488314/488315 terminated. The corrected assertion follows original initialization, and ran in a separate directory. Corrected drivers 491611/491612 and servers 491613/491614 all exited normally. No results from the failed attempt were reused. HTML reports are losslessly gzipped; raw-original.tar.gz preserves original files.

## Exact diagnostic script hashes

- `qvl-m16-matched-hook.py`: `789c5766ce583e8da55c31109a32696437c9acb601458a77b600e791799d1b7f`
- `qvl-m16-matched-hook-fixed.py`: `62442d3f9583ff0bfe54d77d0a7b327aec912623a47570170d49b18f93ed25ff`
- `qvl-m16-matched-gsm-fixed.py`: `13b24e44dd09868b6bdafded26d496f85b0d4e513a6071a8d6845c4c7ef96596`
- `qvl-m16-matched-launch-fixed.py`: `c0e31d466f45651014939a1c26c688c1f4d1e56150a8aafff0059e729433b8f2`
