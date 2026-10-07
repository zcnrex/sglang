# Batch-1 gate/up accuracy screen

Two fresh servers use the same audited 5f60fb8b67-equivalent source. Control runs the production path; candidate replaces only M1 gate/up with FlashInfer `run_direct_dense`, using `default_tactic(1, 19456, 2560)`. BF16 weights, queries and KV cache remain unchanged. No production files are modified.

The first 128 held-out GSM8K questions (dataset rows5–132, after5shots) use concurrency1, temperature0, top_p1 and max2048 tokens. Both score120/128 (93.75%). Control alone answers row122 correctly; candidate alone answers row116 correctly. Completion-token totals differ:23551 control and21925 candidate. This bounded screen does not establish numerical equivalence or full-dataset accuracy.

The candidate records36 distinct gate/up weight pointers,36 captured M1 calls, and actual batch1 graph replay. Independent fresh server states avoid the previous shared-model graph-swapping harness concern. Source audit checks Python/CUDA/header files against466c and finds only the two previously committed public-Lt integration changes, with frozen hashes recorded in `source-audit.json`.

Each server uses a fresh isolated tuning cache and warmed shared compiled-kernel caches. Exact commands, metrics, row outcomes, logs, marker and tuner files are retained. Evaluation times are not performance claims. Both owned servers stopped after completion. Remote root: `/root/qvl/experiments/m1-gateup-accuracy`.
