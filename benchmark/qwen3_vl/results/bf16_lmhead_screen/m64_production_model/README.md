# Production M64 vocabulary two-seed model proof

Both seeds 123/124 pass bitwise prefill, first/fifteenth decode logits and all 16 greedy tokens for 64 sequences. Input length 128 is intentionally short; this is a numerical/path gate with no timing claim or full accuracy equivalence.

Control is frozen /root/qvl/sglang-lmhead-m32-production; isolated candidate /root/qvl/sglang-lmhead-m64-production copies it then overlays exactly logits_processor.py and runner/flashinfer_autotune.py obtained with git show 96ef1b658e. All 4229 runtime Python files plus two inherited .claude benchmark skill scripts are audited. Initial assertion expecting 4229 total .py files failed on those two known non-runtime files; corrected classification passed before any GPU run. Their hashes and other copied non-Python assets are retained. Exactly two overlay hashes differ, matching immutable commit bytes. No future working-tree source was used.

Actual candidate READY includes [0,64,151936,2560]; observer records optimized exact M64 capture and graph 64 replay. Normal public startup tuning is retained in separate fresh caches, with exported records. The observer returns original results and never changes dispatch. BF16 weights/query/KV, TRT HND page 32, mixed 16384, prefill graphs disabled. Model two-seed outputs remain remotely saved; compact comparison records verify every retained tensor bitwise.

Control worker 572893 on GPU4 and candidate 572894 on GPU5 completed. Remote evidence /root/qvl/experiments/lmhead-m64-production/model; original compact archive /tmp/lmhead-m64-production-model.tar.gz retained locally/remotely. C64/N320 serving is a separate conditional gate; no full GSM launched.
