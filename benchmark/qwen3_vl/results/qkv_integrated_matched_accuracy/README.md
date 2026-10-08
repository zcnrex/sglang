# Integrated production GSM8K: matched-startup diagnostic

Control **1216/1314 (92.5419%)**, candidate **1213/1314 (92.3135%)**, net −3. All 1314 held-out prompts align (dataset rows 5–1318, five-shot). There are 13 control-only correct and 10 candidate-only correct rows, 38 extracted-answer differences, and 388 response differences. This result does not demonstrate an accuracy improvement. Remaining disagreement cannot be attributed to the common startup tactic mismatch removed by this diagnostic; no causal attribution to fusion or scheduling is established.

This is an external startup diagnostic, not a production tuning policy. Both arms load the normal control cache and validate actual cache hits plus installed valid-tactic registry membership for M128 gate/up=3, M128 down=4, and M4/M8 vocabulary=2. The override temporarily exits tuning mode only for those four exact targets. Original search_cache is restored immediately after ordinary init_cuda_graphs and before evaluation. Both policy-restored.json files assert restoration. Every actual M128 forward is recorded: six calls in each arm. Candidate public fusion captures all 36 layers.

Frozen M16 control versus exact integrated production overlay; no algorithm replacement or source edits. BF16 weights/query/KV/output, HND page 32, TRT, mixed chunk 16384, C8/max2048, decode graph cap 8 and KV capacity 1,600,000. The established 1314-row evaluation and five-shot examples are unchanged. Drivers 641633/641634 are terminal. Control/candidate total generated tokens are 243044/245973; evaluation elapsed 130.342/130.267 seconds, not a controlled performance comparison.

The separate normal-startup result remains 1220 versus 1218 with active gate/up tactics 3 versus 1. No further accuracy repeat was run. This diagnostic supplements the bitwise standalone/short-model gates and one-image smoke, rather than superseding any of them.

Raw scripts, full source manifests, startup caches, validation/restoration proofs, all per-question rows, compressed HTML and full-row comparison are preserved. Python readable copies end .py.txt. Uncompressed report.html is excluded from the archive because report.html.gz retains identical content. `audit.py.txt` reproduces prompt alignment and row disagreement extraction.

The raw archive is split byte-for-byte into `original.tar.gz.part*`; concatenate in lexical order. `archive-parts.json` records original and part hashes.
