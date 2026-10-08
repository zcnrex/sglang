# Integrated production GSM8K: normal public startup

Control1220/1314 (92.8463%), candidate1218/1314 (92.6941%), net−2. All1314 held-out prompts match in dataset order (rows5–1318, five-shot). There are13 control-only correct,11 candidate-only correct,42 extracted-answer differences and379 response differences. Full per-question outputs are retained as rows.json and compressed report.html.gz.

This pair does not isolate fusion accuracy: normal public startup selected active M128 gate/up tactic3 for control and1 for candidate. M128 MIXED forward occurs in both; down4 and M4/M8 vocabulary2 match. Preserve this ordinary production behavior as evidence; a separate startup-matched diagnostic is authorized and must not replace it.

Frozen M16 versus exact integrated overlay, BF16 weights/query/KV/output, C8/max2048, cap8/pool1.6M, normal isolated caches and existing five-shot harness. Candidate captures all36 public fused layers. Source manifests, evaluation commands, startup cache and observers are retained. Drivers636182/636183 are terminal. Raw archive excludes uncompressed HTML but includes its gzip copy; no per-question content is discarded.

The raw archive is split byte-for-byte into `original.tar.gz.part*`; concatenate in lexical order. `archive-parts.json` records original and part hashes.
