# Integrated B4 production GSM8K accuracy

Normal public startup: control1217/1314 (92.61796%), candidate1220/1314 (92.84627%). Same1314 held-out dataset rows5–1318 after five few-shot examples, C4, temperature0/top_p1/max2048. Prompt hashes align for every row. Ten control-only correct,13 candidate-only correct,43 extracted-answer differences and317 response differences. This single result does not establish a causal accuracy improvement or broad numerical equivalence.

Control exact PR3016f3b591; candidate frozen three-file M4 extension now privately committed5add178cd8. Same BF16 model/query/KV, TRT HND page32, mixed chunk16384, serverseed0, explicit20 graph buckets capped128 and1,600,000 KV capacity; server_info and source hashes retained. Public dispatch only, no algorithm replacement. Candidate admission confirms B4 successful eager/capture36 with actual positions stride(128,1). Startup tuning is normal and isolated per worker.

Common gate/up tactic1 and M4/M8 vocabulary2/2; M128 down differs control1/candidate4. Actual M128 MIXED calls9/10. These tactic and scheduling differences limit attribution of per-question changes. No matched full-accuracy rerun was performed. Kernel and separate fixed-input short-model bitwise gates are retained independently.

Control driver680869/server680872 GPU2; candidate driver680870/server680871 GPU3. Both terminal. Runtime-source post-audit4230 Python files matches the admitted manifests. Full original source manifests are in the separately frozen model gate. Evaluation harness metrics source_revision describes the evaluator checkout, not the serving code; authoritative server hashes are the retained manifests.

All per-question rows and compressed HTML reports are retained. Large extracted readable files are losslessly gzipped; original archive is split into ordered1MiB parts with SHA256 manifest and reconstruction command. Complete originals remain under /root/qvl/experiments/qkv-m4-production/accuracy; local raw archive remains /tmp/qvl-qkv-m4-production/accuracy-original.tar.gz. Observer source and exact launch/evaluation commands are included.
