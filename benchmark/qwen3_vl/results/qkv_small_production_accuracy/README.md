# Full C1/C2 normal-startup production accuracy

Frozen M4 production control versus small-batch extension5a009dde20. No algorithm replacement. Same1314 held-out GSM8K rows5–1318 after five shots; temperature0/top_p1/max2048; C1 orC2 as recorded. BF16 model/query/KV, TRT HND32, mixed16384, cap128 explicit20 buckets, pool1,600,000, serverseed0. Full row outputs and HTML gzip retained, prompt hashes aligned throughout.

C1:1217→1217/1314 (92.61796% both), no correctness or extracted-answer differences;3 response differences. Measured B1 graph/decode counts249310/249308; M128 EXTEND6/6. Startup M128 gateup/down differs2/1 versus1/4; M4/M8 vocab2/2 common.

C2:1216→1213/1314 (92.54186%→92.31355%),8 control-only/5 candidate-only correct,29 answer/213 response differences. Startup tactics common1/1/2/2. Measured graph/decode B1 counts7193/7420 and B2 counts119147/116699; M128 MIXED4/3. This normal result is preserved as a three-question deficit; it is not dismissed or attributed causally to scheduling or kernel arithmetic. Separate bounded fixed-input diagnosis is archived independently, with no full accuracy repeat.

Counters record real ModelRunner decode input rows and graph replay sizes; they agree exactly. Candidate startup proves public B1/B2 capture36 with stride128. Post-run4230 Python source hashes match admitted files; full10,245 source manifests are retained in model evidence. Evaluator source_revision denotes its own checkout, not the serving source.

Drivers C1 control703301/candidate703302 GPUs2/3; C2 control703303/candidate703304 GPUs4/5; all terminal. Each worker used isolated normal public startup cache. Original scripts/observer, commands, server_info, cache/tactic provenance, rows and compressed HTML remain complete. Extracted Python files use.py.txt. Raw archive is split into ordered1MiB parts with SHA256 reconstruction manifest; complete original at /tmp/qvl-qkv-small-production/accuracy-original.tar.gz and remote /root/qvl/experiments/qkv-small-production-accuracy.tar.gz.
