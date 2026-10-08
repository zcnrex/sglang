# Integrated B4 QKV production serving crossover

Exact PR3016f3b591 control versus frozen three-file production M4 extension, now committed privately as5add178cd8. Public dispatch only; observer does not replace algorithms. Two phases swap GPU2/3 assignments. C4/N40, warm64 then flush, nominal8192/1024 (range1), clientseed42/serverseed0; BF16 model/query/KV/output, TRT HND page32, mixed chunk16384, graph cap128 with identical20 buckets, pool1,600,000. Both server capacities and bucket lists were asserted. Each run completed40 requests,327680 input and40960 output tokens. Raw per-request TTFT/ITL/output arrays are retained via --output-details.

Same-GPU throughput gains: GPU2 +3.640925%, GPU3 +3.920578%; geometric gain +3.780657%. Both candidate rates exceed the historical C4 target1237.5 tokens/s. This is a two-pair crossover, not proof of a universal gain.

| Phase/GPU | Arm | Output tok/s | TTFT mean ms | TTFT median ms | TTFT p95 ms | TTFT p99 ms | TPOT median ms |
|---|---|---:|---:|---:|---:|---:|---:|
| A/gpu2 | control | 1235.126360 | 200.903543 | 245.577815 | 262.707501 | 269.918913 | 2.997343 |
| A/gpu3 | candidate | 1298.679674 | 201.368405 | 239.781396 | 256.744191 | 259.952196 | 2.855136 |
| B/gpu2 | candidate | 1280.096383 | 198.592478 | 245.434762 | 263.950126 | 266.560236 | 2.891149 |
| B/gpu3 | control | 1249.684809 | 202.219255 | 239.879353 | 258.500396 | 259.651023 | 2.966876 |

TTFT mean and median improve slightly on both devices; p95 increases1.243ms on GPU2 while p99 increases0.301ms on GPU3. Full TPOT means/tails are in analysis.json. No broad latency regression is indicated by these small samples.

Normal public startup used isolated fresh tuning caches. M128 gate/up/down tactics were1/1 (A/GPU2),2/4 (A/GPU3),1/4 (B/GPU2),1/4 (B/GPU3). M4/M8 vocabulary tactics were2/2 everywhere. Measured epoch1 M128 activity was zero in all four runs, as was warm epoch0. Tactic differences therefore had no observed M128 participation during these benchmarks. Candidate readiness asserted successful shape-qualified B4 eager and graph-capture calls at all36 layers with actual positions stride(128,1), resolving the prior model layout-observer limitation.

Driver672994 and all servers completed. Original archive retains original scripts/logs/cache reports; extracted Python filenames end in.py.txt for evidence only. Large readable JSONL files are losslessly gzipped in extracted copy; original archive and remote originals remain complete. Remote root: /root/qvl/experiments/qkv-m4-production/serving. Source manifests and exact launch commands are retained per worker.
