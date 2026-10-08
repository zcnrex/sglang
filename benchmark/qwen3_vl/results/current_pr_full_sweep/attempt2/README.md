# Exact PR3016f3b591 full concurrency snapshot

All eight runs completed with exact request/input/output counts and unchanged full source hashes. Only C1 reaches the historical handoff baseline ×1.10 target in this sweep. This is one consistent current-head snapshot across distinct GPUs, not a same-device causal optimization or layout comparison. The C128 replicate differs by 0.367991% and is reported separately. No new full accuracy evaluation was run.

| C | GPU | N | Warm | Output tok/s | TTFT mean ms | median ms | p95 ms | p99 ms | TPOT median ms | Throughput / target |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 30 | 64 | 393.564651 | 85.927063 | 87.420941 | 92.606450 | 95.294270 | 2.457600 | 107.1216% |
| 4 | 1 | 40 | 64 | 1229.733480 | 217.889747 | 236.638899 | 287.758646 | 293.720500 | 3.022266 | 99.3724% |
| 8 | 2 | 80 | 64 | 1908.324881 | 354.407505 | 355.149804 | 539.989558 | 549.555896 | 3.844728 | 97.7926% |
| 16 | 3 | 80 | 64 | 2583.188736 | 607.949180 | 644.920366 | 896.671634 | 1058.927951 | 5.576778 | 93.9717% |
| 32 | 4 | 160 | 64 | 3141.890372 | 823.742236 | 804.626946 | 1673.885905 | 2100.255309 | 9.402017 | 92.6156% |
| 64 | 5 | 320 | 64 | 3542.952400 | 1049.678303 | 818.600181 | 3234.144603 | 4143.975004 | 17.267996 | 92.1828% |
| 128 | 6 | 640 | 128 | 3833.221372 | 1421.791746 | 600.927212 | 6725.147851 | 8616.825961 | 32.638741 | 91.3433% |
| 128 rep | 7 | 640 | 128 | 3847.327293 | 1462.023805 | 592.384225 | 6769.086854 | 8594.281213 | 32.532586 | 91.6794% |

Targets for C1/4/8/16/32/64/128 are367.4/1237.5/1951.4/2748.9/3392.4/3843.4/4196.5 tokens/s, from the supplied historical baselines. These target comparisons are private goal tracking, not controlled same-configuration comparisons.

## Immutable source and admission

Exact clean code PR head **3016f3b5917650199c12b47195b9dc1be4c26e4e**, transferred using COPYFILE_DISABLE=1 git archive. All10,245 tracked files match before worker launch and after all benchmarks; no AppleDouble metadata or other extra source files. Snapshot /root/qvl/sglang-current-pr-full-sweep-3016f3b591 is separate from the original failed snapshot. Each worker verifies the complete manifest. Original attempt1 and cap128-model proof remain separate.

All workers verify server-reported KV capacity1,600,000, identical recorded global server configuration, and the explicit20 decode graph buckets [1,2,4,8,12,16,24,32,40,48,56,64,72,80,88,96,104,112,120,128]. Public QKV fusion succeeds for all36 layers with actual positions stride(128,1), in both eager warmup and capture, before benchmark release. Observer wrappers only record successful dispatch and CPU shape counters; no algorithm replacement, fixed tactic or arithmetic change.

## Reproducible protocol

Same model snapshot ebb281ec70b05090aa6165b016eac8ec08e71b17, BF16 weights/query/KV/output, TP1, TRT HND page32, mixed chunk16384, prefill graph disabled, decode cap128, cutedsl BF16 GEMM, pinned1.6M KV tokens. Every server uses seed0; the established sgl-bench client uses explicit seed42 (matching its previous default). Normal public startup tuning uses fresh isolated per-worker caches; compiled caches are shared. GPU0–7 and ports33000–33007 are disjoint. Client requests nominal8192 input/1024 output, random range1, N30/40/80/80/160/320/640/640 and warm max(64,C), with flush after warmup and --output-details. Each run asserts N completed, N×8192 input tokens and N×1024 output tokens.

Exact per-worker server commands and environment are in launch.json, and clients in bench-command.json. A representative server command (substitute the recorded model path/port) is:

```sh
SGLANG_USE_HND_KVCACHE=1 python -m sglang.launch_server \
  --model-path /root/qvl/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17 \
  --dtype bfloat16 --kv-cache-dtype bfloat16 --bf16-gemm-backend cutedsl \
  --attention-backend trtllm_mha --page-size 32 --enable-mixed-chunk \
  --chunked-prefill-size 16384 --cuda-graph-backend-prefill disabled \
  --cuda-graph-max-bs-decode 128 \
  --cuda-graph-bs-decode 1 2 4 8 12 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 \
  --max-total-tokens 1600000 --enforce-disable-flashinfer-allreduce-fusion \
  --random-seed 0 --host 127.0.0.1 --port 33000
```

Client executable /root/qvl/venv-bench/bin/sgl-bench serve, implementation hash recorded in client_provenance.json. Example C1 arguments:

```sh
sgl-bench serve --backend sglang-oai-chat --base-url http://127.0.0.1:33000 \
  --model MODEL_PATH --dataset-name random --random-range-ratio 1 \
  --num-prompts 30 --max-concurrency 1 --warmup-requests 64 --flush-cache \
  --random-input-len 8192 --random-output-len 1024 --seed 42 \
  --output-details --output-file bench.jsonl
```

## Startup choices and measured shapes

| GPU / C | M128 gate/up | M128 down | M4 vocabulary | M8 vocabulary | Measured M128 calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 / 1 | 1 | 1 | 2 | 2 | 0 |
| 1 / 4 | 1 | 4 | 2 | 2 | 0 |
| 2 / 8 | 1 | 4 | 2 | 2 | 0 |
| 3 / 16 | 1 | 4 | 2 | 2 | 0 |
| 4 / 32 | 1 | 1 | 2 | 2 | 0 |
| 5 / 64 | 1 | 4 | 2 | 2 | 0 |
| 6 / 128 | 1 | 4 | 2 | 2 | 4796 |
| 7 / 128 | 1 | 1 | 2 | 2 | 4797 |

Measured M128 calls are DECODE only. The two C128 workers have different normal-public M128 down tactics (4 versus1), so their difference is not an isolated device replicate with identical tactic. All raw startup caches and measured epoch shape counts are retained. Epoch0 is startup/warmup; epoch1 begins with the benchmark cache flush and ends with a post-benchmark flush.

## Telemetry and limitations

Host has256 logical CPUs,256 affinity CPUs, no cgroup CPU quota. 211 samples captured /proc aggregate CPU, load and memory, process PID/PPID/CPU/RSS, and each GPU clock/power/temperature/utilization/memory. Peak one-minute load 32.43; aggregate CPU peak 18.893% during startup/warmup and 8.448% across the union of measured windows. Sampling is roughly one second plus query overhead.

| GPU | Startup/warmup peak process-tree CPU % | Measured peak process-tree CPU % | Measured summed RSS GiB | Measured SM MHz min/median/max | GPU utilization min/median/max % |
| ---: | ---: | ---: | ---: | --- | --- |
| 0 | 1856.4 | 576.9 | 31.663 | 1672.0/2032.0/2032.0 | 0.0/100.0/100.0 |
| 1 | 1719.1 | 583.1 | 31.665 | 1380.0/2032.0/2032.0 | 0.0/100.0/100.0 |
| 2 | 1943.5 | 548.1 | 31.702 | 1275.0/2032.0/2032.0 | 0.0/100.0/100.0 |
| 3 | 1856.8 | 637.4 | 31.705 | 1335.0/2032.0/2032.0 | 0.0/100.0/100.0 |
| 4 | 2305.4 | 639.8 | 31.764 | 1290.0/2032.0/2032.0 | 0.0/100.0/100.0 |
| 5 | 1754.5 | 523.1 | 31.795 | 1327.0/2032.0/2032.0 | 0.0/100.0/100.0 |
| 6 | 1581.1 | 553.8 | 31.832 | 1230.0/2017.0/2032.0 | 0.0/100.0/100.0 |
| 7 | 1239.6 | 605.7 | 31.825 | 1312.0/2032.0/2032.0 | 0.0/100.0/100.0 |

Process CPU values are sampled ps lifetime percentages (100%=one logical CPU), not instantaneous per-interval CPU utilization. Summed RSS can double-count shared pages and is not PSS. GPU utilization is sampled, not proof that no host/GPU interference occurred. Foreign GPU PID audits are empty immediately before and after the sweep; this alone does not prove absence of transient interference. Eight-way startup and differing run durations alter the amount of concurrent host work over time. Full raw telemetry supports further diagnosis without asserting causality.

## Frozen evidence

original.tar.gz preserves complete original scripts/logs/JSONL per-request details, observer records, server_info, commands, cache/provenance and telemetry. The immutable source tar remains remote and its hash/manifest are retained; source.tar.gz and generated __pycache__ are excluded. Readable Python copies end .py.txt. analysis.json adds summaries without rewriting raw benchmark files. Driver656496 and all workers/servers are terminal; no observation timeout caused a restart.
