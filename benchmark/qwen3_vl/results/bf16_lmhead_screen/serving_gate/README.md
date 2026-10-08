# C4 serving crossover

Same frozen prefix-production source, BF16 weights/KV/query, HND page 32, mixed chunk 16384. Only candidate uses the external public cublasLt LM-head hook for M4 and the actual tied embedding pointer. Startup tunes and verifies its public serialized cache record before publishing READY; unsupported calls retain the original method. BF16 output and downstream logits processing remain unchanged. No production edits or full accuracy evaluation.

Two phases swap control/candidate across GPUs 0 and 1. Each worker uses fresh isolated tuner state and shared compilation caches. C4, N40, warmup 64, fixed 8192 input / 1024 output, flush before measurement. All four runs completed exactly 40 requests, 327680 input tokens and 40960 output tokens. Each candidate logged READY tactic 2, actual M4 graph capture and graph 4 replay. All owned servers terminated normally after PHASE B DONE.

| GPU | Control / candidate output tok/s | Gain | TTFT control / candidate (ms) | TPOT control / candidate (ms) |
|---|---|---|---|---|
| 0 | 1238.213 / 1241.900 | +0.2978% | 243.299 / 219.910 | 3.00295 / 3.01221 |
| 1 | 1227.944 / 1229.569 | +0.1324% | 225.586 / 216.768 | 3.04582 / 3.04839 |

Geometric mean throughput gain is +0.2150%, both pairs positive. Median TPOT slightly increased; throughput and latency distributions are different summaries and should not be conflated. Two short pairs are a positive gate, not a precise effect estimate or achievement of the overall target. Actual model logits were checked bitwise in the preceding model gate, not asserted from this serving throughput run. Raw probes and public tuner records are retained.

Normal M128 projection tuning remains enabled and its startup records are archived. Closed-loop C4 rules out a 128-request decode batch, but does not by itself rule out a 128-token prefill. This run did not retain epoch-specific M128 dispatch counts, so measured M128 nonexecution is unproven. No private tuning policy or forced tactic was introduced. Source audit covered all 4229 expected Python files with zero mismatches. Control and candidate use exactly the same source. Parent PID 443043; phase A server PIDs 443048/443049; phase B server PIDs 446023/446024.

An earlier instrumentation attempt (parent 436963, servers 436968/436969) failed before benchmark: marker accessed graph bs before load_batch, and startup hook targeted an entry bypassed by normal initial capture. No result from that attempt is used. Its logs and original source/hook hashes are under raw/lmhead-serving. Corrected hook uses init_cuda_graphs and post-execute marker. The failed shared hook file was replaced only after all owned processes exited; archived .py.txt is the corrected hook.

## Exact final hook and runner hashes

- `qvl-lmhead-serving.py`: `5f91b269bdaad95772a0af81fe83f39fc83211a5f0f256a979c51aaa5db2e524`
- `qvl-lmhead-serving-hook.py`: `83f1d482aa60252df90282ec4fce3a656170ffdaf6459af5f1c290d162324d34`
