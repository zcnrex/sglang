# Full production cast serving gate: inconclusive

The full 640-request crossover does not establish a robust serving gain. Same-GPU effects are −1.07604% and +1.13439%; geometric mean is only +0.02307%. Both variants slow in phase B, by more than the gain observed in the short gate. This result supersedes any promotion conclusion based on the positive short gate. Public production promotion is rejected; retain the implementation commit and evidence for reproducibility, with no further runs.

Control is frozen M16; candidate is the same source plus only the committed cast patch. C128/N640, warmup 128, nominal 8192 input/1024 output, flush before measurement, cap 128 and 1,600,000 KV tokens. Same matched-startup public tactics 1/1/2/2 as the short gate, validated cache hits/registry membership and restored policy before evaluation. No algorithm replacement hook; candidate observations wrap the actual public kernel unchanged. Both 64/128 cast captures preserve caller FP32 buffer identity, and measured B128 decode is observed. All four runs complete 640 requests, 5,242,880 input and 655,360 output tokens exactly.

| Phase/GPU | Variant | Output tok/s | Median TTFT ms | Median TPOT ms |
| --- | --- | ---: | ---: | ---: |
| A/0 | Control | 3829.191 | 741.189 | 32.63299 |
| A/1 | Candidate | 3833.025 | 747.886 | 32.57616 |
| B/0 | Candidate | 3787.987 | 607.079 | 32.73101 |
| B/1 | Control | 3790.031 | 605.594 | 32.75983 |

`summary.json` retains complete mean/median/std/p90/p95/p99 TTFT, TPOT and E2E statistics. For example, GPU 0 mean TTFT worsens 1486.720→1526.907 ms while its median improves; GPU 1 mean improves 1503.936→1475.988 ms while its median worsens. Medians alone do not support a latency-improvement claim.

Read-only telemetry/log audit is in `phase_audit.json` and `TELEMETRY.md`. Memory clock and median SM clocks are stable; phase B has a few lower-power samples and slightly different logged batch grouping. Neither observation identifies a cause. No background CPU/process trace was recorded, and no concrete invalidating failure was found. Do not attribute the common phase drift to cast, thermal throttling or scheduling solely from correlations.

Parent 589912 and servers 589919/589920/609255/609256 terminated normally. The raw tar preserves original bytes. Exact scripts and summaries are archived separately from prior short/model gates. `summarize.py.txt` and `phase_audit.py.txt` regenerate analyses when copied to `.py` beside the data. `source_audit_prior_launch.json` is the previously verified immutable source audit, not a new full-run audit. No new GSM evaluation or retry was launched.
